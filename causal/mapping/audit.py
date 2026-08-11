"""Aggregate-only SQL diagnostics for a registry entry's OMOP mapping.

Every function here returns a BigQuery SQL string that summarizes coverage,
data quality, vocabulary hygiene, or cohort-entry characterization for one
`causal.mapping.registry.ExtractionPlan` -- never a patient-level row. This
is a diagnostic layer, not a second execution path: nothing here runs a
query itself (see `causal.orchestrator.workflow`/`causal.mcp_client.
OMCPClient` for the existing governed execution path -- these strings reach
a database only through OMCP's `Causal_Select_Query`/governance gate, not
through the separate `QueryValidationAgent` pre-check;
`WorkflowManager.run_mapping_audit_mcp` documents why: its regex-based
final-projection check cannot tell `COUNT(DISTINCT person_id)` apart from a
bare `person_id` projection, and several diagnostics here legitimately do
the former). If a real BigQuery/OMCP connection is not available, these
builders are still fully unit-testable on their SQL shape alone (see
`tests/test_mapping_audit.py`), the same way `causal.queries.feasibility`
is tested without executing anything.

`orphan_concepts_sql`, `index_event_breakdown_sql`, and `incidence_rate_sql`
are a simplified adaptation of three diagnostics from OHDSI's
CohortDiagnostics methodology (Rao GA, Shoaibi A, Makadia R, et al.
"CohortDiagnostics: Phenotype evaluation across a network of observational
data sources using population-level characterization." PLOS ONE.
2025;20(1):e0310634. doi:10.1371/journal.pone.0310634). Two simplifications
worth knowing before reading a result: orphan-concept search here is a
`concept_ancestor` sibling proxy (concepts one hop from the same parent as
an approved root), not the paper's non-standard source-code mapping walk --
this codebase has no source-code frequency table to walk; and incidence
rate is a single unstratified rate over the whole database's
`observation_period` person-time, not the paper's age/sex/calendar-year
strata.
"""

from __future__ import annotations

from causal.config import BigQueryDatasets
from causal.mapping.registry import ExtractionPlan, VALUE_DOMAINS
from causal.queries.feasibility import (
    DOMAIN_TABLES,
    ConceptSetSpec,
    concept_ids_literal,
    expanded_concepts_cte,
    unnest_ids,
)


def _safe_divide(datasets: BigQueryDatasets, numerator: str, denominator: str) -> str:
    """Portable null-on-zero-division. BigQuery keeps its existing
    `SAFE_DIVIDE` (unchanged SQL for that backend, which `tests/
    test_mapping_audit.py` asserts on directly); DuckDB has no `SAFE_DIVIDE`,
    so it gets the equivalent `CASE WHEN` instead."""
    if datasets.backend == "duckdb":
        return f"(CASE WHEN ({denominator}) = 0 THEN NULL ELSE ({numerator}) / ({denominator}) END)"
    return f"SAFE_DIVIDE({numerator}, {denominator})"


def _date_diff_days(datasets: BigQueryDatasets, end_expr: str, start_expr: str) -> str:
    """Portable `end_expr - start_expr`, in whole days. BigQuery's
    `DATE_DIFF(end, start, DAY)` and DuckDB's `date_diff('day', start, end)`
    take their two date arguments in opposite order for the same result."""
    if datasets.backend == "duckdb":
        return f"date_diff('day', {start_expr}, {end_expr})"
    return f"DATE_DIFF({end_expr}, {start_expr}, DAY)"


def _concept_spec(plan: ExtractionPlan) -> ConceptSetSpec:
    return ConceptSetSpec(
        domain=plan.domain,
        concept_ids=plan.root_concept_ids,
        include_descendants=plan.include_descendants,
        excluded_concept_ids=plan.excluded_concept_ids,
    )


def _require_table(plan: ExtractionPlan) -> str:
    if len(plan.omop_tables) != 1:
        raise ValueError(f"'{plan.variable_id}' must resolve to exactly one OMOP table for an audit query")
    return plan.omop_tables[0]


def concept_row_and_person_counts_sql(datasets: BigQueryDatasets, plan: ExtractionPlan) -> str:
    """Rows and distinct persons per concept actually included by `plan`
    (root(s) plus expanded descendants minus exclusions) -- aggregate counts
    only, never a `person_id` column."""
    table = _require_table(plan)
    cte = expanded_concepts_cte("included_concepts", _concept_spec(plan), plan.variable_id, datasets)
    return f"""
    WITH {cte}
    SELECT
      t.{plan.concept_id_column} AS concept_id,
      COUNT(*) AS row_count,
      COUNT(DISTINCT t.person_id) AS person_count
    FROM {datasets.table('omop', table)} t
    JOIN included_concepts c ON t.{plan.concept_id_column} = c.concept_id
    GROUP BY t.{plan.concept_id_column}
    ORDER BY row_count DESC
    """


def approved_concepts_missing_from_data_sql(datasets: BigQueryDatasets, plan: ExtractionPlan) -> str:
    """Which of `plan`'s approved *root* concept_ids never occur in the data
    at all -- a mapping being `approved` does not mean the concept is
    actually populated in a given CDM snapshot (see `OMOPVariableDefinition.
    available_in_data`)."""
    table = _require_table(plan)
    ids = concept_ids_literal(_concept_spec(plan), plan.variable_id)
    return f"""
    WITH observed_concepts AS (
      SELECT DISTINCT {plan.concept_id_column} AS concept_id
      FROM {datasets.table('omop', table)}
    )
    SELECT root_concept_id AS missing_concept_id
    FROM {unnest_ids(datasets, ids, 'root_concept_id')}
    LEFT JOIN observed_concepts ON observed_concepts.concept_id = root_concept_id
    WHERE observed_concepts.concept_id IS NULL
    """


def concept_zero_rate_sql(datasets: BigQueryDatasets, domain: str) -> str:
    """Domain-wide rate of `concept_id = 0` (an unmapped source row) -- a
    domain-level hygiene metric, not specific to one variable's concept set."""
    if domain not in DOMAIN_TABLES:
        raise ValueError(f"domain must be one of {tuple(DOMAIN_TABLES)}")
    table, concept_col, _ = DOMAIN_TABLES[domain]
    zero_rate = _safe_divide(datasets, f"COUNTIF({concept_col} = 0)", "COUNT(*)")
    return f"""
    SELECT
      COUNT(*) AS total_rows,
      COUNTIF({concept_col} = 0) AS zero_concept_rows,
      {zero_rate} AS zero_concept_rate
    FROM {datasets.table('omop', table)}
    """


def unmapped_source_values_sql(datasets: BigQueryDatasets, domain: str, top_n: int = 20) -> str:
    """The most frequent source values behind `concept_id = 0` rows in a
    domain -- what a reviewer should look at next to extend a mapping.
    `<domain>_source_value` is a standard OMOP CDM v5.x column name for every
    domain in `DOMAIN_TABLES`, so it is derived from `domain` directly rather
    than kept in a second lookup table."""
    if domain not in DOMAIN_TABLES:
        raise ValueError(f"domain must be one of {tuple(DOMAIN_TABLES)}")
    table, concept_col, _ = DOMAIN_TABLES[domain]
    source_value_col = f"{domain}_source_value"
    return f"""
    SELECT
      {source_value_col} AS source_value,
      COUNT(*) AS row_count
    FROM {datasets.table('omop', table)}
    WHERE {concept_col} = 0
    GROUP BY source_value
    ORDER BY row_count DESC
    LIMIT {int(top_n)}
    """


def concept_mapping_coverage_sql(datasets: BigQueryDatasets, spec: ConceptSetSpec, field: str) -> str:
    """Database-wide check for whether `spec`'s resolved concept(s) actually
    have usable standard-concept coverage in this OMOP build, versus sitting
    unmapped (`concept_id = 0`) under a `source_value` that still spells out
    the concept's own vocabulary name.

    This automates the manual check that first surfaced MIMIC-IV/OMOP's
    near-total oral-drug mapping gap: RxNorm concept 1112807 ("Aspirin") and
    its ~10,662 descendants matched only 3 `drug_exposure` rows database-
    wide, while ~180K unmapped rows had a `drug_source_value` that literally
    read "Aspirin ...". `causal.orchestrator.workflow` turns the two counts
    this returns into a hard block in the feasibility gate when coverage is
    implausibly low -- see `_check_mapping_coverage` there.

    Deliberately database-wide, not restricted to any one study's eligible
    population: whether this dataset build mapped a concept at all is a
    property of the OMOP conversion, not of a specific cohort.

    The unmapped side joins on a `LIKE` predicate (no shared key to equate),
    which BigQuery cannot rewrite as its usual EXISTS/IN semi-join -- "LEFT
    SEMI JOIN cannot be used without a condition that is an equality of
    fields from both sides of the join" -- so this uses a real `JOIN` plus
    `COUNT(DISTINCT <table>_id)` instead of a correlated `EXISTS` subquery.
    `<table>_id` is the standard OMOP CDM v5.x single-column primary key for
    every table in `DOMAIN_TABLES`, matching the `<domain>_source_value`
    derivation already used above and in `unmapped_source_values_sql`; the
    `DISTINCT` absorbs any row matched by more than one of `spec`'s root
    concept names.
    """
    if spec.domain not in DOMAIN_TABLES:
        raise ValueError(f"{field} domain must be one of {tuple(DOMAIN_TABLES)}")
    table, concept_col, _ = DOMAIN_TABLES[spec.domain]
    source_value_col = f"{spec.domain}_source_value"
    row_id_col = f"{table}_id"
    included_cte = expanded_concepts_cte("included_concepts", spec, field, datasets)
    root_ids = concept_ids_literal(spec, field)
    return f"""
    WITH {included_cte},
    root_names AS (
      SELECT DISTINCT LOWER(concept_name) AS name
      FROM {datasets.table('omop', 'concept')}
      WHERE concept_id IN ({root_ids}) AND concept_name IS NOT NULL
    )
    SELECT
      (SELECT COUNT(*)
       FROM {datasets.table('omop', table)} t
       JOIN included_concepts c ON t.{concept_col} = c.concept_id) AS mapped_row_count,
      (SELECT COUNT(DISTINCT t.{row_id_col})
       FROM {datasets.table('omop', table)} t
       JOIN root_names n ON LOWER(t.{source_value_col}) LIKE CONCAT('%', n.name, '%')
       WHERE t.{concept_col} = 0) AS unmapped_name_matched_row_count
    """


def invalid_or_nonstandard_concepts_sql(datasets: BigQueryDatasets, plan: ExtractionPlan) -> str:
    """Included concepts (root + descendants, minus exclusions) that are not
    a standard, valid concept -- `standard_concept != 'S'` or a non-null
    `invalid_reason`. A registry entry may be `mapping_status="approved"` and
    still point at a non-standard or invalid concept if a reviewer typed a
    concept_id directly; this surfaces that for correction."""
    table = _require_table(plan)
    cte = expanded_concepts_cte("included_concepts", _concept_spec(plan), plan.variable_id, datasets)
    return f"""
    WITH {cte}
    SELECT
      c.concept_id,
      con.standard_concept,
      con.invalid_reason,
      COUNT(*) AS row_count
    FROM {datasets.table('omop', table)} t
    JOIN included_concepts c ON t.{plan.concept_id_column} = c.concept_id
    JOIN {datasets.table('omop', 'concept')} con ON con.concept_id = c.concept_id
    WHERE con.standard_concept IS DISTINCT FROM 'S' OR con.invalid_reason IS NOT NULL
    GROUP BY c.concept_id, con.standard_concept, con.invalid_reason
    ORDER BY row_count DESC
    """


def observed_units_sql(datasets: BigQueryDatasets, plan: ExtractionPlan) -> str:
    """Distinct `unit_concept_id`s (with names) observed for a measurement/
    observation variable's included concepts, and how often each occurs --
    the input a reviewer needs to state `OMOPVariableDefinition.
    expected_unit` and judge whether a unit-conversion rule is required."""
    if plan.domain not in VALUE_DOMAINS:
        raise ValueError(f"'{plan.variable_id}' has domain '{plan.domain}'; units only apply to {VALUE_DOMAINS}")
    table = _require_table(plan)
    cte = expanded_concepts_cte("included_concepts", _concept_spec(plan), plan.variable_id, datasets)
    return f"""
    WITH {cte}
    SELECT
      t.unit_concept_id,
      u.concept_name AS unit_name,
      COUNT(*) AS row_count
    FROM {datasets.table('omop', table)} t
    JOIN included_concepts c ON t.{plan.concept_id_column} = c.concept_id
    LEFT JOIN {datasets.table('omop', 'concept')} u ON u.concept_id = t.unit_concept_id
    GROUP BY t.unit_concept_id, unit_name
    ORDER BY row_count DESC
    """


def descendant_expansion_coverage_sql(datasets: BigQueryDatasets, plan: ExtractionPlan) -> str:
    """Distinct persons matched by the approved root concept_ids alone versus
    after `concept_ancestor` expansion (minus exclusions) -- how much
    `include_descendants` actually changes cohort coverage for this
    variable, side by side in one aggregate row."""
    table = _require_table(plan)
    root_only_ids = concept_ids_literal(
        ConceptSetSpec(domain=plan.domain, concept_ids=plan.root_concept_ids), plan.variable_id
    )
    expanded_cte = expanded_concepts_cte("expanded_concepts", _concept_spec(plan), plan.variable_id, datasets)
    return f"""
    WITH root_only_concepts AS (
      SELECT concept_id FROM {unnest_ids(datasets, root_only_ids, 'concept_id')}
    ),
    {expanded_cte}
    SELECT
      (SELECT COUNT(DISTINCT t.person_id) FROM {datasets.table('omop', table)} t
       JOIN root_only_concepts r ON t.{plan.concept_id_column} = r.concept_id) AS persons_root_only,
      (SELECT COUNT(DISTINCT t.person_id) FROM {datasets.table('omop', table)} t
       JOIN expanded_concepts e ON t.{plan.concept_id_column} = e.concept_id) AS persons_with_descendants
    """


def _first_event_cte(alias: str, plan: ExtractionPlan, datasets: BigQueryDatasets) -> str:
    """One row per person: their earliest qualifying event (by
    `plan.datetime_column`) among `included_concepts` -- the CTE this must be
    concatenated after in every caller's `WITH` clause (see
    `index_event_breakdown_sql`/`incidence_rate_sql`). Ties on the same
    timestamp are broken by `concept_id` so the result is deterministic
    rather than depending on BigQuery's unspecified tie order; this mirrors
    the `QUALIFY ROW_NUMBER()` idiom `causal.queries.feasibility._cohort_ctes`
    already uses for the same "first qualifying event per person" shape.
    """
    table = _require_table(plan)
    return f"""{alias} AS (
      SELECT DISTINCT t.person_id, t.{plan.concept_id_column} AS concept_id, t.{plan.datetime_column} AS event_datetime
      FROM {datasets.table('omop', table)} t
      JOIN included_concepts c ON t.{plan.concept_id_column} = c.concept_id
      WHERE t.{plan.datetime_column} IS NOT NULL
      QUALIFY ROW_NUMBER() OVER (
        PARTITION BY t.person_id ORDER BY t.{plan.datetime_column}, t.{plan.concept_id_column}
      ) = 1
    )"""


def orphan_concepts_sql(datasets: BigQueryDatasets, plan: ExtractionPlan, top_n: int = 20) -> str:
    """Concepts one `concept_ancestor` hop from a shared parent of `plan`'s
    approved root concept_ids -- not already included by `plan` (root(s) plus
    expanded descendants, minus exclusions), in the same OMOP domain, and
    actually occurring in the data. A sibling-concept proxy for the orphan-
    concept diagnostic in the CohortDiagnostics paper cited in this module's
    docstring: that method walks non-standard source-code mappings, which
    this codebase has no frequency table for; this instead surfaces standard
    concepts clinically adjacent to what's already included that a reviewer
    may have missed when defining the concept set.
    """
    table = _require_table(plan)
    domain_id = plan.domain.capitalize()
    included_cte = expanded_concepts_cte("included_concepts", _concept_spec(plan), plan.variable_id, datasets)
    root_ids = concept_ids_literal(
        ConceptSetSpec(domain=plan.domain, concept_ids=plan.root_concept_ids), plan.variable_id
    )
    return f"""
    WITH {included_cte},
    sibling_parents AS (
      SELECT DISTINCT ancestor_concept_id
      FROM {datasets.table('omop', 'concept_ancestor')}
      WHERE descendant_concept_id IN ({root_ids})
        AND min_levels_of_separation = 1
    ),
    sibling_candidates AS (
      SELECT DISTINCT ca.descendant_concept_id AS concept_id
      FROM {datasets.table('omop', 'concept_ancestor')} ca
      JOIN sibling_parents sp ON ca.ancestor_concept_id = sp.ancestor_concept_id
      WHERE ca.min_levels_of_separation = 1
    ),
    candidate_pool AS (
      SELECT sc.concept_id
      FROM sibling_candidates sc
      LEFT JOIN included_concepts ic ON sc.concept_id = ic.concept_id
      WHERE ic.concept_id IS NULL
    )
    SELECT
      t.{plan.concept_id_column} AS concept_id,
      con.concept_name,
      COUNT(*) AS row_count,
      COUNT(DISTINCT t.person_id) AS person_count
    FROM {datasets.table('omop', table)} t
    JOIN candidate_pool cp ON t.{plan.concept_id_column} = cp.concept_id
    JOIN {datasets.table('omop', 'concept')} con ON con.concept_id = t.{plan.concept_id_column}
    WHERE con.domain_id = '{domain_id}'
    GROUP BY t.{plan.concept_id_column}, con.concept_name
    ORDER BY row_count DESC
    LIMIT {int(top_n)}
    """


def index_event_breakdown_sql(datasets: BigQueryDatasets, plan: ExtractionPlan, top_n: int = 20) -> str:
    """Which specific concept_id accounted for each person's earliest
    qualifying event for `plan` -- distinct from
    `concept_row_and_person_counts_sql`, which counts every occurrence
    anywhere in the data, not just the one that actually defined cohort
    entry. Matches the "index event breakdown" diagnostic in the
    CohortDiagnostics paper cited in this module's docstring."""
    included_cte = expanded_concepts_cte("included_concepts", _concept_spec(plan), plan.variable_id, datasets)
    first_event_cte = _first_event_cte("first_event", plan, datasets)
    return f"""
    WITH {included_cte},
    {first_event_cte}
    SELECT
      fe.concept_id,
      con.concept_name,
      COUNT(*) AS person_count
    FROM first_event fe
    LEFT JOIN {datasets.table('omop', 'concept')} con ON con.concept_id = fe.concept_id
    GROUP BY fe.concept_id, con.concept_name
    ORDER BY person_count DESC
    LIMIT {int(top_n)}
    """


def incidence_rate_sql(datasets: BigQueryDatasets, plan: ExtractionPlan) -> str:
    """Crude, unstratified incidence rate for `plan`'s earliest qualifying
    event, per 1,000 person-years of `observation_period` coverage across the
    whole database -- not restricted to a narrower at-risk population, and
    not stratified by age/sex/calendar year (see this module's docstring for
    why). Still the same standard person-time incidence-rate definition
    (new entries / person-time at risk) the CohortDiagnostics paper cited
    there uses, just collapsed to one stratum."""
    included_cte = expanded_concepts_cte("included_concepts", _concept_spec(plan), plan.variable_id, datasets)
    first_event_cte = _first_event_cte("first_event", plan, datasets)
    observed_days = _date_diff_days(datasets, "observation_period_end_date", "observation_period_start_date")
    incidence_rate = _safe_divide(datasets, "(SELECT COUNT(*) FROM first_event)", "d.person_years")
    return f"""
    WITH {included_cte},
    {first_event_cte},
    denominator AS (
      SELECT
        SUM({observed_days}) / 365.25
          AS person_years
      FROM {datasets.table('omop', 'observation_period')}
    )
    SELECT
      (SELECT COUNT(*) FROM first_event) AS cohort_entries,
      d.person_years AS total_person_years,
      {incidence_rate} * 1000
        AS incidence_rate_per_1000_person_years
    FROM denominator d
    """
