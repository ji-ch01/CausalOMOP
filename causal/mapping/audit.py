"""Aggregate-only SQL diagnostics for a registry entry's OMOP mapping.

Every function here returns a BigQuery SQL string that summarizes coverage,
data quality, or vocabulary hygiene for one `causal.mapping.registry.
ExtractionPlan` -- never a patient-level row. This is a diagnostic layer, not
a second execution path: nothing here runs a query itself (see
`causal.orchestrator.workflow`/`causal.mcp_client.OMCPClient` for the
existing governed execution path -- these strings are meant to be passed
through the same `QueryValidationAgent`/`Causal_Select_Query` gate before
ever reaching a database). If a real BigQuery/OMCP connection is not
available, these builders are still fully unit-testable on their SQL shape
alone (see `tests/test_mapping_audit.py`), the same way
`causal.queries.feasibility` is tested without executing anything.
"""

from __future__ import annotations

from causal.config import BigQueryDatasets
from causal.mapping.registry import ExtractionPlan, VALUE_DOMAINS
from causal.queries.feasibility import (
    DOMAIN_TABLES,
    ConceptSetSpec,
    concept_ids_literal,
    expanded_concepts_cte,
)


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
    GROUP BY concept_id
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
    FROM UNNEST([{ids}]) AS root_concept_id
    LEFT JOIN observed_concepts ON observed_concepts.concept_id = root_concept_id
    WHERE observed_concepts.concept_id IS NULL
    """


def concept_zero_rate_sql(datasets: BigQueryDatasets, domain: str) -> str:
    """Domain-wide rate of `concept_id = 0` (an unmapped source row) -- a
    domain-level hygiene metric, not specific to one variable's concept set."""
    if domain not in DOMAIN_TABLES:
        raise ValueError(f"domain must be one of {tuple(DOMAIN_TABLES)}")
    table, concept_col, _ = DOMAIN_TABLES[domain]
    return f"""
    SELECT
      COUNT(*) AS total_rows,
      COUNTIF({concept_col} = 0) AS zero_concept_rows,
      SAFE_DIVIDE(COUNTIF({concept_col} = 0), COUNT(*)) AS zero_concept_rate
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
    JOIN {datasets.table('vocabulary', 'concept')} con ON con.concept_id = c.concept_id
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
    LEFT JOIN {datasets.table('vocabulary', 'concept')} u ON u.concept_id = t.unit_concept_id
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
      SELECT concept_id FROM UNNEST([{root_only_ids}]) AS concept_id
    ),
    {expanded_cte}
    SELECT
      (SELECT COUNT(DISTINCT t.person_id) FROM {datasets.table('omop', table)} t
       JOIN root_only_concepts r ON t.{plan.concept_id_column} = r.concept_id) AS persons_root_only,
      (SELECT COUNT(DISTINCT t.person_id) FROM {datasets.table('omop', table)} t
       JOIN expanded_concepts e ON t.{plan.concept_id_column} = e.concept_id) AS persons_with_descendants
    """
