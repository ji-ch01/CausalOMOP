"""General-purpose feasibility/cohort SQL, driven by the concept sets resolved
for one specific causal question (population/treatment/comparator/outcome),
instead of one hardcoded clinical scenario.

Callers resolve `ConceptSetSpec`s from a run's approved `concept_sets` (see
`causal.agents.user_question`); this module never invents or text-matches a
concept itself, and renders only trusted literals -- consistent with the
OMCP transport, which does not bind BigQuery parameters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from causal.config import BigQueryDatasets


# domain -> (table, concept_id_column, start_datetime_column)
DOMAIN_TABLES: dict[str, tuple[str, str, str]] = {
    "condition": ("condition_occurrence", "condition_concept_id", "condition_start_datetime"),
    "drug": ("drug_exposure", "drug_concept_id", "drug_exposure_start_datetime"),
    "procedure": ("procedure_occurrence", "procedure_concept_id", "procedure_datetime"),
    "measurement": ("measurement", "measurement_concept_id", "measurement_datetime"),
    "observation": ("observation", "observation_concept_id", "observation_datetime"),
}

OUTCOME_DOMAINS = tuple(DOMAIN_TABLES) + ("death",)

# How the person-level analytic cohort resolves "treatment" vs "comparator"
# into an inclusion filter and a binary A. This is a real modeling choice,
# not a mechanical detail -- getting it wrong silently changes who is in the
# cohort:
#   "add_on": `comparator` is a required backbone therapy (e.g. crystalloid
#     resuscitation) and `treatment` is an adjunct layered on top of it
#     (e.g. albumin). Cohort = everyone who received the backbone; A=1 marks
#     those who additionally received the adjunct. Matches designs like
#     "crystalloids plus albumin vs crystalloids alone".
#   "mutually_exclusive": standard active-comparator new-user design (drug A
#     vs drug B). Cohort = only persons who received exactly one of
#     treatment/comparator within the grace period; persons who received
#     both (ambiguous/contaminated exposure) or neither are excluded.
EXPOSURE_CONTRASTS = ("add_on", "mutually_exclusive")

# Which aggregation rules make sense for a covariate's OMOP domain. Event
# domains (condition/drug/procedure) carry no numeric value -- only
# presence/count are meaningful. Value domains (measurement/observation)
# aggregate `value_as_number`. Mirrors the "vocab dict next to DOMAIN_TABLES"
# pattern already used for EXPOSURE_CONTRASTS; this is the single source of
# truth both for SQL generation here and for reviewer-facing validation in
# `causal.orchestrator.workflow.review_covariate_aggregation`.
COVARIATE_AGGREGATIONS_BY_DOMAIN: dict[str, tuple[str, ...]] = {
    "condition": ("presence_in_window", "count_in_window"),
    "drug": ("presence_in_window", "count_in_window"),
    "procedure": ("presence_in_window", "count_in_window"),
    "measurement": ("last_value_before_index", "max_in_window", "min_in_window", "mean_in_window"),
    "observation": ("last_value_before_index", "max_in_window", "min_in_window", "mean_in_window"),
}

# Public (not module-private): the single source of truth for "which
# aggregations are event-presence aggregations", reused by
# `causal.mapping.registry` to derive the one clinically-correct missingness
# rule per aggregation (`zero_when_absent` here, `null_when_missing` for
# everything else) instead of re-deriving/duplicating this partition.
PRESENCE_AGGREGATIONS = ("presence_in_window", "count_in_window")
_PRESENCE_AGGREGATIONS = PRESENCE_AGGREGATIONS


@dataclass(frozen=True)
class ConceptSetSpec:
    """One resolved, human-approved OMOP concept set: which clinical domain
    it lives in, plus the standard concept_ids that define it. `domain` must
    be one of `DOMAIN_TABLES` (or "death", outcome only, no concept_ids).

    `include_descendants` (default `True`, preserving this module's
    long-standing behavior) controls whether the concept set expands to its
    `concept_ancestor` descendants or stays exactly the given root ids --
    some registry-reviewed variables (see `causal.mapping.registry`) are
    deliberately narrow (e.g. one specific lab test, not its whole concept
    class) and must not silently pull in every descendant.

    `excluded_concept_ids` (default `()`) removes specific concepts from the
    resolved set *after* expansion -- e.g. a root concept's descendant that a
    reviewer has determined does not belong in this variable's definition.
    Exclusion always wins over inclusion by descendants: a concept listed
    here is removed even if `include_descendants` would otherwise pull it in.
    """

    domain: str
    concept_ids: tuple[int, ...] = ()
    include_descendants: bool = True
    excluded_concept_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class CovariateSpec:
    """One reviewed, human-approved baseline covariate ready for extraction:
    an OMOP domain + concept_ids (see `ConceptSetSpec`) plus the ascertainment
    window (hours relative to `index_datetime`; negative = before) and
    aggregation rule (see `COVARIATE_AGGREGATIONS_BY_DOMAIN`) a reviewer chose
    for it. `column_name` must already be a safe, unique SQL identifier --
    callers (see `causal.orchestrator.workflow.build_cohort_plan`) are
    responsible for turning a human-entered covariate label into one; this
    module only validates it, it does not slugify.

    `include_descendants`/`excluded_concept_ids` mirror `ConceptSetSpec`'s
    fields of the same name -- see that docstring."""

    column_name: str
    domain: str
    concept_ids: tuple[int, ...]
    window_start_hours: int
    window_end_hours: int
    aggregation: str
    include_descendants: bool = True
    excluded_concept_ids: tuple[int, ...] = ()


def _string_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _dt_add(datasets: BigQueryDatasets, expr: str, amount: int, unit: str = "HOUR") -> str:
    """Portable `datetime_expression + amount unit`. BigQuery keeps the
    existing `DATETIME_ADD` form (unchanged SQL for that backend); DuckDB has
    no `DATETIME_ADD` function (only `DATE_ADD`, which requires a DATE, not a
    DATETIME/TIMESTAMP, argument), so it gets interval arithmetic instead.
    `amount` may be negative (a covariate window's `window_start_hours` often
    is); DuckDB's parser rejects a bare negative literal immediately after
    `INTERVAL`, hence the parens.
    """
    amount = int(amount)
    if datasets.backend == "duckdb":
        return f"({expr} + INTERVAL ({amount}) {unit})"
    return f"DATETIME_ADD({expr}, INTERVAL {amount} {unit})"


def unnest_ids(datasets: BigQueryDatasets, ids: str, column_name: str) -> str:
    """Portable `UNNEST([ids]) AS column_name` table-function fragment, for a
    `FROM`/join clause that turns an integer-id literal list into one row per
    id, in a column named `column_name`. BigQuery's `UNNEST(expr) AS alias`
    names the single output *column* `alias`; DuckDB's `AS alias` instead
    names the table (leaving the column named "unnest"), so it needs the
    `AS table_alias(column_name)` form for the same result -- a syntax
    BigQuery does not accept. Public (not module-private) so
    `causal.mapping.audit` can build the same `UNNEST` fragment without
    duplicating this dialect branch.
    """
    if datasets.backend == "duckdb":
        return f"UNNEST([{ids}]) AS _unnest_t({column_name})"
    return f"UNNEST([{ids}]) AS {column_name}"


def concept_ids_literal(spec: ConceptSetSpec, field: str) -> str:
    if spec.domain not in DOMAIN_TABLES:
        raise ValueError(f"{field} domain must be one of {tuple(DOMAIN_TABLES)}")
    if not spec.concept_ids:
        raise ValueError(f"{field}.concept_ids are required for domain '{spec.domain}'")
    return ", ".join(str(int(concept_id)) for concept_id in spec.concept_ids)


_concept_ids_literal = concept_ids_literal


def expanded_concepts_cte(alias: str, spec: ConceptSetSpec, field: str, datasets: BigQueryDatasets) -> str:
    """Resolve a concept set to the concept_ids its SQL should actually match:
    itself plus every `concept_ancestor` descendant by default, or just the
    given root ids when `spec.include_descendants` is False, minus any
    `spec.excluded_concept_ids` either way (exclusion always wins over
    inclusion by descendants -- see `ConceptSetSpec`). Public (not
    module-private) so `causal.mapping.audit` can build the same resolved
    concept-id CTE for aggregate diagnostics without duplicating this logic.
    """
    ids = concept_ids_literal(spec, field)
    if spec.include_descendants:
        base = f"""SELECT descendant_concept_id AS concept_id
      FROM {datasets.table('omop', 'concept_ancestor')}
      WHERE ancestor_concept_id IN ({ids})
      UNION DISTINCT SELECT concept_id FROM {unnest_ids(datasets, ids, 'concept_id')}"""
    else:
        base = f"SELECT concept_id FROM {unnest_ids(datasets, ids, 'concept_id')}"
    if not spec.excluded_concept_ids:
        return f"""{alias} AS (
      {base}
    )"""
    excluded = ", ".join(str(int(concept_id)) for concept_id in spec.excluded_concept_ids)
    return f"""{alias} AS (
      SELECT concept_id FROM ({base})
      WHERE concept_id NOT IN ({excluded})
    )"""


_expanded_concepts_cte = expanded_concepts_cte


# The fixed, always-present columns `analytic_cohort_sql` emits -- a
# covariate's generated column name may not collide with any of these (see
# `_covariate_ctes_columns_and_joins`).
_FIXED_ANALYTIC_COLUMNS = (
    "person_id", "A", "Y", "index_datetime", "follow_up_days", "censored",
    "age", "gender_concept_id", "race_concept_id",
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_SQL_AGGREGATE_FUNCTIONS = {"max_in_window": "MAX", "min_in_window": "MIN", "mean_in_window": "AVG"}


def _covariate_ctes_columns_and_joins(
    datasets: BigQueryDatasets, covariates: tuple[CovariateSpec, ...]
) -> tuple[list[str], list[str], list[str]]:
    """Build one CTE-pair, one final-SELECT column, and one LEFT JOIN per
    reviewed covariate. Presence/count aggregations (event domains) COALESCE
    to 0 -- absence of a coded event is a real "not present" fact. Value
    aggregations (measurement/observation) stay NULL when nothing qualifies
    in the window -- true missingness must never be imputed to 0 here (e.g.
    a missing SOFA score is not clinically a score of 0)."""
    seen_columns = set(_FIXED_ANALYTIC_COLUMNS)
    ctes: list[str] = []
    columns: list[str] = []
    joins: list[str] = []
    for covariate in covariates:
        name = covariate.column_name
        if not _IDENTIFIER_RE.match(name):
            raise ValueError(f"covariate column_name '{name}' is not a safe SQL identifier")
        if name in seen_columns:
            raise ValueError(f"covariate column_name '{name}' collides with another column")
        seen_columns.add(name)
        if covariate.domain not in DOMAIN_TABLES:
            raise ValueError(f"covariate '{name}' domain must be one of {tuple(DOMAIN_TABLES)}")
        allowed = COVARIATE_AGGREGATIONS_BY_DOMAIN[covariate.domain]
        if covariate.aggregation not in allowed:
            raise ValueError(
                f"covariate '{name}' aggregation must be one of {allowed} for domain '{covariate.domain}'"
            )
        table, concept_col, dt_col = DOMAIN_TABLES[covariate.domain]
        concepts_alias = f"{name}_concepts"
        concept_spec = ConceptSetSpec(
            domain=covariate.domain,
            concept_ids=covariate.concept_ids,
            include_descendants=covariate.include_descendants,
            excluded_concept_ids=covariate.excluded_concept_ids,
        )
        ctes.append(_expanded_concepts_cte(concepts_alias, concept_spec, name, datasets))

        window_filter = (
            f"t.{dt_col} >= {_dt_add(datasets, 'e.index_datetime', covariate.window_start_hours)}\n"
            f"       AND t.{dt_col} <= {_dt_add(datasets, 'e.index_datetime', covariate.window_end_hours)}"
        )

        if covariate.aggregation in _PRESENCE_AGGREGATIONS:
            agg_alias = f"{name}_events"
            ctes.append(f"""{agg_alias} AS (
      SELECT e.person_id, COUNT(*) AS event_count
      FROM eligible e
      JOIN {datasets.table('omop', table)} t ON t.person_id = e.person_id
       AND {window_filter}
      JOIN {concepts_alias} c ON t.{concept_col} = c.concept_id
      GROUP BY e.person_id
    )""")
            if covariate.aggregation == "presence_in_window":
                columns.append(f"CASE WHEN {agg_alias}.event_count > 0 THEN 1 ELSE 0 END AS {name}")
            else:
                columns.append(f"COALESCE({agg_alias}.event_count, 0) AS {name}")
        else:
            measurements_alias = f"{name}_measurements"
            agg_alias = f"{name}_agg"
            ctes.append(f"""{measurements_alias} AS (
      SELECT e.person_id, t.{dt_col} AS event_dt, t.value_as_number AS event_value
      FROM eligible e
      JOIN {datasets.table('omop', table)} t ON t.person_id = e.person_id
       AND {window_filter}
      JOIN {concepts_alias} c ON t.{concept_col} = c.concept_id
      WHERE t.value_as_number IS NOT NULL
    )""")
            if covariate.aggregation == "last_value_before_index":
                ctes.append(f"""{agg_alias} AS (
      SELECT person_id, event_value AS agg_value
      FROM {measurements_alias}
      QUALIFY ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY event_dt DESC) = 1
    )""")
            else:
                sql_fn = _SQL_AGGREGATE_FUNCTIONS[covariate.aggregation]
                ctes.append(f"""{agg_alias} AS (
      SELECT person_id, {sql_fn}(event_value) AS agg_value
      FROM {measurements_alias}
      GROUP BY person_id
    )""")
            columns.append(f"{agg_alias}.agg_value AS {name}")
        joins.append(f"LEFT JOIN {agg_alias} USING (person_id)")
    return ctes, columns, joins


def _cohort_ctes(
    datasets: BigQueryDatasets,
    *,
    population: ConceptSetSpec,
    treatment: ConceptSetSpec,
    comparator: ConceptSetSpec,
    outcome: ConceptSetSpec,
    grace_period_hours: int,
    follow_up_days: int,
    min_age_years: int | None,
    care_site_names: tuple[str, ...],
) -> str:
    if population.domain not in DOMAIN_TABLES:
        raise ValueError(f"population domain must be one of {tuple(DOMAIN_TABLES)}")
    if outcome.domain not in OUTCOME_DOMAINS:
        raise ValueError(f"outcome domain must be one of {OUTCOME_DOMAINS}")
    if grace_period_hours <= 0 or follow_up_days <= 0:
        raise ValueError("Study windows must be positive")
    if min_age_years is not None and min_age_years < 0:
        raise ValueError("min_age_years must be non-negative")

    if treatment.domain not in DOMAIN_TABLES:
        raise ValueError(f"treatment domain must be one of {tuple(DOMAIN_TABLES)}")
    if comparator.domain not in DOMAIN_TABLES:
        raise ValueError(f"comparator domain must be one of {tuple(DOMAIN_TABLES)}")

    pop_table, pop_concept_col, pop_dt_col = DOMAIN_TABLES[population.domain]
    trt_table, trt_concept_col, trt_dt_col = DOMAIN_TABLES[treatment.domain]
    cmp_table, cmp_concept_col, cmp_dt_col = DOMAIN_TABLES[comparator.domain]

    site_join = ""
    site_filter = ""
    if care_site_names:
        names = ", ".join(_string_literal(name) for name in care_site_names)
        # A hospitalization (`visit_occurrence`) commonly spans several
        # `visit_detail` stays (e.g. ED -> MICU -> ward). Joining on
        # `visit_occurrence_id` alone would match a person whose
        # population-defining event happened anywhere in that
        # hospitalization as long as *any* stay touched a named unit --
        # including one they were only in before or after the event. The
        # added datetime bounds require the population event to have
        # actually occurred during that specific stay.
        site_join = f"""
      JOIN {datasets.table('omop', 'visit_detail')} vd
        ON vd.visit_occurrence_id = ev.visit_occurrence_id
       AND ev.{pop_dt_col} >= vd.visit_detail_start_datetime
       AND ev.{pop_dt_col} <= COALESCE(vd.visit_detail_end_datetime, vd.visit_detail_start_datetime)
      JOIN {datasets.table('omop', 'care_site')} cs USING (care_site_id)"""
        site_filter = f"AND cs.care_site_name IN ({names})"

    age_filter = ""
    if min_age_years is not None:
        age_filter = (
            f"WHERE EXTRACT(YEAR FROM DATE(el.index_datetime)) - p.year_of_birth >= {int(min_age_years)}"
        )

    ctes = [
        _expanded_concepts_cte("population_concepts", population, "population", datasets),
        f"""first_population_event AS (
      SELECT DISTINCT
        ev.person_id, ev.visit_occurrence_id, ev.{pop_dt_col} AS index_datetime
      FROM {datasets.table('omop', pop_table)} ev
      JOIN population_concepts pc ON ev.{pop_concept_col} = pc.concept_id{site_join}
      WHERE ev.{pop_dt_col} IS NOT NULL {site_filter}
      QUALIFY ROW_NUMBER() OVER (
        PARTITION BY ev.person_id ORDER BY ev.{pop_dt_col}, ev.{pop_concept_col}
      ) = 1
    )""",
        f"""eligible AS (
      SELECT el.person_id, el.visit_occurrence_id, el.index_datetime
      FROM first_population_event el
      JOIN {datasets.table('omop', 'person')} p USING (person_id)
      {age_filter}
    )""",
        _expanded_concepts_cte("treatment_concepts", treatment, "treatment", datasets),
        _expanded_concepts_cte("comparator_concepts", comparator, "comparator", datasets),
        f"""exposure AS (
      SELECT
        e.person_id,
        COUNTIF(tc.concept_id IS NOT NULL) > 0 AS has_treatment,
        COUNTIF(cc.concept_id IS NOT NULL) > 0 AS has_comparator
      FROM eligible e
      LEFT JOIN {datasets.table('omop', trt_table)} t
        ON t.person_id = e.person_id
       AND t.{trt_dt_col} >= e.index_datetime
       AND t.{trt_dt_col} < {_dt_add(datasets, 'e.index_datetime', grace_period_hours)}
      LEFT JOIN treatment_concepts tc ON t.{trt_concept_col} = tc.concept_id
      LEFT JOIN {datasets.table('omop', cmp_table)} c
        ON c.person_id = e.person_id
       AND c.{cmp_dt_col} >= e.index_datetime
       AND c.{cmp_dt_col} < {_dt_add(datasets, 'e.index_datetime', grace_period_hours)}
      LEFT JOIN comparator_concepts cc ON c.{cmp_concept_col} = cc.concept_id
      GROUP BY e.person_id
    )""",
    ]

    # Follow-up (outcome ascertainment) starts at the same index_datetime as
    # exposure ascertainment, by design: this is what a grace-period target
    # trial emulation requires to avoid immortal-time bias (see
    # `design_constraints` in a study protocol). One consequence is that an
    # outcome occurring inside the grace period itself is attributed to
    # whichever arm the person is later classified into, before that
    # person's exposure status was actually determined.
    # `outcome_event_during_grace_period` is a diagnostic flag surfacing how
    # many events fall in that overlap, exposed separately in
    # `feasibility_sql` -- it never changes cohort membership or `outcome_event`.
    if outcome.domain == "death":
        ctes.append(f"""analytic AS (
      SELECT
        e.person_id,
        x.has_treatment,
        x.has_comparator,
        d.death_date IS NOT NULL
          AND d.death_date >= DATE(e.index_datetime)
          AND d.death_date <= DATE_ADD(DATE(e.index_datetime), INTERVAL {int(follow_up_days)} DAY) AS outcome_event,
        -- death_date is DATE-only in the OMOP CDM (no time-of-day), unlike
        -- every other domain's datetime column below, so this can only
        -- approximate the same [index, index + grace_period) window to
        -- day granularity. Kept exclusive on the upper bound to match the
        -- non-death branch's semantics -- an inclusive bound against a
        -- DATE-truncated boundary would silently count deaths up to ~24h
        -- past the true grace-period cutoff as "during" it.
        d.death_date IS NOT NULL
          AND d.death_date >= DATE(e.index_datetime)
          AND d.death_date < DATE({_dt_add(datasets, 'e.index_datetime', grace_period_hours)})
          AS outcome_event_during_grace_period
      FROM eligible e
      JOIN exposure x USING (person_id)
      LEFT JOIN {datasets.table('omop', 'death')} d USING (person_id)
    )""")
    else:
        out_table, out_concept_col, out_dt_col = DOMAIN_TABLES[outcome.domain]
        ctes.append(_expanded_concepts_cte("outcome_concepts", outcome, "outcome", datasets))
        ctes.append(f"""analytic AS (
      SELECT
        e.person_id,
        x.has_treatment,
        x.has_comparator,
        COUNTIF(
          oc.concept_id IS NOT NULL
          AND o.{out_dt_col} >= e.index_datetime
          AND o.{out_dt_col} <= {_dt_add(datasets, 'e.index_datetime', follow_up_days, 'DAY')}
        ) > 0 AS outcome_event,
        COUNTIF(
          oc.concept_id IS NOT NULL
          AND o.{out_dt_col} >= e.index_datetime
          AND o.{out_dt_col} < {_dt_add(datasets, 'e.index_datetime', grace_period_hours)}
        ) > 0 AS outcome_event_during_grace_period
      FROM eligible e
      JOIN exposure x USING (person_id)
      LEFT JOIN {datasets.table('omop', out_table)} o ON o.person_id = e.person_id
      LEFT JOIN outcome_concepts oc ON o.{out_concept_col} = oc.concept_id
      GROUP BY e.person_id, x.has_treatment, x.has_comparator
    )""")

    return ",\n    ".join(ctes)


def feasibility_sql(
    datasets: BigQueryDatasets,
    *,
    population: ConceptSetSpec,
    treatment: ConceptSetSpec,
    comparator: ConceptSetSpec,
    outcome: ConceptSetSpec,
    grace_period_hours: int = 24,
    follow_up_days: int = 28,
    min_age_years: int | None = 18,
    care_site_names: tuple[str, ...] = (),
) -> str:
    """Aggregate-only feasibility counts for an arbitrary target-trial question."""
    ctes = _cohort_ctes(
        datasets,
        population=population,
        treatment=treatment,
        comparator=comparator,
        outcome=outcome,
        grace_period_hours=grace_period_hours,
        follow_up_days=follow_up_days,
        min_age_years=min_age_years,
        care_site_names=care_site_names,
    )
    return f"""
    WITH {ctes}
    SELECT
      COUNT(*) AS eligible_persons,
      COUNTIF(has_treatment) AS any_treatment,
      COUNTIF(has_comparator) AS any_comparator,
      COUNTIF(has_treatment AND has_comparator) AS treatment_and_comparator,
      COUNTIF(has_treatment AND NOT has_comparator) AS treatment_only,
      COUNTIF(NOT has_treatment AND has_comparator) AS comparator_only,
      COUNTIF(NOT has_treatment AND NOT has_comparator) AS neither_exposure,
      COUNTIF(outcome_event) AS outcome_events,
      COUNTIF(outcome_event AND has_treatment) AS outcome_events_treatment,
      COUNTIF(outcome_event AND has_comparator AND NOT has_treatment) AS outcome_events_comparator_only,
      COUNTIF(outcome_event_during_grace_period) AS outcome_events_during_grace_period
    FROM analytic
    """


def analytic_cohort_sql(
    datasets: BigQueryDatasets,
    *,
    population: ConceptSetSpec,
    treatment: ConceptSetSpec,
    comparator: ConceptSetSpec,
    outcome: ConceptSetSpec,
    grace_period_hours: int = 24,
    follow_up_days: int = 28,
    min_age_years: int | None = 18,
    care_site_names: tuple[str, ...] = (),
    exposure_contrast: str = "add_on",
    covariates: tuple[CovariateSpec, ...] = (),
) -> str:
    """Person-level analytic cohort (A/Y columns) for the same target trial.

    `exposure_contrast` (see `EXPOSURE_CONTRASTS`) selects the inclusion
    filter and the meaning of `A`; it does not change how `treatment`/
    `comparator` exposure is ascertained, only who is kept and how the two
    flags are collapsed into one binary arm indicator.

    `covariates` (see `CovariateSpec`) are reviewed, human-approved baseline
    covariates -- each gets one additional column, LEFT JOINed onto the
    cohort by `person_id` so a person with no qualifying event still appears
    (see `_covariate_ctes_columns_and_joins` for aggregation/missingness
    semantics). Demographics (age/gender/race) are unaffected -- they keep
    using the fixed columns below, never this path.
    """
    if exposure_contrast not in EXPOSURE_CONTRASTS:
        raise ValueError(f"exposure_contrast must be one of {EXPOSURE_CONTRASTS}")
    ctes = _cohort_ctes(
        datasets,
        population=population,
        treatment=treatment,
        comparator=comparator,
        outcome=outcome,
        grace_period_hours=grace_period_hours,
        follow_up_days=follow_up_days,
        min_age_years=min_age_years,
        care_site_names=care_site_names,
    )
    where_clause = (
        "WHERE a.has_comparator" if exposure_contrast == "add_on"
        else "WHERE a.has_treatment != a.has_comparator"
    )
    covariate_ctes, covariate_columns, covariate_joins = _covariate_ctes_columns_and_joins(
        datasets, covariates
    )
    all_ctes = ctes + "".join(f",\n    {cte}" for cte in covariate_ctes)
    extra_columns = "".join(f",\n      {column}" for column in covariate_columns)
    extra_joins = "".join(f"\n    {join}" for join in covariate_joins)
    return f"""
    WITH {all_ctes}
    SELECT
      a.person_id,
      CASE WHEN a.has_treatment THEN 1 ELSE 0 END AS A,
      CASE WHEN a.outcome_event THEN 1 ELSE 0 END AS Y,
      e.index_datetime,
      {int(follow_up_days)} AS follow_up_days,
      CAST(NULL AS INT64) AS censored,
      EXTRACT(YEAR FROM DATE(e.index_datetime)) - p.year_of_birth AS age,
      p.gender_concept_id,
      p.race_concept_id{extra_columns}
    FROM analytic a
    JOIN eligible e USING (person_id)
    JOIN {datasets.table('omop', 'person')} p USING (person_id){extra_joins}
    {where_clause}
    """
