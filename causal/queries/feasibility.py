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
#   "add_on": `comparator` is a required backbone therapy (e.g. aspirin) and
#     `treatment` is an adjunct layered on top of it (e.g. clopidogrel).
#     Cohort = everyone who received the backbone; A=1 marks those who
#     additionally received the adjunct. Matches designs like "aspirin plus
#     clopidogrel vs aspirin alone".
#   "mutually_exclusive": standard active-comparator new-user design (drug A
#     vs drug B). Cohort = only persons who received exactly one of
#     treatment/comparator within the grace period; persons who received
#     both (ambiguous/contaminated exposure) or neither are excluded.
#   "vs_unexposed": standard drug-vs-no-treatment design (e.g. "aspirin vs
#     no aspirin"). `comparator` is not queried at all -- there is no OMOP
#     concept for "absence of a drug", so mapping one is neither possible
#     nor required. Cohort = the whole eligible population; A=1 marks
#     `treatment`, A=0 everyone else.
EXPOSURE_CONTRASTS = ("add_on", "mutually_exclusive", "vs_unexposed")

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
    exposure_contrast: str,
    additional_population: tuple[ConceptSetSpec, ...] = (),
    washout_days: int = 0,
) -> str:
    if population.domain not in DOMAIN_TABLES:
        raise ValueError(f"population domain must be one of {tuple(DOMAIN_TABLES)}")
    if outcome.domain not in OUTCOME_DOMAINS:
        raise ValueError(f"outcome domain must be one of {OUTCOME_DOMAINS}")
    # grace_period_hours == 0 passes this check -- roadmap.py's
    # `TargetTrialProtocol.__post_init__`, `hypothesis.py`'s
    # `_positive_int(..., allow_zero=True)`, and the drafting prompt's own
    # example (`causal/prompts/hypothesis.md`) all already treat 0 as a
    # structurally valid integer. But the treatment-matching window built
    # below is the half-open interval `[index_datetime, index_datetime +
    # grace_period_hours)` (see the `t.{trt_dt_col} >=`/`<` pair a few lines
    # down) -- at grace_period_hours=0 that interval has zero width and can
    # never contain any timestamp, so `has_treatment` is guaranteed false for
    # every person regardless of the real drug-exposure data (a silent,
    # 100%-untreated cohort, not a validation error). A reviewer wanting
    # "treatment recorded at essentially the same instant as index_datetime"
    # needs a small positive value (the row's own timestamp precision is
    # rarely finer than seconds/minutes), not 0. Only follow_up_days must be
    # strictly positive: a 0-or-negative follow-up window is never
    # meaningful.
    if grace_period_hours < 0 or follow_up_days <= 0:
        raise ValueError("grace_period_hours must be >= 0 and follow_up_days must be positive")
    if min_age_years is not None and min_age_years < 0:
        raise ValueError("min_age_years must be non-negative")
    if washout_days < 0 or isinstance(washout_days, bool):
        raise ValueError("washout_days must be an integer >= 0")

    if treatment.domain not in DOMAIN_TABLES:
        raise ValueError(f"treatment domain must be one of {tuple(DOMAIN_TABLES)}")
    # "vs_unexposed" (see EXPOSURE_CONTRASTS) has no real comparator exposure
    # to query, so `comparator` is not required to resolve to anything --
    # every other mode needs it exactly like population/treatment/outcome.
    has_real_comparator = exposure_contrast != "vs_unexposed"
    if has_real_comparator and comparator.domain not in DOMAIN_TABLES:
        raise ValueError(f"comparator domain must be one of {tuple(DOMAIN_TABLES)}")

    pop_table, pop_concept_col, pop_dt_col = DOMAIN_TABLES[population.domain]
    trt_table, trt_concept_col, trt_dt_col = DOMAIN_TABLES[treatment.domain]
    if has_real_comparator:
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

    # Every additional required population concept set (see `ConceptSetSpec`'s
    # sibling `additional_population` param -- one entry per independently-
    # required diagnosis, e.g. "type 2 diabetes AND COPD") is ANDed onto
    # `eligible` as its own `EXISTS` semi-join, not a plain `JOIN`: a `JOIN`
    # would fan out one row per matching event in the second domain table
    # (e.g. multiple COPD diagnosis rows), silently duplicating downstream
    # aggregate counts and per-person columns. `EXISTS` only ever contributes
    # a true/false filter. Bounded to `<= el.index_datetime`, the same
    # baseline-only convention covariates use by default: a diagnosis first
    # recorded after the population-defining index event must not
    # retroactively qualify someone (that would be reverse-causation-prone,
    # not a pre-existing baseline condition).
    additional_population_ctes: list[str] = []
    additional_population_filters: list[str] = []
    for index, spec in enumerate(additional_population):
        if spec.domain not in DOMAIN_TABLES:
            raise ValueError(f"additional_population[{index}] domain must be one of {tuple(DOMAIN_TABLES)}")
        alias = f"population_additional_{index}_concepts"
        additional_population_ctes.append(
            _expanded_concepts_cte(alias, spec, f"additional_population[{index}]", datasets)
        )
        ap_table, ap_concept_col, ap_dt_col = DOMAIN_TABLES[spec.domain]
        additional_population_filters.append(f"""EXISTS (
        SELECT 1
        FROM {datasets.table('omop', ap_table)} ap
        JOIN {alias} apc ON ap.{ap_concept_col} = apc.concept_id
        WHERE ap.person_id = el.person_id AND ap.{ap_dt_col} <= el.index_datetime
      )""")

    eligible_filters: list[str] = []
    if min_age_years is not None:
        eligible_filters.append(
            f"EXTRACT(YEAR FROM DATE(el.index_datetime)) - p.year_of_birth >= {int(min_age_years)}"
        )
    eligible_filters.extend(additional_population_filters)
    eligible_where = f"WHERE {' AND '.join(eligible_filters)}" if eligible_filters else ""

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
        *additional_population_ctes,
        f"""eligible AS (
      SELECT el.person_id, el.visit_occurrence_id, el.index_datetime
      FROM first_population_event el
      JOIN {datasets.table('omop', 'person')} p USING (person_id)
      {eligible_where}
    )""",
        _expanded_concepts_cte("treatment_concepts", treatment, "treatment", datasets),
    ]
    if has_real_comparator:
        ctes.append(_expanded_concepts_cte("comparator_concepts", comparator, "comparator", datasets))
        comparator_select = "COUNTIF(cc.concept_id IS NOT NULL) > 0 AS has_comparator"
        comparator_from = f"""
      LEFT JOIN {datasets.table('omop', cmp_table)} c
        ON c.person_id = e.person_id
       AND c.{cmp_dt_col} >= e.index_datetime
       AND c.{cmp_dt_col} < {_dt_add(datasets, 'e.index_datetime', grace_period_hours)}
      LEFT JOIN comparator_concepts cc ON c.{cmp_concept_col} = cc.concept_id"""
    else:
        # No comparator exposure exists to query -- "the comparator arm" is
        # simply everyone the treatment wasn't matched for.
        comparator_select = "NOT (COUNTIF(tc.concept_id IS NOT NULL) > 0) AS has_comparator"
        comparator_from = ""
    ctes.append(f"""exposure AS (
      SELECT
        e.person_id,
        COUNTIF(tc.concept_id IS NOT NULL) > 0 AS has_treatment,
        {comparator_select}
      FROM eligible e
      LEFT JOIN {datasets.table('omop', trt_table)} t
        ON t.person_id = e.person_id
       AND t.{trt_dt_col} >= e.index_datetime
       AND t.{trt_dt_col} < {_dt_add(datasets, 'e.index_datetime', grace_period_hours)}
      LEFT JOIN treatment_concepts tc ON t.{trt_concept_col} = tc.concept_id{comparator_from}
      GROUP BY e.person_id
    )""")

    # New-user / washout eligibility (see `docs/eligibility_refinements_design.md`).
    # `washout_days == 0` (default) emits nothing here and downstream, so the
    # generated SQL is byte-identical to before this feature. When positive,
    # one `washout` CTE carries two person-level booleans that
    # `analytic_cohort_sql` filters on and `feasibility_sql` reports as
    # attrition (never filtering -- same convention as the exposure-combination
    # counts):
    #   `washout_new_user`         -- no qualifying exposure in the half-open
    #     look-back `[index - washout_days, index)`. The look-back concept set
    #     is the treatment class, plus the comparator class only for an
    #     active-comparator (`mutually_exclusive`) design, where "new user"
    #     means new to *either* arm's drug (e.g. PEPTIC: no PPI AND no H2RA).
    #     For `vs_unexposed`/`add_on` the comparator is either not a real drug
    #     or is the backbone every cohort member is on by design.
    #   `washout_lookback_covered` -- an `observation_period` actually spans
    #     `index - washout_days` through `index`, so "no prior exposure" is an
    #     observed fact rather than an artefact of a short record (the standard
    #     OHDSI `priorObservation` rule; critical in MIMIC-IV, where
    #     pre-admission history is thin). Same single-span convention the
    #     `censored` column already uses.
    if washout_days > 0:
        washout_start = _dt_add(datasets, "e.index_datetime", -int(washout_days), "DAY")
        prior_exposure_checks = [
            f"""EXISTS (
        SELECT 1
        FROM {datasets.table('omop', trt_table)} wtx
        JOIN treatment_concepts wtc ON wtx.{trt_concept_col} = wtc.concept_id
        WHERE wtx.person_id = e.person_id
          AND wtx.{trt_dt_col} >= {washout_start}
          AND wtx.{trt_dt_col} < e.index_datetime
      )"""
        ]
        if exposure_contrast == "mutually_exclusive":
            prior_exposure_checks.append(
                f"""EXISTS (
        SELECT 1
        FROM {datasets.table('omop', cmp_table)} wcx
        JOIN comparator_concepts wcc ON wcx.{cmp_concept_col} = wcc.concept_id
        WHERE wcx.person_id = e.person_id
          AND wcx.{cmp_dt_col} >= {washout_start}
          AND wcx.{cmp_dt_col} < e.index_datetime
      )"""
            )
        ctes.append(f"""washout AS (
      SELECT
        e.person_id,
        NOT ({" OR ".join(prior_exposure_checks)}) AS washout_new_user,
        EXISTS (
          SELECT 1
          FROM {datasets.table('omop', 'observation_period')} wop
          WHERE wop.person_id = e.person_id
            AND DATE(e.index_datetime) BETWEEN wop.observation_period_start_date
              AND wop.observation_period_end_date
            AND wop.observation_period_start_date <= DATE({washout_start})
        ) AS washout_lookback_covered
      FROM eligible e
    )""")

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
    #
    # `washout_*` columns are carried through unchanged when the washout is
    # active (see the `washout` CTE above); the non-death branch's GROUP BY
    # gets them too (they are functionally dependent on person_id).
    washout_select = (
        ",\n        w.washout_new_user,\n        w.washout_lookback_covered" if washout_days > 0 else ""
    )
    washout_join = "\n      JOIN washout w USING (person_id)" if washout_days > 0 else ""
    washout_group_by = (
        ", w.washout_new_user, w.washout_lookback_covered" if washout_days > 0 else ""
    )
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
          AS outcome_event_during_grace_period{washout_select}
      FROM eligible e
      JOIN exposure x USING (person_id){washout_join}
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
        ) > 0 AS outcome_event_during_grace_period{washout_select}
      FROM eligible e
      JOIN exposure x USING (person_id){washout_join}
      LEFT JOIN {datasets.table('omop', out_table)} o ON o.person_id = e.person_id
      LEFT JOIN outcome_concepts oc ON o.{out_concept_col} = oc.concept_id
      GROUP BY e.person_id, x.has_treatment, x.has_comparator{washout_group_by}
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
    exposure_contrast: str = "add_on",
    additional_population: tuple[ConceptSetSpec, ...] = (),
    washout_days: int = 0,
) -> str:
    """Aggregate-only feasibility counts for an arbitrary target-trial question.

    Reports every has_treatment/has_comparator combination unfiltered,
    regardless of `exposure_contrast` -- that parameter is used only to
    decide whether `comparator` needs to resolve to a real OMOP concept at
    all (see `EXPOSURE_CONTRASTS`'s `"vs_unexposed"`), not to filter these
    counts the way `analytic_cohort_sql`'s inclusion filter does.

    `additional_population` (see `_cohort_ctes`) ANDs one or more further
    independently-required population concept sets onto `population` --
    empty by default, unchanged behavior for a study with a single-concept
    population.

    `washout_days` (see `_cohort_ctes` and `docs/eligibility_refinements_design.md`)
    is likewise reported here as attrition, never filtered: when positive,
    three extra columns (`washout_excluded_prevalent_users`,
    `washout_excluded_insufficient_lookback`, `eligible_after_washout`) show
    how the new-user look-back would reshape the cohort before a reviewer
    approves it. Zero (default) emits no extra SQL.
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
        exposure_contrast=exposure_contrast,
        additional_population=additional_population,
        washout_days=washout_days,
    )
    washout_counts = (
        """,
      COUNTIF(NOT washout_new_user) AS washout_excluded_prevalent_users,
      COUNTIF(washout_new_user AND NOT washout_lookback_covered) AS washout_excluded_insufficient_lookback,
      COUNTIF(washout_new_user AND washout_lookback_covered) AS eligible_after_washout"""
        if washout_days > 0
        else ""
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
      COUNTIF(outcome_event_during_grace_period) AS outcome_events_during_grace_period{washout_counts}
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
    censoring_definition_approved: bool = False,
    additional_population: tuple[ConceptSetSpec, ...] = (),
    washout_days: int = 0,
) -> str:
    """Person-level analytic cohort (A/Y columns) for the same target trial.

    `additional_population` (see `_cohort_ctes`) ANDs one or more further
    independently-required population concept sets onto `population` --
    empty by default, unchanged behavior for a study with a single-concept
    population.

    `washout_days` (see `_cohort_ctes` and
    `docs/eligibility_refinements_design.md`) restricts the cohort to
    new users: persons with no qualifying exposure in
    `[index - washout_days, index)` whose observation record actually covers
    that look-back. Both conditions are ANDed into the inclusion filter.
    Zero (default) emits no extra SQL.

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

    `censoring_definition_approved` (default `False`, unchanged behavior:
    `censored` stays `NULL` for every row) gates a real per-row
    administrative-censoring indicator -- the exact same nominal-completeness
    definition `causal.data_builder.analytic.AnalyticCohortBuilder.
    followup_audit_sql` already computes in aggregate (death within
    follow-up, or observation-period coverage through the follow-up horizon
    -> not censored; neither -> censored), now emitted per person instead of
    only counted. See `causal.orchestrator.workflow.
    WorkflowManager.approve_censoring_definition` for the human-review gate
    that flips this on -- a reviewer must have already seen the aggregate
    followup_audit.json numbers first.
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
        exposure_contrast=exposure_contrast,
        additional_population=additional_population,
        washout_days=washout_days,
    )
    inclusion_filters = {
        "add_on": ["a.has_comparator"],
        "mutually_exclusive": ["a.has_treatment != a.has_comparator"],
        # Everyone eligible is kept -- no comparator exposure to filter on.
        "vs_unexposed": [],
    }[exposure_contrast]
    if washout_days > 0:
        # New-user restriction: no prior exposure in the look-back AND the
        # look-back is actually observed (see the `washout` CTE).
        inclusion_filters = [*inclusion_filters, "a.washout_new_user", "a.washout_lookback_covered"]
    where_clause = f"WHERE {' AND '.join(inclusion_filters)}" if inclusion_filters else ""
    covariate_ctes, covariate_columns, covariate_joins = _covariate_ctes_columns_and_joins(
        datasets, covariates
    )
    all_ctes = ctes + "".join(f",\n    {cte}" for cte in covariate_ctes)
    extra_columns = "".join(f",\n      {column}" for column in covariate_columns)
    extra_joins = "".join(f"\n    {join}" for join in covariate_joins)
    if censoring_definition_approved:
        horizon = f"DATE_ADD(DATE(e.index_datetime), INTERVAL {int(follow_up_days)} DAY)"
        censored_expr = f"""IF(
        (SELECT MIN(d.death_date) FROM {datasets.table('omop', 'death')} d WHERE d.person_id = a.person_id)
          BETWEEN DATE(e.index_datetime) AND {horizon}
        OR (
          SELECT MAX(op.observation_period_end_date)
          FROM {datasets.table('omop', 'observation_period')} op
          WHERE op.person_id = a.person_id
            AND DATE(e.index_datetime) BETWEEN op.observation_period_start_date AND op.observation_period_end_date
        ) >= {horizon},
        0,
        1
      )"""
    else:
        censored_expr = "CAST(NULL AS INT64)"
    return f"""
    WITH {all_ctes}
    SELECT
      a.person_id,
      CASE WHEN a.has_treatment THEN 1 ELSE 0 END AS A,
      CASE WHEN a.outcome_event THEN 1 ELSE 0 END AS Y,
      e.index_datetime,
      {int(follow_up_days)} AS follow_up_days,
      {censored_expr} AS censored,
      EXTRACT(YEAR FROM DATE(e.index_datetime)) - p.year_of_birth AS age,
      p.gender_concept_id,
      p.race_concept_id{extra_columns}
    FROM analytic a
    JOIN eligible e USING (person_id)
    JOIN {datasets.table('omop', 'person')} p USING (person_id){extra_joins}
    {where_clause}
    """
