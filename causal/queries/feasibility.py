"""General-purpose feasibility/cohort SQL, driven by the concept sets resolved
for one specific causal question (population/treatment/comparator/outcome),
instead of one hardcoded clinical scenario.

Callers resolve `ConceptSetSpec`s from a run's approved `concept_sets` (see
`causal.agents.user_question`); this module never invents or text-matches a
concept itself, and renders only trusted literals -- consistent with the
OMCP transport, which does not bind BigQuery parameters.
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class ConceptSetSpec:
    """One resolved, human-approved OMOP concept set: which clinical domain
    it lives in, plus the standard concept_ids that define it. `domain` must
    be one of `DOMAIN_TABLES` (or "death", outcome only, no concept_ids)."""

    domain: str
    concept_ids: tuple[int, ...] = ()


def _string_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _concept_ids_literal(spec: ConceptSetSpec, field: str) -> str:
    if spec.domain not in DOMAIN_TABLES:
        raise ValueError(f"{field} domain must be one of {tuple(DOMAIN_TABLES)}")
    if not spec.concept_ids:
        raise ValueError(f"{field}.concept_ids are required for domain '{spec.domain}'")
    return ", ".join(str(int(concept_id)) for concept_id in spec.concept_ids)


def _expanded_concepts_cte(alias: str, spec: ConceptSetSpec, field: str, datasets: BigQueryDatasets) -> str:
    """Expand a concept set to itself plus every `concept_ancestor` descendant."""
    ids = _concept_ids_literal(spec, field)
    return f"""{alias} AS (
      SELECT descendant_concept_id AS concept_id
      FROM {datasets.table('omop', 'concept_ancestor')}
      WHERE ancestor_concept_id IN ({ids})
      UNION DISTINCT SELECT concept_id FROM UNNEST([{ids}]) AS concept_id
    )"""


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
        site_join = f"""
      JOIN {datasets.table('omop', 'visit_detail')} vd
        ON vd.visit_occurrence_id = ev.visit_occurrence_id
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
        PARTITION BY ev.person_id ORDER BY ev.{pop_dt_col}
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
       AND t.{trt_dt_col} < DATETIME_ADD(e.index_datetime, INTERVAL {int(grace_period_hours)} HOUR)
      LEFT JOIN treatment_concepts tc ON t.{trt_concept_col} = tc.concept_id
      LEFT JOIN {datasets.table('omop', cmp_table)} c
        ON c.person_id = e.person_id
       AND c.{cmp_dt_col} >= e.index_datetime
       AND c.{cmp_dt_col} < DATETIME_ADD(e.index_datetime, INTERVAL {int(grace_period_hours)} HOUR)
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
        d.death_date IS NOT NULL
          AND d.death_date >= DATE(e.index_datetime)
          AND d.death_date <= DATE(DATETIME_ADD(e.index_datetime, INTERVAL {int(grace_period_hours)} HOUR))
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
          AND o.{out_dt_col} <= DATETIME_ADD(e.index_datetime, INTERVAL {int(follow_up_days)} DAY)
        ) > 0 AS outcome_event,
        COUNTIF(
          oc.concept_id IS NOT NULL
          AND o.{out_dt_col} >= e.index_datetime
          AND o.{out_dt_col} < DATETIME_ADD(e.index_datetime, INTERVAL {int(grace_period_hours)} HOUR)
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
) -> str:
    """Person-level analytic cohort (A/Y columns) for the same target trial.

    `exposure_contrast` (see `EXPOSURE_CONTRASTS`) selects the inclusion
    filter and the meaning of `A`; it does not change how `treatment`/
    `comparator` exposure is ascertained, only who is kept and how the two
    flags are collapsed into one binary arm indicator.
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
    return f"""
    WITH {ctes}
    SELECT
      a.person_id,
      CASE WHEN a.has_treatment THEN 1 ELSE 0 END AS A,
      CASE WHEN a.outcome_event THEN 1 ELSE 0 END AS Y,
      e.index_datetime,
      {int(follow_up_days)} AS follow_up_days,
      CAST(NULL AS INT64) AS censored,
      EXTRACT(YEAR FROM DATE(e.index_datetime)) - p.year_of_birth AS age,
      p.gender_concept_id,
      p.race_concept_id
    FROM analytic a
    JOIN eligible e USING (person_id)
    JOIN {datasets.table('omop', 'person')} p USING (person_id)
    {where_clause}
    """
