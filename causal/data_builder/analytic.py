"""Deterministic builder for the first BigQuery analytic-cohort contract."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from causal.config import BigQueryDatasets
from causal.queries.feasibility import (
    COVARIATE_AGGREGATIONS_BY_DOMAIN,
    EXPOSURE_CONTRASTS,
    ConceptSetSpec,
    CovariateSpec,
    analytic_cohort_sql,
)

_VALUE_AGGREGATIONS = frozenset(
    aggregation
    for domain in ("measurement", "observation")
    for aggregation in COVARIATE_AGGREGATIONS_BY_DOMAIN[domain]
)


@dataclass(frozen=True)
class AnalyticColumn:
    name: str
    role: str
    data_type: str
    temporal_status: str


@dataclass(frozen=True)
class CohortBuildManifest:
    schema_version: str
    study_id: str
    status: str
    query_sha256: str
    contains_person_level_data: bool
    delivery_to_llm_allowed: bool
    execution_allowed: bool
    exposure_contrast: str
    columns: tuple[AnalyticColumn, ...]
    unresolved_issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnalyticCohortBuilder:
    """Generate reviewed SQL without executing or exporting patient-level rows."""

    COLUMNS = (
        AnalyticColumn("person_id", "identifier", "INT64", "index_linkage_only"),
        AnalyticColumn("A", "treatment", "INT64", "within_grace_period"),
        AnalyticColumn("Y", "outcome", "INT64", "within_follow_up"),
        AnalyticColumn("index_datetime", "time_zero", "DATETIME", "population_defining_event"),
        AnalyticColumn("follow_up_days", "follow_up", "INT64", "prespecified"),
        AnalyticColumn("censored", "censoring", "INT64", "not_implemented"),
        AnalyticColumn("age", "baseline_covariate", "INT64", "at_index"),
        AnalyticColumn("gender_concept_id", "baseline_covariate", "INT64", "time_invariant"),
        AnalyticColumn("race_concept_id", "baseline_covariate", "INT64", "time_invariant"),
    )

    def build(
        self,
        study_id: str,
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
        execution_allowed: bool,
    ) -> tuple[str, CohortBuildManifest]:
        """`covariates` are reviewed, human-approved baseline covariates (see
        `CovariateSpec`) -- typically only the ones an analysis plan marks
        `status="ready"` (`mapping_status="approved"` and an aggregation
        already reviewed; see `causal.orchestrator.workflow.build_cohort_plan`).
        Demographics (age/gender/race) are unaffected -- they always come
        from the fixed columns below, never from this parameter.
        """
        if not study_id.strip():
            raise ValueError("study_id is required")
        if exposure_contrast not in EXPOSURE_CONTRASTS:
            raise ValueError(f"exposure_contrast must be one of {EXPOSURE_CONTRASTS}")
        sql = analytic_cohort_sql(
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
            covariates=covariates,
        )
        covariate_columns = tuple(
            AnalyticColumn(
                covariate.column_name,
                "baseline_covariate",
                "FLOAT64" if covariate.aggregation in _VALUE_AGGREGATIONS else "INT64",
                f"{covariate.window_start_hours}h_to_{covariate.window_end_hours}h_relative_to_index",
            )
            for covariate in covariates
        )
        manifest = CohortBuildManifest(
            schema_version="1.0",
            study_id=study_id.strip(),
            status="ready_for_governed_build" if execution_allowed else "draft_not_executable",
            query_sha256=hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            contains_person_level_data=True,
            delivery_to_llm_allowed=False,
            execution_allowed=execution_allowed,
            exposure_contrast=exposure_contrast,
            columns=self.COLUMNS + covariate_columns,
            unresolved_issues=(
                "censoring and observation-window completeness are not implemented",
                "effect modifiers are not implemented in this cohort query",
            ),
        )
        return sql, manifest

    def profile_sql(
        self,
        study_id: str,
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
        """Return an aggregate-only query safe for MCP feasibility output."""
        cohort_sql, _ = self.build(
            study_id, datasets,
            population=population, treatment=treatment, comparator=comparator, outcome=outcome,
            grace_period_hours=grace_period_hours, follow_up_days=follow_up_days,
            min_age_years=min_age_years, care_site_names=care_site_names,
            exposure_contrast=exposure_contrast,
            execution_allowed=False,
        )
        return f"""
        WITH analytic_cohort AS (
          {cohort_sql}
        )
        SELECT
          COUNT(*) AS cohort_persons,
          COUNTIF(A = 1) AS treated_persons,
          COUNTIF(A = 0) AS comparator_persons,
          COUNTIF(Y = 1) AS outcome_events,
          COUNTIF(Y = 1 AND A = 1) AS treated_events,
          COUNTIF(Y = 1 AND A = 0) AS comparator_events,
          COUNTIF(age IS NULL) AS missing_age,
          COUNTIF(gender_concept_id IS NULL OR gender_concept_id = 0) AS missing_gender,
          COUNTIF(race_concept_id IS NULL OR race_concept_id = 0) AS missing_race,
          COUNTIF(censored IS NULL) AS unresolved_censoring,
          ROUND(AVG(age), 2) AS mean_age,
          MIN(age) AS min_age,
          MAX(age) AS max_age
        FROM analytic_cohort
        """

    def followup_audit_sql(
        self,
        study_id: str,
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
        """Audit nominal follow-up observability and return aggregate counts only."""
        cohort_sql, _ = self.build(
            study_id, datasets,
            population=population, treatment=treatment, comparator=comparator, outcome=outcome,
            grace_period_hours=grace_period_hours, follow_up_days=follow_up_days,
            min_age_years=min_age_years, care_site_names=care_site_names,
            exposure_contrast=exposure_contrast,
            execution_allowed=False,
        )
        return f"""
        WITH analytic_cohort AS (
          {cohort_sql}
        ), followup AS (
          SELECT
            ac.person_id,
            DATE(ac.index_datetime) AS index_date,
            MAX(
              IF(
                DATE(ac.index_datetime) BETWEEN op.observation_period_start_date
                  AND op.observation_period_end_date,
                op.observation_period_end_date,
                NULL
              )
            ) AS observation_end_date,
            MIN(
              IF(d.death_date >= DATE(ac.index_datetime), d.death_date, NULL)
            ) AS death_date
          FROM analytic_cohort ac
          LEFT JOIN {datasets.table('omop', 'observation_period')} op USING (person_id)
          LEFT JOIN {datasets.table('omop', 'death')} d USING (person_id)
          GROUP BY ac.person_id, index_date
        )
        SELECT
          COUNT(*) AS cohort_persons,
          COUNTIF(observation_end_date IS NULL) AS no_observation_period_at_index,
          COUNTIF(
            observation_end_date >= DATE_ADD(index_date, INTERVAL {int(follow_up_days)} DAY)
          ) AS observation_period_covers_follow_up,
          COUNTIF(
            death_date BETWEEN index_date AND DATE_ADD(index_date, INTERVAL {int(follow_up_days)} DAY)
          ) AS known_death_within_follow_up,
          COUNTIF(
            death_date BETWEEN index_date AND DATE_ADD(index_date, INTERVAL {int(follow_up_days)} DAY)
            OR observation_end_date >= DATE_ADD(index_date, INTERVAL {int(follow_up_days)} DAY)
          ) AS nominal_followup_complete,
          COUNTIF(
            (death_date IS NULL OR death_date > DATE_ADD(index_date, INTERVAL {int(follow_up_days)} DAY))
            AND (
              observation_end_date IS NULL
              OR observation_end_date < DATE_ADD(index_date, INTERVAL {int(follow_up_days)} DAY)
            )
          ) AS nominal_followup_incomplete
        FROM followup
        """
