"""Deterministic builder for the first BigQuery analytic-cohort contract."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from causal.config import BigQueryDatasets
from causal.queries.albumin_sepsis import albumin_sepsis_feasibility_sql


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
    columns: tuple[AnalyticColumn, ...]
    unresolved_issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnalyticCohortBuilder:
    """Generate reviewed SQL without executing or exporting patient-level rows."""

    COLUMNS = (
        AnalyticColumn("person_id", "identifier", "INT64", "index_linkage_only"),
        AnalyticColumn("A", "treatment", "INT64", "within_24h_grace_period"),
        AnalyticColumn("Y", "outcome", "INT64", "within_28d_follow_up"),
        AnalyticColumn("index_datetime", "time_zero", "DATETIME", "icu_admission"),
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
        execution_allowed: bool,
    ) -> tuple[str, CohortBuildManifest]:
        if not study_id.strip():
            raise ValueError("study_id is required")
        feasibility = albumin_sepsis_feasibility_sql(datasets)
        marker = "\n    SELECT\n      COUNT(*) AS eligible_persons"
        if marker not in feasibility:
            raise RuntimeError("Unexpected feasibility SQL structure")
        common_ctes = feasibility.split(marker, 1)[0]
        sql = common_ctes + f"""
    SELECT
      a.person_id,
      CASE WHEN a.has_albumin THEN 1 ELSE 0 END AS A,
      CASE WHEN a.death_28d THEN 1 ELSE 0 END AS Y,
      e.index_datetime,
      28 AS follow_up_days,
      CAST(NULL AS INT64) AS censored,
      EXTRACT(YEAR FROM DATE(e.index_datetime)) - p.year_of_birth AS age,
      p.gender_concept_id,
      p.race_concept_id
    FROM analytic a
    JOIN eligible e USING (person_id)
    JOIN {datasets.table('omop', 'person')} p USING (person_id)
    WHERE a.has_crystalloid
    """
        manifest = CohortBuildManifest(
            schema_version="1.0",
            study_id=study_id.strip(),
            status="ready_for_governed_build" if execution_allowed else "draft_not_executable",
            query_sha256=hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            contains_person_level_data=True,
            delivery_to_llm_allowed=False,
            execution_allowed=execution_allowed,
            columns=self.COLUMNS,
            unresolved_issues=(
                "censoring and observation-window completeness are not implemented",
                "non-demographic baseline covariates are not implemented",
                "Sepsis-3 phenotype is not implemented",
                "albumin and crystalloid mappings remain provisional",
                "septic-shock effect modifier is not implemented",
            ),
        )
        return sql, manifest

    def profile_sql(self, study_id: str, datasets: BigQueryDatasets) -> str:
        """Return an aggregate-only query safe for MCP feasibility output."""
        cohort_sql, _ = self.build(study_id, datasets, execution_allowed=False)
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

    def followup_audit_sql(self, study_id: str, datasets: BigQueryDatasets) -> str:
        """Audit nominal 28-day observability and return aggregate counts only."""
        cohort_sql, _ = self.build(study_id, datasets, execution_allowed=False)
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
            observation_end_date >= DATE_ADD(index_date, INTERVAL 28 DAY)
          ) AS observation_period_covers_28d,
          COUNTIF(
            death_date BETWEEN index_date AND DATE_ADD(index_date, INTERVAL 28 DAY)
          ) AS known_death_within_28d,
          COUNTIF(
            death_date BETWEEN index_date AND DATE_ADD(index_date, INTERVAL 28 DAY)
            OR observation_end_date >= DATE_ADD(index_date, INTERVAL 28 DAY)
          ) AS nominal_followup_complete,
          COUNTIF(
            (death_date IS NULL OR death_date > DATE_ADD(index_date, INTERVAL 28 DAY))
            AND (
              observation_end_date IS NULL
              OR observation_end_date < DATE_ADD(index_date, INTERVAL 28 DAY)
            )
          ) AS nominal_followup_incomplete
        FROM followup
        """
