"""Read-only BigQuery access and OMOP preflight checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from causal.agents.query_validation import QueryValidationAgent
from causal.config import BigQueryDatasets
from causal.mapping.audit import concept_mapping_coverage_sql
from causal.queries.feasibility import ConceptSetSpec, feasibility_sql


REQUIRED_OMOP_TABLES = (
    "person", "observation_period", "visit_occurrence", "condition_occurrence",
    "drug_exposure", "measurement", "procedure_occurrence", "death", "concept",
    "concept_ancestor",
)


@dataclass(frozen=True)
class PreflightResult:
    accessible_tables: tuple[str, ...]
    missing_tables: tuple[str, ...]
    person_count: int | None = None

    @property
    def ready(self) -> bool:
        return not self.missing_tables


class BigQueryOMOPGateway:
    """Narrow gateway: analysis reads OMOP; writes must target metrics."""

    def __init__(self, datasets: BigQueryDatasets, client: Any) -> None:
        self.datasets = datasets.validated()
        self.client = client

    def preflight(self, count_persons: bool = False) -> PreflightResult:
        dataset = self.client.get_dataset(self.datasets.fq_dataset("omop"))
        available = {table.table_id for table in self.client.list_tables(dataset)}
        present = tuple(name for name in REQUIRED_OMOP_TABLES if name in available)
        missing = tuple(name for name in REQUIRED_OMOP_TABLES if name not in available)
        count = None
        if count_persons and not missing:
            sql = f"SELECT COUNT(*) AS n FROM {self.datasets.table('omop', 'person')}"
            count = int(next(iter(self.client.query(sql).result())).n)
        return PreflightResult(present, missing, count)

    def search_concepts(self, term: str, limit: int = 20) -> list[dict[str, Any]]:
        """Find candidate OMOP concepts without interpolating user text into SQL."""
        if not term.strip():
            raise ValueError("term is required")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        try:
            from google.cloud import bigquery
        except ImportError as exc:
            raise RuntimeError("Install requirements-bigquery.txt") from exc

        sql = f"""
        SELECT concept_id, concept_name, domain_id, vocabulary_id,
               concept_class_id, standard_concept, invalid_reason
        FROM {self.datasets.table('omop', 'concept')}
        WHERE LOWER(concept_name) LIKE CONCAT('%', LOWER(@term), '%')
          AND invalid_reason IS NULL
        ORDER BY
          CASE WHEN standard_concept = 'S' THEN 0 ELSE 1 END,
          CASE WHEN LOWER(concept_name) = LOWER(@term) THEN 0 ELSE 1 END,
          concept_name
        LIMIT @limit
        """
        config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("term", "STRING", term),
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
        ])
        return [dict(row.items()) for row in self.client.query(sql, job_config=config).result()]

    def search_used_concepts(self, domain: str, term: str, limit: int = 20) -> list[dict[str, Any]]:
        """Search concepts that actually occur in the converted clinical tables."""
        locations = {
            "condition": ("condition_occurrence", "condition_concept_id"),
            "drug": ("drug_exposure", "drug_concept_id"),
        }
        if domain not in locations:
            raise ValueError("domain must be 'condition' or 'drug'")
        if not term.strip() or not 1 <= limit <= 100:
            raise ValueError("term is required and limit must be between 1 and 100")
        try:
            from google.cloud import bigquery
        except ImportError as exc:
            raise RuntimeError("Install requirements-bigquery.txt") from exc
        table, column = locations[domain]
        sql = f"""
        SELECT c.concept_id, c.concept_name, c.domain_id, c.vocabulary_id,
               c.concept_class_id, c.standard_concept,
               COUNT(*) AS row_count,
               COUNT(DISTINCT e.person_id) AS person_count
        FROM {self.datasets.table('omop', table)} e
        JOIN {self.datasets.table('omop', 'concept')} c
          ON e.{column} = c.concept_id
        WHERE LOWER(c.concept_name) LIKE CONCAT('%', LOWER(@term), '%')
        GROUP BY 1, 2, 3, 4, 5, 6
        ORDER BY person_count DESC, row_count DESC
        LIMIT @limit
        """
        config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("term", "STRING", term),
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
        ])
        return [dict(row.items()) for row in self.client.query(sql, job_config=config).result()]

    def search_source_values(self, domain: str, term: str, limit: int = 20) -> list[dict[str, Any]]:
        """Audit source values, including rows whose standard concept is unmapped."""
        locations = {
            "condition": ("condition_occurrence", "condition_source_value", "condition_concept_id"),
            "drug": ("drug_exposure", "drug_source_value", "drug_concept_id"),
        }
        if domain not in locations:
            raise ValueError("domain must be 'condition' or 'drug'")
        if not term.strip() or not 1 <= limit <= 100:
            raise ValueError("term is required and limit must be between 1 and 100")
        try:
            from google.cloud import bigquery
        except ImportError as exc:
            raise RuntimeError("Install requirements-bigquery.txt") from exc
        table, source_column, standard_column = locations[domain]
        sql = f"""
        SELECT e.{source_column} AS source_value, e.{standard_column} AS standard_concept_id,
               COALESCE(c.concept_name, 'Unmapped') AS standard_concept_name,
               COUNT(*) AS row_count, COUNT(DISTINCT e.person_id) AS person_count
        FROM {self.datasets.table('omop', table)} e
        LEFT JOIN {self.datasets.table('omop', 'concept')} c
          ON e.{standard_column} = c.concept_id
        WHERE LOWER(e.{source_column}) LIKE CONCAT('%', LOWER(@term), '%')
        GROUP BY 1, 2, 3
        ORDER BY person_count DESC, row_count DESC
        LIMIT @limit
        """
        config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("term", "STRING", term),
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
        ])
        return [dict(row.items()) for row in self.client.query(sql, job_config=config).result()]

    def audit_visit_details(self, limit: int = 100) -> list[dict[str, Any]]:
        """Summarize visit-detail locations used to define ICU time zero."""
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        try:
            from google.cloud import bigquery
        except ImportError as exc:
            raise RuntimeError("Install requirements-bigquery.txt") from exc
        sql = f"""
        SELECT
          vd.visit_detail_concept_id,
          COALESCE(c.concept_name, 'Unmapped') AS visit_detail_concept_name,
          cs.care_site_name,
          cs.care_site_source_value,
          COUNT(*) AS detail_count,
          COUNT(DISTINCT vd.person_id) AS person_count,
          COUNT(DISTINCT vd.visit_occurrence_id) AS visit_count,
          COUNTIF(vd.visit_detail_start_datetime IS NULL) AS missing_start_datetime,
          COUNTIF(vd.visit_detail_end_datetime IS NULL) AS missing_end_datetime
        FROM {self.datasets.table('omop', 'visit_detail')} vd
        LEFT JOIN {self.datasets.table('omop', 'concept')} c
          ON vd.visit_detail_concept_id = c.concept_id
        LEFT JOIN {self.datasets.table('omop', 'care_site')} cs
          ON vd.care_site_id = cs.care_site_id
        GROUP BY 1, 2, 3, 4
        ORDER BY person_count DESC, detail_count DESC
        LIMIT @limit
        """
        config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
        ])
        return [dict(row.items()) for row in self.client.query(sql, job_config=config).result()]

    def mapping_coverage(self, spec: ConceptSetSpec, field: str) -> dict[str, int]:
        """Mapped-vs-unmapped-name-matched row counts for one concept set
        (see `causal.mapping.audit.concept_mapping_coverage_sql`) -- the
        diagnostic `causal.orchestrator.workflow`'s feasibility gate uses to
        block on an unmapped exposure/outcome/population before any cohort
        SQL runs.
        """
        try:
            from google.cloud import bigquery  # noqa: F401  (import-guard only)
        except ImportError as exc:
            raise RuntimeError("Install requirements-bigquery.txt") from exc
        sql = concept_mapping_coverage_sql(self.datasets, spec, field)
        rows = list(self.client.query(sql).result())
        return dict(rows[0].items()) if rows else {}

    def feasibility(
        self,
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
    ) -> dict[str, Any]:
        """Run provisional cohort counts for one target-trial question.

        This deliberately estimates no causal effect: it returns aggregate
        eligibility/exposure/outcome counts only, for pipeline feasibility.
        """
        try:
            from google.cloud import bigquery  # noqa: F401  (import-guard only)
        except ImportError as exc:
            raise RuntimeError("Install requirements-bigquery.txt") from exc
        sql = feasibility_sql(
            self.datasets,
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
        validation = QueryValidationAgent().run(sql, purpose="aggregate_feasibility")
        if not validation.passed:
            raise RuntimeError(
                f"Feasibility SQL failed governance validation: {', '.join(validation.violations)}"
            )
        rows = list(self.client.query(sql).result())
        counts = dict(rows[0].items()) if rows else {}
        return {
            "status": "provisional_cohort_feasibility",
            "causal_effect_estimated": False,
            "grace_period_hours": grace_period_hours,
            "follow_up_days": follow_up_days,
            "washout_days": washout_days,
            "counts": counts,
            "warnings": [
                *(
                    [
                        "washout_days > 0: a large washout_excluded_insufficient_lookback "
                        "count reflects MIMIC-IV's thin pre-admission history, not a real "
                        "prevalent-user population -- read it before approving the new-user rule."
                    ]
                    if washout_days > 0
                    else []
                ),
                "Concept sets reflect current human-approved mappings only; unmapped "
                "roles will fail with a clear error rather than silently degrading.",
                "Counts are for pipeline feasibility and must not be interpreted causally.",
                "outcome_events_during_grace_period counts events that occurred inside "
                "the exposure-ascertainment window itself, before treatment status was "
                "determined; a large share of outcome_events falling here warrants review "
                "before trusting the arm each such person was assigned to.",
            ],
        }


def make_google_client(project: str):
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise RuntimeError(
            "BigQuery support is optional. Install with: pip install google-cloud-bigquery"
        ) from exc
    return bigquery.Client(project=project)
