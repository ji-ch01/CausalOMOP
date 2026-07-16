"""Read-only BigQuery access and OMOP preflight checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from causal.config import BigQueryDatasets


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


DEFAULT_ICU_CARE_SITES = (
    "Medical Intensive Care Unit (MICU)",
    "Cardiac Vascular Intensive Care Unit (CVICU)",
    "Medical/Surgical Intensive Care Unit (MICU/SICU)",
    "Surgical Intensive Care Unit (SICU)",
    "Trauma SICU (TSICU)",
    "Coronary Care Unit (CCU)",
    "Neuro Surgical Intensive Care Unit (Neuro SICU)",
)


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

    def albumin_sepsis_feasibility(
        self,
        icu_care_sites: tuple[str, ...] = DEFAULT_ICU_CARE_SITES,
        sepsis_seed_concept_id: int = 132797,
        grace_period_hours: int = 24,
        follow_up_days: int = 28,
    ) -> dict[str, Any]:
        """Run provisional cohort counts; this deliberately estimates no effect."""
        if not icu_care_sites or sepsis_seed_concept_id <= 0:
            raise ValueError("ICU care sites and a positive sepsis seed are required")
        try:
            from google.cloud import bigquery
        except ImportError as exc:
            raise RuntimeError("Install requirements-bigquery.txt") from exc
        sql = f"""
        WITH sepsis_concepts AS (
          SELECT descendant_concept_id AS concept_id
          FROM {self.datasets.table('omop', 'concept_ancestor')}
          WHERE ancestor_concept_id = @sepsis_seed
          UNION DISTINCT SELECT @sepsis_seed
        ),
        first_icu AS (
          SELECT
            vd.person_id, vd.visit_occurrence_id, vd.visit_detail_id,
            vd.visit_detail_start_datetime AS index_datetime,
            cs.care_site_name
          FROM {self.datasets.table('omop', 'visit_detail')} vd
          JOIN {self.datasets.table('omop', 'care_site')} cs USING (care_site_id)
          WHERE cs.care_site_name IN UNNEST(@icu_names)
            AND vd.visit_detail_start_datetime IS NOT NULL
          QUALIFY ROW_NUMBER() OVER (
            PARTITION BY vd.person_id ORDER BY vd.visit_detail_start_datetime, vd.visit_detail_id
          ) = 1
        ),
        eligible AS (
          SELECT DISTINCT i.*
          FROM first_icu i
          JOIN {self.datasets.table('omop', 'person')} p USING (person_id)
          JOIN {self.datasets.table('omop', 'condition_occurrence')} co
            ON co.person_id = i.person_id
           AND co.visit_occurrence_id = i.visit_occurrence_id
          JOIN sepsis_concepts sc
            ON co.condition_concept_id = sc.concept_id
          WHERE EXTRACT(YEAR FROM DATE(i.index_datetime)) - p.year_of_birth >= 18
        ),
        exposure AS (
          SELECT
            e.person_id,
            COUNTIF(REGEXP_CONTAINS(de.drug_source_value, r'(?i)^Albumin(?:, Human)?\\b')) > 0 AS has_albumin,
            COUNTIF(REGEXP_CONTAINS(
              de.drug_source_value,
              r'(?i)(Lactated Ringers|Sodium Chloride).*(Bag|Bottle|Injection)$'
            )) > 0 AS has_crystalloid
          FROM eligible e
          LEFT JOIN {self.datasets.table('omop', 'drug_exposure')} de
            ON de.person_id = e.person_id
           AND de.visit_occurrence_id = e.visit_occurrence_id
           AND de.drug_exposure_start_datetime >= e.index_datetime
           AND de.drug_exposure_start_datetime < DATETIME_ADD(e.index_datetime, INTERVAL @grace HOUR)
          GROUP BY e.person_id
        ),
        analytic AS (
          SELECT
            e.person_id,
            x.has_albumin,
            x.has_crystalloid,
            d.death_date IS NOT NULL
              AND d.death_date >= DATE(e.index_datetime)
              AND d.death_date <= DATE_ADD(DATE(e.index_datetime), INTERVAL @follow_up DAY) AS death_28d
          FROM eligible e
          JOIN exposure x USING (person_id)
          LEFT JOIN {self.datasets.table('omop', 'death')} d USING (person_id)
        )
        SELECT
          COUNT(*) AS eligible_persons,
          COUNTIF(has_crystalloid) AS any_crystalloid,
          COUNTIF(has_albumin) AS any_albumin,
          COUNTIF(has_crystalloid AND has_albumin) AS albumin_plus_crystalloid,
          COUNTIF(has_crystalloid AND NOT has_albumin) AS crystalloid_only,
          COUNTIF(NOT has_crystalloid AND NOT has_albumin) AS neither_exposure,
          COUNTIF(death_28d) AS deaths_28d,
          COUNTIF(death_28d AND has_crystalloid AND has_albumin) AS deaths_albumin_plus_crystalloid,
          COUNTIF(death_28d AND has_crystalloid AND NOT has_albumin) AS deaths_crystalloid_only
        FROM analytic
        """
        config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("icu_names", "STRING", list(icu_care_sites)),
            bigquery.ScalarQueryParameter("sepsis_seed", "INT64", sepsis_seed_concept_id),
            bigquery.ScalarQueryParameter("grace", "INT64", grace_period_hours),
            bigquery.ScalarQueryParameter("follow_up", "INT64", follow_up_days),
        ])
        rows = list(self.client.query(sql, job_config=config).result())
        counts = dict(rows[0].items()) if rows else {}
        return {
            "status": "provisional_source_value_phenotype",
            "causal_effect_estimated": False,
            "icu_care_sites": list(icu_care_sites),
            "sepsis_seed_concept_id": sepsis_seed_concept_id,
            "grace_period_hours": grace_period_hours,
            "follow_up_days": follow_up_days,
            "counts": counts,
            "warnings": [
                "Sepsis uses a diagnosis concept set, not the published Sepsis-3 phenotype.",
                "Albumin and crystalloid exposure use unvalidated source-value regular expressions.",
                "Sodium-chloride products may include non-resuscitation administrations.",
                "Counts are for pipeline feasibility and must not be interpreted causally.",
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
