"""Reviewed BigQuery SQL for the provisional albumin/sepsis feasibility case."""

from __future__ import annotations

from causal.bigquery import DEFAULT_ICU_CARE_SITES
from causal.config import BigQueryDatasets


def albumin_sepsis_feasibility_sql(
    datasets: BigQueryDatasets,
    *,
    icu_care_sites: tuple[str, ...] = DEFAULT_ICU_CARE_SITES,
    sepsis_seed_concept_id: int = 132797,
    grace_period_hours: int = 24,
    follow_up_days: int = 28,
) -> str:
    """Render trusted literals for OMCP, which does not bind BigQuery parameters."""
    if not icu_care_sites or sepsis_seed_concept_id <= 0:
        raise ValueError("ICU care sites and a positive sepsis seed are required")
    if grace_period_hours <= 0 or follow_up_days <= 0:
        raise ValueError("Study windows must be positive")
    names = ", ".join(_string_literal(value) for value in icu_care_sites)
    return f"""
    WITH sepsis_concepts AS (
      SELECT descendant_concept_id AS concept_id
      FROM {datasets.table('omop', 'concept_ancestor')}
      WHERE ancestor_concept_id = {int(sepsis_seed_concept_id)}
      UNION DISTINCT SELECT {int(sepsis_seed_concept_id)}
    ),
    first_icu AS (
      SELECT
        vd.person_id, vd.visit_occurrence_id, vd.visit_detail_id,
        vd.visit_detail_start_datetime AS index_datetime,
        cs.care_site_name
      FROM {datasets.table('omop', 'visit_detail')} vd
      JOIN {datasets.table('omop', 'care_site')} cs USING (care_site_id)
      WHERE cs.care_site_name IN ({names})
        AND vd.visit_detail_start_datetime IS NOT NULL
      QUALIFY ROW_NUMBER() OVER (
        PARTITION BY vd.person_id ORDER BY vd.visit_detail_start_datetime, vd.visit_detail_id
      ) = 1
    ),
    eligible AS (
      SELECT DISTINCT i.*
      FROM first_icu i
      JOIN {datasets.table('omop', 'person')} p USING (person_id)
      JOIN {datasets.table('omop', 'condition_occurrence')} co
        ON co.person_id = i.person_id
       AND co.visit_occurrence_id = i.visit_occurrence_id
      JOIN sepsis_concepts sc ON co.condition_concept_id = sc.concept_id
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
      LEFT JOIN {datasets.table('omop', 'drug_exposure')} de
        ON de.person_id = e.person_id
       AND de.visit_occurrence_id = e.visit_occurrence_id
       AND de.drug_exposure_start_datetime >= e.index_datetime
       AND de.drug_exposure_start_datetime < DATETIME_ADD(
         e.index_datetime, INTERVAL {int(grace_period_hours)} HOUR
       )
      GROUP BY e.person_id
    ),
    analytic AS (
      SELECT
        e.person_id,
        x.has_albumin,
        x.has_crystalloid,
        d.death_date IS NOT NULL
          AND d.death_date >= DATE(e.index_datetime)
          AND d.death_date <= DATE_ADD(
            DATE(e.index_datetime), INTERVAL {int(follow_up_days)} DAY
          ) AS death_28d
      FROM eligible e
      JOIN exposure x USING (person_id)
      LEFT JOIN {datasets.table('omop', 'death')} d USING (person_id)
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


def _string_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"
