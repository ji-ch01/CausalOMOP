import unittest

from causal.config import BigQueryDatasets
from causal.queries import ConceptSetSpec, feasibility_sql

SEPSIS = ConceptSetSpec(domain="condition", concept_ids=(132797,))
ALBUMIN = ConceptSetSpec(domain="drug", concept_ids=(1,))
CRYSTALLOID = ConceptSetSpec(domain="drug", concept_ids=(2,))
DEATH = ConceptSetSpec(domain="death")


class FeasibilityQueryTests(unittest.TestCase):
    def test_query_contains_no_unbound_parameters(self):
        sql = feasibility_sql(
            BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
        )
        self.assertNotIn("@", sql)
        self.assertIn("omop_cdm", sql)
        self.assertIn("COUNT(*) AS eligible_persons", sql)

    def test_non_death_outcome_uses_its_own_domain_table(self):
        outcome = ConceptSetSpec(domain="measurement", concept_ids=(3016723,))
        sql = feasibility_sql(
            BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=outcome,
        )
        self.assertIn("measurement_concept_id", sql)
        self.assertNotIn("death_date", sql)

    def test_rejects_unmapped_concept_ids(self):
        unmapped = ConceptSetSpec(domain="drug", concept_ids=())
        with self.assertRaises(ValueError):
            feasibility_sql(
                BigQueryDatasets(),
                population=SEPSIS, treatment=unmapped, comparator=CRYSTALLOID, outcome=DEATH,
            )

    def test_rejects_unknown_domain(self):
        bad = ConceptSetSpec(domain="not_a_domain", concept_ids=(1,))
        with self.assertRaises(ValueError):
            feasibility_sql(
                BigQueryDatasets(),
                population=bad, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            )

    def test_reports_outcome_events_during_grace_period_for_death_outcome(self):
        sql = feasibility_sql(
            BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            grace_period_hours=24,
        )
        self.assertIn("outcome_event_during_grace_period", sql)
        self.assertIn("COUNTIF(outcome_event_during_grace_period) AS outcome_events_during_grace_period", sql)
        self.assertIn("INTERVAL 24 HOUR", sql)

    def test_reports_outcome_events_during_grace_period_for_domain_outcome(self):
        outcome = ConceptSetSpec(domain="measurement", concept_ids=(3016723,))
        sql = feasibility_sql(
            BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=outcome,
        )
        self.assertIn("outcome_event_during_grace_period", sql)

    def test_optional_care_site_filter(self):
        sql = feasibility_sql(
            BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            care_site_names=("Medical Intensive Care Unit (MICU)",),
        )
        self.assertIn("care_site_name IN", sql)
        self.assertIn("Medical Intensive Care Unit (MICU)", sql)


DUCKDB_DATASETS = BigQueryDatasets(backend="duckdb", duckdb_path="data/mimiciv_omop.duckdb", duckdb_schema="main")


class DuckDBFeasibilityQueryTests(unittest.TestCase):
    """Same SQL generator, `backend="duckdb"` -- must drop every
    BigQuery-only construct (`DATETIME_ADD`, backtick-qualified
    `project.dataset.table` identifiers) that DuckDB's parser rejects."""

    def test_no_datetime_add_or_backtick_identifiers(self):
        sql = feasibility_sql(
            DUCKDB_DATASETS,
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
        )
        self.assertNotIn("DATETIME_ADD", sql)
        self.assertNotIn("`", sql)
        self.assertIn('"main"."person"', sql)
        self.assertIn("INTERVAL (24) HOUR", sql)

    def test_negative_covariate_window_hours_are_parenthesized(self):
        from causal.queries.feasibility import CovariateSpec, analytic_cohort_sql

        covariate = CovariateSpec(
            column_name="diabetes",
            domain="condition",
            concept_ids=(201820,),
            window_start_hours=-876000,
            window_end_hours=0,
            aggregation="presence_in_window",
        )
        sql = analytic_cohort_sql(
            DUCKDB_DATASETS,
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            covariates=(covariate,),
        )
        self.assertIn("INTERVAL (-876000) HOUR", sql)
        self.assertNotIn("DATETIME_ADD", sql)

    def test_bigquery_output_is_unchanged_by_the_duckdb_branch(self):
        sql = feasibility_sql(
            BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
        )
        self.assertIn("DATETIME_ADD(e.index_datetime, INTERVAL 24 HOUR)", sql)


if __name__ == "__main__":
    unittest.main()
