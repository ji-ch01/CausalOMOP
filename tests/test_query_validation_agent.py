import unittest

from causal.agents.query_validation import QueryValidationAgent
from causal.config import BigQueryDatasets
from causal.queries import ConceptSetSpec, analytic_cohort_sql, feasibility_sql

SEPSIS = ConceptSetSpec(domain="condition", concept_ids=(132797,))
ALBUMIN = ConceptSetSpec(domain="drug", concept_ids=(1,))
CRYSTALLOID = ConceptSetSpec(domain="drug", concept_ids=(2,))
DEATH = ConceptSetSpec(domain="death")


class FakeDryRunJob:
    def __init__(self, total_bytes_processed: int) -> None:
        self.total_bytes_processed = total_bytes_processed


class FakeDryRunClient:
    def __init__(self, total_bytes_processed: int = 12345) -> None:
        self.total_bytes_processed = total_bytes_processed
        self.last_job_config = None

    def query(self, sql, job_config=None):
        self.last_job_config = job_config
        return FakeDryRunJob(self.total_bytes_processed)


class FailingDryRunClient:
    def query(self, sql, job_config=None):
        raise RuntimeError("Syntax error at line 3")


class QueryValidationAgentTests(unittest.TestCase):
    def setUp(self):
        self.feasibility_sql = feasibility_sql(
            BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
        )
        self.cohort_sql = analytic_cohort_sql(
            BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
        )

    def test_passes_a_well_formed_aggregate_query(self):
        result = QueryValidationAgent().run(self.feasibility_sql, purpose="aggregate_feasibility")
        self.assertTrue(result.passed)
        self.assertEqual(result.violations, ())
        self.assertFalse(result.dry_run_performed)

    def test_passes_a_well_formed_person_level_query(self):
        result = QueryValidationAgent().run(self.cohort_sql, purpose="person_level_dataset")
        self.assertTrue(result.passed)

    def test_rejects_unbound_parameter_placeholder(self):
        result = QueryValidationAgent().run(
            self.feasibility_sql + " WHERE person_id = @person_id",
            purpose="aggregate_feasibility",
        )
        self.assertFalse(result.passed)
        self.assertIn("unbound_parameter_placeholder_present", result.violations)

    def test_rejects_select_star(self):
        result = QueryValidationAgent().run("SELECT * FROM t", purpose="person_level_dataset")
        self.assertFalse(result.passed)
        self.assertIn("select_star_present", result.violations)

    def test_rejects_unexpected_table_reference(self):
        sql = self.feasibility_sql.replace(
            "`example-gcp-project.omop_cdm.death`",
            "`example-gcp-project.omop_cdm.some_other_table`",
        )
        result = QueryValidationAgent().run(sql, purpose="aggregate_feasibility")
        self.assertFalse(result.passed)
        self.assertTrue(any("unexpected_table_reference" in v for v in result.violations))

    def test_rejects_person_id_leaking_into_aggregate_final_projection(self):
        leaking_sql = self.feasibility_sql.replace(
            "COUNT(*) AS eligible_persons,", "COUNT(*) AS eligible_persons, e.person_id,"
        )
        result = QueryValidationAgent().run(leaking_sql, purpose="aggregate_feasibility")
        self.assertFalse(result.passed)
        self.assertIn("forbidden_column_in_final_projection:person_id", result.violations)

    def test_person_id_in_a_cte_does_not_fail_an_aggregate_query(self):
        """person_id appears throughout the CTEs of a normal feasibility
        query (joins, GROUP BY); only the final projection matters."""
        result = QueryValidationAgent().run(self.feasibility_sql, purpose="aggregate_feasibility")
        self.assertTrue(result.passed)

    def test_dry_run_reports_estimated_bytes(self):
        client = FakeDryRunClient(total_bytes_processed=999)
        result = QueryValidationAgent().run(
            self.feasibility_sql, purpose="aggregate_feasibility", dry_run_client=client
        )
        self.assertTrue(result.passed)
        self.assertTrue(result.dry_run_performed)
        self.assertEqual(result.estimated_bytes_processed, 999)
        self.assertTrue(client.last_job_config.dry_run)

    def test_dry_run_failure_is_a_violation_not_an_exception(self):
        result = QueryValidationAgent().run(
            self.feasibility_sql, purpose="aggregate_feasibility", dry_run_client=FailingDryRunClient()
        )
        self.assertFalse(result.passed)
        self.assertTrue(any("dry_run_failed" in v for v in result.violations))

    def test_rejects_unknown_purpose(self):
        with self.assertRaises(ValueError):
            QueryValidationAgent().run(self.feasibility_sql, purpose="not_a_real_purpose")

    def test_rejects_empty_sql(self):
        with self.assertRaises(ValueError):
            QueryValidationAgent().run("   ", purpose="aggregate_feasibility")


if __name__ == "__main__":
    unittest.main()
