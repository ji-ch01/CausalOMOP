import unittest

from causal.config import BigQueryDatasets
from causal.data_builder import AnalyticCohortBuilder


class AnalyticCohortBuilderTests(unittest.TestCase):
    def test_builds_non_executable_person_level_contract(self):
        sql, manifest = AnalyticCohortBuilder().build(
            "study", BigQueryDatasets(), execution_allowed=False
        )
        self.assertIn("CASE WHEN a.has_albumin THEN 1 ELSE 0 END AS A", sql)
        self.assertIn("CASE WHEN a.death_28d THEN 1 ELSE 0 END AS Y", sql)
        self.assertIn("CAST(NULL AS INT64) AS censored", sql)
        self.assertTrue(manifest.contains_person_level_data)
        self.assertFalse(manifest.delivery_to_llm_allowed)
        self.assertFalse(manifest.execution_allowed)
        self.assertEqual(manifest.status, "draft_not_executable")

    def test_profile_is_aggregate_only(self):
        sql = AnalyticCohortBuilder().profile_sql("study", BigQueryDatasets())
        self.assertIn("COUNT(*) AS cohort_persons", sql)
        self.assertIn("COUNTIF(censored IS NULL) AS unresolved_censoring", sql)
        self.assertNotIn("SELECT *", sql.upper())

    def test_followup_audit_is_aggregate_and_uses_observation_period(self):
        sql = AnalyticCohortBuilder().followup_audit_sql(
            "study", BigQueryDatasets()
        )

        self.assertIn("observation_period", sql)
        self.assertIn("nominal_followup_complete", sql)
        self.assertIn("DATE_ADD(index_date, INTERVAL 28 DAY)", sql)
        self.assertIn("COUNT(*) AS cohort_persons", sql)
        self.assertNotIn("SELECT *", sql.upper())


if __name__ == "__main__":
    unittest.main()
