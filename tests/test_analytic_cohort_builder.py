import unittest

from causal.config import BigQueryDatasets
from causal.data_builder import AnalyticCohortBuilder
from causal.queries import ConceptSetSpec

SEPSIS = ConceptSetSpec(domain="condition", concept_ids=(132797,))
ALBUMIN = ConceptSetSpec(domain="drug", concept_ids=(1,))
CRYSTALLOID = ConceptSetSpec(domain="drug", concept_ids=(2,))
DEATH = ConceptSetSpec(domain="death")


class AnalyticCohortBuilderTests(unittest.TestCase):
    def test_builds_non_executable_person_level_contract(self):
        sql, manifest = AnalyticCohortBuilder().build(
            "study", BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            execution_allowed=False,
        )
        self.assertIn("CASE WHEN a.has_treatment THEN 1 ELSE 0 END AS A", sql)
        self.assertIn("CASE WHEN a.outcome_event THEN 1 ELSE 0 END AS Y", sql)
        self.assertIn("CAST(NULL AS INT64) AS censored", sql)
        self.assertTrue(manifest.contains_person_level_data)
        self.assertFalse(manifest.delivery_to_llm_allowed)
        self.assertFalse(manifest.execution_allowed)
        self.assertEqual(manifest.status, "draft_not_executable")
        self.assertEqual(manifest.exposure_contrast, "add_on")

    def test_add_on_contrast_requires_the_comparator_backbone(self):
        """Default design (e.g. crystalloids +/- albumin): everyone in the
        cohort must have received the comparator; A marks the add-on."""
        sql, manifest = AnalyticCohortBuilder().build(
            "study", BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            exposure_contrast="add_on", execution_allowed=False,
        )
        self.assertIn("WHERE a.has_comparator", sql)
        self.assertEqual(manifest.exposure_contrast, "add_on")

    def test_mutually_exclusive_contrast_excludes_both_and_neither(self):
        """Active-comparator design (drug A vs drug B): only persons with
        exactly one of treatment/comparator are kept."""
        sql, manifest = AnalyticCohortBuilder().build(
            "study", BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            exposure_contrast="mutually_exclusive", execution_allowed=False,
        )
        self.assertIn("WHERE a.has_treatment != a.has_comparator", sql)
        self.assertNotIn("WHERE a.has_comparator\n", sql)
        self.assertEqual(manifest.exposure_contrast, "mutually_exclusive")

    def test_rejects_unknown_exposure_contrast(self):
        with self.assertRaises(ValueError):
            AnalyticCohortBuilder().build(
                "study", BigQueryDatasets(),
                population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
                exposure_contrast="head_to_head", execution_allowed=False,
            )

    def test_profile_is_aggregate_only(self):
        sql = AnalyticCohortBuilder().profile_sql(
            "study", BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
        )
        self.assertIn("COUNT(*) AS cohort_persons", sql)
        self.assertIn("COUNTIF(censored IS NULL) AS unresolved_censoring", sql)
        self.assertNotIn("SELECT *", sql.upper())

    def test_followup_audit_is_aggregate_and_uses_observation_period(self):
        sql = AnalyticCohortBuilder().followup_audit_sql(
            "study", BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
        )

        self.assertIn("observation_period", sql)
        self.assertIn("nominal_followup_complete", sql)
        self.assertIn("DATE_ADD(index_date, INTERVAL 28 DAY)", sql)
        self.assertIn("COUNT(*) AS cohort_persons", sql)
        self.assertNotIn("SELECT *", sql.upper())

    def test_builder_generalizes_to_a_different_question(self):
        """The whole point of the generic builder: a different domain mix
        (measurement population, procedure outcome) works without touching
        any Python code, just different ConceptSetSpecs."""
        population = ConceptSetSpec(domain="measurement", concept_ids=(3004249,))
        treatment = ConceptSetSpec(domain="procedure", concept_ids=(4230167,))
        comparator = ConceptSetSpec(domain="procedure", concept_ids=(4163872,))
        outcome = ConceptSetSpec(domain="condition", concept_ids=(316866,))
        sql, manifest = AnalyticCohortBuilder().build(
            "study", BigQueryDatasets(),
            population=population, treatment=treatment, comparator=comparator, outcome=outcome,
            exposure_contrast="mutually_exclusive", execution_allowed=False,
        )
        self.assertIn("measurement_concept_id", sql)
        self.assertIn("procedure_occurrence", sql)
        self.assertIn("condition_occurrence", sql)
        # This is exactly the design the default "add_on" contrast gets wrong:
        # two mutually exclusive procedures, so both arms must be reachable
        # as A=1, not silently dropped by an add_on-style backbone filter.
        self.assertIn("WHERE a.has_treatment != a.has_comparator", sql)
        self.assertEqual(manifest.exposure_contrast, "mutually_exclusive")


if __name__ == "__main__":
    unittest.main()
