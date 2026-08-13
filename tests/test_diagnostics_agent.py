import unittest

import pandas as pd

from causal.agents.diagnostics import DiagnosticsAgent


def _balanced_and_imbalanced_dataset() -> pd.DataFrame:
    """20 persons per arm. `age` is identical between arms (balanced);
    `cov_sofa_score` is shifted 5 points higher in the treated arm
    (imbalanced); `gender_concept_id` is evenly split in both arms
    (balanced categorical)."""
    n_per_arm = 20
    rows = []
    for i in range(n_per_arm):
        rows.append({
            "A": 1, "age": 50 + (i % 5), "cov_sofa_score": 10.0 + (i % 3),
            "gender_concept_id": 8507 if i % 2 == 0 else 8532,
        })
    for i in range(n_per_arm):
        rows.append({
            "A": 0, "age": 50 + (i % 5), "cov_sofa_score": 4.0 + (i % 3),
            "gender_concept_id": 8507 if i % 2 == 0 else 8532,
        })
    return pd.DataFrame(rows)


class DiagnosticsAgentTests(unittest.TestCase):
    def setUp(self):
        self.dataset = _balanced_and_imbalanced_dataset()
        self.data_types = {"age": "INT64", "cov_sofa_score": "FLOAT64", "gender_concept_id": "INT64"}

    def test_flags_a_meaningfully_imbalanced_continuous_covariate(self):
        artifact = DiagnosticsAgent().run(
            self.dataset, study_id="test_study",
            covariate_columns=("cov_sofa_score",), covariate_data_types=self.data_types,
        )
        entry = artifact.balance[0]
        self.assertEqual(entry.column_name, "cov_sofa_score")
        self.assertGreater(abs(entry.smd), 0.1)
        self.assertTrue(entry.flagged)
        self.assertEqual(artifact.n_covariates_flagged, 1)

    def test_does_not_flag_a_balanced_continuous_covariate(self):
        artifact = DiagnosticsAgent().run(
            self.dataset, study_id="test_study",
            covariate_columns=("age",), covariate_data_types=self.data_types,
        )
        entry = artifact.balance[0]
        self.assertAlmostEqual(entry.smd, 0.0, places=6)
        self.assertFalse(entry.flagged)

    def test_categorical_covariate_produces_one_row_per_level(self):
        artifact = DiagnosticsAgent().run(
            self.dataset, study_id="test_study",
            covariate_columns=("gender_concept_id",), covariate_data_types=self.data_types,
        )
        names = {entry.column_name for entry in artifact.balance}
        self.assertEqual(names, {"gender_concept_id=8507", "gender_concept_id=8532"})
        for entry in artifact.balance:
            self.assertFalse(entry.flagged)

    def test_binary_presence_covariate_uses_proportion_smd(self):
        dataset = self.dataset.copy()
        dataset["cov_has_condition"] = [1] * 20 + [0] * 20
        artifact = DiagnosticsAgent().run(
            dataset, study_id="test_study",
            covariate_columns=("cov_has_condition",),
            covariate_data_types={"cov_has_condition": "INT64"},
        )
        entry = artifact.balance[0]
        self.assertEqual(entry.treated_mean, 1.0)
        self.assertEqual(entry.comparator_mean, 0.0)
        self.assertTrue(entry.flagged)

    def test_citation_and_threshold_are_reported(self):
        artifact = DiagnosticsAgent().run(
            self.dataset, study_id="test_study",
            covariate_columns=("age",), covariate_data_types=self.data_types,
        )
        self.assertIn("Austin", artifact.citation)
        self.assertEqual(artifact.smd_flag_threshold, 0.1)
        self.assertEqual(artifact.n_total, 40)
        self.assertEqual(artifact.n_treated, 20)
        self.assertEqual(artifact.n_comparator, 20)

    def test_rejects_a_single_treatment_arm(self):
        dataset = self.dataset[self.dataset["A"] == 1]
        with self.assertRaises(ValueError):
            DiagnosticsAgent().run(
                dataset, study_id="test_study",
                covariate_columns=("age",), covariate_data_types=self.data_types,
            )

    def test_rejects_missing_covariate_column(self):
        with self.assertRaises(ValueError):
            DiagnosticsAgent().run(
                self.dataset, study_id="test_study",
                covariate_columns=("does_not_exist",), covariate_data_types={},
            )


if __name__ == "__main__":
    unittest.main()
