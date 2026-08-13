import unittest

import numpy as np
import pandas as pd

from causal.agents.estimation import EstimationAgent


def _synthetic_dataset(n: int, true_risk_difference: float, seed: int = 0) -> pd.DataFrame:
    """Rows where A is randomized (so no confounding) and Y's probability is
    a simple function of A and one continuous covariate -- lets a test check
    that g-computation recovers a known risk difference at a fixed seed."""
    rng = np.random.default_rng(seed)
    covariate = rng.normal(0, 1, size=n)
    treatment = rng.integers(0, 2, size=n)
    baseline_risk = 0.3 + 0.05 * covariate
    risk = np.clip(baseline_risk + true_risk_difference * treatment, 0.01, 0.99)
    outcome = rng.binomial(1, risk)
    return pd.DataFrame({"A": treatment, "Y": outcome, "age": covariate})


def _informatively_censored_dataset(n: int, true_risk_difference: float, seed: int = 0) -> pd.DataFrame:
    """Like `_synthetic_dataset`, but with informative administrative
    censoring: the censoring probability depends on the same covariate that
    drives outcome risk, and a censored row's observed `Y` is coded 0
    regardless of the true outcome -- the real `causal.queries.feasibility.
    analytic_cohort_sql` semantics (a row that leaves observation before the
    follow-up horizon without a recorded event gets `Y=0`, not "unknown").
    Lets a test show `ignore_censoring_administrative_waiver` is biased by
    this and `ipcw_administrative_censoring` corrects for it.
    """
    rng = np.random.default_rng(seed)
    covariate = rng.normal(0, 1, size=n)
    treatment = rng.integers(0, 2, size=n)
    baseline_risk = 0.3 + 0.1 * covariate
    risk = np.clip(baseline_risk + true_risk_difference * treatment, 0.02, 0.98)
    true_outcome = rng.binomial(1, risk)
    p_censored = 1 / (1 + np.exp(-covariate))
    censored = rng.binomial(1, p_censored)
    observed_outcome = np.where(censored == 1, 0, true_outcome)
    return pd.DataFrame({"A": treatment, "Y": observed_outcome, "age": covariate, "censored": censored})


class EstimationAgentTests(unittest.TestCase):
    def test_recovers_a_known_risk_difference_within_tolerance(self):
        dataset = _synthetic_dataset(4000, true_risk_difference=0.15, seed=1)
        artifact = EstimationAgent().run(
            dataset, study_id="test_study", covariate_columns=("age",),
            n_bootstrap=200, random_seed=1,
        )
        estimand = artifact.estimand
        self.assertAlmostEqual(estimand.point_estimate, 0.15, delta=0.03)
        self.assertLess(estimand.ci_lower, estimand.point_estimate)
        self.assertGreater(estimand.ci_upper, estimand.point_estimate)
        self.assertEqual(estimand.effect_measure, "risk_difference")
        self.assertEqual(estimand.method, "g_computation_standardization")
        self.assertIn("Hernan", estimand.citation)
        self.assertEqual(estimand.covariates_adjusted, ("age",))
        self.assertEqual(artifact.status, "estimated")
        self.assertEqual(artifact.schema_version, "1.0")

    def test_is_reproducible_under_a_fixed_seed(self):
        dataset = _synthetic_dataset(500, true_risk_difference=0.1, seed=2)
        first = EstimationAgent().run(
            dataset, study_id="test_study", covariate_columns=("age",),
            n_bootstrap=50, random_seed=7,
        )
        second = EstimationAgent().run(
            dataset, study_id="test_study", covariate_columns=("age",),
            n_bootstrap=50, random_seed=7,
        )
        self.assertEqual(first.estimand.point_estimate, second.estimand.point_estimate)
        self.assertEqual(first.estimand.ci_lower, second.estimand.ci_lower)
        self.assertEqual(first.estimand.ci_upper, second.estimand.ci_upper)

    def test_one_hot_encodes_categorical_covariates(self):
        dataset = _synthetic_dataset(500, true_risk_difference=0.1, seed=3)
        dataset["gender_concept_id"] = np.where(dataset["age"] > 0, 8507, 8532)
        artifact = EstimationAgent().run(
            dataset, study_id="test_study", covariate_columns=("age", "gender_concept_id"),
            n_bootstrap=50, random_seed=3,
        )
        self.assertIn("gender_concept_id", artifact.estimand.covariates_adjusted)

    def test_warns_about_dropped_incomplete_rows(self):
        dataset = _synthetic_dataset(500, true_risk_difference=0.1, seed=4)
        dataset.loc[0:9, "age"] = np.nan
        artifact = EstimationAgent().run(
            dataset, study_id="test_study", covariate_columns=("age",),
            n_bootstrap=50, random_seed=4,
        )
        self.assertEqual(artifact.estimand.n_dropped_incomplete_covariates, 10)
        self.assertTrue(any("dropped" in warning for warning in artifact.estimand.warnings))

    def test_rejects_missing_treatment_or_outcome_column(self):
        dataset = pd.DataFrame({"A": [1, 0], "age": [1, 2]})
        with self.assertRaises(ValueError):
            EstimationAgent().run(dataset, study_id="test_study", covariate_columns=("age",))

    def test_rejects_a_single_treatment_arm(self):
        dataset = pd.DataFrame({"A": [1, 1, 1], "Y": [0, 1, 0], "age": [1, 2, 3]})
        with self.assertRaises(ValueError):
            EstimationAgent().run(dataset, study_id="test_study", covariate_columns=("age",))

    def test_rejects_a_constant_outcome(self):
        dataset = pd.DataFrame({"A": [1, 0, 1, 0], "Y": [1, 1, 1, 1], "age": [1, 2, 3, 4]})
        with self.assertRaises(ValueError):
            EstimationAgent().run(dataset, study_id="test_study", covariate_columns=("age",))

    def test_rejects_a_blank_study_id(self):
        dataset = _synthetic_dataset(100, 0.1, seed=5)
        with self.assertRaises(ValueError):
            EstimationAgent().run(dataset, study_id="  ", covariate_columns=("age",))

    def test_ignores_censoring_by_default(self):
        dataset = _synthetic_dataset(500, true_risk_difference=0.1, seed=7)
        dataset["censored"] = 1  # every row administratively censored
        artifact = EstimationAgent().run(
            dataset, study_id="test_study", covariate_columns=("age",),
            n_bootstrap=50, random_seed=7,
        )
        self.assertEqual(artifact.estimand.censoring_strategy, "ignore_censoring_administrative_waiver")
        self.assertEqual(artifact.estimand.n_excluded_censored, 0)
        self.assertEqual(artifact.estimand.n_total, 500)
        self.assertTrue(any("ignore_censoring" in warning for warning in artifact.estimand.warnings))

    def test_excludes_censored_rows_when_declared(self):
        dataset = _synthetic_dataset(500, true_risk_difference=0.1, seed=8)
        dataset["censored"] = 0
        dataset.loc[0:49, "censored"] = 1  # 50 rows administratively censored
        artifact = EstimationAgent().run(
            dataset, study_id="test_study", covariate_columns=("age",),
            censoring_strategy="exclude_censored_before_analysis",
            n_bootstrap=50, random_seed=8,
        )
        self.assertEqual(artifact.estimand.n_excluded_censored, 50)
        self.assertEqual(artifact.estimand.n_total, 450)
        self.assertTrue(any("exclude_censored" in warning for warning in artifact.estimand.warnings))

    def test_exclude_censored_requires_the_column(self):
        dataset = _synthetic_dataset(200, 0.1, seed=9)
        with self.assertRaisesRegex(ValueError, "censored"):
            EstimationAgent().run(
                dataset, study_id="test_study", covariate_columns=("age",),
                censoring_strategy="exclude_censored_before_analysis",
            )

    def test_exclude_censored_rejects_unpopulated_column(self):
        dataset = _synthetic_dataset(200, 0.1, seed=10)
        dataset["censored"] = np.nan
        with self.assertRaisesRegex(ValueError, "null"):
            EstimationAgent().run(
                dataset, study_id="test_study", covariate_columns=("age",),
                censoring_strategy="exclude_censored_before_analysis",
            )

    def test_rejects_unsupported_censoring_strategy(self):
        dataset = _synthetic_dataset(200, 0.1, seed=11)
        with self.assertRaisesRegex(ValueError, "censoring_strategy"):
            EstimationAgent().run(
                dataset, study_id="test_study", covariate_columns=("age",),
                censoring_strategy="ipcw",
            )

    def test_rejects_unsupported_missing_data_strategy(self):
        dataset = _synthetic_dataset(200, 0.1, seed=12)
        with self.assertRaisesRegex(ValueError, "missing_data_strategy"):
            EstimationAgent().run(
                dataset, study_id="test_study", covariate_columns=("age",),
                missing_data_strategy="multiple_imputation",
            )

    def test_metadata_records_inputs(self):
        dataset = _synthetic_dataset(200, 0.1, seed=6)
        artifact = EstimationAgent().run(
            dataset, study_id="test_study", covariate_columns=("age",), n_bootstrap=30, random_seed=6,
        )
        self.assertEqual(artifact.metadata["agent_name"], "estimation_agent")
        self.assertIn("input_sha256", artifact.metadata)
        as_dict = artifact.to_dict()
        self.assertEqual(as_dict["estimand"]["method"], "g_computation_standardization")


class IPCWCensoringTests(unittest.TestCase):
    def test_requires_the_censored_column(self):
        dataset = _synthetic_dataset(200, 0.1, seed=20)
        with self.assertRaisesRegex(ValueError, "censored"):
            EstimationAgent().run(
                dataset, study_id="test_study", covariate_columns=("age",),
                censoring_strategy="ipcw_administrative_censoring",
            )

    def test_rejects_unpopulated_censored_column(self):
        dataset = _synthetic_dataset(200, 0.1, seed=21)
        dataset["censored"] = np.nan
        with self.assertRaisesRegex(ValueError, "null"):
            EstimationAgent().run(
                dataset, study_id="test_study", covariate_columns=("age",),
                censoring_strategy="ipcw_administrative_censoring",
            )

    def test_rejects_no_variation_in_censored(self):
        dataset = _synthetic_dataset(200, 0.1, seed=22)
        dataset["censored"] = 0
        with self.assertRaisesRegex(ValueError, "same censored value"):
            EstimationAgent().run(
                dataset, study_id="test_study", covariate_columns=("age",),
                censoring_strategy="ipcw_administrative_censoring",
            )

    def test_corrects_bias_from_informative_censoring(self):
        dataset = _informatively_censored_dataset(6000, true_risk_difference=0.15, seed=42)
        naive = EstimationAgent().run(
            dataset, study_id="test_study", covariate_columns=("age",),
            censoring_strategy="ignore_censoring_administrative_waiver",
            n_bootstrap=100, random_seed=1,
        )
        ipcw = EstimationAgent().run(
            dataset, study_id="test_study", covariate_columns=("age",),
            censoring_strategy="ipcw_administrative_censoring",
            n_bootstrap=100, random_seed=1,
        )
        # Treating a censored row's unreliable Y=0 as truth (the default
        # waiver) meaningfully underestimates the true 0.15 risk difference
        # here, since higher-covariate rows are both more likely censored
        # and more likely to have a true event masked as Y=0.
        self.assertLess(naive.estimand.point_estimate, 0.12)
        # IPCW reweighting recovers something much closer to the truth.
        self.assertAlmostEqual(ipcw.estimand.point_estimate, 0.15, delta=0.05)

        estimand = ipcw.estimand
        self.assertEqual(estimand.censoring_strategy, "ipcw_administrative_censoring")
        self.assertGreater(estimand.n_censored, 0)
        self.assertGreater(estimand.censoring_model_auc, 0.5)
        self.assertGreater(estimand.ipcw_weight_mean, 0)
        self.assertGreaterEqual(estimand.ipcw_weight_max, estimand.ipcw_weight_mean)
        self.assertTrue(
            any("ipcw_administrative_censoring" in warning for warning in estimand.warnings)
        )
        self.assertTrue(any("Ch. 12" in warning for warning in estimand.warnings))


class OutcomeModelCrossValidationTests(unittest.TestCase):
    def test_reports_cross_validated_auc_by_default(self):
        dataset = _synthetic_dataset(1000, 0.15, seed=30)
        artifact = EstimationAgent().run(
            dataset, study_id="test_study", covariate_columns=("age",),
            n_bootstrap=50, random_seed=30,
        )
        estimand = artifact.estimand
        self.assertGreaterEqual(estimand.outcome_model_auc, 0.0)
        self.assertLessEqual(estimand.outcome_model_auc, 1.0)
        self.assertTrue(any("cross-validation" in warning for warning in estimand.warnings))

    def test_falls_back_to_in_sample_with_too_few_events(self):
        frame = pd.DataFrame({
            "A": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "Y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            "age": [0.1, 0.2, -0.1, 0.3, -0.2, 0.05, 0.15, -0.05, 0.25, -0.15],
        })
        auc, brier, reason = EstimationAgent._cross_validated_outcome_diagnostics(
            frame, ("A", "age"), (), None, sample_weight=None, random_seed=0,
        )
        self.assertIsNone(auc)
        self.assertIsNone(brier)
        self.assertIn("fewer than 2", reason)


class RiskRatioTests(unittest.TestCase):
    """`risk_ratio_point_estimate`/`risk_ratio_ci_lower`/`risk_ratio_ci_upper`
    are reported alongside the primary risk-difference estimate purely so
    `causal.sensitivity.e_value` has a risk ratio to work with -- see
    `GComputationEstimate`'s field-group comment."""

    def test_recovers_a_known_risk_ratio_within_tolerance(self):
        dataset = _synthetic_dataset(4000, true_risk_difference=0.15, seed=1)
        artifact = EstimationAgent().run(
            dataset, study_id="test_study", covariate_columns=("age",), n_bootstrap=200, random_seed=1,
        )
        estimand = artifact.estimand
        # Baseline risk ~0.3, true_risk_difference=0.15 -> true RR ~1.5.
        self.assertAlmostEqual(estimand.risk_ratio_point_estimate, 1.5, delta=0.15)
        self.assertLess(estimand.risk_ratio_ci_lower, estimand.risk_ratio_point_estimate)
        self.assertGreater(estimand.risk_ratio_ci_upper, estimand.risk_ratio_point_estimate)
        self.assertGreater(estimand.risk_ratio_ci_lower, 1.0)

    def test_risk_ratio_is_reproducible_under_a_fixed_seed(self):
        dataset = _synthetic_dataset(500, true_risk_difference=0.1, seed=2)
        first = EstimationAgent().run(
            dataset, study_id="test_study", covariate_columns=("age",), n_bootstrap=50, random_seed=7,
        )
        second = EstimationAgent().run(
            dataset, study_id="test_study", covariate_columns=("age",), n_bootstrap=50, random_seed=7,
        )
        self.assertEqual(first.estimand.risk_ratio_point_estimate, second.estimand.risk_ratio_point_estimate)
        self.assertEqual(first.estimand.risk_ratio_ci_lower, second.estimand.risk_ratio_ci_lower)
        self.assertEqual(first.estimand.risk_ratio_ci_upper, second.estimand.risk_ratio_ci_upper)


if __name__ == "__main__":
    unittest.main()
