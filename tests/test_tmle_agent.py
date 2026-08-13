import unittest

import numpy as np
import pandas as pd

from causal.agents.tmle import TMLEAgent, _risk_ratio_from_targeted_arms, _target


def _synthetic_dataset(n: int, true_risk_difference: float, seed: int = 0) -> pd.DataFrame:
    """Rows where A is randomized (so no confounding) and Y's probability is
    a simple function of A and one continuous covariate -- same shape as
    `tests/test_estimation_agent.py`'s fixture, so a TMLE recovery test is
    directly comparable to `EstimationAgent`'s equivalent test."""
    rng = np.random.default_rng(seed)
    covariate = rng.normal(0, 1, size=n)
    treatment = rng.integers(0, 2, size=n)
    baseline_risk = 0.3 + 0.05 * covariate
    risk = np.clip(baseline_risk + true_risk_difference * treatment, 0.01, 0.99)
    outcome = rng.binomial(1, risk)
    return pd.DataFrame({"A": treatment, "Y": outcome, "age": covariate})


def _near_deterministic_treatment_dataset(n: int, seed: int) -> pd.DataFrame:
    """Treatment near-deterministically follows the sign of one covariate
    (99% agreement) -- a flexible propensity model can push its predicted
    g(W) very close to 0/1 away from the decision boundary, letting a test
    check the practical-positivity clip actually engages."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    flip = rng.random(n) < 0.01
    treatment = np.where(flip, 1 - (x > 0).astype(int), (x > 0).astype(int))
    risk = np.clip(0.3 + 0.1 * x + 0.1 * treatment, 0.02, 0.98)
    outcome = rng.binomial(1, risk)
    return pd.DataFrame({"A": treatment, "Y": outcome, "age": x})


class TMLEAgentRecoveryTests(unittest.TestCase):
    def test_recovers_a_known_risk_difference_within_tolerance(self):
        dataset = _synthetic_dataset(4000, true_risk_difference=0.15, seed=1)
        artifact = TMLEAgent().run(dataset, study_id="test_study", covariate_columns=("age",), random_seed=1)
        estimand = artifact.estimand
        self.assertAlmostEqual(estimand.point_estimate, 0.15, delta=0.03)
        self.assertLess(estimand.ci_lower, estimand.point_estimate)
        self.assertGreater(estimand.ci_upper, estimand.point_estimate)
        self.assertEqual(estimand.effect_measure, "risk_difference")
        self.assertEqual(estimand.method, "targeted_minimum_loss_based_estimation")
        self.assertEqual(estimand.covariates_adjusted, ("age",))
        self.assertEqual(artifact.status, "estimated")
        self.assertEqual(artifact.schema_version, "1.0")

    def test_is_reproducible_under_a_fixed_seed(self):
        dataset = _synthetic_dataset(600, true_risk_difference=0.1, seed=2)
        first = TMLEAgent().run(dataset, study_id="test_study", covariate_columns=("age",), random_seed=7)
        second = TMLEAgent().run(dataset, study_id="test_study", covariate_columns=("age",), random_seed=7)
        self.assertEqual(first.estimand.point_estimate, second.estimand.point_estimate)
        self.assertEqual(first.estimand.ci_lower, second.estimand.ci_lower)
        self.assertEqual(first.estimand.ci_upper, second.estimand.ci_upper)

    def test_efficient_influence_curve_mean_near_zero_and_variance_nonnegative(self):
        dataset = _synthetic_dataset(2000, true_risk_difference=0.1, seed=8)
        artifact = TMLEAgent().run(dataset, study_id="test_study", covariate_columns=("age",), random_seed=8)
        estimand = artifact.estimand
        self.assertGreaterEqual(estimand.efficient_influence_curve_variance, 0.0)
        se = (estimand.efficient_influence_curve_variance / estimand.n_total) ** 0.5
        self.assertLess(abs(estimand.efficient_influence_curve_mean), 0.1 * se)
        self.assertFalse(any("not close to zero" in warning for warning in estimand.warnings))

    def test_reports_citation_and_metadata(self):
        dataset = _synthetic_dataset(500, true_risk_difference=0.1, seed=9)
        artifact = TMLEAgent().run(dataset, study_id="test_study", covariate_columns=("age",), random_seed=9)
        self.assertIn("Van der Laan", artifact.estimand.citation)
        self.assertIn("2011", artifact.estimand.citation)
        self.assertEqual(artifact.metadata["agent_name"], "tmle_agent")
        self.assertIn("input_sha256", artifact.metadata)
        as_dict = artifact.to_dict()
        self.assertEqual(as_dict["estimand"]["method"], "targeted_minimum_loss_based_estimation")
        self.assertEqual(artifact.estimand.outcome_model, "super_learner_ensemble")
        self.assertEqual(artifact.estimand.treatment_model, "super_learner_ensemble")
        self.assertIn("van der Laan", artifact.estimand.outcome_superlearner_diagnostics["citation"])
        self.assertIn("van der Laan", artifact.estimand.treatment_superlearner_diagnostics["citation"])

    def test_one_hot_encodes_categorical_covariates(self):
        dataset = _synthetic_dataset(600, 0.1, seed=3)
        dataset["gender_concept_id"] = np.where(dataset["age"] > 0, 8507, 8532)
        artifact = TMLEAgent().run(
            dataset, study_id="test_study", covariate_columns=("age", "gender_concept_id"), random_seed=3,
        )
        self.assertIn("gender_concept_id", artifact.estimand.covariates_adjusted)


class TMLEAgentValidationTests(unittest.TestCase):
    def test_rejects_a_blank_study_id(self):
        dataset = _synthetic_dataset(100, 0.1, seed=5)
        with self.assertRaises(ValueError):
            TMLEAgent().run(dataset, study_id="  ", covariate_columns=("age",))

    def test_rejects_missing_treatment_or_outcome_column(self):
        dataset = pd.DataFrame({"A": [1, 0], "age": [1, 2]})
        with self.assertRaises(ValueError):
            TMLEAgent().run(dataset, study_id="test_study", covariate_columns=("age",))

    def test_rejects_a_single_treatment_arm(self):
        dataset = pd.DataFrame({"A": [1, 1, 1], "Y": [0, 1, 0], "age": [1, 2, 3]})
        with self.assertRaises(ValueError):
            TMLEAgent().run(dataset, study_id="test_study", covariate_columns=("age",))

    def test_rejects_a_constant_outcome(self):
        dataset = pd.DataFrame({"A": [1, 0, 1, 0], "Y": [1, 1, 1, 1], "age": [1, 2, 3, 4]})
        with self.assertRaises(ValueError):
            TMLEAgent().run(dataset, study_id="test_study", covariate_columns=("age",))

    def test_rejects_unsupported_missing_data_strategy(self):
        dataset = _synthetic_dataset(200, 0.1, seed=12)
        with self.assertRaisesRegex(ValueError, "missing_data_strategy"):
            TMLEAgent().run(
                dataset, study_id="test_study", covariate_columns=("age",),
                missing_data_strategy="multiple_imputation",
            )

    def test_excludes_censored_rows_when_declared(self):
        dataset = _synthetic_dataset(600, true_risk_difference=0.1, seed=8)
        dataset["censored"] = 0
        dataset.loc[0:49, "censored"] = 1
        artifact = TMLEAgent().run(
            dataset, study_id="test_study", covariate_columns=("age",),
            censoring_strategy="exclude_censored_before_analysis", random_seed=8,
        )
        self.assertEqual(artifact.estimand.n_excluded_censored, 50)
        self.assertEqual(artifact.estimand.n_total, 550)
        self.assertTrue(any("exclude_censored" in warning for warning in artifact.estimand.warnings))

    def test_exclude_censored_requires_the_column(self):
        dataset = _synthetic_dataset(200, 0.1, seed=9)
        with self.assertRaisesRegex(ValueError, "censored"):
            TMLEAgent().run(
                dataset, study_id="test_study", covariate_columns=("age",),
                censoring_strategy="exclude_censored_before_analysis",
            )


class TMLEIPCWScopeTests(unittest.TestCase):
    def test_rejects_ipcw_administrative_censoring_with_a_scope_explanation(self):
        dataset = _synthetic_dataset(200, 0.1, seed=20)
        dataset["censored"] = np.where(dataset["age"] > 0, 1, 0)
        with self.assertRaisesRegex(ValueError, "ch\\. 5"):
            TMLEAgent().run(
                dataset, study_id="test_study", covariate_columns=("age",),
                censoring_strategy="ipcw_administrative_censoring",
            )


class TMLEPositivityClippingTests(unittest.TestCase):
    def test_clips_extreme_propensity_scores_and_warns(self):
        dataset = _near_deterministic_treatment_dataset(1500, seed=3)
        artifact = TMLEAgent().run(dataset, study_id="test_study", covariate_columns=("age",), random_seed=2)
        estimand = artifact.estimand
        self.assertGreater(estimand.n_g_clipped, 0)
        self.assertEqual(estimand.g_clip_lower, 0.01)
        self.assertEqual(estimand.g_clip_upper, 0.99)
        self.assertTrue(any("clipped" in warning for warning in estimand.warnings))
        self.assertTrue(any("Petersen" in warning for warning in estimand.warnings))

    def test_reports_no_clipping_on_a_well_overlapped_sample(self):
        dataset = _synthetic_dataset(2000, true_risk_difference=0.1, seed=15)
        artifact = TMLEAgent().run(dataset, study_id="test_study", covariate_columns=("age",), random_seed=15)
        self.assertEqual(artifact.estimand.n_g_clipped, 0)


def _confounded_dgp(n: int, seed: int):
    """A confounded DGP with a known, closed-form true ATE (marginalized
    over the sampled W): treatment probability and outcome risk both
    depend on the same covariate `w`, so a treatment-effect estimate that
    ignores confounding is biased. Returns `(y, a, w, q1w, q0w, g1w, true_ate)`
    -- the true nuisance functions evaluated at each row, so a test can
    plug in either the true or a deliberately misspecified version of each
    without needing a fitted model at all."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1, n)
    g1w = 1 / (1 + np.exp(-(0.8 * w)))
    a = rng.binomial(1, g1w)
    q1w = 1 / (1 + np.exp(-(0.5 * w + 0.3)))
    q0w = 1 / (1 + np.exp(-(0.5 * w)))
    qaw = np.where(a == 1, q1w, q0w)
    y = rng.binomial(1, qaw)
    true_ate = float(np.mean(q1w - q0w))
    return y, a, w, qaw, q1w, q0w, g1w, true_ate


class TMLEDoubleRobustnessTests(unittest.TestCase):
    """The single most important test in this module: TMLE must recover the
    true effect when *either* nuisance model is correctly specified, even
    if the other is badly wrong -- unlike a plug-in estimator (needs Q
    correct) or an IPW estimator (needs g correct), each of which is only
    consistent under its own model. Exercises `_target` directly, on
    precomputed nuisance predictions, so misspecification is exact and
    doesn't depend on which candidate `SuperLearner`'s fixed library
    happens to pick."""

    def test_consistent_when_only_the_outcome_model_is_correct(self):
        y, a, w, qaw, q1w, q0w, g1w, true_ate = _confounded_dgp(20000, seed=0)
        g_wrong = np.full(len(y), 0.5)  # ignores w entirely -- badly misspecified

        tmle = _target(y, a, qaw, q1w, q0w, g_wrong)
        ipw_using_wrong_g = float(np.mean(a * y / g_wrong - (1 - a) * y / (1 - g_wrong)))

        self.assertAlmostEqual(tmle.point_estimate, true_ate, delta=0.02)
        # The single-robust IPW estimator, which needs g correct, is far
        # more biased here than TMLE is despite sharing the same wrong g.
        self.assertGreater(abs(ipw_using_wrong_g - true_ate), 2 * abs(tmle.point_estimate - true_ate))

    def test_consistent_when_only_the_treatment_model_is_correct(self):
        y, a, w, qaw, q1w, q0w, g1w, true_ate = _confounded_dgp(20000, seed=0)
        q_wrong = np.full(len(y), float(y.mean()))  # ignores w and A entirely

        tmle = _target(y, a, q_wrong, q_wrong, q_wrong, g1w)
        plugin_using_wrong_q = float(np.mean(q_wrong - q_wrong))  # exactly 0 by construction

        self.assertAlmostEqual(tmle.point_estimate, true_ate, delta=0.02)
        # The single-robust plug-in estimator, which needs Q correct,
        # completely misses the effect here (a constant Q implies no
        # effect), while TMLE recovers it via the correct propensity model.
        self.assertGreater(abs(plugin_using_wrong_q - true_ate), 2 * abs(tmle.point_estimate - true_ate))

    def test_efficient_influence_curve_variance_is_nonnegative_under_misspecification(self):
        y, a, w, qaw, q1w, q0w, g1w, true_ate = _confounded_dgp(2000, seed=1)
        q_wrong = np.full(len(y), float(y.mean()))
        result = _target(y, a, q_wrong, q_wrong, q_wrong, g1w)
        self.assertGreaterEqual(result.efficient_influence_curve_variance, 0.0)


class TMLERiskRatioTests(unittest.TestCase):
    """`risk_ratio_point_estimate`/`risk_ratio_ci_lower`/`risk_ratio_ci_upper`
    are reported alongside the primary risk-difference estimate purely so
    `causal.sensitivity.e_value` has a risk ratio to work with -- see
    `_target`'s docstring for the delta-method derivation."""

    def test_recovers_a_known_risk_ratio_within_tolerance(self):
        dataset = _synthetic_dataset(4000, true_risk_difference=0.15, seed=1)
        artifact = TMLEAgent().run(dataset, study_id="test_study", covariate_columns=("age",), random_seed=1)
        estimand = artifact.estimand
        # Baseline risk ~0.3, true_risk_difference=0.15 -> true RR ~1.5.
        self.assertAlmostEqual(estimand.risk_ratio_point_estimate, 1.5, delta=0.15)
        self.assertLess(estimand.risk_ratio_ci_lower, estimand.risk_ratio_point_estimate)
        self.assertGreater(estimand.risk_ratio_ci_upper, estimand.risk_ratio_point_estimate)
        self.assertGreater(estimand.risk_ratio_ci_lower, 1.0)

    def test_matches_the_risk_difference_eic_decomposition(self):
        """`H(A,W)*(Y-Q1(A,W))` (the risk-difference EIC's first term) must
        equal the difference of the two per-arm EIC terms the risk-ratio
        delta method uses -- the algebraic identity `_target`'s docstring
        claims, checked directly rather than only indirectly through the
        fitted numbers above."""
        y, a, w = np.array([1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0.1, 0.2, -0.1, 0.3])
        g1w = np.array([0.6, 0.5, 0.4, 0.7])
        q1w = np.array([0.7, 0.6, 0.5, 0.8])
        q0w = np.array([0.3, 0.2, 0.4, 0.1])
        qaw = np.where(a == 1, q1w, q0w)
        h_aw = a / g1w - (1 - a) / (1 - g1w)
        lhs = h_aw * (y - qaw)
        d1_term = (a / g1w) * (y - q1w)
        d0_term = ((1 - a) / (1 - g1w)) * (y - q0w)
        np.testing.assert_allclose(lhs, d1_term - d0_term)

    def test_risk_ratio_is_nan_when_an_arm_mean_is_zero(self):
        """A degenerate but legitimate case (a targeted arm mean of exactly
        0 -- practically unreachable through `_target`'s own sigmoid output
        given its outcome-clipping floor, but a real possibility for
        `_risk_ratio_from_targeted_arms` as a standalone function) must
        report NaN rather than raising a division error or silently
        producing an infinite/garbage ratio."""
        y = np.zeros(50)
        a = np.array([0, 1] * 25)
        g1w = np.full(50, 0.5)
        q1_1w = np.zeros(50)
        q1_0w = np.full(50, 0.3)
        point, ci_lower, ci_upper = _risk_ratio_from_targeted_arms(y, a, q1_1w, q1_0w, g1w)
        self.assertTrue(np.isnan(point))
        self.assertTrue(np.isnan(ci_lower))
        self.assertTrue(np.isnan(ci_upper))


if __name__ == "__main__":
    unittest.main()
