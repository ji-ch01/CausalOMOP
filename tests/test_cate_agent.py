import unittest

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from causal.agents.cate import CATEAgent, _holm_stepdown_fwer_correction, _subgroup_strata


def _synthetic_dataset(n: int, effect_sex0: float, effect_sex1: float, seed: int) -> pd.DataFrame:
    """Binary outcome with a planted effect-modifier: the treatment effect
    differs by `sex` (equal values means no true effect modification)."""
    rng = np.random.default_rng(seed)
    age = rng.normal(0, 1, n)
    sex = rng.integers(0, 2, n)
    a = rng.integers(0, 2, n)
    baseline_risk = 0.3 + 0.05 * age
    effect = np.where(sex == 1, effect_sex1, effect_sex0)
    risk = np.clip(baseline_risk + effect * a, 0.02, 0.98)
    y = rng.binomial(1, risk)
    return pd.DataFrame({"A": a, "Y": y, "age": age, "sex": sex})


class HolmStepdownFWERTests(unittest.TestCase):
    """`_holm_stepdown_fwer_correction` is the module's replacement for a
    resampling-based stepdown that was tried first and found to be
    statistically broken (see the function's own docstring) -- these tests
    exist specifically because that bug slipped past a single smoke test
    and was only caught by inspecting output, not by an assertion."""

    def test_matches_a_hand_computed_worked_example(self):
        adjusted, _ = _holm_stepdown_fwer_correction({"a": 0.01, "b": 0.02, "c": 0.04}, alpha=0.05)
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.04)
        self.assertAlmostEqual(adjusted["c"], 0.04)

    def test_a_clearly_null_hypothesis_is_not_spuriously_significant(self):
        """The exact bug this function replaced: a hypothesis with a
        clearly non-significant raw p-value must never come out of the
        correction with a near-zero adjusted p-value."""
        adjusted, _ = _holm_stepdown_fwer_correction({"strong": 0.001, "null": 0.6}, alpha=0.05)
        self.assertGreater(adjusted["null"], 0.5)

    def test_adjusted_pvalues_are_never_smaller_than_raw(self):
        raw = {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.5}
        adjusted, _ = _holm_stepdown_fwer_correction(raw, alpha=0.05)
        for name in raw:
            self.assertGreaterEqual(adjusted[name], raw[name])

    def test_adjusted_pvalues_are_monotone_in_ascending_raw_order(self):
        raw = {"a": 0.001, "b": 0.01, "c": 0.02, "d": 0.3, "e": 0.9}
        adjusted, _ = _holm_stepdown_fwer_correction(raw, alpha=0.05)
        ordered = sorted(raw, key=lambda name: raw[name])
        adjusted_in_order = [adjusted[name] for name in ordered]
        self.assertEqual(adjusted_in_order, sorted(adjusted_in_order))

    def test_family_wise_error_rate_is_controlled_under_the_global_null_independent(self):
        rng = np.random.default_rng(0)
        k, alpha, n_replicates = 5, 0.05, 3000
        false_rejections = 0
        for _ in range(n_replicates):
            z = rng.normal(0, 1, k)
            pvalues = {f"g{i}": float(2 * (1 - norm.cdf(abs(z[i])))) for i in range(k)}
            adjusted, _ = _holm_stepdown_fwer_correction(pvalues, alpha=alpha)
            if any(p < alpha for p in adjusted.values()):
                false_rejections += 1
        empirical_fwer = false_rejections / n_replicates
        # Generous margin around alpha for Monte Carlo noise at 3000
        # replicates; the concrete claim is "controlled", not "exactly
        # nominal" -- Holm's guarantee is an upper bound, not an equality.
        self.assertLess(empirical_fwer, alpha + 0.02)

    def test_family_wise_error_rate_is_controlled_under_the_global_null_correlated(self):
        """Holm's guarantee holds under *arbitrary* dependence -- checked
        here with test statistics sharing a common factor (correlated),
        not just the independent case above."""
        rng = np.random.default_rng(1)
        k, alpha, n_replicates = 5, 0.05, 3000
        false_rejections = 0
        for _ in range(n_replicates):
            common = rng.normal(0, 1)
            z = 0.6 * common + 0.8 * rng.normal(0, 1, k)
            pvalues = {f"g{i}": float(2 * (1 - norm.cdf(abs(z[i])))) for i in range(k)}
            adjusted, _ = _holm_stepdown_fwer_correction(pvalues, alpha=alpha)
            if any(p < alpha for p in adjusted.values()):
                false_rejections += 1
        empirical_fwer = false_rejections / n_replicates
        self.assertLess(empirical_fwer, alpha + 0.02)

    def test_simultaneous_ci_multiplier_widens_as_k_grows(self):
        _, multiplier_k2 = _holm_stepdown_fwer_correction({"a": 0.1, "b": 0.2}, alpha=0.05)
        _, multiplier_k5 = _holm_stepdown_fwer_correction(
            {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4, "e": 0.5}, alpha=0.05,
        )
        self.assertGreater(multiplier_k5, multiplier_k2)
        self.assertGreater(multiplier_k2, 1.96)  # wider than an unadjusted 95% CI


class SubgroupStrataTests(unittest.TestCase):
    def test_splits_a_binary_column_by_level(self):
        frame = pd.DataFrame({"sex": [0, 0, 1, 1, 1]})
        strata = _subgroup_strata(frame, "sex", cutpoint=None)
        labels = [label for label, _ in strata]
        self.assertEqual(labels, ["sex=0", "sex=1"])
        masks = dict(strata)
        self.assertEqual(int(masks["sex=0"].sum()), 2)
        self.assertEqual(int(masks["sex=1"].sum()), 3)

    def test_continuous_column_requires_a_cutpoint(self):
        frame = pd.DataFrame({"sofa": np.linspace(0, 20, 50)})
        with self.assertRaisesRegex(ValueError, "cutpoint"):
            _subgroup_strata(frame, "sofa", cutpoint=None)

    def test_continuous_column_splits_at_the_given_cutpoint(self):
        frame = pd.DataFrame({"sofa": np.linspace(0, 20, 50)})
        strata = _subgroup_strata(frame, "sofa", cutpoint=10.0)
        labels = [label for label, _ in strata]
        self.assertEqual(labels, ["sofa<=10.0", "sofa>10.0"])

    def test_rejects_a_column_with_no_variation(self):
        frame = pd.DataFrame({"const": [1, 1, 1, 1]})
        with self.assertRaisesRegex(ValueError, "no variation"):
            _subgroup_strata(frame, "const", cutpoint=None)


class CATEAgentValidationTests(unittest.TestCase):
    def test_rejects_a_blank_study_id(self):
        dataset = _synthetic_dataset(200, 0.1, 0.1, seed=1)
        with self.assertRaises(ValueError):
            CATEAgent().run(
                dataset, study_id="  ", covariate_columns=("age", "sex"), subgroup_columns=("sex",),
            )

    def test_rejects_missing_treatment_or_outcome_column(self):
        dataset = pd.DataFrame({"A": [1, 0], "age": [1, 2], "sex": [0, 1]})
        with self.assertRaises(ValueError):
            CATEAgent().run(
                dataset, study_id="t", covariate_columns=("age", "sex"), subgroup_columns=("sex",),
            )

    def test_requires_at_least_one_prespecified_subgroup(self):
        dataset = _synthetic_dataset(200, 0.1, 0.1, seed=2)
        with self.assertRaises(ValueError):
            CATEAgent().run(
                dataset, study_id="t", covariate_columns=("age", "sex"), subgroup_columns=(),
            )

    def test_rejects_a_subgroup_column_not_in_covariate_columns(self):
        dataset = _synthetic_dataset(200, 0.1, 0.1, seed=3)
        with self.assertRaises(ValueError):
            CATEAgent().run(
                dataset, study_id="t", covariate_columns=("age",), subgroup_columns=("sex",),
            )

    def test_rejects_overlapping_prespecified_and_exploratory_subgroups(self):
        dataset = _synthetic_dataset(200, 0.1, 0.1, seed=4)
        with self.assertRaises(ValueError):
            CATEAgent().run(
                dataset, study_id="t", covariate_columns=("age", "sex"),
                subgroup_columns=("sex",), exploratory_subgroup_columns=("sex",),
            )

    def test_rejects_missing_dataset_covariate_column(self):
        dataset = _synthetic_dataset(200, 0.1, 0.1, seed=5).drop(columns=["sex"])
        with self.assertRaises(ValueError):
            CATEAgent().run(
                dataset, study_id="t", covariate_columns=("age", "sex"), subgroup_columns=("sex",),
            )


@pytest.mark.slow
class CATEAgentRecoveryTests(unittest.TestCase):
    """Real `CausalForestDML` fits -- expensive (~40s each), kept to the
    minimum needed to demonstrate the concrete claims this agent exists to
    make."""

    @classmethod
    def setUpClass(cls):
        cls.planted_dataset = _synthetic_dataset(1500, effect_sex0=0.02, effect_sex1=0.28, seed=1)
        cls.planted_artifact = CATEAgent().run(
            cls.planted_dataset, study_id="planted_test", covariate_columns=("age", "sex"),
            subgroup_columns=("sex",), random_seed=1,
        )
        cls.null_dataset = _synthetic_dataset(1500, effect_sex0=0.15, effect_sex1=0.15, seed=2)
        cls.null_artifact = CATEAgent().run(
            cls.null_dataset, study_id="null_test", covariate_columns=("age", "sex"),
            subgroup_columns=("sex",), random_seed=2,
        )

    def test_recovers_the_direction_of_a_planted_effect_modifier(self):
        estimand = self.planted_artifact.estimand
        by_stratum = {r.stratum: r for r in estimand.prespecified_subgroup_results}
        self.assertLess(by_stratum["sex=0"].point_estimate, by_stratum["sex=1"].point_estimate)
        # The strong-effect stratum should be closer to its planted 0.28
        # than to the weak stratum's 0.02.
        self.assertAlmostEqual(by_stratum["sex=1"].point_estimate, 0.28, delta=0.15)

    def test_overall_ate_from_forest_is_between_the_two_strata_effects(self):
        estimand = self.planted_artifact.estimand
        by_stratum = {r.stratum: r for r in estimand.prespecified_subgroup_results}
        low, high = by_stratum["sex=0"].point_estimate, by_stratum["sex=1"].point_estimate
        self.assertGreaterEqual(estimand.overall_ate_from_forest, min(low, high) - 0.1)
        self.assertLessEqual(estimand.overall_ate_from_forest, max(low, high) + 0.1)

    def test_null_effect_modification_is_not_spuriously_significant_after_correction(self):
        estimand = self.null_artifact.estimand
        for result in estimand.prespecified_subgroup_results:
            self.assertGreater(result.adjusted_pvalue, 0.05)

    def test_reports_citations_and_metadata(self):
        artifact = self.planted_artifact
        estimand = artifact.estimand
        self.assertIn("Athey", estimand.citation)
        self.assertIn("Wager", estimand.citation)
        self.assertIn("List", estimand.multiple_testing_citation)
        self.assertIn("Holm", estimand.multiple_testing_citation)
        self.assertEqual(artifact.metadata["agent_name"], "cate_agent")
        self.assertIn("input_sha256", artifact.metadata)
        self.assertEqual(estimand.method, "causal_forest_dml")
        self.assertEqual(artifact.status, "estimated")

    def test_exploratory_subgroups_are_structurally_separated_and_never_carry_adjusted_values(self):
        artifact = CATEAgent().run(
            self.planted_dataset, study_id="exploratory_test", covariate_columns=("age", "sex"),
            subgroup_columns=("sex",), exploratory_subgroup_columns=(), random_seed=1,
        )
        # No exploratory columns requested here -- confirms the field
        # exists and is empty, not silently merged into the prespecified
        # list.
        self.assertEqual(artifact.estimand.exploratory_subgroup_results, ())


if __name__ == "__main__":
    unittest.main()
