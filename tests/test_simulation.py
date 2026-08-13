import unittest
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from causal.simulation import (
    DGPSpec,
    confounded_logistic_dgp,
    confounded_nonlinear_dgp,
    randomized_logistic_dgp,
    run_simulation,
)


@dataclass(frozen=True)
class _FakeEstimate:
    """Minimal `EstimateLike` stand-in -- exercises `run_simulation`'s own
    bias/coverage/failure-counting logic without paying for a real model
    fit, same "duck-typed, no adapter needed" point the module docstring
    makes about `GComputationEstimate`/`TMLEEstimate`."""

    point_estimate: float
    ci_lower: float
    ci_upper: float


def _tiny_dgp(true_ate: float = 0.1) -> DGPSpec:
    def generate(rng: np.random.Generator) -> pd.DataFrame:
        return pd.DataFrame({"A": [0, 1], "Y": [0, 1], "w1": [0.0, 1.0]})
    return DGPSpec(name="tiny", citation="test fixture", true_ate=true_ate, covariate_columns=("w1",), generate=generate)


class RunSimulationMechanicsTests(unittest.TestCase):
    """Fast tests of `run_simulation`'s own bookkeeping (bias/coverage/
    failure-counting/validation), using fake estimators so they run in
    milliseconds -- the real-estimator behavior (does TMLE/g-computation
    actually recover a known effect) is `SimulationEstimatorTests` below,
    marked slow."""

    def test_computes_bias_and_coverage_for_a_perfect_estimator(self):
        dgp = _tiny_dgp(true_ate=0.1)
        results = run_simulation(
            dgp, {"perfect": lambda dataset: _FakeEstimate(0.1, 0.05, 0.15)}, n_replicates=5, random_seed=0,
        )
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertAlmostEqual(result.bias, 0.0)
        self.assertEqual(result.mean_estimate, 0.1)
        self.assertEqual(result.coverage, 1.0)
        self.assertEqual(result.n_replicates, 5)
        self.assertEqual(result.n_replicates_failed, 0)
        self.assertEqual(result.dgp_name, "tiny")
        self.assertEqual(result.estimator_name, "perfect")

    def test_coverage_is_zero_when_the_ci_never_contains_the_truth(self):
        dgp = _tiny_dgp(true_ate=0.5)
        results = run_simulation(
            dgp, {"biased": lambda dataset: _FakeEstimate(0.1, 0.05, 0.15)}, n_replicates=5, random_seed=0,
        )
        self.assertEqual(results[0].coverage, 0.0)
        self.assertAlmostEqual(results[0].bias, 0.1 - 0.5)

    def test_counts_failures_without_dropping_them_from_the_denominator(self):
        dgp = _tiny_dgp()
        calls = {"n": 0}

        def flaky(dataset: pd.DataFrame) -> _FakeEstimate:
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise ValueError("degenerate resample")
            return _FakeEstimate(0.1, 0.0, 0.2)

        results = run_simulation(dgp, {"flaky": flaky}, n_replicates=10, random_seed=0)
        result = results[0]
        self.assertEqual(result.n_replicates, 5)
        self.assertEqual(result.n_replicates_failed, 5)

    def test_raises_when_every_replicate_fails(self):
        dgp = _tiny_dgp()

        def always_fails(dataset: pd.DataFrame) -> _FakeEstimate:
            raise RuntimeError("nope")

        with self.assertRaises(ValueError):
            run_simulation(dgp, {"broken": always_fails}, n_replicates=3, random_seed=0)

    def test_rejects_zero_replicates(self):
        dgp = _tiny_dgp()
        with self.assertRaises(ValueError):
            run_simulation(dgp, {"x": lambda dataset: _FakeEstimate(0.0, 0.0, 0.0)}, n_replicates=0)

    def test_rejects_no_estimators(self):
        dgp = _tiny_dgp()
        with self.assertRaises(ValueError):
            run_simulation(dgp, {}, n_replicates=5)

    def test_summarizes_every_named_estimator_independently(self):
        dgp = _tiny_dgp(true_ate=0.1)
        results = run_simulation(
            dgp,
            {
                "good": lambda dataset: _FakeEstimate(0.1, 0.05, 0.15),
                "bad": lambda dataset: _FakeEstimate(0.9, 0.85, 0.95),
            },
            n_replicates=4, random_seed=0,
        )
        by_name = {result.estimator_name: result for result in results}
        self.assertAlmostEqual(by_name["good"].bias, 0.0)
        self.assertAlmostEqual(by_name["bad"].bias, 0.8)

    def test_reports_citation(self):
        dgp = _tiny_dgp()
        results = run_simulation(
            dgp, {"perfect": lambda dataset: _FakeEstimate(0.1, 0.05, 0.15)}, n_replicates=2, random_seed=0,
        )
        self.assertIn("Rudolph", results[0].citation)
        self.assertIn("2021", results[0].citation)


class DGPTrueATETests(unittest.TestCase):
    """The three prespecified DGPs' `true_ate` must be the actual
    population risk difference their `generate` function draws from --
    cross-checked here against an independent computation method (a large
    Monte Carlo draw), not just re-derivation of the same closed-form sum
    the module itself uses."""

    def test_randomized_dgp_true_ate_matches_a_large_monte_carlo_draw(self):
        # A randomized DGP's naive difference-in-means is itself an
        # unbiased estimator of the ATE -- an independent check that
        # doesn't depend on any adjustment/estimation logic at all.
        dgp = randomized_logistic_dgp(n=300_000)
        dataset = dgp.generate(np.random.default_rng(99))
        naive_estimate = (
            dataset.loc[dataset["A"] == 1, "Y"].mean() - dataset.loc[dataset["A"] == 0, "Y"].mean()
        )
        self.assertAlmostEqual(dgp.true_ate, naive_estimate, delta=0.01)

    def test_confounded_logistic_dgp_shares_the_randomized_dgps_true_ate(self):
        # Same outcome model as the randomized DGP -- confounding treatment
        # assignment must not change the true (counterfactual) ATE.
        self.assertEqual(
            confounded_logistic_dgp().true_ate, randomized_logistic_dgp().true_ate,
        )

    def test_nonlinear_dgp_true_ate_matches_a_potential_outcome_monte_carlo_draw(self):
        # Independent check via explicit potential-outcome simulation
        # (draw W1/W2, compute Y(1) and Y(0) for the *same* draws, average
        # the difference) rather than the module's own closed-form
        # enumeration over the four discrete strata.
        from scipy.special import expit

        from causal import simulation as sim_module

        rng = np.random.default_rng(7)
        n = 300_000
        w1 = rng.integers(0, 2, n).astype(float)
        w2 = rng.integers(0, 2, n).astype(float)
        xor = (w1 != w2).astype(float)
        risk1 = expit(sim_module._DGP3_DELTA0 + sim_module._DGP3_DELTA1 * xor + sim_module._DGP3_DELTA2 * 1)
        risk0 = expit(sim_module._DGP3_DELTA0 + sim_module._DGP3_DELTA1 * xor + sim_module._DGP3_DELTA2 * 0)
        monte_carlo_ate = float(np.mean(risk1 - risk0))
        self.assertAlmostEqual(confounded_nonlinear_dgp().true_ate, monte_carlo_ate, delta=0.005)


@pytest.mark.slow
class SimulationEstimatorTests(unittest.TestCase):
    """Real `EstimationAgent`/`TMLEAgent` fits over many replicates --
    the concrete, falsifiable claim this harness exists to make. Modest
    replicate counts/sample sizes relative to the plan's illustrative
    200-replicate example (TMLE's SuperLearner ensemble is expensive per
    replicate); tolerance bands are widened accordingly rather than
    asserting near-nominal behavior at a small replicate count.
    """

    @classmethod
    def setUpClass(cls):
        from causal.agents.estimation import EstimationAgent
        from causal.agents.tmle import TMLEAgent

        def make_gcomp(dgp: DGPSpec):
            def _run(dataset: pd.DataFrame):
                return EstimationAgent().run(
                    dataset, study_id="sim", covariate_columns=dgp.covariate_columns,
                    n_bootstrap=30, random_seed=0,
                ).estimand
            return _run

        def make_tmle(dgp: DGPSpec):
            def _run(dataset: pd.DataFrame):
                return TMLEAgent().run(
                    dataset, study_id="sim", covariate_columns=dgp.covariate_columns, random_seed=0,
                ).estimand
            return _run

        cls.make_gcomp = staticmethod(make_gcomp)
        cls.make_tmle = staticmethod(make_tmle)

    def test_recovers_the_true_ate_under_randomized_treatment(self):
        dgp = randomized_logistic_dgp(n=1200)
        estimators = {"gcomp": self.make_gcomp(dgp), "tmle": self.make_tmle(dgp)}
        results = run_simulation(dgp, estimators, n_replicates=20, random_seed=1)
        for result in results:
            self.assertAlmostEqual(result.bias, 0.0, delta=0.04)

    def test_recovers_the_true_ate_under_confounded_treatment(self):
        dgp = confounded_logistic_dgp(n=1200)
        estimators = {"gcomp": self.make_gcomp(dgp), "tmle": self.make_tmle(dgp)}
        results = run_simulation(dgp, estimators, n_replicates=20, random_seed=1)
        for result in results:
            self.assertAlmostEqual(result.bias, 0.0, delta=0.04)

    def test_coverage_is_in_a_generous_tolerance_band_around_nominal(self):
        # gcomp only (cheaper than TMLE) at a larger replicate count for a
        # more informative coverage check. At R=50 replicates, the
        # coverage estimate's own standard error under 95% nominal
        # coverage is sqrt(0.95*0.05/50) ~= 0.031 -- [0.85, 1.0] is a
        # ~3-SE lower cushion, not an assertion of exact nominal coverage.
        dgp = randomized_logistic_dgp(n=1000)
        results = run_simulation(dgp, {"gcomp": self.make_gcomp(dgp)}, n_replicates=50, random_seed=2)
        self.assertGreaterEqual(results[0].coverage, 0.85)
        self.assertLessEqual(results[0].coverage, 1.0)

    def test_tmle_is_less_biased_than_g_computation_on_the_nonlinear_dgp(self):
        """The concrete, falsifiable claim this harness exists to make:
        with a threshold/XOR outcome mechanism a logistic-in-the-raw-
        covariates model cannot represent, TMLE's SuperLearner ensemble
        (which includes tree-based candidates) must show smaller bias than
        g-computation's single logistic regression -- asserted, not just
        reported."""
        dgp = confounded_nonlinear_dgp(n=1500)
        estimators = {"gcomp": self.make_gcomp(dgp), "tmle": self.make_tmle(dgp)}
        results = run_simulation(dgp, estimators, n_replicates=25, random_seed=3)
        by_name = {result.estimator_name: result for result in results}
        self.assertLess(abs(by_name["tmle"].bias), abs(by_name["gcomp"].bias))


if __name__ == "__main__":
    unittest.main()
