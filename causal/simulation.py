"""Outcome-blind simulation harness for validating estimator finite-sample
behavior (bias, variance calibration, confidence-interval coverage)
against a known ground truth.

Rudolph JE, Fox MP, Naimi AI. Simulation as a Tool for Teaching and
Learning Epidemiologic Methods. Am J Epidemiol. 2021;190(5):900-907.
Real-data diagnostics (SMD covariate balance, AUC/Brier calibration) can
never reveal the true counterfactual effect a study estimates -- only a
data-generating process with a documented true effect can show whether an
estimator's point estimate, variance, and CI coverage behave correctly in
finite samples. This module is that validation harness.

Not part of the governed pipeline: pure numpy/pandas plus direct calls to
`causal.agents.estimation.EstimationAgent`/`causal.agents.tmle.TMLEAgent`.
Never touches `WorkflowManager`, never writes into `artifacts/runs/<run_id>/`
-- an offline tool for maintainers validating this codebase's own
estimators, not a workflow step any clinical study run touches.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd
from scipy.special import expit

_CITATION = (
    "Rudolph JE, Fox MP, Naimi AI. Simulation as a Tool for Teaching and Learning Epidemiologic "
    "Methods. Am J Epidemiol. 2021;190(5):900-907."
)

# Standard normal 97.5th percentile, hardcoded rather than importing
# scipy.stats.norm for one constant -- same convention as the 1.96 already
# hardcoded in `causal.agents.tmle`/`causal.agents.estimation`'s own Wald
# CIs, which both fix `ci_level=0.95`.
_Z_95 = 1.96


class EstimateLike(Protocol):
    """The minimal shape `run_simulation` needs from an estimator's return
    value -- `causal.agents.estimation.GComputationEstimate` and
    `causal.agents.tmle.TMLEEstimate` already satisfy this by shared field
    names (see that design decision in `causal.agents.tmle`'s module
    docstring), so no adapter is needed to plug either one in here."""

    point_estimate: float
    ci_lower: float
    ci_upper: float


@dataclass(frozen=True)
class DGPSpec:
    """A closed-form data-generating process with an exactly-known true
    ATE (risk-difference scale). `true_ate` is derived analytically as a
    finite weighted sum over `outcome_model`'s discrete covariate strata
    (see the `_dgp_*` factories below), not estimated from a large Monte
    Carlo draw -- so a simulation replicate's bias is measured against a
    genuine ground truth, not against another estimate.

    `generate(rng)` returns one dataset (`A`, `Y`, plus `covariate_columns`)
    of this DGP's fixed sample size -- the sample size is baked into the
    closure, not a `run_simulation` parameter, so every replicate of a
    given DGP is comparable.
    """

    name: str
    citation: str
    true_ate: float
    covariate_columns: tuple[str, ...]
    generate: Callable[[np.random.Generator], pd.DataFrame]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "citation": self.citation, "true_ate": self.true_ate,
            "covariate_columns": list(self.covariate_columns),
        }


@dataclass(frozen=True)
class SimulationResult:
    dgp_name: str
    estimator_name: str
    true_ate: float
    mean_estimate: float
    bias: float
    empirical_variance: float
    mean_reported_variance: float
    coverage: float
    ci_level: float
    n_replicates: int
    n_replicates_failed: int
    citation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _implied_variance_from_ci(ci_lower: float, ci_upper: float) -> float:
    """Back out an approximate variance from a reported 95% CI width
    (`se = (upper-lower)/(2*1.96)`, `variance = se**2`) -- exact for a Wald
    CI (TMLE), an approximation for a percentile-bootstrap CI (g-
    computation), but a standard, sufficient diagnostic for the calibration
    check `SimulationResult.mean_reported_variance` exists for: does an
    estimator's *reported* uncertainty match its *empirical* variance
    across replicates."""
    se = (ci_upper - ci_lower) / (2 * _Z_95)
    return se * se


def run_simulation(
    dgp: DGPSpec,
    estimators: dict[str, Callable[[pd.DataFrame], EstimateLike]],
    n_replicates: int,
    random_seed: int = 0,
) -> tuple[SimulationResult, ...]:
    """Draw `n_replicates` independent datasets from `dgp`, fit every named
    estimator (a callable taking the generated `DataFrame` and returning
    an `EstimateLike`) on each, and summarize bias/variance-calibration/
    coverage per estimator against `dgp.true_ate`.

    A replicate that raises `ValueError`/`RuntimeError` for a given
    estimator (e.g. too few successful bootstrap resamples, a degenerate
    nuisance fit) is counted in `n_replicates_failed`, not silently dropped
    from the denominator -- same discipline as `EstimationAgent`'s own
    bootstrap failure counting. Bias/variance/coverage are computed over
    the successful replicates only; raises if every replicate failed for a
    given estimator (nothing to summarize).
    """
    if n_replicates < 1:
        raise ValueError("n_replicates must be at least 1")
    if not estimators:
        raise ValueError("at least one estimator is required")

    rng = np.random.default_rng(random_seed)
    point_estimates: dict[str, list[float]] = {name: [] for name in estimators}
    implied_variances: dict[str, list[float]] = {name: [] for name in estimators}
    n_covered: dict[str, int] = {name: 0 for name in estimators}
    n_failed: dict[str, int] = {name: 0 for name in estimators}

    for _ in range(n_replicates):
        dataset = dgp.generate(rng)
        for name, estimator in estimators.items():
            try:
                estimate = estimator(dataset)
            except (ValueError, RuntimeError):
                n_failed[name] += 1
                continue
            point_estimates[name].append(estimate.point_estimate)
            implied_variances[name].append(_implied_variance_from_ci(estimate.ci_lower, estimate.ci_upper))
            if estimate.ci_lower <= dgp.true_ate <= estimate.ci_upper:
                n_covered[name] += 1

    results = []
    for name in estimators:
        successes = point_estimates[name]
        n_success = len(successes)
        if n_success == 0:
            raise ValueError(f"Every replicate failed for estimator {name!r} on DGP {dgp.name!r}")
        mean_estimate = float(np.mean(successes))
        results.append(SimulationResult(
            dgp_name=dgp.name,
            estimator_name=name,
            true_ate=dgp.true_ate,
            mean_estimate=mean_estimate,
            bias=mean_estimate - dgp.true_ate,
            empirical_variance=float(np.var(successes, ddof=1)) if n_success > 1 else float("nan"),
            mean_reported_variance=float(np.mean(implied_variances[name])),
            coverage=n_covered[name] / n_success,
            ci_level=0.95,
            n_replicates=n_success,
            n_replicates_failed=n_failed[name],
            citation=_CITATION,
        ))
    return tuple(results)


def _dgp_true_ate_one_covariate(beta0: float, beta1: float, beta2: float) -> float:
    """Exact risk-difference ATE for `P(Y=1|A,W) = expit(beta0 + beta1*W +
    beta2*A)`, `W ~ Bernoulli(0.5)` -- a closed-form finite sum over the
    two covariate strata, not a Monte Carlo approximation."""
    return float(sum(
        0.5 * (expit(beta0 + beta1 * w + beta2) - expit(beta0 + beta1 * w))
        for w in (0, 1)
    ))


def _dgp_true_ate_xor(delta0: float, delta1: float, delta2: float) -> float:
    """Exact risk-difference ATE for `P(Y=1|A,W1,W2) = expit(delta0 +
    delta1*(W1 XOR W2) + delta2*A)`, `W1, W2 ~ Bernoulli(0.5)` independent
    -- a closed-form finite sum over the four covariate strata."""
    total = 0.0
    for w1 in (0, 1):
        for w2 in (0, 1):
            xor = float(w1 != w2)
            total += 0.25 * (expit(delta0 + delta1 * xor + delta2) - expit(delta0 + delta1 * xor))
    return float(total)


# DGP 1: randomized treatment, correctly-specified logistic outcome (sanity
# check -- no confounding, so even a naive unadjusted comparison would be
# unbiased here; validates the estimator's basic mechanics before DGP 2
# introduces confounding).
_DGP1_BETA0, _DGP1_BETA1, _DGP1_BETA2 = -0.85, 0.5, 0.7


def _generate_dgp1(n: int) -> Callable[[np.random.Generator], pd.DataFrame]:
    def _generate(rng: np.random.Generator) -> pd.DataFrame:
        w = rng.integers(0, 2, n).astype(float)
        a = rng.integers(0, 2, n)
        risk = expit(_DGP1_BETA0 + _DGP1_BETA1 * w + _DGP1_BETA2 * a)
        y = rng.binomial(1, risk)
        return pd.DataFrame({"A": a, "Y": y, "w1": w})
    return _generate


def randomized_logistic_dgp(n: int = 2000) -> DGPSpec:
    return DGPSpec(
        name="randomized_logistic",
        citation=(
            "Randomized treatment, correctly-specified logistic outcome model -- a sanity check, "
            "no confounding to adjust for."
        ),
        true_ate=_dgp_true_ate_one_covariate(_DGP1_BETA0, _DGP1_BETA1, _DGP1_BETA2),
        covariate_columns=("w1",),
        generate=_generate_dgp1(n),
    )


# DGP 2: confounded treatment, correctly-specified logistic outcome (the
# textbook case -- the same outcome model as DGP 1, but W now also drives
# treatment assignment, so an estimator must actually adjust for W to
# recover the same true ATE).
_DGP2_GAMMA0, _DGP2_GAMMA1 = -0.4, 1.2
_DGP2_BETA0, _DGP2_BETA1, _DGP2_BETA2 = _DGP1_BETA0, _DGP1_BETA1, _DGP1_BETA2


def _generate_dgp2(n: int) -> Callable[[np.random.Generator], pd.DataFrame]:
    def _generate(rng: np.random.Generator) -> pd.DataFrame:
        w = rng.integers(0, 2, n).astype(float)
        propensity = expit(_DGP2_GAMMA0 + _DGP2_GAMMA1 * w)
        a = rng.binomial(1, propensity)
        risk = expit(_DGP2_BETA0 + _DGP2_BETA1 * w + _DGP2_BETA2 * a)
        y = rng.binomial(1, risk)
        return pd.DataFrame({"A": a, "Y": y, "w1": w})
    return _generate


def confounded_logistic_dgp(n: int = 2000) -> DGPSpec:
    return DGPSpec(
        name="confounded_logistic",
        citation=(
            "Confounded treatment (P(A=1|W) depends on W), correctly-specified logistic outcome "
            "model -- the textbook confounding-adjustment case."
        ),
        true_ate=_dgp_true_ate_one_covariate(_DGP2_BETA0, _DGP2_BETA1, _DGP2_BETA2),
        covariate_columns=("w1",),
        generate=_generate_dgp2(n),
    )


# DGP 3: confounded treatment, nonlinear/threshold outcome mechanism a
# logistic-in-the-raw-covariates model cannot represent -- outcome risk
# and treatment propensity both depend on the XOR of two binary covariates
# (same "invisible to a linear-in-the-logit model" idea validated in
# `tests/test_nuisance.py`'s `_xor_like_dataset`, adapted here into a
# closed-form causal DGP with a confounded treatment and a documented true
# ATE, rather than reusing that helper directly -- it has no treatment
# variable at all).
_DGP3_GAMMA0, _DGP3_GAMMA1 = -0.3, 1.4
_DGP3_DELTA0, _DGP3_DELTA1, _DGP3_DELTA2 = -1.0, 1.8, 0.9


def _generate_dgp3(n: int) -> Callable[[np.random.Generator], pd.DataFrame]:
    def _generate(rng: np.random.Generator) -> pd.DataFrame:
        w1 = rng.integers(0, 2, n).astype(float)
        w2 = rng.integers(0, 2, n).astype(float)
        xor = (w1 != w2).astype(float)
        propensity = expit(_DGP3_GAMMA0 + _DGP3_GAMMA1 * xor)
        a = rng.binomial(1, propensity)
        risk = expit(_DGP3_DELTA0 + _DGP3_DELTA1 * xor + _DGP3_DELTA2 * a)
        y = rng.binomial(1, risk)
        return pd.DataFrame({"A": a, "Y": y, "w1": w1, "w2": w2})
    return _generate


def confounded_nonlinear_dgp(n: int = 2000) -> DGPSpec:
    return DGPSpec(
        name="confounded_nonlinear",
        citation=(
            "Confounded treatment and outcome risk both depend on the XOR of two binary "
            "covariates -- a threshold interaction a logistic-in-the-raw-covariates model "
            "structurally cannot represent, unlike a SuperLearner ensemble with tree-based "
            "candidates (cf. tests/test_nuisance.py's _xor_like_dataset)."
        ),
        true_ate=_dgp_true_ate_xor(_DGP3_DELTA0, _DGP3_DELTA1, _DGP3_DELTA2),
        covariate_columns=("w1", "w2"),
        generate=_generate_dgp3(n),
    )
