"""Targeted maximum likelihood estimation (TMLE) of the population average
treatment effect (ATE), risk-difference scale, binary outcome.

Van der Laan MJ, Rose S. Targeted Learning: Causal Inference for
Observational and Experimental Data. New York: Springer, 2011.

This is `docs/causal_roadmap.md` milestone #4's second estimator, alongside
`causal.agents.estimation.EstimationAgent`'s parametric g-computation
baseline: same identifying functional (g-formula standardization), a
different -- doubly robust, efficiency-theory-grounded -- estimation
procedure (`causal.roadmap.ESTIMATION_PROCEDURES`). Nuisance models are
`causal.nuisance.SuperLearner` ensembles rather than a single unpenalized
logistic regression, and the confidence interval is the efficient influence
curve's closed-form variance estimator rather than a nonparametric
bootstrap -- TMLE's whole point is that this is asymptotically efficient and
doubly robust (consistent if *either* the outcome or the treatment nuisance
model is correctly specified, not only when both are), which a bootstrap
over a single misspecified model cannot provide.

Version 0.1 implements exactly one prespecified, citable method for exactly
one estimand (ATE, risk-difference scale), same discipline as
`EstimationAgent`. Does NOT implement, and must not be extended ad hoc to
half-implement: TMLE for informatively censored data (`censoring_strategy=
"ipcw_administrative_censoring"` is rejected outright -- see `run`'s
docstring for why), CATE/effect-modification, or any quantitative
sensitivity analysis -- see `docs/causal_roadmap.md` milestone #6 and the
CATE roadmap item for why those are separate, later work.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

from causal.covariates import CATEGORICAL_COLUMNS, build_design_matrix, fit_categorical_encoder
from causal.nuisance import SuperLearner
from causal.provenance import artifact_metadata
from causal.roadmap import CENSORING_STRATEGIES, MISSING_DATA_STRATEGIES

_CITATION = (
    "Van der Laan MJ, Rose S. Targeted Learning: Causal Inference for "
    "Observational and Experimental Data. New York: Springer, 2011."
)
_POSITIVITY_CITATION = (
    "Petersen ML, Porter KE, Gruber S, Wang Y, van der Laan MJ. Diagnosing "
    "and responding to violations in the positivity assumption. Stat "
    "Methods Med Res. 2012;21(1):31-54."
)

# Practical-positivity g-bound (Petersen et al. 2012): a raw predicted
# propensity outside this range is clipped before computing the clever
# covariate, rather than left to blow up 1/g(W) or 1/(1-g(W)) into an
# extreme weight -- flagged via `n_g_clipped`, not silently produced.
_G_CLIP_LOWER = 0.01
_G_CLIP_UPPER = 0.99

# Clipped away from exact 0/1 before `logit`/the log-likelihood used in the
# targeting step, which are otherwise +/-inf for a fully confident
# prediction -- same rationale and magnitude as `causal.nuisance.
# _LOG_LOSS_EPS`.
_OUTCOME_CLIP_EPS = 1e-6

# The efficient influence curve has mean ~0 at convergence by construction
# (it is literally the estimating equation the targeting step solves) --
# a value exceeding this fraction of the estimate's own standard error
# signals the 1-D targeting-step optimization did not fully converge,
# rather than being silently accepted as if it were exactly 0.
_EIC_MEAN_CONVERGENCE_FACTOR = 0.1


@dataclass(frozen=True)
class _TargetingResult:
    """Pure-math result of the TMLE targeting step, decoupled from how
    `q_aw`/`q_1w`/`q_0w`/`g_1w_raw` were fit -- lets a test plug in
    deliberately misspecified nuisance predictions directly (see
    `tests/test_tmle_agent.py`'s double-robustness test) without going
    through `SuperLearner`."""

    point_estimate: float
    epsilon: float
    ci_lower: float
    ci_upper: float
    efficient_influence_curve_mean: float
    efficient_influence_curve_variance: float
    n_g_clipped: int
    optimizer_converged: bool
    # Risk ratio (psi1/psi0) point estimate and delta-method Wald CI on the
    # log scale -- see `_target`'s docstring for the derivation. NaN when
    # either arm's targeted mean is 0 (a genuine degenerate case, not
    # papered over).
    risk_ratio_point_estimate: float
    risk_ratio_ci_lower: float
    risk_ratio_ci_upper: float


def _risk_ratio_from_targeted_arms(
    y: np.ndarray, a: np.ndarray, q1_1w: np.ndarray, q1_0w: np.ndarray, g_1w: np.ndarray,
) -> tuple[float, float, float]:
    """Risk ratio (`psi1/psi0`) point estimate and delta-method Wald CI on
    the log scale, from the already-targeted arm-specific predictions
    `_target` computes. A separate, directly-testable function (not inlined
    into `_target`) so a test can hand it an exact degenerate arm mean of 0
    without needing the full targeting-step optimization to happen to land
    on one -- floating-point sigmoid output essentially never underflows to
    exactly 0.0 given the practical-positivity-bounded `g_1w` and the
    outcome-clipping floor (`_OUTCOME_CLIP_EPS`) `_target` applies upstream.

    Derived from the standard per-arm TMLE efficient influence curves, not
    a second fluctuation: `D1_i = (A_i/g(W_i))*(Y_i-Q1(1,W_i)) + Q1(1,W_i)
    - psi1` and `D0_i` symmetric for the comparator arm. These decompose
    `_target`'s risk-difference EIC exactly (`H(A,W)*(Y-Q1(A,W)) ==
    D1_i_term - D0_i_term` pointwise, since `A_i/g(W_i)` is 0 whenever
    `A_i=0` and vice versa) -- a delta-method Wald CI on `log(psi1/psi0)`
    then follows from `Var(log RR) = Var(D1/psi1 - D0/psi0)/n`, the
    standard construction for a ratio of two TMLE-estimated means (Van der
    Laan & Rose 2011). Returns `(nan, nan, nan)` when either arm's targeted
    mean is exactly 0 -- a genuine degenerate case, not papered over."""
    n = len(y)
    psi1, psi0 = float(np.mean(q1_1w)), float(np.mean(q1_0w))
    if psi1 <= 0 or psi0 <= 0:
        return float("nan"), float("nan"), float("nan")
    d1 = (a / g_1w) * (y - q1_1w) + q1_1w - psi1
    d0 = ((1 - a) / (1 - g_1w)) * (y - q1_0w) + q1_0w - psi0
    log_rr_influence = d1 / psi1 - d0 / psi0
    log_rr_se = float(np.sqrt(np.var(log_rr_influence, ddof=1) / n))
    log_rr = float(np.log(psi1 / psi0))
    return psi1 / psi0, float(np.exp(log_rr - 1.96 * log_rr_se)), float(np.exp(log_rr + 1.96 * log_rr_se))


def _target(
    y: np.ndarray,
    a: np.ndarray,
    q_aw: np.ndarray,
    q_1w: np.ndarray,
    q_0w: np.ndarray,
    g_1w_raw: np.ndarray,
    *,
    g_clip_lower: float = _G_CLIP_LOWER,
    g_clip_upper: float = _G_CLIP_UPPER,
) -> _TargetingResult:
    """The TMLE targeting step: clip the raw propensity, form the clever
    covariate `H(A,W) = A/g(W) - (1-A)/(1-g(W))`, fit a 1-D no-intercept
    logistic fluctuation of `Y` on `H` with offset `logit(Q(A,W))`, update
    `Q` for the observed and both counterfactual arms, and return the
    point estimate together with its efficient-influence-curve-based Wald
    CI (`Var(psi) = Var(D)/n`). Also returns a risk ratio alongside the risk
    difference -- see `_risk_ratio_from_targeted_arms` for that derivation."""
    n = len(y)
    n_g_clipped = int(((g_1w_raw < g_clip_lower) | (g_1w_raw > g_clip_upper)).sum())
    g_1w = np.clip(g_1w_raw, g_clip_lower, g_clip_upper)

    h_aw = a / g_1w - (1 - a) / (1 - g_1w)
    h_1w = 1 / g_1w
    h_0w = -1 / (1 - g_1w)

    offset = logit(np.clip(q_aw, _OUTCOME_CLIP_EPS, 1 - _OUTCOME_CLIP_EPS))
    offset_1w = logit(np.clip(q_1w, _OUTCOME_CLIP_EPS, 1 - _OUTCOME_CLIP_EPS))
    offset_0w = logit(np.clip(q_0w, _OUTCOME_CLIP_EPS, 1 - _OUTCOME_CLIP_EPS))

    def _negative_log_likelihood(epsilon: float) -> float:
        p = np.clip(expit(offset + epsilon * h_aw), 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    result = minimize_scalar(_negative_log_likelihood, method="brent")
    epsilon_hat = float(result.x)
    optimizer_converged = bool(getattr(result, "success", True))

    q1_aw = expit(offset + epsilon_hat * h_aw)
    q1_1w = expit(offset_1w + epsilon_hat * h_1w)
    q1_0w = expit(offset_0w + epsilon_hat * h_0w)

    point_estimate = float(np.mean(q1_1w - q1_0w))
    eic = h_aw * (y - q1_aw) + (q1_1w - q1_0w) - point_estimate
    eic_variance = float(np.var(eic, ddof=1))
    eic_mean = float(np.mean(eic))
    se = float(np.sqrt(eic_variance / n))

    risk_ratio_point_estimate, risk_ratio_ci_lower, risk_ratio_ci_upper = _risk_ratio_from_targeted_arms(
        y, a, q1_1w, q1_0w, g_1w
    )

    return _TargetingResult(
        point_estimate=point_estimate,
        epsilon=epsilon_hat,
        ci_lower=point_estimate - 1.96 * se,
        ci_upper=point_estimate + 1.96 * se,
        efficient_influence_curve_mean=eic_mean,
        efficient_influence_curve_variance=eic_variance,
        n_g_clipped=n_g_clipped,
        optimizer_converged=optimizer_converged,
        risk_ratio_point_estimate=risk_ratio_point_estimate,
        risk_ratio_ci_lower=risk_ratio_ci_lower,
        risk_ratio_ci_upper=risk_ratio_ci_upper,
    )


@dataclass(frozen=True)
class TMLEEstimate:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    effect_measure: str
    method: str
    citation: str
    outcome_model: str
    treatment_model: str
    epsilon: float
    propensity_model_auc: float
    g_clip_lower: float
    g_clip_upper: float
    n_g_clipped: int
    efficient_influence_curve_mean: float
    efficient_influence_curve_variance: float
    # Risk ratio alongside the risk-difference `point_estimate`/`ci_lower`/
    # `ci_upper` above -- see `_target`'s docstring for the delta-method
    # derivation; feeds `causal.sensitivity.e_value`.
    risk_ratio_point_estimate: float
    risk_ratio_ci_lower: float
    risk_ratio_ci_upper: float
    outcome_superlearner_diagnostics: dict[str, Any]
    treatment_superlearner_diagnostics: dict[str, Any]
    n_total: int
    n_treated: int
    n_comparator: int
    n_outcome_events: int
    n_dropped_incomplete_covariates: int
    covariates_adjusted: tuple[str, ...]
    missing_data_strategy: str
    censoring_strategy: str
    n_excluded_censored: int
    n_censored: int
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TMLEArtifact:
    metadata: dict[str, str]
    schema_version: str
    study_id: str
    status: str
    estimand: TMLEEstimate

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TMLEAgent:
    """Targeted maximum likelihood estimation of the population ATE for a
    binary outcome. See the module docstring for the citation and scope.

    `censoring_strategy`/`missing_data_strategy` (see `causal.roadmap.
    StatisticalEstimand`): missing adjustment covariates only support
    complete-case analysis, same as `EstimationAgent`. Censoring supports
    only `exclude_censored_before_analysis` and `ignore_censoring_
    administrative_waiver` -- `ipcw_administrative_censoring` raises
    `ValueError` immediately. Censored-data TMLE needs a materially
    different efficient-influence-function construction that accounts for
    the censoring mechanism in the influence curve itself (Van der Laan &
    Rose 2011, ch. 5), not a reuse of `EstimationAgent`'s stabilized-weight
    IPCW bolted onto an otherwise-uncensored-data TMLE -- that would be
    exactly the kind of half-implemented extension this project avoids.

    Does NOT verify identification assumptions, same caveat as
    `EstimationAgent`: the adjustment set handed to it is assumed already
    approved via DAG/identification review.
    """

    def run(
        self,
        dataset: pd.DataFrame,
        *,
        study_id: str,
        covariate_columns: tuple[str, ...],
        censoring_strategy: str = "ignore_censoring_administrative_waiver",
        missing_data_strategy: str = "complete_case_analysis",
        random_seed: int = 0,
    ) -> TMLEArtifact:
        if not study_id.strip():
            raise ValueError("study_id is required")
        if "A" not in dataset.columns or "Y" not in dataset.columns:
            raise ValueError("dataset must contain a treatment column 'A' and outcome column 'Y'")
        missing = [name for name in covariate_columns if name not in dataset.columns]
        if missing:
            raise ValueError(f"dataset is missing covariate column(s): {', '.join(missing)}")
        if censoring_strategy not in CENSORING_STRATEGIES:
            raise ValueError(f"censoring_strategy must be one of {CENSORING_STRATEGIES}")
        if censoring_strategy == "ipcw_administrative_censoring":
            raise ValueError(
                "TMLEAgent does not support censoring_strategy='ipcw_administrative_censoring' -- "
                "censored-data TMLE needs a materially different efficient-influence-function "
                "construction (Van der Laan & Rose 2011, ch. 5), not a reuse of EstimationAgent's "
                "stabilized-weight IPCW; this is an explicit scope decision, not an oversight. Use "
                "'exclude_censored_before_analysis' or 'ignore_censoring_administrative_waiver'."
            )
        if missing_data_strategy not in MISSING_DATA_STRATEGIES:
            raise ValueError(f"missing_data_strategy must be one of {MISSING_DATA_STRATEGIES}")

        n_censored = int((dataset["censored"] == 1).sum()) if "censored" in dataset.columns else 0
        n_before_censoring = len(dataset)
        if censoring_strategy == "exclude_censored_before_analysis":
            if "censored" not in dataset.columns:
                raise ValueError(
                    "censoring_strategy is exclude_censored_before_analysis but dataset has no "
                    "`censored` column"
                )
            if dataset["censored"].isna().any():
                raise ValueError(
                    "censoring_strategy is exclude_censored_before_analysis but the dataset's "
                    "`censored` column has null values -- rebuild analytic_dataset.parquet after "
                    "approving the per-row censoring definition (WorkflowManager."
                    "approve_censoring_definition)"
                )
            working = dataset[dataset["censored"] != 1][["A", "Y", *covariate_columns]].copy()
        else:
            working = dataset[["A", "Y", *covariate_columns]].copy()
        n_excluded_censored = (
            n_before_censoring - len(working) if censoring_strategy == "exclude_censored_before_analysis" else 0
        )

        n_total_raw = len(working)
        complete = working.dropna().reset_index(drop=True)
        n_dropped = n_total_raw - len(complete)
        if complete.empty:
            raise ValueError("No complete-case rows remain after dropping missing covariates")
        if complete["A"].nunique() < 2:
            raise ValueError("Both treatment arms (A=0 and A=1) must be present to estimate an effect")
        if complete["Y"].nunique() < 2:
            raise ValueError("Outcome Y has no variance in the complete-case sample")

        categorical_columns = tuple(name for name in covariate_columns if name in CATEGORICAL_COLUMNS)
        numeric_columns = tuple(name for name in covariate_columns if name not in CATEGORICAL_COLUMNS)
        encoder = fit_categorical_encoder(complete, categorical_columns)

        a = complete["A"].astype(int).to_numpy()
        y = complete["Y"].astype(int).to_numpy()
        design_w = build_design_matrix(complete, numeric_columns, categorical_columns, encoder)
        design_aw = build_design_matrix(complete, ("A",) + numeric_columns, categorical_columns, encoder)
        forced_treated = complete.copy()
        forced_treated["A"] = 1
        forced_comparator = complete.copy()
        forced_comparator["A"] = 0
        design_a1w = build_design_matrix(forced_treated, ("A",) + numeric_columns, categorical_columns, encoder)
        design_a0w = build_design_matrix(forced_comparator, ("A",) + numeric_columns, categorical_columns, encoder)

        outcome_learner = SuperLearner(random_seed=random_seed)
        outcome_diagnostics = outcome_learner.fit(design_aw, y)
        q_aw = outcome_learner.predict_proba(design_aw)
        q_1w = outcome_learner.predict_proba(design_a1w)
        q_0w = outcome_learner.predict_proba(design_a0w)

        treatment_learner = SuperLearner(random_seed=random_seed)
        treatment_diagnostics = treatment_learner.fit(design_w, a)
        g_1w_raw = treatment_learner.predict_proba(design_w)
        # In-sample only (informational), same discipline and rationale as
        # `EstimationAgent._fit_and_standardize_ipcw`'s reported censoring-
        # model AUC: `SuperLearnerFit` reports out-of-fold log-loss/Brier,
        # not AUC, so this is the ensemble's final refit-on-all-data fit.
        propensity_model_auc = float(roc_auc_score(a, g_1w_raw))

        targeting = _target(y, a, q_aw, q_1w, q_0w, g_1w_raw)

        if censoring_strategy == "exclude_censored_before_analysis":
            censoring_warning = (
                f"Censoring strategy: exclude_censored_before_analysis -- {n_excluded_censored} "
                "row(s) with censored=1 were excluded before fitting (a reviewed choice, see "
                "statistical_estimand.json); this changes who the estimate is over and does not "
                "adjust for informative censoring via IPCW or a similar weighting method."
            )
        else:
            censoring_warning = (
                "Censoring strategy: ignore_censoring_administrative_waiver (a reviewed decision, "
                "see statistical_estimand.json, not an oversight) -- the analytic cohort's "
                "`censored` column was not used to exclude or reweight rows; does not adjust for "
                "informative censoring."
            )

        warnings = [
            "Assumes no unmeasured confounding and consistency; the adjustment set is taken as "
            "given, not re-verified here.",
            "Positivity is checked only univariately elsewhere "
            "(AnalyticCohortBuilder.positivity_audit_sql); joint positivity for this "
            "covariate set is not verified.",
            f"Missing-data strategy: {missing_data_strategy} -- rows with any missing covariate "
            "value were dropped, not imputed.",
            censoring_warning,
            "Estimates the average treatment effect (ATE) only -- no subgroup or CATE estimate.",
            "Outcome and treatment nuisance models are SuperLearner ensembles -- see "
            "outcome_superlearner_diagnostics/treatment_superlearner_diagnostics for per-candidate "
            "cross-validated log-loss/Brier and combination weights.",
        ]
        if n_dropped:
            warnings.append(f"{n_dropped} row(s) dropped for missing covariate values.")
        if outcome_diagnostics.fallback_reason:
            warnings.append(f"Outcome SuperLearner fallback: {outcome_diagnostics.fallback_reason}")
        if treatment_diagnostics.fallback_reason:
            warnings.append(f"Treatment SuperLearner fallback: {treatment_diagnostics.fallback_reason}")
        if targeting.n_g_clipped:
            warnings.append(
                f"{targeting.n_g_clipped} row(s) had an estimated propensity score outside "
                f"[{_G_CLIP_LOWER}, {_G_CLIP_UPPER}] and were clipped to this range (a practical-"
                f"positivity bound) before computing the clever covariate. {_POSITIVITY_CITATION}"
            )
        if not targeting.optimizer_converged:
            warnings.append(
                "The 1-D targeting-step optimization (scipy.optimize.minimize_scalar) did not "
                "report convergence -- inspect the fitted epsilon and efficient-influence-curve "
                "mean before trusting this estimate."
            )
        se = float(np.sqrt(targeting.efficient_influence_curve_variance / len(complete)))
        if abs(targeting.efficient_influence_curve_mean) > _EIC_MEAN_CONVERGENCE_FACTOR * max(se, 1e-12):
            warnings.append(
                "The efficient influence curve's mean is not close to zero "
                f"({targeting.efficient_influence_curve_mean:.6f} vs. a standard error of "
                f"{se:.6f}) -- the targeting step's estimating equation should be solved at "
                "convergence; treat this estimate with caution."
            )
        if np.isnan(targeting.risk_ratio_point_estimate):
            warnings.append(
                "risk_ratio_point_estimate could not be computed (a targeted arm mean was "
                "exactly 0)."
            )

        estimand = TMLEEstimate(
            point_estimate=targeting.point_estimate,
            ci_lower=targeting.ci_lower,
            ci_upper=targeting.ci_upper,
            ci_level=0.95,
            effect_measure="risk_difference",
            method="targeted_minimum_loss_based_estimation",
            citation=_CITATION,
            outcome_model="super_learner_ensemble",
            treatment_model="super_learner_ensemble",
            epsilon=targeting.epsilon,
            propensity_model_auc=propensity_model_auc,
            g_clip_lower=_G_CLIP_LOWER,
            g_clip_upper=_G_CLIP_UPPER,
            n_g_clipped=targeting.n_g_clipped,
            efficient_influence_curve_mean=targeting.efficient_influence_curve_mean,
            efficient_influence_curve_variance=targeting.efficient_influence_curve_variance,
            risk_ratio_point_estimate=targeting.risk_ratio_point_estimate,
            risk_ratio_ci_lower=targeting.risk_ratio_ci_lower,
            risk_ratio_ci_upper=targeting.risk_ratio_ci_upper,
            outcome_superlearner_diagnostics=outcome_diagnostics.to_dict(),
            treatment_superlearner_diagnostics=treatment_diagnostics.to_dict(),
            n_total=int(len(complete)),
            n_treated=int((complete["A"] == 1).sum()),
            n_comparator=int((complete["A"] == 0).sum()),
            n_outcome_events=int((complete["Y"] == 1).sum()),
            n_dropped_incomplete_covariates=int(n_dropped),
            covariates_adjusted=tuple(covariate_columns),
            missing_data_strategy=missing_data_strategy,
            censoring_strategy=censoring_strategy,
            n_excluded_censored=int(n_excluded_censored),
            n_censored=n_censored,
            warnings=tuple(warnings),
        )
        return TMLEArtifact(
            metadata=artifact_metadata(
                "tmle_agent", "0.1.0",
                {
                    "study_id": study_id,
                    "covariate_columns": list(covariate_columns),
                    "censoring_strategy": censoring_strategy,
                    "missing_data_strategy": missing_data_strategy,
                    "random_seed": random_seed,
                    "n_total": int(len(complete)),
                },
            ),
            schema_version="1.0",
            study_id=study_id.strip(),
            status="estimated",
            estimand=estimand,
        )
