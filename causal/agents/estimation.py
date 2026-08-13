"""Parametric g-computation (standardization) estimator for the population
average treatment effect (ATE), risk-difference scale, binary outcome.

This is the "transparent g-computation baseline" `docs/causal_roadmap.md`
milestone #4 calls for -- a single unpenalized logistic outcome model and a
nonparametric bootstrap CI. `causal.agents.tmle.TMLEAgent` is the doubly
robust sibling estimator for the identical `g_formula_standardization`
functional (SuperLearner-ensembled nuisance models, an efficient-influence-
curve variance estimator); CATE/effect-modification remains future work, not
half-implemented here. See `EstimationAgent` for the full scope statement
and citation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from causal.covariates import CATEGORICAL_COLUMNS as _CATEGORICAL_COLUMNS
from causal.covariates import build_design_matrix, fit_categorical_encoder
from causal.nuisance import stratified_folds
from causal.provenance import artifact_metadata
from causal.roadmap import CENSORING_STRATEGIES, MISSING_DATA_STRATEGIES

# The two `CENSORING_STRATEGIES` (`causal.roadmap`) that need a real, fully
# populated per-row `censored` column rather than treating it as unused --
# `exclude_censored_before_analysis` to filter rows, `ipcw_administrative_
# censoring` to fit a censoring model from it.
_CENSORING_STRATEGIES_REQUIRING_COLUMN = frozenset(
    {"exclude_censored_before_analysis", "ipcw_administrative_censoring"}
)

_CITATION = (
    "Hernan MA, Robins JM. Causal Inference: What If. Boca Raton: Chapman & "
    "Hall/CRC, 2020, Ch. 13 (Standardization and the parametric g-formula)."
)
_IPCW_CITATION = (
    "Hernan MA, Robins JM. Causal Inference: What If. Boca Raton: Chapman & "
    "Hall/CRC, 2020, Ch. 12 (Inverse probability weighting to adjust for "
    "informative censoring)."
)
_IPCW_WEIGHT_CITATION = (
    "Cole SR, Hernan MA. Constructing inverse probability weights for "
    "marginal structural models. Am J Epidemiol. 2008;168(6):656-664."
)
# Common rule-of-thumb flag for a likely positivity violation in the
# censoring model (Cole & Hernan 2008 discuss stabilization/truncation for
# exactly this symptom) -- a screening heuristic, not a hard cutoff, same
# spirit as `causal.agents.diagnostics._SMD_FLAG_THRESHOLD`.
_IPCW_WEIGHT_WARN_THRESHOLD = 10.0


@dataclass(frozen=True)
class GComputationEstimate:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    effect_measure: str
    method: str
    citation: str
    # `effect_measure`/`point_estimate`/`ci_lower`/`ci_upper` above are
    # always the reviewed risk-difference-scale estimate -- the risk ratio
    # is reported alongside it (not as an alternative `effect_measure`)
    # purely so `causal.sensitivity.e_value` has a risk ratio to work with,
    # per VanderWeele & Ding 2017's own formula (defined on that scale).
    # Percentile-bootstrap CI over the same resamples as the risk difference
    # (see `run`); NaN when the comparator arm's standardized risk was 0 in
    # too many resamples to form one (a genuine small-sample limit).
    risk_ratio_point_estimate: float
    risk_ratio_ci_lower: float
    risk_ratio_ci_upper: float
    n_bootstrap: int
    bootstrap_seed: int
    outcome_model: str
    outcome_model_auc: float
    outcome_model_brier: float
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
    censoring_model_auc: float
    ipcw_weight_mean: float
    ipcw_weight_max: float
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EstimationArtifact:
    metadata: dict[str, str]
    schema_version: str
    study_id: str
    status: str
    estimand: GComputationEstimate

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EstimationAgent:
    """Parametric g-computation (standardization) estimator of the
    population ATE for a binary outcome. See `_CITATION` for the method.

    Version 0.1 implements exactly one prespecified, citable method for
    exactly one estimand (ATE, risk-difference scale). Does NOT implement,
    and must not be extended ad hoc to half-implement: SuperLearner nuisance-
    model ensembling, CATE/effect-modification, or any quantitative
    sensitivity analysis -- see `docs/causal_roadmap.md` milestones #4/#6 for
    why those are separate, later work. TMLE -- the doubly robust sibling
    estimator for the same functional -- is `causal.agents.tmle.TMLEAgent`,
    a separate class rather than an option grafted onto this one.

    `censoring_strategy`/`missing_data_strategy` (see `causal.roadmap.
    StatisticalEstimand`, the persisted, human-reviewed contract these must
    match): missing adjustment covariates only support complete-case
    analysis today (multiple imputation is future work). Censoring supports
    three reviewed strategies -- excluding administratively censored rows
    (`dataset["censored"] == 1`) before fitting, an explicit waiver not to
    use that column at all, or `ipcw_administrative_censoring`, a
    stabilized inverse-probability-of-censoring-weighted g-formula (see
    `_IPCW_CITATION`): the outcome model is fit on uncensored rows only,
    weighted by the stabilized inverse probability of remaining uncensored
    given treatment and the adjustment set, then standardized over every
    row's baseline covariates (censored or not) -- unlike exclusion, this
    keeps the estimate over the original eligible population rather than
    changing who it is over, at the cost of assuming censoring is ignorable
    given the modeled covariates.

    Does NOT verify identification assumptions (no unmeasured confounding,
    positivity, consistency) -- it assumes the adjustment set handed to it
    is the one already approved via DAG/identification review, and only
    reports what it fits, never whether the assumptions hold.

    This agent is purely statistical: it does not know about gates or
    readiness flags at all (unlike `ExecutionAgent`/`QueryValidationAgent`,
    which return a graded pass/fail policy artifact). A caller not
    authorized to call this must never call it in the first place --
    `WorkflowManager.run_estimation` is solely responsible for that guard,
    exactly as `HypothesisAgent`/`DataScienceAgent` also just raise on
    invalid input rather than encode a "blocked" status themselves.
    """

    def run(
        self,
        dataset: pd.DataFrame,
        *,
        study_id: str,
        covariate_columns: tuple[str, ...],
        censoring_strategy: str = "ignore_censoring_administrative_waiver",
        missing_data_strategy: str = "complete_case_analysis",
        n_bootstrap: int = 1000,
        random_seed: int = 0,
    ) -> EstimationArtifact:
        if not study_id.strip():
            raise ValueError("study_id is required")
        if "A" not in dataset.columns or "Y" not in dataset.columns:
            raise ValueError("dataset must contain a treatment column 'A' and outcome column 'Y'")
        missing = [name for name in covariate_columns if name not in dataset.columns]
        if missing:
            raise ValueError(f"dataset is missing covariate column(s): {', '.join(missing)}")
        if censoring_strategy not in CENSORING_STRATEGIES:
            raise ValueError(f"censoring_strategy must be one of {CENSORING_STRATEGIES}")
        if missing_data_strategy not in MISSING_DATA_STRATEGIES:
            raise ValueError(f"missing_data_strategy must be one of {MISSING_DATA_STRATEGIES}")
        if n_bootstrap < 1:
            raise ValueError("n_bootstrap must be at least 1")

        n_censored = int((dataset["censored"] == 1).sum()) if "censored" in dataset.columns else 0

        n_before_censoring = len(dataset)
        if censoring_strategy in _CENSORING_STRATEGIES_REQUIRING_COLUMN:
            if "censored" not in dataset.columns:
                raise ValueError(
                    f"censoring_strategy is {censoring_strategy} but dataset has no `censored` "
                    "column"
                )
            if dataset["censored"].isna().any():
                raise ValueError(
                    f"censoring_strategy is {censoring_strategy} but the dataset's `censored` "
                    "column has null values -- rebuild analytic_dataset.parquet after approving "
                    "the per-row censoring definition (WorkflowManager."
                    "approve_censoring_definition)"
                )
        if censoring_strategy == "exclude_censored_before_analysis":
            working = dataset[dataset["censored"] != 1][["A", "Y", *covariate_columns]].copy()
        elif censoring_strategy == "ipcw_administrative_censoring":
            working = dataset[["A", "Y", "censored", *covariate_columns]].copy()
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
        if censoring_strategy == "ipcw_administrative_censoring":
            if complete["censored"].nunique() < 2:
                raise ValueError(
                    "censoring_strategy is ipcw_administrative_censoring but every complete-case "
                    "row has the same censored value -- a censoring model needs both censored and "
                    "uncensored rows"
                )
            uncensored_complete = complete[complete["censored"] == 0]
            if uncensored_complete["A"].nunique() < 2:
                raise ValueError(
                    "Both treatment arms must be present among uncensored rows to estimate an effect"
                )
            if uncensored_complete["Y"].nunique() < 2:
                raise ValueError("Outcome Y has no variance among uncensored rows")

        categorical_columns = tuple(name for name in covariate_columns if name in _CATEGORICAL_COLUMNS)
        numeric_columns = ("A",) + tuple(
            name for name in covariate_columns if name not in _CATEGORICAL_COLUMNS
        )
        encoder = fit_categorical_encoder(complete, categorical_columns)

        censoring_model_auc = 0.0
        ipcw_weight_mean = 0.0
        ipcw_weight_max = 0.0
        if censoring_strategy == "ipcw_administrative_censoring":
            point_estimate, model, design, y, ipcw_weights, censoring_model_auc, mu1_mean, mu0_mean = (
                self._fit_and_standardize_ipcw(complete, numeric_columns, categorical_columns, encoder)
            )
            ipcw_weight_mean, ipcw_weight_max = float(ipcw_weights.mean()), float(ipcw_weights.max())
            outcome_diagnostic_frame = complete[complete["censored"] == 0].reset_index(drop=True)
            outcome_diagnostic_weight = ipcw_weights
        else:
            point_estimate, model, design, y, mu1_mean, mu0_mean = self._fit_and_standardize(
                complete, numeric_columns, categorical_columns, encoder
            )
            outcome_diagnostic_frame = complete
            outcome_diagnostic_weight = None
        risk_ratio_point_estimate = mu1_mean / mu0_mean if mu0_mean > 0 else float("nan")

        auc, brier, cv_skip_reason = self._cross_validated_outcome_diagnostics(
            outcome_diagnostic_frame, numeric_columns, categorical_columns, encoder,
            sample_weight=outcome_diagnostic_weight, random_seed=random_seed,
        )
        if auc is None:
            in_sample_pred = model.predict_proba(design)[:, 1]
            auc = float(roc_auc_score(y, in_sample_pred))
            brier = float(brier_score_loss(y, in_sample_pred))
            outcome_diagnostic_note = (
                f"Outcome-model AUC/Brier fell back to in-sample values ({cv_skip_reason}) -- "
                "cross-validation was not possible with this sample."
            )
        else:
            outcome_diagnostic_note = (
                "Outcome-model AUC/Brier are computed via stratified k-fold cross-validation, "
                "not in-sample."
            )

        rng = np.random.default_rng(random_seed)
        n_rows = len(complete)
        boot_estimates: list[float] = []
        boot_risk_ratios: list[float] = []
        for _ in range(n_bootstrap):
            resample = complete.iloc[rng.integers(0, n_rows, size=n_rows)].reset_index(drop=True)
            if censoring_strategy == "ipcw_administrative_censoring":
                uncensored_resample = resample[resample["censored"] == 0]
                if (
                    resample["A"].nunique() < 2
                    or resample["censored"].nunique() < 2
                    or uncensored_resample["A"].nunique() < 2
                    or uncensored_resample["Y"].nunique() < 2
                ):
                    # A degenerate resample (missing a treatment arm overall or
                    # among the uncensored rows, or no variation left in who
                    # was censored) cannot fit both the censoring and outcome
                    # models -- skipped rather than crashing the whole
                    # bootstrap, same rationale as the non-IPCW branch below.
                    continue
                estimate, _, _, _, _, _, boot_mu1_mean, boot_mu0_mean = self._fit_and_standardize_ipcw(
                    resample, numeric_columns, categorical_columns, encoder
                )
            else:
                if resample["A"].nunique() < 2 or resample["Y"].nunique() < 2:
                    # A degenerate resample (only one arm, or a constant outcome)
                    # cannot be standardized -- skipped rather than crashing the
                    # whole bootstrap, same rationale as any resampling method
                    # applied to a small or imbalanced cohort.
                    continue
                estimate, _, _, _, boot_mu1_mean, boot_mu0_mean = self._fit_and_standardize(
                    resample, numeric_columns, categorical_columns, encoder
                )
            boot_estimates.append(estimate)
            if boot_mu0_mean > 0:
                boot_risk_ratios.append(boot_mu1_mean / boot_mu0_mean)

        min_required = max(10, n_bootstrap // 2)
        if len(boot_estimates) < min_required:
            raise ValueError(
                "Too few successful bootstrap resamples to compute a confidence interval "
                f"({len(boot_estimates)}/{n_bootstrap} succeeded)"
            )
        ci_lower, ci_upper = (float(bound) for bound in np.percentile(boot_estimates, [2.5, 97.5]))
        if len(boot_risk_ratios) >= min_required:
            risk_ratio_ci_lower, risk_ratio_ci_upper = (
                float(bound) for bound in np.percentile(boot_risk_ratios, [2.5, 97.5])
            )
        else:
            # A comparator-arm standardized risk of exactly 0 is rare (would
            # need a resample with a perfectly separable outcome model) but
            # not impossible -- reported as NaN rather than a divide-by-zero
            # crash or a silently misleading interval, same "genuine small-
            # sample limit, not papered over" discipline as the AUC/Brier
            # cross-validation fallback below.
            risk_ratio_ci_lower, risk_ratio_ci_upper = float("nan"), float("nan")

        if censoring_strategy == "exclude_censored_before_analysis":
            censoring_warning = (
                f"Censoring strategy: exclude_censored_before_analysis -- {n_excluded_censored} "
                "row(s) with censored=1 were excluded before fitting (a reviewed choice, see "
                "statistical_estimand.json); this changes who the estimate is over and does not "
                "adjust for informative censoring via IPCW or a similar weighting method."
            )
        elif censoring_strategy == "ipcw_administrative_censoring":
            censoring_warning = (
                f"Censoring strategy: ipcw_administrative_censoring -- {n_censored} row(s) with "
                "censored=1 were reweighted via stabilized inverse probability of censoring "
                "weighting rather than excluded or ignored (a reviewed choice, see "
                f"statistical_estimand.json). {_IPCW_CITATION} Assumes censoring is independent "
                "of the outcome given treatment and the adjustment set ('censoring at random'); "
                "this is not verifiable from the data alone."
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
            outcome_diagnostic_note,
            "Estimates the average treatment effect (ATE) only -- no subgroup or CATE estimate.",
        ]
        if n_dropped:
            warnings.append(f"{n_dropped} row(s) dropped for missing covariate values.")
        if np.isnan(risk_ratio_point_estimate):
            warnings.append(
                "risk_ratio_point_estimate could not be computed (the comparator arm's "
                "standardized risk was exactly 0)."
            )
        elif np.isnan(risk_ratio_ci_lower):
            warnings.append(
                "risk_ratio_ci_lower/risk_ratio_ci_upper could not be computed (too few "
                "bootstrap resamples had a nonzero comparator-arm standardized risk)."
            )
        if len(boot_estimates) < n_bootstrap:
            warnings.append(
                f"Only {len(boot_estimates)}/{n_bootstrap} bootstrap resamples were usable; "
                "the rest had only one treatment arm or a constant outcome."
            )
        if censoring_strategy == "ipcw_administrative_censoring":
            warnings.append(
                f"Censoring-model AUC ({censoring_model_auc:.3f}, in-sample) and IPCW weights "
                f"(mean {ipcw_weight_mean:.2f}, max {ipcw_weight_max:.2f}) are reported for "
                "transparency only -- neither validates the censoring-at-random assumption IPCW "
                "relies on."
            )
            if ipcw_weight_max > _IPCW_WEIGHT_WARN_THRESHOLD:
                warnings.append(
                    f"Maximum IPCW weight is {ipcw_weight_max:.1f}, above the "
                    f"{_IPCW_WEIGHT_WARN_THRESHOLD:.0f} rule-of-thumb flag for a likely "
                    "positivity violation in the censoring model -- inspect covariate strata "
                    f"with a near-zero probability of remaining uncensored. {_IPCW_WEIGHT_CITATION}"
                )

        estimand = GComputationEstimate(
            point_estimate=float(point_estimate),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=0.95,
            effect_measure="risk_difference",
            method="g_computation_standardization",
            citation=_CITATION,
            risk_ratio_point_estimate=float(risk_ratio_point_estimate),
            risk_ratio_ci_lower=risk_ratio_ci_lower,
            risk_ratio_ci_upper=risk_ratio_ci_upper,
            n_bootstrap=len(boot_estimates),
            bootstrap_seed=random_seed,
            outcome_model="logistic_regression_unpenalized",
            outcome_model_auc=auc,
            outcome_model_brier=brier,
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
            censoring_model_auc=censoring_model_auc,
            ipcw_weight_mean=ipcw_weight_mean,
            ipcw_weight_max=ipcw_weight_max,
            warnings=tuple(warnings),
        )
        return EstimationArtifact(
            metadata=artifact_metadata(
                "estimation_agent", "0.1.0",
                {
                    "study_id": study_id,
                    "covariate_columns": list(covariate_columns),
                    "censoring_strategy": censoring_strategy,
                    "missing_data_strategy": missing_data_strategy,
                    "n_bootstrap": n_bootstrap,
                    "random_seed": random_seed,
                    "n_total": int(len(complete)),
                },
            ),
            schema_version="1.0",
            study_id=study_id.strip(),
            status="estimated",
            estimand=estimand,
        )

    @staticmethod
    def _design_matrix(
        frame: pd.DataFrame,
        numeric_columns: tuple[str, ...],
        categorical_columns: tuple[str, ...],
        encoder: OneHotEncoder | None,
    ) -> np.ndarray:
        return build_design_matrix(frame, numeric_columns, categorical_columns, encoder)

    @classmethod
    def _fit_and_standardize(
        cls,
        frame: pd.DataFrame,
        numeric_columns: tuple[str, ...],
        categorical_columns: tuple[str, ...],
        encoder: OneHotEncoder | None,
    ) -> tuple[float, LogisticRegression, np.ndarray, np.ndarray, float, float]:
        """Fit the outcome-regression model E[Y | A, covariates] on `frame`,
        then standardize: predict every row's outcome under a forced A=1 and
        a forced A=0, and return the averaged difference (the g-formula
        estimate of the ATE) together with the fitted model and its design
        matrix/labels, so the caller can also compute in-sample diagnostics
        without refitting. Also returns the two arm-specific standardized
        means (`mu1.mean()`, `mu0.mean()`) so a caller can compute the risk
        ratio on the identical fit -- free, no second model -- rather than
        approximating it from the risk difference."""
        design = cls._design_matrix(frame, numeric_columns, categorical_columns, encoder)
        y = frame["Y"].astype(int).to_numpy()
        # C=np.inf disables regularization (equivalent to the unpenalized
        # model this estimator needs -- regularization would itself need a
        # citation/justification this baseline does not have). Preferred
        # over the deprecated `penalty=None` per scikit-learn's own guidance.
        model = LogisticRegression(C=np.inf, max_iter=1000)
        model.fit(design, y)
        forced_treated = frame.copy()
        forced_treated["A"] = 1
        forced_comparator = frame.copy()
        forced_comparator["A"] = 0
        mu1 = model.predict_proba(
            cls._design_matrix(forced_treated, numeric_columns, categorical_columns, encoder)
        )[:, 1]
        mu0 = model.predict_proba(
            cls._design_matrix(forced_comparator, numeric_columns, categorical_columns, encoder)
        )[:, 1]
        mu1_mean, mu0_mean = float(mu1.mean()), float(mu0.mean())
        return mu1_mean - mu0_mean, model, design, y, mu1_mean, mu0_mean

    @classmethod
    def _fit_and_standardize_ipcw(
        cls,
        frame: pd.DataFrame,
        numeric_columns: tuple[str, ...],
        categorical_columns: tuple[str, ...],
        encoder: OneHotEncoder | None,
    ) -> tuple[float, LogisticRegression, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
        """IPCW-weighted g-formula: fit the same outcome-regression model
        `_fit_and_standardize` does, but restricted to administratively
        uncensored rows (`frame["censored"] == 0`) and weighted by the
        stabilized inverse probability of remaining uncensored -- a
        denominator model, P(uncensored | A, covariates), divided by a
        numerator model, P(uncensored | A) alone (stabilization per Cole &
        Hernan 2008, `_IPCW_WEIGHT_CITATION`). Standardizes over *every*
        row's baseline covariates, censored or not: censoring cannot depend
        on covariates measured before it, and the whole point of the
        weights is to let the uncensored-only fit generalize back to the
        full eligible population. Returns the same first four values as
        `_fit_and_standardize` (for standardization/diagnostics on the
        uncensored, weighted rows the outcome model actually fit on) plus
        the per-row weight vector, the denominator model's in-sample AUC
        (informational only -- see the caller's warning), and the two arm-
        specific standardized means (see `_fit_and_standardize`)."""
        uncensored_mask = (frame["censored"] == 0).to_numpy()
        responded = 1 - frame["censored"].astype(int).to_numpy()

        denominator_design = cls._design_matrix(frame, numeric_columns, categorical_columns, encoder)
        denominator_model = LogisticRegression(C=np.inf, max_iter=1000)
        denominator_model.fit(denominator_design, responded)
        p_uncensored_denominator = denominator_model.predict_proba(denominator_design)[:, 1]
        censoring_model_auc = float(roc_auc_score(responded, p_uncensored_denominator))

        numerator_design = frame[["A"]].astype(float).to_numpy()
        numerator_model = LogisticRegression(C=np.inf, max_iter=1000)
        numerator_model.fit(numerator_design, responded)
        p_uncensored_numerator = numerator_model.predict_proba(numerator_design)[:, 1]

        # Clipped away from 0 so a near-positivity-violation row gets a
        # large but finite weight instead of a division blow-up -- flagged
        # via the reported max weight (see `_IPCW_WEIGHT_WARN_THRESHOLD`),
        # not silently produced as inf.
        p_uncensored_denominator = np.clip(p_uncensored_denominator, 1e-3, 1.0)
        weights = (p_uncensored_numerator / p_uncensored_denominator)[uncensored_mask]

        uncensored_frame = frame[uncensored_mask].reset_index(drop=True)
        design = cls._design_matrix(uncensored_frame, numeric_columns, categorical_columns, encoder)
        y = uncensored_frame["Y"].astype(int).to_numpy()
        model = LogisticRegression(C=np.inf, max_iter=1000)
        model.fit(design, y, sample_weight=weights)

        forced_treated = frame.copy()
        forced_treated["A"] = 1
        forced_comparator = frame.copy()
        forced_comparator["A"] = 0
        mu1 = model.predict_proba(
            cls._design_matrix(forced_treated, numeric_columns, categorical_columns, encoder)
        )[:, 1]
        mu0 = model.predict_proba(
            cls._design_matrix(forced_comparator, numeric_columns, categorical_columns, encoder)
        )[:, 1]
        mu1_mean, mu0_mean = float(mu1.mean()), float(mu0.mean())
        point_estimate = mu1_mean - mu0_mean
        return point_estimate, model, design, y, weights, censoring_model_auc, mu1_mean, mu0_mean

    @classmethod
    def _cross_validated_outcome_diagnostics(
        cls,
        frame: pd.DataFrame,
        numeric_columns: tuple[str, ...],
        categorical_columns: tuple[str, ...],
        encoder: OneHotEncoder | None,
        *,
        sample_weight: np.ndarray | None,
        random_seed: int,
    ) -> tuple[float | None, float | None, str | None]:
        """Out-of-fold AUC/Brier for the outcome-regression model: refit on
        each training fold (with `sample_weight`, for the IPCW path) and
        score only that fold's held-out rows, then pool across folds -- an
        honest estimate of predictive performance, unlike scoring the same
        rows the model was fit on. Returns `(None, None, reason)` instead of
        raising when there are too few outcome events (or non-events) to
        stratify into at least 2 folds -- a genuine small-sample limit, not
        something to paper over with in-sample numbers instead. Delegates
        the fold-capping logic to `causal.nuisance.stratified_folds`,
        shared with `causal.nuisance.SuperLearner.fit` rather than
        reimplemented here."""
        y = frame["Y"].astype(int).to_numpy()
        splitter, skip_reason = stratified_folds(y, random_seed=random_seed)
        if splitter is None:
            return None, None, skip_reason
        design = cls._design_matrix(frame, numeric_columns, categorical_columns, encoder)
        out_of_fold_pred = np.empty(len(frame))
        for train_index, test_index in splitter.split(design, y):
            fold_weight = sample_weight[train_index] if sample_weight is not None else None
            fold_model = LogisticRegression(C=np.inf, max_iter=1000)
            fold_model.fit(design[train_index], y[train_index], sample_weight=fold_weight)
            out_of_fold_pred[test_index] = fold_model.predict_proba(design[test_index])[:, 1]
        return (
            float(roc_auc_score(y, out_of_fold_pred)),
            float(brier_score_loss(y, out_of_fold_pred)),
            None,
        )
