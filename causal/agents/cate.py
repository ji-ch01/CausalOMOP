"""Conditional average treatment effect (CATE) / effect-modification
estimation via causal forests, for a prespecified family of subgroup
covariates (Athey & Imbens 2016's "prespecify subgroups, control the
family-wise error rate" discipline for credible heterogeneous-effect
claims -- an unprespecified, undirected search over every covariate for
"the subgroup with the biggest effect" is the textbook way to manufacture
a false positive).

Athey S, Imbens G. Recursive partitioning for heterogeneous causal
effects. Proc Natl Acad Sci USA. 2016;113(27):7353-7360.
Wager S, Athey S. Estimation and Inference of Heterogeneous Treatment
Effects using Random Forests. J Am Stat Assoc. 2018;113(523):1228-1242.

Version 0.1 implements exactly one prespecified, citable method
(`econml.dml.CausalForestDML`) for exactly one estimand (CATE by
prespecified subgroup, risk-difference scale). Does NOT implement, and
must not be extended ad hoc to half-implement: an LLM-narrative
interpretation of subgroup results (`docs/causal_roadmap.md` -- v1 report
integration is a deterministic markdown table only, see
`causal.orchestrator.workflow.build_causal_report`), or any subgroup
family not already reviewed as `causal_role="effect_modifier"` in
`analysis_plan.json`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from scipy.stats import norm

from causal.covariates import CATEGORICAL_COLUMNS, build_design_matrix, fit_categorical_encoder
from causal.nuisance import SuperLearner, SuperLearnerClassifier, SuperLearnerRegressor
from causal.provenance import artifact_metadata

_ATHEY_IMBENS_CITATION = (
    "Athey S, Imbens G. Recursive partitioning for heterogeneous causal effects. Proc Natl Acad "
    "Sci USA. 2016;113(27):7353-7360."
)
_WAGER_ATHEY_CITATION = (
    "Wager S, Athey S. Estimation and Inference of Heterogeneous Treatment Effects using Random "
    "Forests. J Am Stat Assoc. 2018;113(523):1228-1242."
)
_MULTIPLE_TESTING_CITATION = (
    "List JA, Shaikh AM, Xu Y. Multiple hypothesis testing in experimental economics. Exp Econ. "
    "2019;22(4):773-793. Holm S. A simple sequentially rejective multiple test procedure. Scand J "
    "Stat. 1979;6(2):65-70."
)
_CITATION = f"{_ATHEY_IMBENS_CITATION} {_WAGER_ATHEY_CITATION}"

# A subgroup covariate with at most this many distinct values is split by
# level (one stratum per value, e.g. a presence flag's {0,1}); above it, a
# reviewer-entered cutpoint is required (see `cutpoints`) and the covariate
# is split into two strata at that point. Matches the plan's "split by
# level (categorical) or at a reviewer-entered cutpoint (continuous)"
# distinction using cardinality rather than the covariate's OMOP domain,
# so this module stays protocol-agnostic (see class docstring) the same
# way `EstimationAgent`/`TMLEAgent` take plain covariate columns, not raw
# analysis-plan dicts.
_MAX_CATEGORICAL_LEVELS = 6

_ALPHA = 0.05


def _subgroup_strata(
    frame: pd.DataFrame, column: str, cutpoint: float | None,
) -> tuple[tuple[str, np.ndarray], ...]:
    """Split `frame[column]` into named boolean-mask strata: by level for a
    low-cardinality (`<= _MAX_CATEGORICAL_LEVELS`) column, or at `cutpoint`
    otherwise. Raises `ValueError` if a continuous column has no cutpoint,
    or a column has no variation at all."""
    values = frame[column]
    distinct = sorted(values.dropna().unique().tolist())
    if len(distinct) < 2:
        raise ValueError(f"subgroup column '{column}' has no variation -- cannot split into strata")
    if len(distinct) <= _MAX_CATEGORICAL_LEVELS:
        return tuple((f"{column}={level}", (values == level).to_numpy()) for level in distinct)
    if cutpoint is None:
        raise ValueError(
            f"subgroup column '{column}' has {len(distinct)} distinct values (continuous) -- a "
            "reviewer-entered cutpoint is required (see `cutpoints`)"
        )
    return (
        (f"{column}<={cutpoint}", (values <= cutpoint).to_numpy()),
        (f"{column}>{cutpoint}", (values > cutpoint).to_numpy()),
    )


@dataclass(frozen=True)
class SubgroupCATEResult:
    subgroup_column: str
    stratum: str
    n: int
    point_estimate: float
    ci_lower: float
    ci_upper: float
    pvalue: float
    # NaN for `exploratory_subgroup_results` -- multiple-testing correction
    # only applies to the prespecified family (see module docstring).
    adjusted_ci_lower: float
    adjusted_ci_upper: float
    adjusted_pvalue: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CATEEstimate:
    method: str
    citation: str
    ci_level: float
    overall_ate_from_forest: float
    overall_ate_ci_lower: float
    overall_ate_ci_upper: float
    outcome_model: str
    treatment_model: str
    outcome_superlearner_diagnostics: dict[str, Any]
    treatment_superlearner_diagnostics: dict[str, Any]
    # `econml.dml.CausalForestDML`'s own internal cross-fitted nuisance
    # scores (see `run`'s docstring for why this, and not a best-linear-
    # predictor calibration test, is reported as "the forest's own
    # diagnostic" in this version).
    forest_nuisance_score_y: float
    forest_nuisance_score_t: float
    prespecified_subgroup_results: tuple[SubgroupCATEResult, ...]
    exploratory_subgroup_results: tuple[SubgroupCATEResult, ...]
    multiple_testing_method: str
    multiple_testing_citation: str
    n_prespecified_subgroups_tested: int
    n_total: int
    n_treated: int
    n_comparator: int
    n_dropped_incomplete_covariates: int
    covariates_adjusted: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CATEArtifact:
    metadata: dict[str, str]
    schema_version: str
    study_id: str
    status: str
    estimand: CATEEstimate

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _holm_stepdown_fwer_correction(
    pvalues: dict[str, float], *, alpha: float = _ALPHA,
) -> tuple[dict[str, float], float]:
    """Holm's (1979) sequentially-rejective stepdown FWER correction.

    A resampling-based stepdown (List, Shaikh & Xu 2019's headline method,
    closely related to Romano & Wolf 2005) was the first design tried here
    and rejected after direct verification: bootstrapping the causal
    forest's *already-fitted* per-unit point estimates (cheap -- no
    refitting) does not capture genuine estimation uncertainty, only the
    sampling noise of averaging already-fixed numbers, which is far
    smaller than the forest's own reported uncertainty -- studentizing
    against it produced spuriously significant adjusted p-values even for
    a subgroup with a clearly null raw p-value of 0.6, caught by directly
    inspecting output before shipping this. Refitting the forest once per
    bootstrap replicate would fix that but is computationally prohibitive.
    Holm's procedure sidesteps the problem entirely: it needs only the
    already-valid marginal p-values `econml`'s own asymptotic inference
    provides (`effect_inference(...).population_summary()`), and -- unlike
    a common misconception about Bonferroni-family procedures -- is
    provably valid for FWER control under *arbitrary* dependence between
    the K test statistics, not just independence. List, Shaikh & Xu (2019)
    themselves discuss this family of stepdown procedures as a valid
    alternative to their bootstrap method.

    Returns `(adjusted_pvalues, simultaneous_ci_multiplier)`:
    `adjusted_pvalues` is keyed the same as `pvalues`, monotone non-
    decreasing in ascending-p order (Holm's own requirement).
    `simultaneous_ci_multiplier` is the Bonferroni normal-quantile
    multiplier `z_(1 - alpha/(2K))` -- the standard, valid (if
    conservative) way to widen every hypothesis's CI into a simultaneous
    confidence region at the same family-wise level Holm's procedure
    controls.
    """
    names = list(pvalues)
    k = len(names)
    order = sorted(names, key=lambda name: pvalues[name])
    adjusted_pvalue: dict[str, float] = {}
    running_max = 0.0
    stopped = False
    for rank, name in enumerate(order, start=1):
        if stopped:
            adjusted_pvalue[name] = 1.0
            continue
        holm_pvalue = min(1.0, (k - rank + 1) * pvalues[name])
        running_max = max(running_max, holm_pvalue)
        adjusted_pvalue[name] = running_max
        if pvalues[name] > alpha / (k - rank + 1):
            stopped = True
    simultaneous_ci_multiplier = float(norm.ppf(1 - alpha / (2 * k)))
    return adjusted_pvalue, simultaneous_ci_multiplier


class CATEAgent:
    """Conditional average treatment effect estimator (`econml.dml.
    CausalForestDML`, Wager & Athey 2018) for a prespecified family of
    subgroup covariates. See the module docstring for scope and citations.

    Deliberately takes plain `covariate_columns`/`subgroup_columns` tuples,
    not a raw `analysis_plan.json`/registry dict -- same "protocol-
    agnostic agent" design as `EstimationAgent`/`TMLEAgent`. The caller
    (`WorkflowManager.run_cate`) is responsible for resolving the reviewed
    `causal_role == "effect_modifier"` subgroup family from `analysis_
    plan.json` before calling this.

    Nuisance models are `causal.nuisance.SuperLearner`-backed sklearn
    adapters: `SuperLearnerClassifier` for `model_t` (`discrete_treatment=
    True` requires a classifier) and `SuperLearnerRegressor` for `model_y`
    (a regressor even though `Y` is binary -- verified directly against
    `econml`: passing a classifier for `model_y` raises `AttributeError:
    Cannot use a classifier as a first stage model when the target is
    continuous!`; standard double-ML practice residualizes a binary
    outcome via continuous regression, not classification).

    "The forest's own calibration diagnostic" this reports is `econml`'s
    own cross-fitted `nuisance_scores_y`/`nuisance_scores_t` (directly
    available on a fitted `CausalForestDML`), not a best-linear-predictor
    calibration test (`econml.validate`) -- deferred as a separate,
    heavier integration this version does not attempt.
    """

    def run(
        self,
        dataset: pd.DataFrame,
        *,
        study_id: str,
        covariate_columns: tuple[str, ...],
        subgroup_columns: tuple[str, ...],
        exploratory_subgroup_columns: tuple[str, ...] = (),
        cutpoints: dict[str, float] | None = None,
        random_seed: int = 0,
    ) -> CATEArtifact:
        if not study_id.strip():
            raise ValueError("study_id is required")
        if "A" not in dataset.columns or "Y" not in dataset.columns:
            raise ValueError("dataset must contain a treatment column 'A' and outcome column 'Y'")
        missing = [name for name in covariate_columns if name not in dataset.columns]
        if missing:
            raise ValueError(f"dataset is missing covariate column(s): {', '.join(missing)}")
        if not subgroup_columns:
            raise ValueError("at least one prespecified subgroup_column is required")
        overlap = set(subgroup_columns) & set(exploratory_subgroup_columns)
        if overlap:
            raise ValueError(
                f"column(s) {sorted(overlap)} appear in both subgroup_columns and "
                "exploratory_subgroup_columns -- a subgroup is either prespecified or exploratory, "
                "never both"
            )
        all_subgroup_columns = tuple(dict.fromkeys(subgroup_columns + exploratory_subgroup_columns))
        missing_subgroups = [name for name in all_subgroup_columns if name not in covariate_columns]
        if missing_subgroups:
            raise ValueError(f"subgroup column(s) not in covariate_columns: {', '.join(missing_subgroups)}")
        cutpoints = cutpoints or {}

        complete = dataset[["A", "Y", *covariate_columns]].dropna().reset_index(drop=True)
        n_dropped = len(dataset) - len(complete)
        if complete.empty:
            raise ValueError("No complete-case rows remain after dropping missing covariates")
        if complete["A"].nunique() < 2:
            raise ValueError("Both treatment arms (A=0 and A=1) must be present to estimate an effect")
        if complete["Y"].nunique() < 2:
            raise ValueError("Outcome Y has no variance in the complete-case sample")

        categorical_columns = tuple(name for name in covariate_columns if name in CATEGORICAL_COLUMNS)
        numeric_columns = tuple(name for name in covariate_columns if name not in CATEGORICAL_COLUMNS)
        encoder = fit_categorical_encoder(complete, categorical_columns)
        design_w = build_design_matrix(complete, numeric_columns, categorical_columns, encoder)
        a = complete["A"].astype(int).to_numpy()
        y = complete["Y"].astype(int).to_numpy()

        # Diagnostics-only fits, mirroring `TMLEAgent`'s pattern: a single
        # full-sample `SuperLearner` fit purely to report per-candidate
        # cross-validated log-loss/Brier and combination weights.
        # `CausalForestDML` below does its own, separate internal cross-
        # fitting via fresh `SuperLearnerClassifier`/`SuperLearnerRegressor`
        # instances -- these diagnostics describe how well this library
        # fits this data in general, not literally the forest's internal
        # folds.
        outcome_diagnostics = SuperLearner(random_seed=random_seed).fit(design_w, y)
        treatment_diagnostics = SuperLearner(random_seed=random_seed).fit(design_w, a)

        estimator = CausalForestDML(
            model_y=SuperLearnerRegressor(random_seed=random_seed),
            model_t=SuperLearnerClassifier(random_seed=random_seed),
            discrete_treatment=True,
            random_state=random_seed,
        )
        estimator.fit(y, a, X=design_w, W=None)

        overall = estimator.ate_inference(design_w)
        overall_point = float(overall.mean_point)
        overall_ci_lower, overall_ci_upper = (float(bound) for bound in overall.conf_int_mean())
        nuisance_score_y = float(np.mean(estimator.nuisance_scores_y))
        nuisance_score_t = float(np.mean(estimator.nuisance_scores_t))

        def _build_raw_results(columns: tuple[str, ...]) -> dict[str, dict]:
            raw: dict[str, dict] = {}
            for column in columns:
                strata = _subgroup_strata(complete, column, cutpoints.get(column))
                for stratum_label, mask in strata:
                    n_stratum = int(mask.sum())
                    if n_stratum < 2:
                        raise ValueError(
                            f"stratum '{stratum_label}' has fewer than 2 rows -- cannot estimate a "
                            "subgroup CATE"
                        )
                    summary = estimator.effect_inference(design_w[mask]).population_summary()
                    key = f"{column}::{stratum_label}"
                    raw[key] = {
                        "subgroup_column": column, "stratum": stratum_label, "n": n_stratum,
                        "point": float(summary.mean_point), "se": float(summary.stderr_mean),
                        "pvalue": float(summary.pvalue()),
                        "ci": tuple(float(bound) for bound in summary.conf_int_mean()),
                    }
            return raw

        raw_prespecified = _build_raw_results(subgroup_columns)
        raw_exploratory = _build_raw_results(exploratory_subgroup_columns)

        pvalues = {key: entry["pvalue"] for key, entry in raw_prespecified.items()}
        adjusted_pvalue, simultaneous_ci_multiplier = _holm_stepdown_fwer_correction(pvalues)

        prespecified_results = tuple(
            SubgroupCATEResult(
                subgroup_column=entry["subgroup_column"], stratum=entry["stratum"], n=entry["n"],
                point_estimate=entry["point"], ci_lower=entry["ci"][0], ci_upper=entry["ci"][1],
                pvalue=entry["pvalue"],
                adjusted_ci_lower=entry["point"] - simultaneous_ci_multiplier * entry["se"],
                adjusted_ci_upper=entry["point"] + simultaneous_ci_multiplier * entry["se"],
                adjusted_pvalue=adjusted_pvalue[key],
            )
            for key, entry in raw_prespecified.items()
        )
        exploratory_results = tuple(
            SubgroupCATEResult(
                subgroup_column=entry["subgroup_column"], stratum=entry["stratum"], n=entry["n"],
                point_estimate=entry["point"], ci_lower=entry["ci"][0], ci_upper=entry["ci"][1],
                pvalue=entry["pvalue"],
                adjusted_ci_lower=float("nan"), adjusted_ci_upper=float("nan"), adjusted_pvalue=float("nan"),
            )
            for entry in raw_exploratory.values()
        )

        warnings = [
            "Assumes no unmeasured confounding and consistency; the adjustment set is taken as "
            "given, not re-verified here.",
            "Estimates the CATE for a prespecified subgroup family only -- the population ATE is "
            "reported for cross-checking (overall_ate_from_forest), not as this artifact's primary "
            "estimand.",
            f"Multiple-testing correction ({_MULTIPLE_TESTING_CITATION}) applies only to "
            "prespecified_subgroup_results; exploratory_subgroup_results (adjusted fields are NaN) "
            "must never be reported as if family-wise error controlled.",
        ]
        if outcome_diagnostics.fallback_reason:
            warnings.append(f"Outcome SuperLearner fallback: {outcome_diagnostics.fallback_reason}")
        if treatment_diagnostics.fallback_reason:
            warnings.append(f"Treatment SuperLearner fallback: {treatment_diagnostics.fallback_reason}")
        if n_dropped:
            warnings.append(f"{n_dropped} row(s) dropped for missing covariate values.")
        if exploratory_subgroup_columns:
            warnings.append(
                f"{len(exploratory_results)} exploratory subgroup result(s) reported for "
                "transparency, structurally separated from the prespecified family -- hypothesis-"
                "generating only, not confirmatory."
            )

        estimand = CATEEstimate(
            method="causal_forest_dml",
            citation=_CITATION,
            ci_level=0.95,
            overall_ate_from_forest=overall_point,
            overall_ate_ci_lower=overall_ci_lower,
            overall_ate_ci_upper=overall_ci_upper,
            outcome_model="super_learner_ensemble",
            treatment_model="super_learner_ensemble",
            outcome_superlearner_diagnostics=outcome_diagnostics.to_dict(),
            treatment_superlearner_diagnostics=treatment_diagnostics.to_dict(),
            forest_nuisance_score_y=nuisance_score_y,
            forest_nuisance_score_t=nuisance_score_t,
            prespecified_subgroup_results=prespecified_results,
            exploratory_subgroup_results=exploratory_results,
            multiple_testing_method="holm_stepdown_fwer",
            multiple_testing_citation=_MULTIPLE_TESTING_CITATION,
            n_prespecified_subgroups_tested=len(prespecified_results),
            n_total=int(len(complete)),
            n_treated=int((complete["A"] == 1).sum()),
            n_comparator=int((complete["A"] == 0).sum()),
            n_dropped_incomplete_covariates=int(n_dropped),
            covariates_adjusted=tuple(covariate_columns),
            warnings=tuple(warnings),
        )
        return CATEArtifact(
            metadata=artifact_metadata(
                "cate_agent", "0.1.0",
                {
                    "study_id": study_id,
                    "covariate_columns": list(covariate_columns),
                    "subgroup_columns": list(subgroup_columns),
                    "exploratory_subgroup_columns": list(exploratory_subgroup_columns),
                    "random_seed": random_seed,
                    "n_total": int(len(complete)),
                },
            ),
            schema_version="1.0",
            study_id=study_id.strip(),
            status="estimated",
            estimand=estimand,
        )
