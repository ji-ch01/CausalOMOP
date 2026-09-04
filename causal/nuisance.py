"""SuperLearner-style nuisance-model ensembling: a prespecified, cross-
validated combination of a small candidate library, chosen by non-negative
least squares (NNLS) over out-of-fold predictions (van der Laan, Polley &
Hubbard 2007 -- see `SuperLearner`'s docstring for the full citation).

`SuperLearner` itself is deliberately not scikit-learn-compatible (see its
own docstring) -- `causal.agents.tmle.TMLEAgent` calls `fit`/`predict_proba`
directly. `SuperLearnerClassifier`/`SuperLearnerRegressor` below are
separate, additive sklearn-compatible adapters over the same class, for a
caller (`causal.agents.cate.CATEAgent`, via `econml.dml.CausalForestDML`)
that needs `get_params`/`set_params`/`clone()` support this module's own
callers never did -- one canonical nuisance-fitting implementation behind
both TMLE and CATE, not two.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

# `LogisticRegressionCV`'s defaults are mid-deprecation-cycle in the
# `scikit-learn>=1.2,<2` range this package pins (`requirements-estimation.
# txt`): `l1_ratios` (None -> `[0.0]`) and, from ~1.9, `use_legacy_
# attributes` (True -> False, i.e. a simplified fitted-attribute layout).
# We only ever call `predict_proba`, never read `C_`/`coef_`, so pin both
# to the post-1.10 behaviour explicitly -- deterministic across versions
# and silent (no `FutureWarning` in the test suite). `use_legacy_
# attributes` is signature-guarded so the pin is a no-op on the older
# supported scikit-learn versions that predate the parameter.
_LOGISTIC_REGRESSION_CV_KWARGS: dict[str, Any] = {"l1_ratios": [0.0]}
if "use_legacy_attributes" in inspect.signature(LogisticRegressionCV).parameters:
    _LOGISTIC_REGRESSION_CV_KWARGS["use_legacy_attributes"] = False

_CITATION = (
    "van der Laan MJ, Polley EC, Hubbard AE. Super Learner. Stat Appl Genet "
    "Mol Biol. 2007;6:Article 25."
)

# Cross-validated ensembling needs at least this many folds to be worth the
# name; callers may pass a smaller cap, but never more than this.
MAX_SPLITS = 5

# Clipped away from exact 0/1 before `log_loss`, which is otherwise -inf for
# a fully confident wrong out-of-fold prediction (can happen with a small
# fold and a deterministic candidate like `DummyClassifier`).
_LOG_LOSS_EPS = 1e-6


def stratified_folds(
    y: np.ndarray, max_splits: int = MAX_SPLITS, random_seed: int = 0,
) -> tuple[StratifiedKFold | None, str | None]:
    """A `StratifiedKFold` splitter capped at `max_splits` and at `y`'s
    minority-class row count, so no fold's training set can end up missing
    a class entirely. Returns `(None, reason)` instead of raising when
    there are fewer than 2 minority-class rows to split at all -- a
    genuine small-sample limit, shared by `SuperLearner.fit` and by
    `causal.agents.estimation.EstimationAgent.
    _cross_validated_outcome_diagnostics`, which reuses this helper rather
    than reimplementing the same cap."""
    counts = np.bincount(np.asarray(y).astype(int), minlength=2)
    n_splits = min(max_splits, int(counts.min()))
    if n_splits < 2:
        return None, "fewer than 2 minority-class rows after splitting"
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed), None


def _default_library(random_seed: int) -> dict[str, Any]:
    """The fixed, prespecified candidate library -- no hyperparameter
    search, same "prespecified, not data-adaptively tuned" rationale as the
    single-logistic-regression baseline this ensemble is meant to improve
    on (`causal.agents.estimation.EstimationAgent`). All five are already
    within `scikit-learn>=1.2,<2` (`requirements-estimation.txt`); no new
    dependency needed.

    - `logistic_regression_unpenalized`: today's baseline, kept so the
      ensemble can never do worse than the status quo.
    - `logistic_regression_l2_cv`: an L2-shrinkage variant (its own
      internal CV picks the penalty strength).
    - `random_forest` / `hist_gradient_boosting`: two different inductive
      biases for nonlinearity/interactions a logistic model cannot fit.
    - `mean_only`: the conventional Super Learner "null model" -- the
      ensemble can never do worse than the marginal outcome rate.
    """
    return {
        "logistic_regression_unpenalized": LogisticRegression(C=np.inf, max_iter=1000),
        "logistic_regression_l2_cv": LogisticRegressionCV(
            max_iter=1000, cv=3, scoring="neg_log_loss", random_state=random_seed,
            **_LOGISTIC_REGRESSION_CV_KWARGS,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_leaf=10, random_state=random_seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=random_seed),
        "mean_only": DummyClassifier(strategy="prior"),
    }


def _normalize_weights(
    names: list[str], raw_coefficients: np.ndarray,
) -> tuple[dict[str, float], str | None]:
    """Turn NNLS-combination coefficients into a normalized (sum-to-1)
    per-candidate weight dict. Falls back to equal weights across the
    library -- with the fallback reason recorded, not silently produced --
    when every coefficient comes back non-positive (a degenerate
    combination, e.g. every candidate predicts identically)."""
    weights = np.clip(np.asarray(raw_coefficients, dtype=float), 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        equal = np.full(len(names), 1.0 / len(names))
        return (
            dict(zip(names, (float(w) for w in equal), strict=True)),
            "NNLS combination degenerate (all-zero weights); using equal weights across the library",
        )
    return dict(zip(names, (float(w) for w in weights / total), strict=True)), None


@dataclass(frozen=True)
class SuperLearnerFit:
    """Diagnostics from one `SuperLearner.fit` call: the combination
    weights and each candidate's (plus the ensemble's) out-of-fold
    cross-validated risk -- transparency for a reviewer to judge whether
    ensembling actually helped on this fit, same spirit as
    `causal.agents.estimation.GComputationEstimate`'s reported diagnostics."""

    weights: dict[str, float]
    candidate_cv_log_loss: dict[str, float]
    candidate_cv_brier: dict[str, float]
    ensemble_cv_log_loss: float
    ensemble_cv_brier: float
    n_splits: int
    citation: str
    fallback_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SuperLearner:
    """Cross-validated, non-negative-least-squares ensemble of a small
    prespecified candidate library (`_default_library`) for a binary
    target. See this module's docstring for the citation and for why it is
    not yet wired into any estimator's reported causal effect.

    Not a scikit-learn `Estimator` (no `get_params`/`set_params`) --
    deliberately minimal: this module's only planned callers (a future
    `TMLEAgent`, and eventually a richer IPCW censoring model) call `fit`/
    `predict_proba` directly and need `SuperLearnerFit`'s diagnostics,
    not scikit-learn-pipeline compatibility.
    """

    def __init__(self, random_seed: int = 0, max_splits: int = MAX_SPLITS):
        self._random_seed = random_seed
        self._max_splits = max_splits
        self._library = _default_library(random_seed)
        self._fitted_candidates: dict[str, Any] | None = None
        self._weights: dict[str, float] | None = None

    def fit(
        self,
        design: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> SuperLearnerFit:
        y = np.asarray(y).astype(int)
        splitter, skip_reason = stratified_folds(y, self._max_splits, self._random_seed)
        if splitter is None:
            raise ValueError(f"Cannot fit SuperLearner: {skip_reason}")

        names = list(self._library)
        out_of_fold = {name: np.empty(len(y)) for name in names}
        for train_index, test_index in splitter.split(design, y):
            fold_weight = sample_weight[train_index] if sample_weight is not None else None
            for name in names:
                candidate = clone(self._library[name])
                candidate.fit(design[train_index], y[train_index], sample_weight=fold_weight)
                out_of_fold[name][test_index] = candidate.predict_proba(design[test_index])[:, 1]

        oof_matrix = np.column_stack([out_of_fold[name] for name in names])
        combiner = LinearRegression(positive=True, fit_intercept=False)
        combiner.fit(oof_matrix, y, sample_weight=sample_weight)
        weights, fallback_reason = _normalize_weights(names, combiner.coef_)
        weight_vector = np.array([weights[name] for name in names])

        clipped = {name: np.clip(out_of_fold[name], _LOG_LOSS_EPS, 1 - _LOG_LOSS_EPS) for name in names}
        candidate_log_loss = {name: float(log_loss(y, clipped[name])) for name in names}
        candidate_brier = {name: float(brier_score_loss(y, out_of_fold[name])) for name in names}
        ensemble_pred = oof_matrix @ weight_vector
        ensemble_log_loss = float(log_loss(y, np.clip(ensemble_pred, _LOG_LOSS_EPS, 1 - _LOG_LOSS_EPS)))
        ensemble_brier = float(brier_score_loss(y, ensemble_pred))

        self._fitted_candidates = {}
        for name in names:
            candidate = clone(self._library[name])
            candidate.fit(design, y, sample_weight=sample_weight)
            self._fitted_candidates[name] = candidate
        self._weights = weights

        return SuperLearnerFit(
            weights=weights,
            candidate_cv_log_loss=candidate_log_loss,
            candidate_cv_brier=candidate_brier,
            ensemble_cv_log_loss=ensemble_log_loss,
            ensemble_cv_brier=ensemble_brier,
            n_splits=splitter.get_n_splits(),
            citation=_CITATION,
            fallback_reason=fallback_reason,
        )

    def predict_proba(self, design: np.ndarray) -> np.ndarray:
        if self._fitted_candidates is None or self._weights is None:
            raise RuntimeError("SuperLearner.predict_proba called before fit")
        prediction = np.zeros(len(design))
        for name, candidate in self._fitted_candidates.items():
            prediction += self._weights[name] * candidate.predict_proba(design)[:, 1]
        return prediction


class SuperLearnerClassifier(BaseEstimator, ClassifierMixin):
    """scikit-learn-compatible binary-classifier adapter over `SuperLearner`
    -- `fit`/`predict_proba`/`predict`, plus `get_params`/`set_params`/
    `clone()` support inherited from `BaseEstimator` (introspects `__init__`
    parameters; heavy lifting stays in `fit`, the standard sklearn
    convention this class follows so third-party frameworks that expect a
    real `Estimator` can use it as a drop-in nuisance model).

    `econml.dml.CausalForestDML`'s `model_t` (the treatment/propensity
    nuisance) requires exactly this shape when `discrete_treatment=True`
    -- see `causal.agents.cate.CATEAgent`. `predict_proba` returns the
    conventional two-column `[P(Y=0), P(Y=1)]` array; `SuperLearner.
    predict_proba` itself only returns the single `P(Y=1)` column.
    """

    def __init__(self, random_seed: int = 0, max_splits: int = MAX_SPLITS):
        self.random_seed = random_seed
        self.max_splits = max_splits

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "SuperLearnerClassifier":
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        self._learner = SuperLearner(random_seed=self.random_seed, max_splits=self.max_splits)
        self.fit_ = self._learner.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = self._learner.predict_proba(np.asarray(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class SuperLearnerRegressor(BaseEstimator, RegressorMixin):
    """scikit-learn-compatible regressor adapter over `SuperLearner`'s
    binary-classification machinery, exposing its `P(Y=1)` prediction as
    `predict()`'s continuous output.

    Needed alongside `SuperLearnerClassifier`, not instead of it:
    `econml.dml.CausalForestDML`'s `model_y` (the outcome nuisance) must be
    a regressor even when `Y` is binary -- standard double-ML practice
    residualizes a binary outcome via continuous regression on `Y in
    {0,1}` (`Y - E[Y|X,W]`), not via a classifier's hard-label `predict()`,
    which cannot residualize a doubly-robust orthogonal score. Verified
    directly against `econml`: passing a classifier as `model_y` raises
    `AttributeError: Cannot use a classifier as a first stage model when
    the target is continuous!` -- this is not a stylistic choice.
    """

    def __init__(self, random_seed: int = 0, max_splits: int = MAX_SPLITS):
        self.random_seed = random_seed
        self.max_splits = max_splits

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "SuperLearnerRegressor":
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        self._learner = SuperLearner(random_seed=self.random_seed, max_splits=self.max_splits)
        self.fit_ = self._learner.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._learner.predict_proba(np.asarray(X))
