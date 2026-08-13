import unittest

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from causal.nuisance import SuperLearner, SuperLearnerClassifier, SuperLearnerRegressor, _normalize_weights, stratified_folds


def _xor_like_dataset(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A binary outcome driven by a threshold interaction of two covariates
    that a linear-in-the-logit model structurally cannot fit (the sign of
    x1 relative to x2 matters, not either covariate's own magnitude) --
    lets a test check that ensembling in a tree-based candidate actually
    earns its cross-validated weight, rather than asserting the ensemble
    is "better" by construction."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    logit = 2.5 * ((x1 > 0) != (x2 > 0)).astype(float) - 1.25
    risk = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, risk)
    return np.column_stack([x1, x2]), y


class SuperLearnerNonlinearFitTests(unittest.TestCase):
    """One expensive `SuperLearner.fit` call, shared across several cheap
    assertions on its result -- avoids refitting the same candidate library
    five times over for what are really facets of a single question (did
    the fit behave correctly)."""

    @classmethod
    def setUpClass(cls):
        cls.design, cls.y = _xor_like_dataset(2500, seed=7)
        cls.learner = SuperLearner(random_seed=1)
        cls.fit = cls.learner.fit(cls.design, cls.y)

        splitter, _ = stratified_folds(cls.y, random_seed=1)
        out_of_fold = np.empty(len(cls.y))
        for train_index, test_index in splitter.split(cls.design, cls.y):
            model = LogisticRegression(C=np.inf, max_iter=1000)
            model.fit(cls.design[train_index], cls.y[train_index])
            out_of_fold[test_index] = model.predict_proba(cls.design[test_index])[:, 1]
        cls.plain_logistic_cv_log_loss = float(
            log_loss(cls.y, np.clip(out_of_fold, 1e-6, 1 - 1e-6))
        )

    def test_ensemble_beats_plain_logistic_regression_on_a_nonlinear_dgp(self):
        # The XOR-like interaction is invisible to a linear-in-the-logit
        # model but not to the random-forest/gradient-boosting candidates
        # in the library -- the ensemble's own out-of-fold risk should be
        # meaningfully lower, not just numerically different.
        self.assertLess(self.fit.ensemble_cv_log_loss, self.plain_logistic_cv_log_loss - 0.05)

    def test_tree_based_candidates_get_most_of_the_weight(self):
        tree_weight = (
            self.fit.weights["random_forest"] + self.fit.weights["hist_gradient_boosting"]
        )
        self.assertGreater(tree_weight, 0.5)

    def test_weights_are_nonnegative_and_sum_to_one(self):
        values = list(self.fit.weights.values())
        self.assertTrue(all(value >= 0.0 for value in values))
        self.assertAlmostEqual(sum(values), 1.0, places=6)

    def test_reports_the_citation_and_fold_count(self):
        self.assertIn("van der Laan", self.fit.citation)
        self.assertEqual(self.fit.n_splits, 5)
        self.assertIsNone(self.fit.fallback_reason)

    def test_predict_proba_matches_the_fitted_library(self):
        prediction = self.learner.predict_proba(self.design)
        self.assertEqual(prediction.shape, (len(self.y),))
        self.assertTrue(np.all((prediction >= 0.0) & (prediction <= 1.0)))


class SuperLearnerSampleWeightTests(unittest.TestCase):
    def test_sample_weight_shifts_predictions_toward_the_upweighted_outcome(self):
        rng = np.random.default_rng(3)
        n = 200
        # A near-constant feature -- with no real covariate signal, the
        # fitted ensemble's prediction is essentially the (weighted) mean
        # outcome rate, making the direction of a weight-induced shift easy
        # to reason about.
        design = rng.normal(0, 0.01, n).reshape(-1, 1)
        y = rng.binomial(1, 0.5, n)

        unweighted = SuperLearner(random_seed=0, max_splits=2)
        unweighted.fit(design, y)
        unweighted_mean_pred = float(unweighted.predict_proba(design).mean())

        up_weight_positive = np.where(y == 1, 10.0, 1.0)
        weighted = SuperLearner(random_seed=0, max_splits=2)
        weighted.fit(design, y, sample_weight=up_weight_positive)
        weighted_mean_pred = float(weighted.predict_proba(design).mean())

        self.assertGreater(weighted_mean_pred, unweighted_mean_pred + 0.2)


class SuperLearnerReproducibilityTests(unittest.TestCase):
    def test_is_reproducible_under_a_fixed_seed(self):
        design, y = _xor_like_dataset(300, seed=11)

        first = SuperLearner(random_seed=5, max_splits=2)
        first_fit = first.fit(design, y)
        second = SuperLearner(random_seed=5, max_splits=2)
        second_fit = second.fit(design, y)

        self.assertEqual(first_fit.weights, second_fit.weights)
        self.assertTrue(np.array_equal(first.predict_proba(design), second.predict_proba(design)))


class SuperLearnerEdgeCaseTests(unittest.TestCase):
    def test_predict_proba_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            SuperLearner().predict_proba(np.zeros((5, 1)))

    def test_fit_raises_with_too_few_minority_rows(self):
        design = np.zeros((10, 1))
        y = np.array([0] * 9 + [1])
        with self.assertRaises(ValueError):
            SuperLearner().fit(design, y)


class NormalizeWeightsTests(unittest.TestCase):
    def test_normalizes_positive_coefficients_to_sum_to_one(self):
        weights, reason = _normalize_weights(["a", "b", "c"], np.array([2.0, 2.0, 0.0]))
        self.assertEqual(weights, {"a": 0.5, "b": 0.5, "c": 0.0})
        self.assertIsNone(reason)

    def test_falls_back_to_equal_weights_when_degenerate(self):
        weights, reason = _normalize_weights(["a", "b"], np.array([0.0, 0.0]))
        self.assertEqual(weights, {"a": 0.5, "b": 0.5})
        self.assertIn("degenerate", reason)

    def test_clips_negative_coefficients_before_normalizing(self):
        weights, reason = _normalize_weights(["a", "b"], np.array([-1.0, 1.0]))
        self.assertEqual(weights, {"a": 0.0, "b": 1.0})
        self.assertIsNone(reason)


class StratifiedFoldsTests(unittest.TestCase):
    def test_caps_splits_at_the_minority_class_count(self):
        y = np.array([0] * 20 + [1] * 3)
        splitter, reason = stratified_folds(y, max_splits=5)
        self.assertIsNone(reason)
        self.assertEqual(splitter.get_n_splits(), 3)

    def test_signals_when_too_few_minority_rows(self):
        y = np.array([0] * 20 + [1])
        splitter, reason = stratified_folds(y, max_splits=5)
        self.assertIsNone(splitter)
        self.assertIn("fewer than 2", reason)


class SuperLearnerClassifierAdapterTests(unittest.TestCase):
    """`SuperLearnerClassifier` is `SuperLearner` wrapped to look like a
    real scikit-learn `Estimator` -- for `causal.agents.cate.CATEAgent`'s
    `econml.dml.CausalForestDML` `model_t` (requires a classifier when
    `discrete_treatment=True`)."""

    def test_fit_predict_proba_matches_the_wrapped_superlearner(self):
        design, y = _xor_like_dataset(600, seed=21)
        clf = SuperLearnerClassifier(random_seed=5, max_splits=2)
        clf.fit(design, y)
        proba = clf.predict_proba(design)
        self.assertEqual(proba.shape, (len(y), 2))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)
        direct = SuperLearner(random_seed=5, max_splits=2)
        direct.fit(design, y)
        np.testing.assert_allclose(proba[:, 1], direct.predict_proba(design))

    def test_predict_returns_hard_labels(self):
        design, y = _xor_like_dataset(300, seed=22)
        clf = SuperLearnerClassifier(random_seed=1, max_splits=2)
        clf.fit(design, y)
        predictions = clf.predict(design)
        self.assertTrue(set(np.unique(predictions)) <= {0, 1})

    def test_is_sklearn_clone_compatible(self):
        clf = SuperLearnerClassifier(random_seed=3, max_splits=2)
        cloned = clone(clf)
        self.assertEqual(cloned.get_params(), clf.get_params())

    def test_set_params_updates_constructor_arguments(self):
        clf = SuperLearnerClassifier()
        clf.set_params(random_seed=7, max_splits=3)
        self.assertEqual(clf.random_seed, 7)
        self.assertEqual(clf.max_splits, 3)


class SuperLearnerRegressorAdapterTests(unittest.TestCase):
    """`SuperLearnerRegressor` exposes `SuperLearner`'s `P(Y=1)` prediction
    as a regressor's continuous `predict()` output -- needed for
    `CausalForestDML`'s `model_y`, which must be a regressor even for a
    binary `Y` (verified directly against `econml`: a classifier there
    raises `AttributeError`, see `causal.nuisance`'s module docstring)."""

    def test_predict_matches_the_wrapped_superlearners_predict_proba(self):
        design, y = _xor_like_dataset(600, seed=23)
        reg = SuperLearnerRegressor(random_seed=5, max_splits=2)
        reg.fit(design, y)
        prediction = reg.predict(design)
        direct = SuperLearner(random_seed=5, max_splits=2)
        direct.fit(design, y)
        np.testing.assert_allclose(prediction, direct.predict_proba(design))
        self.assertTrue(np.all((prediction >= 0.0) & (prediction <= 1.0)))

    def test_is_sklearn_clone_compatible(self):
        reg = SuperLearnerRegressor(random_seed=3, max_splits=2)
        cloned = clone(reg)
        self.assertEqual(cloned.get_params(), reg.get_params())

    def test_rejects_predict_before_fit(self):
        reg = SuperLearnerRegressor()
        with self.assertRaises(Exception):
            reg.predict(np.zeros((5, 1)))


if __name__ == "__main__":
    unittest.main()
