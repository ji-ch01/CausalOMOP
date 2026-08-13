"""Shared design-matrix construction for nuisance/outcome models.

Extracted from `causal.agents.estimation.EstimationAgent` once a second
caller (`causal.agents.tmle.TMLEAgent`) needed the identical one-hot-encode-
categoricals-then-concatenate logic -- same "extract once it has a second
caller" precedent as `causal.nuisance.stratified_folds`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# OMOP concept_id-typed baseline covariates in the fixed analytic-cohort
# schema (`causal.data_builder.analytic.AnalyticCohortBuilder.COLUMNS`) --
# integers with no ordinal meaning, so they must be one-hot encoded rather
# than fed to an outcome/treatment model as raw numbers. Reviewed covariates
# (`cov_*`) are always FLOAT64 value aggregates or INT64 presence flags,
# neither of which needs this treatment.
CATEGORICAL_COLUMNS = ("gender_concept_id", "race_concept_id")


def build_design_matrix(
    frame: pd.DataFrame,
    numeric_columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
    encoder: OneHotEncoder | None,
) -> np.ndarray:
    """Concatenate `numeric_columns` (cast to float) with a one-hot
    encoding of `categorical_columns` via an already-fit `encoder`
    (`fit_categorical_encoder`). `encoder` may be `None` only when
    `categorical_columns` is empty."""
    parts = [frame[list(numeric_columns)].astype(float).to_numpy()]
    if categorical_columns and encoder is not None:
        parts.append(encoder.transform(frame[list(categorical_columns)].astype("Int64").astype(str)))
    return np.concatenate(parts, axis=1)


def fit_categorical_encoder(frame: pd.DataFrame, categorical_columns: tuple[str, ...]) -> OneHotEncoder | None:
    """Fit a `OneHotEncoder` over `categorical_columns` on `frame`, or
    return `None` when there are none -- the shared fit-time counterpart to
    `build_design_matrix`'s transform-time use of the same encoder."""
    if not categorical_columns:
        return None
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(frame[list(categorical_columns)].astype("Int64").astype(str))
    return encoder
