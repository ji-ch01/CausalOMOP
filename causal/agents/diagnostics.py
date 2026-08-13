"""Standardized mean difference (SMD) covariate-balance diagnostics between
treatment arms. See `DiagnosticsAgent` for citation and scope.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from causal.provenance import artifact_metadata

_SMD_FLAG_THRESHOLD = 0.1
_CITATION = (
    "Austin PC. Balance diagnostics for comparing the distribution of baseline "
    "covariates between treatment groups in propensity-score matched samples. "
    "Stat Med. 2009;28(25):3083-3107."
)

# Fixed analytic-cohort columns (causal.data_builder.analytic.
# AnalyticCohortBuilder.COLUMNS) that need special handling beyond their
# declared INT64/FLOAT64 data_type: `age` is a continuous integer, not a
# binary presence flag, and gender_concept_id/race_concept_id are OMOP
# concept_id categories, not numbers -- Austin (2009) covers a categorical
# covariate as one proportion-based SMD per level, not a single numeric SMD.
# Reviewed covariates (`cov_*`) need no such override: FLOAT64 already means
# a continuous value aggregation and INT64 already means a presence/count
# flag (see `causal.data_builder.analytic._VALUE_AGGREGATIONS`).
_CONTINUOUS_COLUMNS = ("age",)
_CATEGORICAL_COLUMNS = ("gender_concept_id", "race_concept_id")


@dataclass(frozen=True)
class CovariateBalance:
    column_name: str
    smd: float
    treated_mean: float
    comparator_mean: float
    flagged: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiagnosticsArtifact:
    metadata: dict[str, str]
    schema_version: str
    study_id: str
    n_total: int
    n_treated: int
    n_comparator: int
    balance: tuple[CovariateBalance, ...]
    n_covariates_flagged: int
    smd_flag_threshold: float
    citation: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DiagnosticsAgent:
    """Standardized mean difference (SMD) covariate-balance table between
    treatment arms, per Austin (2009), `|SMD| > 0.1` as the standard flag
    threshold. Requires no fitted model -- a pure property of the
    materialized covariate distributions vs. treatment assignment, so it can
    run independently of (and before) `EstimationAgent`.

    Deliberately excludes joint/multivariate balance metrics, propensity-
    score weighting diagnostics, negative controls, and quantitative
    sensitivity analysis (E-values etc.) -- see `docs/causal_roadmap.md`
    milestones #5/#6 for why those are separate, later work.
    """

    def run(
        self,
        dataset: pd.DataFrame,
        *,
        study_id: str,
        covariate_columns: tuple[str, ...],
        covariate_data_types: dict[str, str],
    ) -> DiagnosticsArtifact:
        if not study_id.strip():
            raise ValueError("study_id is required")
        if "A" not in dataset.columns:
            raise ValueError("dataset must contain a treatment column 'A'")
        missing = [name for name in covariate_columns if name not in dataset.columns]
        if missing:
            raise ValueError(f"dataset is missing covariate column(s): {', '.join(missing)}")
        if dataset["A"].nunique() < 2:
            raise ValueError("Both treatment arms (A=0 and A=1) must be present to compute balance")

        treated = dataset[dataset["A"] == 1]
        comparator = dataset[dataset["A"] == 0]

        balance: list[CovariateBalance] = []
        for column in covariate_columns:
            if column in _CATEGORICAL_COLUMNS:
                balance.extend(self._categorical_balance(column, treated, comparator))
            elif column in _CONTINUOUS_COLUMNS or covariate_data_types.get(column) == "FLOAT64":
                balance.append(self._continuous_balance(column, treated, comparator))
            else:
                balance.append(self._binary_balance(column, treated, comparator))

        n_flagged = sum(1 for entry in balance if entry.flagged)
        warnings = (
            "|SMD| > 0.1 is the standard flag threshold (Austin 2009); it is a screening "
            "heuristic, not a hypothesis test or a guarantee of unconfoundedness.",
            "Only marginal (per-covariate) balance is assessed -- joint/multivariate balance "
            "across the full adjustment set is not checked.",
        )
        return DiagnosticsArtifact(
            metadata=artifact_metadata(
                "diagnostics_agent", "0.1.0",
                {
                    "study_id": study_id,
                    "covariate_columns": list(covariate_columns),
                    "n_total": int(len(dataset)),
                },
            ),
            schema_version="1.0",
            study_id=study_id.strip(),
            n_total=int(len(dataset)),
            n_treated=int(len(treated)),
            n_comparator=int(len(comparator)),
            balance=tuple(balance),
            n_covariates_flagged=n_flagged,
            smd_flag_threshold=_SMD_FLAG_THRESHOLD,
            citation=_CITATION,
            warnings=warnings,
        )

    @staticmethod
    def _smd_from_means(mean1: float, mean0: float, var1: float, var0: float) -> float:
        pooled_sd = float(np.sqrt((var1 + var0) / 2))
        if pooled_sd == 0:
            # Zero pooled variance with equal means is a genuinely balanced
            # (often constant) covariate -- SMD 0. Zero pooled variance with
            # unequal means (e.g. a binary covariate present in 100% of one
            # arm and 0% of the other) is a real, maximal imbalance that the
            # 0/0 form of the standard formula must not silently report as
            # "balanced" -- signed infinity, still correctly flagged by
            # `abs(smd) > _SMD_FLAG_THRESHOLD`.
            if mean1 == mean0:
                return 0.0
            return float("inf") if mean1 > mean0 else float("-inf")
        return (mean1 - mean0) / pooled_sd

    @classmethod
    def _smd_from_proportions(cls, p1: float, p0: float) -> float:
        return cls._smd_from_means(p1, p0, p1 * (1 - p1), p0 * (1 - p0))

    @classmethod
    def _continuous_balance(cls, column: str, treated: pd.DataFrame, comparator: pd.DataFrame) -> CovariateBalance:
        t = treated[column].dropna().astype(float)
        c = comparator[column].dropna().astype(float)
        t_mean, c_mean = float(t.mean()), float(c.mean())
        smd = cls._smd_from_means(t_mean, c_mean, float(t.var(ddof=1)), float(c.var(ddof=1)))
        return CovariateBalance(column, float(smd), t_mean, c_mean, abs(smd) > _SMD_FLAG_THRESHOLD)

    @classmethod
    def _binary_balance(cls, column: str, treated: pd.DataFrame, comparator: pd.DataFrame) -> CovariateBalance:
        t = treated[column].dropna().astype(float)
        c = comparator[column].dropna().astype(float)
        p1, p0 = float(t.mean()), float(c.mean())
        smd = cls._smd_from_proportions(p1, p0)
        return CovariateBalance(column, float(smd), p1, p0, abs(smd) > _SMD_FLAG_THRESHOLD)

    @classmethod
    def _categorical_balance(
        cls, column: str, treated: pd.DataFrame, comparator: pd.DataFrame
    ) -> list[CovariateBalance]:
        t_col = treated[column].dropna()
        c_col = comparator[column].dropna()
        levels = sorted(set(t_col.unique().tolist()) | set(c_col.unique().tolist()))
        results = []
        for level in levels:
            p1 = float((t_col == level).mean())
            p0 = float((c_col == level).mean())
            smd = cls._smd_from_proportions(p1, p0)
            results.append(
                CovariateBalance(f"{column}={level}", float(smd), p1, p0, abs(smd) > _SMD_FLAG_THRESHOLD)
            )
        return results
