"""Materializes the governed analytic cohort SQL into a local parquet file.

`AnalyticCohortBuilder`/`WorkflowManager.build_cohort_plan` already generate
and validate person-level cohort SQL but never execute it. This module is
the pure, testable half of actually pulling those rows into memory once
`WorkflowManager.materialize_analytic_dataset` has confirmed it is
authorized to (see that method's own docstring for the governance path):
parsing OMCP's CSV response with the query's own declared column contract as
the dtype authority, then writing it to a parquet file that never leaves the
governed run directory.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from causal.data_builder.analytic import AnalyticColumn
from causal.provenance import artifact_metadata

DATASET_FILENAME = "analytic_dataset.parquet"

# AnalyticColumn.data_type -> pandas dtype. Nullable "Int64"/"Float64" so a
# NULL in the analytic cohort (an unresolved `censored` column, a missing
# covariate) survives the CSV round-trip as a real null instead of silently
# becoming 0 or forcing the whole column to float.
_DTYPE_BY_ANALYTIC_TYPE = {"INT64": "Int64", "FLOAT64": "Float64"}


@dataclass(frozen=True)
class DatasetManifest:
    """Describes `analytic_dataset.parquet` without repeating any value from
    it -- row/column counts and hashes only. Unlike the parquet file it
    describes, this manifest is safe to include in a report or hand to an
    LLM: `parquet_contains_person_level_data` documents that the
    *referenced file* holds patient-level rows, this JSON object never does.
    """

    metadata: dict[str, str]
    schema_version: str
    study_id: str
    run_id: str
    row_count: int
    column_count: int
    columns: tuple[str, ...]
    parquet_relpath: str
    parquet_sha256: str
    source_cohort_query_sha256: str
    parquet_contains_person_level_data: bool
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_cohort_csv_to_dataframe(csv_text: str, columns: tuple[AnalyticColumn, ...]) -> pd.DataFrame:
    """Parse OMCP's CSV response for `AnalyticCohortBuilder.build()`'s SQL,
    using `columns` (`CohortBuildManifest.columns`, the query's own declared
    contract) as the dtype authority rather than pandas' own type inference
    off the CSV text -- a covariate that happens to be all-zero or entirely
    missing in one run's cohort must not silently end up with a different
    dtype than a run where it varies. Returns columns in `columns`' order.
    """
    if not csv_text.strip():
        raise ValueError("Cohort CSV response is empty")
    dtype_map = {
        column.name: _DTYPE_BY_ANALYTIC_TYPE[column.data_type]
        for column in columns
        if column.data_type in _DTYPE_BY_ANALYTIC_TYPE
    }
    frame = pd.read_csv(StringIO(csv_text), dtype=dtype_map)
    for column in columns:
        if column.data_type == "DATETIME" and column.name in frame.columns:
            frame[column.name] = pd.to_datetime(frame[column.name], errors="coerce")
    expected = [column.name for column in columns]
    missing = [name for name in expected if name not in frame.columns]
    if missing:
        raise ValueError(f"Cohort CSV is missing expected column(s): {', '.join(missing)}")
    return frame[expected]


def write_analytic_dataset(
    dataframe: pd.DataFrame,
    run_dir: Path,
    *,
    study_id: str,
    run_id: str,
    source_query_sha256: str,
) -> DatasetManifest:
    """Write `analytic_dataset.parquet` under `run_dir` and return its
    manifest. Pure I/O over already-in-memory data -- no state/locking here,
    matching `AnalyticCohortBuilder.build()`'s own split from
    `WorkflowManager`; the caller (`WorkflowManager.materialize_analytic_
    dataset`) owns `_RunLock` and `state.json`.
    """
    parquet_path = run_dir / DATASET_FILENAME
    dataframe.to_parquet(parquet_path, index=False)
    parquet_sha256 = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    warnings: list[str] = [
        "This manifest is safe to report or hand to an LLM; the parquet file it "
        "describes is not -- it holds patient-level rows and must stay inside the "
        "governed run directory.",
    ]
    if dataframe.empty:
        warnings.append("The materialized cohort has zero rows.")
    return DatasetManifest(
        metadata=artifact_metadata(
            "dataset_materializer", "0.1.0",
            {
                "study_id": study_id,
                "run_id": run_id,
                "source_query_sha256": source_query_sha256,
                "row_count": len(dataframe),
            },
        ),
        schema_version="1.0",
        study_id=study_id,
        run_id=run_id,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        columns=tuple(dataframe.columns),
        parquet_relpath=DATASET_FILENAME,
        parquet_sha256=parquet_sha256,
        source_cohort_query_sha256=source_query_sha256,
        parquet_contains_person_level_data=True,
        warnings=tuple(warnings),
    )
