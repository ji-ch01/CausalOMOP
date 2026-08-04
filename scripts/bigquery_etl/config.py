"""Standalone BigQuery dataset-role config for the bigquery_etl scripts.

Deliberately self-contained (no import from `causal/`) so this ETL tool has
no dependency on the causal-inference package -- same posture as
`scripts/snuh_etl`, which does not import from `causal/` either. Reads the
same `CAUSALOMOP_BQ_*` variables and the same repo-root `.env` file as
`causal/config.py`, so one `.env` still configures both.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


_BQ_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_local_env(path: str | Path = _PROJECT_ROOT / ".env") -> None:
    """Load simple local KEY=VALUE settings without logging their contents."""
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            os.environ.setdefault(key, value.strip().strip("'\""))


@dataclass(frozen=True)
class BigQueryDatasets:
    """The four dataset roles used by the MIMIC-IV OMOP pipeline."""

    project: str = "example-gcp-project"
    omop: str = "omop_cdm"
    vocabulary: str = "omop_vocabulary"
    etl: str = "omop_etl"
    metrics: str = "omop_metrics"

    @classmethod
    def from_env(cls) -> "BigQueryDatasets":
        _load_local_env()
        return cls(
            project=os.getenv("CAUSALOMOP_BQ_PROJECT", cls.project),
            omop=os.getenv("CAUSALOMOP_BQ_OMOP_DATASET", cls.omop),
            vocabulary=os.getenv("CAUSALOMOP_BQ_VOCAB_DATASET", cls.vocabulary),
            etl=os.getenv("CAUSALOMOP_BQ_ETL_DATASET", cls.etl),
            metrics=os.getenv("CAUSALOMOP_BQ_METRICS_DATASET", cls.metrics),
        ).validated()

    def validated(self) -> "BigQueryDatasets":
        for name, value in self.__dict__.items():
            if not value or not _BQ_ID.fullmatch(value):
                raise ValueError(f"Invalid BigQuery {name}: {value!r}")
        return self

    def fq_dataset(self, role: str) -> str:
        if role not in {"omop", "vocabulary", "etl", "metrics"}:
            raise ValueError(f"Unknown dataset role: {role}")
        return f"{self.project}.{getattr(self, role)}"
