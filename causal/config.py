"""Runtime configuration for the CausalOMOP BigQuery/DuckDB deployment."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


_BQ_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BACKENDS = ("bigquery", "duckdb")
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_local_env(path: str | Path = ".env") -> None:
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
    """The four BigQuery dataset roles used by the MIMIC-IV OMOP pipeline,
    plus the DuckDB backend that stands in for them during local development
    (see `scripts.bigquery_etl.export_duckdb`). `backend` selects which of
    the two groups of fields below is actually used; the inactive group
    keeps its defaults but is ignored by `table()`/`describe()`.

    For DuckDB, `omop`/`vocabulary`/`etl`/`metrics` collapse to one
    `duckdb_schema`: `export_duckdb.py` exports the single BigQuery "omop"
    dataset role (clinical tables and vocabulary already live together
    there, per `causal.mcp_client.OMCPClient`'s own docstring) into one
    schema of one local file, so there is no separate physical dataset per
    role to point at locally.
    """

    project: str = "example-gcp-project"
    omop: str = "omop_cdm"
    vocabulary: str = "omop_vocabulary"
    etl: str = "omop_etl"
    metrics: str = "omop_metrics"
    backend: str = "bigquery"
    duckdb_path: str = "data/mimiciv_omop.duckdb"
    duckdb_schema: str = "main"

    @classmethod
    def from_env(cls) -> "BigQueryDatasets":
        _load_local_env()
        datasets = cls(
            project=os.getenv("CAUSALOMOP_BQ_PROJECT", cls.project),
            omop=os.getenv("CAUSALOMOP_BQ_OMOP_DATASET", cls.omop),
            vocabulary=os.getenv("CAUSALOMOP_BQ_VOCAB_DATASET", cls.vocabulary),
            etl=os.getenv("CAUSALOMOP_BQ_ETL_DATASET", cls.etl),
            metrics=os.getenv("CAUSALOMOP_BQ_METRICS_DATASET", cls.metrics),
            backend=os.getenv("CAUSALOMOP_DB_BACKEND", cls.backend),
            duckdb_path=os.getenv("CAUSALOMOP_DUCKDB_PATH", cls.duckdb_path),
            duckdb_schema=os.getenv("CAUSALOMOP_DUCKDB_SCHEMA", cls.duckdb_schema),
        ).validated()
        # `project` is a regex-valid but meaningless placeholder when
        # `CAUSALOMOP_BQ_PROJECT` was never actually set -- e.g. a missing or
        # misplaced repo-root `.env` (`_load_local_env` only ever *fills in*
        # unset variables, so a bad path fails silently here, not loudly).
        # Left unchecked, this class still builds a fully "valid" client that
        # only fails later, deep inside a BigQuery API call, in a way that
        # gives no hint the real MIMIC-IV OMOP dataset config never loaded.
        if datasets.backend == "bigquery" and datasets.project == cls.project:
            raise RuntimeError(
                f"CAUSALOMOP_BQ_PROJECT is not set -- refusing to run against the "
                f"placeholder project {cls.project!r}. Set CAUSALOMOP_BQ_PROJECT and the "
                "other CAUSALOMOP_BQ_*_DATASET variables in a repo-root .env (see "
                "docs/datasets.md), or set CAUSALOMOP_DB_BACKEND=duckdb to use a local "
                "DuckDB export instead."
            )
        return datasets

    def validated(self) -> "BigQueryDatasets":
        if self.backend not in _BACKENDS:
            raise ValueError(f"Invalid backend: {self.backend!r}; must be one of {_BACKENDS}")
        if self.backend == "bigquery":
            for name in ("project", "omop", "vocabulary", "etl", "metrics"):
                value = getattr(self, name)
                if not value or not _BQ_ID.fullmatch(value):
                    raise ValueError(f"Invalid BigQuery {name}: {value!r}")
        else:
            if not self.duckdb_path.strip():
                raise ValueError("duckdb_path is required when backend='duckdb'")
            if not _SQL_IDENTIFIER.fullmatch(self.duckdb_schema):
                raise ValueError(f"Invalid DuckDB schema: {self.duckdb_schema!r}")
        return self

    def resolved_duckdb_path(self) -> Path:
        """Absolute path to the local DuckDB file, resolved against the repo
        root (not the caller's cwd) -- the OMCP subprocess is launched with
        `--directory <omcp_dir>`, so a relative `DB_PATH` would otherwise
        resolve against `omcp/`, not this repo."""
        path = Path(self.duckdb_path)
        return path if path.is_absolute() else (_PROJECT_ROOT / path).resolve()

    def fq_dataset(self, role: str) -> str:
        if role not in {"omop", "vocabulary", "etl", "metrics"}:
            raise ValueError(f"Unknown dataset role: {role}")
        if self.backend == "duckdb":
            return self.duckdb_schema
        return f"{self.project}.{getattr(self, role)}"

    def table(self, role: str, table_name: str) -> str:
        if not _BQ_ID.fullmatch(table_name):
            raise ValueError(f"Invalid table name: {table_name!r}")
        if self.backend == "duckdb":
            if role not in {"omop", "vocabulary", "etl", "metrics"}:
                raise ValueError(f"Unknown dataset role: {role}")
            return f'"{self.duckdb_schema}"."{table_name}"'
        return f"`{self.fq_dataset(role)}.{table_name}`"

    def describe(self) -> str:
        """Human-readable data-source label for LLM prompts/UI display."""
        if self.backend == "duckdb":
            return f"duckdb:{self.resolved_duckdb_path()}#{self.duckdb_schema}"
        return self.fq_dataset("omop")
