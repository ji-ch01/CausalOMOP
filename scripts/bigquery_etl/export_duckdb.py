"""Export the BigQuery OMOP CDM dataset to a local DuckDB file.

`OMCPClient` (in `causal/mcp_client.py`) always points `CDM_SCHEMA` and
`VOCAB_SCHEMA` at the same BigQuery dataset role ("omop"), because the
converted MIMIC-IV clinical tables and the vocabulary tables already live
together in that one dataset. This module pulls every table out of that
dataset (read-only, via `list_rows`/`to_arrow_iterable`, so no query bytes
are billed) and materializes it as a table in a local DuckDB file. Pointing
`DB_TYPE=duckdb`, `CDM_SCHEMA=main`, `VOCAB_SCHEMA=main` at the resulting
file at that point is a drop-in stand-in for `DB_TYPE=bigquery` during local
development, without hitting BigQuery for every query.

The exported .duckdb file carries the same governed clinical data as the
source BigQuery dataset (MIMIC-IV requires credentialed access); treat it
with the same care -- never commit it (already covered by `*.duckdb` in
.gitignore) and keep it out of shared/public locations.

This is a standalone ETL tool (like `scripts/snuh_etl`): no import from
`causal/`, run directly with:

    python -m scripts.bigquery_etl.export_duckdb data/mimiciv_omop.duckdb
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from scripts.bigquery_etl.config import BigQueryDatasets

logger = logging.getLogger(__name__)

# Schema/table names are spliced directly into DDL strings below (DuckDB's
# Python driver has no identifier-binding parameter); restrict them to a safe
# character set rather than trusting BigQuery dataset listings verbatim.
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_]+$")


def _validate_identifier(kind: str, name: str) -> None:
    if not _IDENTIFIER.fullmatch(name):
        raise ValueError(f"Invalid {kind} name for a DuckDB identifier: {name!r}")


# Conservative BigQuery standard-SQL type -> Arrow type mapping, used only as
# a fallback to give a zero-row table a schema. Populated tables instead use
# the exact schema off their first real Arrow batch, since the BigQuery
# client's own to-Arrow conversion (decimal precision, timestamp units, ...)
# is authoritative and must match what `to_arrow_iterable` actually yields.
_BQ_TO_ARROW: dict[str, str] = {
    "STRING": "string", "BYTES": "binary", "INTEGER": "int64", "INT64": "int64",
    "FLOAT": "float64", "FLOAT64": "float64", "BOOLEAN": "bool", "BOOL": "bool",
    "TIMESTAMP": "timestamp[us, tz=UTC]", "DATE": "date32", "TIME": "time64[us]",
    "DATETIME": "timestamp[us]", "NUMERIC": "decimal128(38, 9)", "BIGNUMERIC": "decimal128(38, 9)",
}


@dataclass(frozen=True)
class TableExportResult:
    table: str
    source_row_count: int
    dest_row_count: int

    @property
    def verified(self) -> bool:
        return self.source_row_count == self.dest_row_count


@dataclass(frozen=True)
class ExportSummary:
    output_path: str
    schema: str
    results: tuple[TableExportResult, ...]

    @property
    def all_verified(self) -> bool:
        return all(result.verified for result in self.results)

    @property
    def mismatched_tables(self) -> tuple[str, ...]:
        return tuple(result.table for result in self.results if not result.verified)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path,
            "schema": self.schema,
            "all_verified": self.all_verified,
            "mismatched_tables": list(self.mismatched_tables),
            "tables": [
                {
                    "table": result.table,
                    "source_row_count": result.source_row_count,
                    "dest_row_count": result.dest_row_count,
                    "verified": result.verified,
                }
                for result in self.results
            ],
        }


def list_dataset_tables(client: Any, datasets: BigQueryDatasets, role: str = "omop") -> tuple[str, ...]:
    """Discover every table currently in the given BigQuery dataset role."""
    dataset = client.get_dataset(datasets.fq_dataset(role))
    return tuple(sorted(table.table_id for table in client.list_tables(dataset)))


def _bq_schema_to_arrow_fields(schema: Any) -> list[tuple[str, str, bool]]:
    return [
        (field.name, _BQ_TO_ARROW.get(field.field_type, "string"), field.mode != "REQUIRED")
        for field in schema
    ]


def _fetch_arrow_batches(
    client: Any, table_ref: str, batch_size: int, bqstorage_client: Any = None
) -> tuple["pyarrow.Schema", Iterable["pyarrow.RecordBatch"]]:  # noqa: F821
    """Return (schema, lazy batch iterator) for one BigQuery table.

    Reads via `list_rows`, which is a storage read (no bytes billed), not a
    query. The schema comes off the first real batch so it exactly matches
    what the batches themselves carry; only a genuinely empty table falls
    back to the hand-built BigQuery-type mapping.
    """
    row_iterator = client.list_rows(table_ref, page_size=batch_size)
    batch_iter = iter(row_iterator.to_arrow_iterable(bqstorage_client=bqstorage_client))
    first_batch = next(batch_iter, None)
    if first_batch is None:
        import pyarrow as pa

        fields = _bq_schema_to_arrow_fields(row_iterator.schema)
        schema = pa.schema([
            pa.field(name, _arrow_type(pa, kind), nullable=nullable) for name, kind, nullable in fields
        ])
        return schema, iter(())

    def _chain() -> Iterable["pyarrow.RecordBatch"]:  # noqa: F821
        yield first_batch
        yield from batch_iter

    return first_batch.schema, _chain()


def _arrow_type(pa: Any, kind: str) -> Any:
    if kind == "decimal128(38, 9)":
        return pa.decimal128(38, 9)
    if kind.startswith("timestamp"):
        if "tz=UTC" in kind:
            return pa.timestamp("us", tz="UTC")
        return pa.timestamp("us")
    if kind == "date32":
        return pa.date32()
    if kind == "time64[us]":
        return pa.time64("us")
    return getattr(pa, kind)()


def _source_row_count(client: Any, table_ref: str) -> int:
    rows = list(client.query(f"SELECT COUNT(*) AS n FROM `{table_ref}`").result())
    return int(rows[0].n) if rows else 0


def _table_exists(con: Any, schema: str, table: str) -> bool:
    result = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_schema = ? AND table_name = ?",
        [schema, table],
    ).fetchone()
    return result is not None


def _write_table(
    con: Any,
    schema: str,
    table: str,
    arrow_schema: "pyarrow.Schema",  # noqa: F821
    batches: Iterable["pyarrow.RecordBatch"],  # noqa: F821
    overwrite: bool,
) -> int:
    import pyarrow as pa

    qualified = f'"{schema}"."{table}"' if schema != "main" else f'"{table}"'
    if schema != "main":
        con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    if _table_exists(con, schema, table):
        if not overwrite:
            raise FileExistsError(
                f"{qualified} already exists in the output DuckDB file; pass overwrite=True to replace it"
            )
        con.execute(f"DROP TABLE {qualified}")

    reader = pa.RecordBatchReader.from_batches(arrow_schema, batches)
    con.execute(f"CREATE TABLE {qualified} AS SELECT * FROM reader")
    return con.execute(f"SELECT COUNT(*) FROM {qualified}").fetchone()[0]


def _make_bqstorage_client() -> Any:
    try:
        from google.cloud import bigquery_storage
    except ImportError:
        logger.warning(
            "google-cloud-bigquery-storage is not installed; falling back to the "
            "slower REST API for large tables. Install requirements-bigquery.txt "
            "(includes it) for faster export."
        )
        return None
    return bigquery_storage.BigQueryReadClient()


def make_bigquery_client(project: str) -> Any:
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise RuntimeError(
            "BigQuery support is optional. Install with: pip install -r requirements-bigquery.txt"
        ) from exc
    return bigquery.Client(project=project)


def export_dataset_to_duckdb(
    client: Any,
    datasets: BigQueryDatasets,
    output_path: str | Path,
    *,
    role: str = "omop",
    tables: Sequence[str] | None = None,
    schema: str = "main",
    batch_size: int = 100_000,
    verify_row_counts: bool = True,
    overwrite: bool = False,
    bqstorage_client: Any = "auto",
) -> ExportSummary:
    """Export `tables` (default: every table in the dataset) to a local DuckDB file.

    Args:
        client: an authenticated `google.cloud.bigquery.Client` (or a
            duck-typed fake exposing `get_dataset`/`list_tables`/`list_rows`/`query`).
        datasets: resolved dataset roles (see `scripts.bigquery_etl.config.BigQueryDatasets`).
        output_path: path to the DuckDB file to create/append to.
        role: which dataset role to read from; "omop" (the default) is what
            the rest of this codebase treats as the analysis-ready source.
        tables: restrict the export to this subset; default is every table
            currently in the dataset.
        schema: DuckDB schema to write into. "main" (the default) matches
            the simplest local dev setup: point `CDM_SCHEMA=main` and
            `VOCAB_SCHEMA=main` at the resulting file.
        verify_row_counts: run a cheap `COUNT(*)` on both sides per table
            (BigQuery optimizes a bare `COUNT(*)` to a metadata read, so this
            does not bill query bytes) and record any mismatch.
        overwrite: replace tables that already exist in the output file
            instead of raising.
        bqstorage_client: pass an existing `BigQueryReadClient` to reuse one
            across calls, `None` to force the slower REST fallback, or leave
            "auto" to create one if `google-cloud-bigquery-storage` is
            installed.
    """
    import duckdb

    resolved_tables = tuple(tables) if tables else list_dataset_tables(client, datasets, role)
    _validate_identifier("schema", schema)
    for table in resolved_tables:
        _validate_identifier("table", table)
    dataset_fqn = datasets.fq_dataset(role)
    storage_client = _make_bqstorage_client() if bqstorage_client == "auto" else bqstorage_client

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(output_path))
    results: list[TableExportResult] = []
    try:
        for table in resolved_tables:
            table_ref = f"{dataset_fqn}.{table}"
            logger.info("Exporting %s -> %s.%s", table_ref, schema, table)
            arrow_schema, batches = _fetch_arrow_batches(client, table_ref, batch_size, storage_client)
            dest_count = _write_table(con, schema, table, arrow_schema, batches, overwrite)
            source_count = _source_row_count(client, table_ref) if verify_row_counts else dest_count
            result = TableExportResult(table=table, source_row_count=source_count, dest_row_count=dest_count)
            if not result.verified:
                logger.warning(
                    "Row count mismatch for %s: source=%d dest=%d",
                    table, result.source_row_count, result.dest_row_count,
                )
            results.append(result)
    finally:
        con.close()

    return ExportSummary(output_path=str(output_path), schema=schema, results=tuple(results))


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.bigquery_etl.export_duckdb",
        description="Export the BigQuery OMOP dataset to a local DuckDB file.",
    )
    parser.add_argument("output", help="path to the DuckDB file to create")
    parser.add_argument(
        "--tables", nargs="+", default=None,
        help="restrict the export to these tables (default: every table in the dataset)",
    )
    parser.add_argument(
        "--role", default="omop",
        help="BigQuery dataset role to read from (default: omop, the analysis-ready dataset)",
    )
    parser.add_argument(
        "--schema", default="main",
        help="DuckDB schema to write into (default: main; point CDM_SCHEMA/VOCAB_SCHEMA at the same value)",
    )
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument(
        "--no-verify", action="store_true", help="skip the per-table source/dest row-count check"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace tables that already exist in the output file"
    )
    parser.add_argument(
        "--no-bqstorage", action="store_true",
        help=(
            "skip the BigQuery Storage Read API and fall back to the slower, single-threaded "
            "REST API; use this if the Storage API's parallel reads are hitting OAuth "
            "token-refresh errors (e.g. 401 ACCESS_TOKEN_TYPE_UNSUPPORTED / RemoteDisconnected)"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    datasets = BigQueryDatasets.from_env()
    client = make_bigquery_client(datasets.project)
    summary = export_dataset_to_duckdb(
        client,
        datasets,
        args.output,
        role=args.role,
        tables=args.tables,
        schema=args.schema,
        batch_size=args.batch_size,
        verify_row_counts=not args.no_verify,
        overwrite=args.overwrite,
        bqstorage_client=None if args.no_bqstorage else "auto",
    )
    print(json.dumps(summary.to_dict(), indent=2))
    return 0 if summary.all_verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
