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

Full-CDM exports can take hours, so this is built to survive a crash: each
table retries transient network/auth errors with backoff, and re-running the
exact same command against the same output file resumes by skipping tables
that already made it in (pass --overwrite to force a full redo instead).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from scripts.bigquery_etl.config import BigQueryDatasets

logger = logging.getLogger(__name__)

# Exporting the full MIMIC-IV OMOP CDM can take hours; a machine sleeping or a
# network blip mid-export breaks DNS resolution / the TLS connection for OAuth
# token refresh long enough that the underlying library's own internal retry
# (a few minutes) gives up and raises. Retry those transient failures at the
# per-table level instead of letting one blip kill an entire run's progress.
# `OSError` alone covers most of it -- `ssl.SSLError` and every
# `requests.exceptions.*` (ConnectionError, SSLError, Timeout, ...) subclass
# it -- but google-auth and grpc define their own exception hierarchies that
# do not, so those are added explicitly when the packages are installed.
_TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (OSError,)

try:
    from google.api_core import exceptions as _gax_exceptions

    _TRANSIENT_EXCEPTIONS += (
        _gax_exceptions.RetryError,
        _gax_exceptions.ServiceUnavailable,
        _gax_exceptions.DeadlineExceeded,
        _gax_exceptions.TooManyRequests,
    )
except ImportError:
    pass

try:
    from google.auth import exceptions as _gauth_exceptions

    _TRANSIENT_EXCEPTIONS += (_gauth_exceptions.TransportError, _gauth_exceptions.RefreshError)
except ImportError:
    pass

try:
    import grpc

    _TRANSIENT_EXCEPTIONS += (grpc.RpcError,)
except ImportError:
    pass

# Force an explicit checkpoint after this many rows within one table, so a
# huge table's WAL never accumulates more than this much unflushed data
# before DuckDB has to reconcile it -- see `_write_table`.
_CHECKPOINT_EVERY_ROWS = 1_000_000

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


def _qualified(schema: str, table: str) -> str:
    return f'"{schema}"."{table}"' if schema != "main" else f'"{table}"'


def _table_exists(con: Any, schema: str, table: str) -> bool:
    result = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_schema = ? AND table_name = ?",
        [schema, table],
    ).fetchone()
    return result is not None


def _with_retries(func: Callable[[], Any], *, max_retries: int, base_delay: float, description: str) -> Any:
    """Call `func`, retrying transient network/auth errors with exponential backoff."""
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except _TRANSIENT_EXCEPTIONS as exc:
            if attempt == max_retries:
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), 300.0)
            logger.warning(
                "Transient error %s (attempt %d/%d): %s -- retrying in %.0fs",
                description, attempt, max_retries, exc, delay,
            )
            time.sleep(delay)


def _write_table(
    con: Any,
    schema: str,
    table: str,
    arrow_schema: "pyarrow.Schema",  # noqa: F821
    batches: Iterable["pyarrow.RecordBatch"],  # noqa: F821
    overwrite: bool,
) -> int:
    """Write `batches` to `schema.table`, one batch (and DuckDB transaction) at a time.

    A single `CREATE TABLE ... AS SELECT * FROM reader` spanning the whole
    lazy batch stream keeps the entire table in one uncommitted transaction;
    for the largest OMOP tables (e.g. `measurement`, tens/hundreds of
    millions of rows) that accumulates enough buffer-manager memory to raise
    DuckDB's `Out of Memory Error: Allocation failure`. Committing per batch
    bounds a single INSERT's memory, but on a RAM-constrained machine even
    DuckDB's own automatic WAL checkpoint -- triggered once enough
    uncheckpointed inserts pile up -- can itself OOM if it is left to
    reconcile many accumulated batches at once. Checkpointing explicitly
    every `_CHECKPOINT_EVERY_ROWS` rows keeps each checkpoint's own working
    set small and predictable instead of leaving the threshold to chance.
    """
    import pyarrow as pa

    qualified = _qualified(schema, table)
    if schema != "main":
        con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    if _table_exists(con, schema, table):
        if not overwrite:
            raise FileExistsError(
                f"{qualified} already exists in the output DuckDB file; pass overwrite=True to replace it"
            )
        con.execute(f"DROP TABLE {qualified}")

    created = False
    rows_since_checkpoint = 0
    for batch in batches:
        reader = pa.RecordBatchReader.from_batches(arrow_schema, [batch])
        if created:
            con.execute(f"INSERT INTO {qualified} SELECT * FROM reader")
        else:
            con.execute(f"CREATE TABLE {qualified} AS SELECT * FROM reader")
            created = True
        rows_since_checkpoint += batch.num_rows
        if rows_since_checkpoint >= _CHECKPOINT_EVERY_ROWS:
            con.execute("CHECKPOINT")
            rows_since_checkpoint = 0
    if not created:
        # `reader` is resolved by name inside the SQL below via DuckDB's Python
        # replacement scan -- it is used, just not in a way ruff can see.
        reader = pa.RecordBatchReader.from_batches(arrow_schema, [])  # noqa: F841
        con.execute(f"CREATE TABLE {qualified} AS SELECT * FROM reader")

    con.execute("CHECKPOINT")
    return con.execute(f"SELECT COUNT(*) FROM {qualified}").fetchone()[0]


def _make_file_refreshed_credentials(token_file: str | Path) -> Any:
    """OAuth2 credentials whose token is re-read from a local file on each refresh.

    For a host that must not hold a long-lived key (e.g. a rented, ephemeral
    compute pod): `token_file` holds a short-lived impersonated access token
    (`gcloud auth print-access-token --impersonate-service-account=...`) as
    two lines, `<token>\\n<ISO-8601 UTC expiry>\\n`, rewritten periodically by
    whoever controls the impersonating identity. This never mints or
    refreshes a token itself -- it only re-reads whatever the file currently
    holds, so the pod itself never needs impersonation rights. The base
    `Credentials.before_request`/`valid`/`expired` already do the right thing
    once `token`/`expiry` are populated; only `refresh` needs overriding.
    """
    import google.auth.credentials

    token_path = Path(token_file)

    class _FileRefreshedCredentials(google.auth.credentials.Credentials):
        def refresh(self, request: Any) -> None:
            token, _, expiry_iso = token_path.read_text(encoding="utf-8").strip().partition("\n")
            self.token = token.strip()
            expiry = datetime.datetime.fromisoformat(expiry_iso.strip())
            if expiry.tzinfo is not None:
                # google-auth's own expired-check compares against a naive UTC
                # now() (true across current google-auth versions), so a
                # timezone-aware expiry here would raise on comparison.
                expiry = expiry.astimezone(datetime.timezone.utc).replace(tzinfo=None)
            self.expiry = expiry

    return _FileRefreshedCredentials()


def _make_bqstorage_client(credentials: Any = None) -> Any:
    try:
        from google.cloud import bigquery_storage
    except ImportError:
        logger.warning(
            "google-cloud-bigquery-storage is not installed; falling back to the "
            "slower REST API for large tables. Install requirements-bigquery.txt "
            "(includes it) for faster export."
        )
        return None
    return bigquery_storage.BigQueryReadClient(credentials=credentials)


def make_bigquery_client(project: str, credentials: Any = None) -> Any:
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise RuntimeError(
            "BigQuery support is optional. Install with: pip install -r requirements-bigquery.txt"
        ) from exc
    return bigquery.Client(project=project, credentials=credentials)


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
    max_retries: int = 5,
    retry_base_delay: float = 30.0,
    duckdb_memory_limit: str | None = None,
    duckdb_temp_directory: str | None = None,
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
            instead of skipping them. Default is resume behavior: a table
            already present in `output_path` (e.g. left over from a run that
            crashed partway through) is treated as already exported and
            skipped, so re-running the same command after a crash resumes
            from the failure point instead of redoing the whole export.
        max_retries: per-table retry budget for transient network/auth
            errors (e.g. a machine sleeping mid-export breaks DNS resolution
            for OAuth token refresh). Uses exponential backoff, capped at 5
            minutes between attempts.
        retry_base_delay: base delay in seconds before the first retry.
        duckdb_memory_limit: DuckDB `memory_limit` PRAGMA value (e.g.
            `"3GB"`). DuckDB otherwise defaults to ~80% of detected system
            RAM, which on a machine with little free memory to begin with
            can still be enough to trigger an out-of-memory error during a
            large table's checkpoint; setting an explicit, conservative
            limit here keeps DuckDB inside a known-safe budget.
        duckdb_temp_directory: DuckDB `temp_directory` PRAGMA value, a
            filesystem path DuckDB may spill to when over `memory_limit`.
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
    if duckdb_memory_limit is not None:
        con.execute("PRAGMA memory_limit=?", [duckdb_memory_limit])
    if duckdb_temp_directory is not None:
        Path(duckdb_temp_directory).mkdir(parents=True, exist_ok=True)
        con.execute("PRAGMA temp_directory=?", [duckdb_temp_directory])
    results: list[TableExportResult] = []
    try:
        # The `lambda`/`_do_export` closures below capture `table`/`table_ref`,
        # but each is handed straight to `_with_retries` and fully consumed
        # inside its own iteration -- none outlives the loop, so the
        # late-binding hazard B023 warns about cannot occur here.
        for table in resolved_tables:
            table_ref = f"{dataset_fqn}.{table}"
            table_present = not overwrite and _table_exists(con, schema, table)

            # Chunked writes (see `_write_table`) mean a crash mid-table can leave
            # a partially-populated table behind, so presence alone no longer
            # implies completeness -- verify the row count matches before
            # trusting a resume-skip; a mismatch means re-export it.
            if table_present and verify_row_counts:
                dest_count = con.execute(f"SELECT COUNT(*) FROM {_qualified(schema, table)}").fetchone()[0]
                source_count = _with_retries(
                    lambda: _source_row_count(client, table_ref),  # noqa: B023
                    max_retries=max_retries, base_delay=retry_base_delay,
                    description=f"counting {table_ref}",
                )
                if source_count == dest_count:
                    logger.info(
                        "Resuming: %s.%s already complete (%d rows), skipping re-fetch",
                        schema, table, dest_count,
                    )
                    results.append(
                        TableExportResult(table=table, source_row_count=source_count, dest_row_count=dest_count)
                    )
                    continue
                logger.warning(
                    "%s.%s exists but is incomplete (dest=%d, source=%d) -- likely a partial write "
                    "from an earlier crash; re-exporting",
                    schema, table, dest_count, source_count,
                )
            elif table_present:
                logger.info(
                    "Resuming: %s.%s already present in %s, skipping re-fetch (row-count verification "
                    "is disabled, so completeness was not checked)",
                    schema, table, output_path,
                )
                dest_count = con.execute(f"SELECT COUNT(*) FROM {_qualified(schema, table)}").fetchone()[0]
                results.append(TableExportResult(table=table, source_row_count=dest_count, dest_row_count=dest_count))
                continue

            logger.info("Exporting %s -> %s.%s", table_ref, schema, table)

            def _do_export() -> int:
                arrow_schema, batches = _fetch_arrow_batches(client, table_ref, batch_size, storage_client)  # noqa: B023
                return _write_table(con, schema, table, arrow_schema, batches, overwrite=True)  # noqa: B023

            dest_count = _with_retries(
                _do_export, max_retries=max_retries, base_delay=retry_base_delay,
                description=f"exporting {table_ref}",
            )
            source_count = (
                _with_retries(
                    lambda: _source_row_count(client, table_ref),  # noqa: B023
                    max_retries=max_retries, base_delay=retry_base_delay,
                    description=f"counting {table_ref}",
                )
                if verify_row_counts
                else dest_count
            )
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
        "--overwrite", action="store_true",
        help=(
            "replace tables that already exist in the output file, instead of the default "
            "resume behavior (skip tables already present, e.g. after a crashed run)"
        ),
    )
    parser.add_argument(
        "--max-retries", type=int, default=5,
        help="per-table retry budget for transient network/auth errors (default: 5)",
    )
    parser.add_argument(
        "--memory-limit",
        help=(
            "DuckDB memory_limit PRAGMA (e.g. '3GB'); DuckDB otherwise defaults to ~80%% of "
            "detected system RAM, which on a memory-constrained machine can still be enough "
            "to OOM while checkpointing a large table. Recommended if exports OOM."
        ),
    )
    parser.add_argument(
        "--temp-directory",
        help="DuckDB temp_directory PRAGMA: a filesystem path DuckDB may spill to when over --memory-limit",
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
    # CAUSALOMOP_BQ_TOKEN_FILE: opt-in path for a host that must not hold a
    # long-lived key (e.g. a rented compute pod) -- authenticate from a
    # short-lived impersonated access token that gets rewritten periodically
    # by whoever holds impersonation rights, instead of from a downloadable
    # service-account key file / ADC. Unset (the default) is unaffected: both
    # clients fall back to normal ADC discovery exactly as before.
    token_file = os.environ.get("CAUSALOMOP_BQ_TOKEN_FILE")
    credentials = _make_file_refreshed_credentials(token_file) if token_file else None

    client = make_bigquery_client(datasets.project, credentials=credentials)
    bqstorage_client: Any
    if args.no_bqstorage:
        bqstorage_client = None
    elif credentials is not None:
        bqstorage_client = _make_bqstorage_client(credentials)
    else:
        bqstorage_client = "auto"
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
        max_retries=args.max_retries,
        duckdb_memory_limit=args.memory_limit,
        duckdb_temp_directory=args.temp_directory,
        bqstorage_client=bqstorage_client,
    )
    print(json.dumps(summary.to_dict(), indent=2))
    return 0 if summary.all_verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
