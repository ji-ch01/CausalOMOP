"""Deterministic governance checks for SQL already produced by a trusted
builder (`causal.queries.feasibility`, `AnalyticCohortBuilder`, ...).

This agent never writes and never judges *clinical* correctness of a query
-- it only enforces fixed, auditable rules: no unbound parameters, no
`SELECT *`, only expected OMOP tables, and (for aggregate-only queries) no
row-level identifier in the final projection. An optional BigQuery dry-run
adds syntax validation and a cost estimate without executing the query. No
LLM is involved anywhere in this module, by design -- see
`causal.queries.feasibility` module docstring for why.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from causal.provenance import artifact_metadata
from causal.queries.feasibility import DOMAIN_TABLES


_UNBOUND_PARAMETER = re.compile(r"@\w+")
_TABLE_REFERENCE = re.compile(r"`[^`]+\.([a-zA-Z0-9_]+)`")
_FINAL_SELECT_MARKER = "\n    SELECT"

ALLOWED_TABLES = {table for table, _, _ in DOMAIN_TABLES.values()} | {
    "person", "death", "concept_ancestor", "visit_occurrence", "visit_detail",
    "care_site", "observation_period",
}

# purpose -> columns the final projection must not expose. Row-level
# identifiers are fine in a "person_level_dataset" query (that's the point
# of AnalyticCohortBuilder.build) but must never leak into an
# "aggregate_feasibility" query, which is what OMCP is allowed to run.
FORBIDDEN_COLUMNS_BY_PURPOSE: dict[str, tuple[str, ...]] = {
    "aggregate_feasibility": ("person_id",),
}
KNOWN_PURPOSES = ("aggregate_feasibility", "person_level_dataset")


class DryRunClient(Protocol):
    def query(self, sql: str, job_config: Any = None) -> Any: ...


@dataclass(frozen=True)
class QueryValidationArtifact:
    metadata: dict[str, str]
    schema_version: str
    purpose: str
    status: str
    passed: bool
    dry_run_performed: bool
    estimated_bytes_processed: int | None
    violations: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QueryValidationAgent:
    """Governance checks for one already-rendered SQL string."""

    def run(
        self,
        sql: str,
        *,
        purpose: str,
        dry_run_client: DryRunClient | None = None,
    ) -> QueryValidationArtifact:
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("sql is required")
        if purpose not in KNOWN_PURPOSES:
            raise ValueError(f"purpose must be one of {KNOWN_PURPOSES}")

        violations = self._static_violations(sql, purpose)

        estimated_bytes = None
        dry_run_performed = dry_run_client is not None
        if dry_run_client is not None:
            try:
                estimated_bytes = self._dry_run(sql, dry_run_client)
            except Exception as exc:  # noqa: BLE001 -- surfaced as a violation, never raised
                violations.append(f"dry_run_failed:{exc}")

        passed = not violations
        return QueryValidationArtifact(
            metadata=artifact_metadata(
                "query_validation_agent", "0.1.0", {"sql": sql, "purpose": purpose}
            ),
            schema_version="1.0",
            purpose=purpose,
            status="passed" if passed else "failed",
            passed=passed,
            dry_run_performed=dry_run_performed,
            estimated_bytes_processed=estimated_bytes,
            violations=tuple(violations),
            warnings=(
                "Checks syntax, governance rules, and estimated cost only; it does not "
                "verify that the underlying concept sets or windows are clinically correct.",
            ),
        )

    @classmethod
    def _static_violations(cls, sql: str, purpose: str) -> list[str]:
        violations = []
        if _UNBOUND_PARAMETER.search(sql):
            violations.append("unbound_parameter_placeholder_present")
        if "select *" in sql.lower():
            violations.append("select_star_present")

        referenced_tables = {match.group(1) for match in _TABLE_REFERENCE.finditer(sql)}
        unexpected_tables = referenced_tables - ALLOWED_TABLES
        if unexpected_tables:
            violations.append(f"unexpected_table_reference:{','.join(sorted(unexpected_tables))}")

        forbidden_columns = FORBIDDEN_COLUMNS_BY_PURPOSE.get(purpose, ())
        if forbidden_columns:
            final_select = cls._final_select(sql)
            if final_select is None:
                violations.append("could_not_isolate_final_projection")
            else:
                for column in forbidden_columns:
                    if re.search(rf"\b{re.escape(column)}\b", final_select, re.IGNORECASE):
                        violations.append(f"forbidden_column_in_final_projection:{column}")
        return violations

    @staticmethod
    def _final_select(sql: str) -> str | None:
        """Isolate the query's outermost SELECT (after the WITH clause).

        Relies on the fixed indentation this codebase's builders always use
        (CTE-internal SELECTs are indented 6+ spaces; the final SELECT is
        indented exactly 4) -- not a general SQL parser. Returns None rather
        than guessing when the marker is absent, so a malformed/unexpected
        query fails validation instead of silently skipping the check.
        """
        index = sql.rfind(_FINAL_SELECT_MARKER)
        return sql[index:] if index != -1 else None

    @staticmethod
    def _dry_run(sql: str, client: DryRunClient) -> int:
        try:
            from google.cloud import bigquery
        except ImportError as exc:
            raise RuntimeError("Install requirements-bigquery.txt") from exc
        config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        job = client.query(sql, job_config=config)
        return int(job.total_bytes_processed)
