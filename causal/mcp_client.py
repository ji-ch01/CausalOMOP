"""Client adapter for the shared OMCP stdio server."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causal.config import BigQueryDatasets

# Mirrors `omcp.governance.SIGNED_CONTEXT_FIELDS`/`sign_causal_context`.
# Deliberately duplicated rather than imported: `causal` treats OMCP as an
# out-of-process service (see `_INHERITED_ENV_NAMES` below), never an
# in-process library, so the two packages share this tiny contract by
# convention instead of a Python import across the process boundary.
_SIGNED_CONTEXT_FIELDS = (
    "run_id",
    "study_id",
    "mode",
    "clinical_mapping_status",
    "analysis_plan_status",
    "aggregate_feasibility_allowed",
    "analytic_dataset_build_allowed",
    "causal_estimation_allowed",
)


def _sign_causal_context(causal_context: dict[str, Any], secret: str) -> str:
    ordered = {name: causal_context[name] for name in _SIGNED_CONTEXT_FIELDS}
    payload = json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class MCPToolResponse:
    text: str
    metadata: dict[str, Any]


# The OMCP subprocess is a third-party server (a separate submodule/service);
# it gets only what it plausibly needs to run and to authenticate to
# BigQuery/Google Cloud/Langfuse, not a blanket copy of this process's
# environment (which could otherwise carry unrelated credentials -- LLM
# provider keys, other services' tokens -- into a process this codebase does
# not control). Matches the "narrow gateway" posture already documented in
# `causal.bigquery.BigQueryOMOPGateway`.
_INHERITED_ENV_NAMES = (
    "PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "SYSTEMDRIVE",
    "TEMP", "TMP", "APPDATA", "LOCALAPPDATA", "PATHEXT", "COMSPEC",
)
_INHERITED_ENV_PREFIXES = ("GOOGLE_", "GCLOUD_", "CLOUDSDK_", "LANGFUSE_")


def _subprocess_environment(overrides: dict[str, str]) -> dict[str, str]:
    inherited = {
        key: value for key, value in os.environ.items()
        if key in _INHERITED_ENV_NAMES or key.startswith(_INHERITED_ENV_PREFIXES)
    }
    inherited.update(overrides)
    return inherited


def _omcp_db_overrides(datasets: BigQueryDatasets) -> tuple[dict[str, str], str]:
    """The `DB_*`/schema env overrides and `uv run --extra` group for
    `datasets.backend`. A pure function of `datasets` (no env/subprocess
    access) so the backend-selection logic is unit-testable on its own,
    without spawning the OMCP subprocess (see `_call_tool`)."""
    common = {
        "DB_READ_ONLY": "true",
        "MCP_TRANSPORT": "stdio",
        "ENABLE_LANGFUSE": os.environ.get("ENABLE_LANGFUSE", "false"),
    }
    if datasets.backend == "duckdb":
        return {
            "DB_TYPE": "duckdb",
            "DB_PATH": str(datasets.resolved_duckdb_path()),
            "CDM_SCHEMA": datasets.duckdb_schema,
            "VOCAB_SCHEMA": datasets.duckdb_schema,
            **common,
        }, "duckdb"
    return {
        "DB_TYPE": "bigquery",
        "DB_PROJECT": datasets.project,
        "DB_DATASET": datasets.omop,
        "CDM_SCHEMA": datasets.omop,
        "VOCAB_SCHEMA": datasets.omop,
        **common,
    }, "bigquery"


class OMCPClient:
    """Small synchronous facade over the official asynchronous MCP client."""

    def __init__(
        self,
        datasets: BigQueryDatasets,
        *,
        omcp_dir: str | Path = "omcp",
        timeout_seconds: float = 180.0,
    ) -> None:
        self.datasets = datasets.validated()
        self.omcp_dir = Path(omcp_dir).resolve()
        self.timeout_seconds = timeout_seconds
        if not self.omcp_dir.is_dir():
            raise FileNotFoundError(f"OMCP directory not found: {self.omcp_dir}")

    def causal_select(
        self,
        query: str,
        operation: str,
        causal_context: dict[str, Any],
    ) -> MCPToolResponse:
        return asyncio.run(self._causal_select(query, operation, causal_context))

    def lookup_drug(self, term: str, *, limit: int = 10, offset: int = 0) -> MCPToolResponse:
        return asyncio.run(
            self._call_tool("Lookup_Drug", {"term": term, "limit": limit, "offset": offset})
        )

    def lookup_condition(self, term: str, *, limit: int = 10, offset: int = 0) -> MCPToolResponse:
        return asyncio.run(
            self._call_tool("Lookup_Condition", {"term": term, "limit": limit, "offset": offset})
        )

    def lookup_concept(
        self, term: str, domain: str, *, vocabulary_id: str | None = None, limit: int = 10, offset: int = 0
    ) -> MCPToolResponse:
        arguments: dict[str, Any] = {"term": term, "domain": domain, "limit": limit, "offset": offset}
        if vocabulary_id is not None:
            arguments["vocabulary_id"] = vocabulary_id
        return asyncio.run(self._call_tool("Lookup_Concept", arguments))

    def get_concepts_by_id(self, concept_ids: list[int]) -> MCPToolResponse:
        return asyncio.run(self._call_tool("Get_Concepts_By_Id", {"concept_ids": concept_ids}))

    async def _causal_select(
        self,
        query: str,
        operation: str,
        causal_context: dict[str, Any],
    ) -> MCPToolResponse:
        # A fresh secret per subprocess spawn: this stdio server is only
        # ever reachable over its own stdin/stdout pipe, so there is no one
        # else to share it with, but it still stops OMCP from accepting a
        # `causal_context` that wasn't produced by this exact call -- see
        # `omcp.governance.verify_causal_context_signature`.
        secret = secrets.token_hex(32)
        signed_context = {
            **causal_context,
            "signature": _sign_causal_context(causal_context, secret),
        }
        return await self._call_tool(
            "Causal_Select_Query",
            {
                "query": query,
                "operation": operation,
                "causal_context": signed_context,
            },
            env_overrides={"OMCP_CONTEXT_SECRET": secret},
        )

    async def _call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        env_overrides: dict[str, str] | None = None,
    ) -> MCPToolResponse:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise RuntimeError(
                "Install the MCP client with: pip install -r requirements-mcp.txt"
            ) from exc

        overrides, extra = _omcp_db_overrides(self.datasets)
        if env_overrides:
            overrides = {**overrides, **env_overrides}
        environment = _subprocess_environment(overrides)
        parameters = StdioServerParameters(
            command="uv",
            args=[
                "run", "--extra", extra, "--directory", str(self.omcp_dir),
                "python", "src/omcp/main.py",
            ],
            env=environment,
        )
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await asyncio.wait_for(session.initialize(), self.timeout_seconds)
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments=arguments),
                    self.timeout_seconds,
                )
        text_parts = [
            item.text for item in result.content
            if getattr(item, "type", None) == "text" and hasattr(item, "text")
        ]
        text = "\n".join(text_parts)
        if result.isError:
            raise RuntimeError(text or f"OMCP tool failed: {tool_name}")
        metadata = getattr(result, "meta", None) or {}
        return _normalize_tool_response(text, dict(metadata))


def _normalize_tool_response(text: str, metadata: dict[str, Any]) -> MCPToolResponse:
    """Unwrap CallToolResult JSON emitted by older FastMCP combinations."""
    try:
        nested = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return MCPToolResponse(text=text, metadata=metadata)
    if not isinstance(nested, dict) or not isinstance(nested.get("content"), list):
        return MCPToolResponse(text=text, metadata=metadata)
    nested_text = "\n".join(
        str(item.get("text", ""))
        for item in nested["content"]
        if isinstance(item, dict) and item.get("type") == "text"
    )
    if nested.get("isError"):
        raise RuntimeError(nested_text or "OMCP rejected the causal query")
    nested_metadata = nested.get("_meta")
    if isinstance(nested_metadata, dict):
        metadata = {**metadata, **nested_metadata}
    return MCPToolResponse(text=nested_text, metadata=metadata)
