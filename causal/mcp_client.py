"""Client adapter for the shared OMCP stdio server."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MCPToolResponse:
    text: str
    metadata: dict[str, Any]


class OMCPClient:
    """Small synchronous facade over the official asynchronous MCP client."""

    def __init__(
        self,
        *,
        project: str,
        dataset: str,
        omcp_dir: str | Path = "omcp",
        timeout_seconds: float = 180.0,
    ) -> None:
        self.project = project
        self.dataset = dataset
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

    def lookup_drug(self, term: str) -> MCPToolResponse:
        return asyncio.run(self._call_tool("Lookup_Drug", {"term": term}))

    def lookup_condition(self, term: str) -> MCPToolResponse:
        return asyncio.run(self._call_tool("Lookup_Condition", {"term": term}))

    async def _causal_select(
        self,
        query: str,
        operation: str,
        causal_context: dict[str, Any],
    ) -> MCPToolResponse:
        return await self._call_tool(
            "Causal_Select_Query",
            {
                "query": query,
                "operation": operation,
                "causal_context": causal_context,
            },
        )

    async def _call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> MCPToolResponse:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise RuntimeError(
                "Install the MCP client with: pip install -r requirements-mcp.txt"
            ) from exc

        environment = dict(os.environ)
        environment.update({
            "DB_TYPE": "bigquery",
            "DB_PROJECT": self.project,
            "DB_DATASET": self.dataset,
            "DB_READ_ONLY": "true",
            "CDM_SCHEMA": self.dataset,
            "VOCAB_SCHEMA": self.dataset,
            "MCP_TRANSPORT": "stdio",
            "ENABLE_LANGFUSE": environment.get("ENABLE_LANGFUSE", "false"),
        })
        parameters = StdioServerParameters(
            command="uv",
            args=[
                "run", "--extra", "bigquery", "--directory", str(self.omcp_dir),
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
