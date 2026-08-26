"""Bridge from CausalOMOP's causal-question gate to the FastOMOP NL-to-SQL agent.

`agno_fastomop/` is a separate submodule with its own Python 3.13 + `uv`-managed
environment (agno, an Ollama/Anthropic/OpenAI client, langfuse...) that is
deliberately not installed into this project's own `.venv` -- the same
"out-of-process service" posture `causal.mcp_client.OMCPClient` already takes
toward OMCP (see that module's docstring), extended here because FastOMOP pulls
in a large, independently-versioned dependency tree of its own. So instead of
importing `agno_fastomop.workflows.omop_workflow` in-process, this module shells
out to the `agno_fastomop ask` CLI command added alongside it for this purpose.

The result (the SQL FastOMOP executed, plus its own natural-language answer) is
meant for direct display to the human user -- CLI/Streamlit -- and must never be
routed into `causal.llm`/CausalOMOP's own hypothesis-drafting LLM: FastOMOP
queries can surface patient-level content, and docs/datasets.md's data-handling
rules forbid sending patient-level data to an LLM provider.

WARNING: unlike `causal.mcp_client.OMCPClient` (which this module mirrors),
`agno_fastomop` was built and has only run against the DuckDB backend.
`_omcp_db_overrides` below will happily derive `DB_TYPE=bigquery` overrides
too, and OMCP's own optional `--extra bigquery` uv group (`omcp/pyproject.toml`,
auto-selected by `omop_workflow._omcp_command_for_database`) will get pulled in
the same way it is for a causal-side `OMCPClient` call -- but FastOMOP's own
semantic/database agents have never actually been exercised against a BigQuery
backend end to end. Keep `CAUSALOMOP_DB_BACKEND=duckdb` for this path until
that's confirmed.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from causal.config import BigQueryDatasets
from causal.mcp_client import _omcp_db_overrides

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RESULT_PREFIX = "FASTOMOP_JSON:"


class FastOMOPError(RuntimeError):
    """Raised when the FastOMOP subprocess fails, times out, or is unavailable."""


@dataclass(frozen=True)
class FastOMOPAnswer:
    answer: str
    sql_query: str | None


def ask_fastomop(
    question: str,
    *,
    datasets: BigQueryDatasets | None = None,
    fastomop_dir: str | Path | None = None,
    omcp_dir: str | Path | None = None,
    timeout_seconds: float = 180.0,
) -> FastOMOPAnswer:
    """Run one descriptive question through FastOMOP end to end.

    Reuses `causal.mcp_client._omcp_db_overrides` to derive the same `DB_*`
    environment OMCP itself expects, so FastOMOP's own OMCP subprocess (see
    `omop_workflow.initialize_workflow`) points at the same database this
    process resolved from `.env`/`BigQueryDatasets`. That helper always
    includes `DB_READ_ONLY=true` -- important because DuckDB allows only one
    writer at a time, and a causal-side `OMCPClient` connection may be open
    against the same file concurrently.
    """
    fastomop_dir = Path(fastomop_dir).resolve() if fastomop_dir else (_PROJECT_ROOT / "agno_fastomop")
    if not fastomop_dir.is_dir():
        raise FileNotFoundError(f"FastOMOP directory not found: {fastomop_dir}")
    omcp_dir = Path(omcp_dir).resolve() if omcp_dir else (_PROJECT_ROOT / "omcp")
    if not omcp_dir.is_dir():
        raise FileNotFoundError(f"OMCP directory not found: {omcp_dir}")

    datasets = (datasets or BigQueryDatasets.from_env()).validated()
    db_overrides, _extra = _omcp_db_overrides(datasets)
    environment = {
        **os.environ,
        **db_overrides,
        # Expanded by FastOMOP's own config.toml `[omcp].command` at call time
        # (`${OMCP_SERVER_DIR}`) to launch its OMCP subprocess -- mirrors the
        # `omcp_dir` this process itself would use via `OMCPClient`.
        "OMCP_SERVER_DIR": str(omcp_dir),
    }

    try:
        process = subprocess.run(
            ["uv", "run", "--directory", str(fastomop_dir), "agno_fastomop", "ask", question],
            cwd=fastomop_dir,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise FastOMOPError("uv is not installed or not on PATH -- required to run FastOMOP") from exc
    except subprocess.TimeoutExpired as exc:
        raise FastOMOPError(f"FastOMOP did not respond within {timeout_seconds:.0f}s") from exc

    if process.returncode != 0:
        raise FastOMOPError(process.stderr.strip() or f"FastOMOP exited with status {process.returncode}")

    payload = None
    for line in process.stdout.splitlines():
        if line.startswith(_RESULT_PREFIX):
            payload = line[len(_RESULT_PREFIX):]
            break
    if payload is None:
        raise FastOMOPError(f"FastOMOP produced no result payload. stdout: {process.stdout[-2000:]!r}")

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise FastOMOPError(f"Could not parse FastOMOP result: {payload!r}") from exc

    if data.get("error"):
        raise FastOMOPError(data["error"])

    return FastOMOPAnswer(answer=data.get("answer") or "", sql_query=data.get("sql_query"))
