# FastOMOP Integration

This project uses FastOMOP as the descriptive OMOP query layer and as the base
architecture for the planned causal reasoning extension.

## Included FastOMOP Repositories

| Path | Upstream role | Used for |
|---|---|---|
| `agno_fastomop/` | FastOMOP implementation using Agno agents | NL-to-SQL clinical query workflow |
| `omcp/` | Model Context Protocol server for OMOP CDM | database tools exposed to agents |
| `foem/` | FastOMOP Evaluation and Monitoring | benchmark SQL query evaluation |

These are submodules. They are intentionally kept separate from project-owned
causal and ETL code.

## Runtime Relationship

```text
User query
  |
  v
agno_fastomop
  |
  +-- Semantic Agent maps clinical terms to OMOP concepts
  |
  +-- Database Agent generates and executes SQL
          |
          v
        omcp
          |
          v
        OMOP database
```

`foem` is not required for normal runtime. It is useful for evaluation.

## Why Causal Work Should Stay Near NL-to-SQL

Causal questions still need:

- concept lookup;
- exposure/comparator mapping;
- outcome concept mapping;
- cohort SQL;
- covariate SQL;
- database execution.

For that reason, the causal extension should not be a completely separate
top-level agent team that bypasses NL-to-SQL. A practical architecture is:

```text
FastOMOP orchestrator
  |
  v
NL-to-SQL team
  |
  +-- descriptive mode
  |     semantic context -> SQL answer
  |
  +-- causal mode
        causal study spec -> OMOP cohort/covariate SQL -> estimate/report
```

The orchestrator should route intent. The causal planner should create a target
trial specification before SQL is generated.

## Local Configuration

Key files:

| File | Purpose |
|---|---|
| `agno_fastomop/config.toml` | model/provider and agent settings |
| `agno_fastomop/.env` | local LLM and tracing environment |
| `omcp/sample.env` | OMCP environment example |
| `docker-compose.yml` | integrated local runtime |

Check that the configured database path points to the file that actually exists.
For example, if the mounted file is:

```text
data/synthetic_snuh_curated.duckdb
```

then the container path should be:

```text
DB_PATH=/app/data/synthetic_snuh_curated.duckdb
```

## Docker Run

From the repository root:

```bash
docker compose up --build
```

Before running, verify:

- `OPENAI_API_KEY` or the chosen provider key is available;
- `DB_PATH` matches the mounted DuckDB filename;
- `DB_HOST_PATH` points to the local folder containing the database.

## Local Run

Install and run FastOMOP from the submodule:

```bash
cd agno_fastomop
uv sync
uv run python -m agno_fastomop.run_agent
```

`agno_fastomop` launches `omcp` through its MCP command configuration.

## Evaluation

`foem` contains benchmark queries:

```bash
cd foem
uv run python src/foem/sql_test.py
```

Use this for NL-to-SQL regression checks, not for causal validity.

## Submodule Hygiene

Inspect submodule status regularly:

```bash
git submodule status --recursive
git -C agno_fastomop status -sb
git -C omcp status -sb
git -C foem status -sb
```

If a submodule is edited, commit inside that submodule first, then commit the
updated pointer in the root repo.

