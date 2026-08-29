# Repository Map

CausalOMOP is an integration repository. Project-owned research code and
documentation live beside pinned external Git submodules and ignored local
data.

## Ownership

| Path | Owner and purpose | Git policy |
|---|---|---|
| `README.md`, `docs/` | CausalOMOP documentation | tracked |
| `causal/` | CausalOMOP workflow, agents, contracts, mapping, queries, the conversation layer (`causal/companion/`), and UI | tracked |
| `studies/` | versioned study specifications and reference audits | tracked, no patient data |
| `scripts/bigquery_etl/` | project-owned BigQuery-to-DuckDB exporter | tracked |
| `scripts/snuh_etl/` | project-owned SYN-SNUH ETL and validation | tracked |
| `scripts/stage_demo_run.py` | dev-only: stage a fully synthetic run to `estimand_review` / `results_review` for offline UI-dialogue testing (no OMCP/BigQuery) | tracked |
| `notebooks/` | exploratory analysis | tracked selectively; strip large outputs |
| `tests/` | deterministic project test suite | tracked |
| `artifacts/runs/` | generated workflow artifacts | generated; review before any commit or sharing |
| `data/` | local databases, vocabularies, and backups | `data/README.md` only; data ignored |
| `agno_fastomop/` | [FastOMOP](https://github.com/ji-ch01/agno_fastomop) | Git submodule |
| `omcp/` | [OMCP](https://github.com/ji-ch01/omcp) | Git submodule |
| `foem/` | [FOEM](https://github.com/ji-ch01/foem) | Git submodule |
| `external/ohdsi_mimic/` | [OHDSI/MIMIC](https://github.com/OHDSI/MIMIC) | Git submodule |

Runtime prompt templates under `causal/prompts/` are part of application
behavior even though they use Markdown syntax. Treat their edits like code
changes, with tests and review; they are not general documentation pages.

## Submodule rules

The root repository records only a commit pointer for each submodule. Read the
upstream README for that component and avoid duplicating its installation guide
in CausalOMOP documentation.

```powershell
git submodule status --recursive
git -C agno_fastomop status -sb
git -C omcp status -sb
git -C foem status -sb
git -C external/ohdsi_mimic status -sb
```

An intentional submodule change must be committed in that repository first;
the root repository can then record the new pointer.

## Generated and sensitive files

Do not commit:

- `.env` files, tokens, credentials, or real cloud identifiers;
- DuckDB files, WAL files, BigQuery exports, or patient-level extracts;
- Athena vocabulary CSVs and generated backups;
- virtual environments, caches, and Python bytecode;
- generated reports or LLM traces without an explicit reproducibility and
  disclosure decision.

Generated `demo_report.md` files under `artifacts/runs/` are snapshots, not
maintained documentation. Regenerate them from the workflow instead of editing
them by hand.

## Documentation rules

- Begin at [docs/index.md](index.md).
- Keep operational status in `README.md`, `causal/README.md`, and
  `docs/causal_roadmap.md`.
- Keep dataset access and handling rules in `docs/datasets.md`.
- Keep external component details in their official repositories.
- Preserve superseded proposals as short historical notes or rely on Git
  history; do not maintain a second, conflicting architecture.
