# Repository Map

This repository is a project-level integration repo. It combines local research
code, documentation, notebooks, local data conventions, and pinned FastOMOP
submodules.

## Top-Level Structure

| Path | Purpose | Git policy |
|---|---|---|
| `README.md` | short project entry point | tracked |
| `document_design.md` | long-form causal reasoning design draft | tracked |
| `docs/` | focused explanatory docs | tracked |
| `scripts/snuh_etl/` | SNUH synthetic ETL and data-quality scripts | tracked |
| `causal/` | causal prototype code owned by this project | tracked |
| `notebooks/` | exploratory analyses | tracked selectively |
| `data/` | local DuckDB files, backups, vocabulary CSVs | mostly ignored |
| `agno_fastomop/` | FastOMOP NL-to-SQL implementation | submodule |
| `omcp/` | OMOP MCP server | submodule |
| `foem/` | FastOMOP evaluation toolkit | submodule |
| `Dockerfile`, `docker-compose.yml` | local integrated runtime | tracked |

## Submodules

The FastOMOP-related folders are not ordinary copied source folders. They are
Git submodules. The root repository tracks a commit pointer for each submodule.

```text
agno_fastomop/  -> FastOMOP agent implementation
omcp/           -> OMOP MCP server
foem/           -> evaluation and monitoring
```

This means:

- edits inside a submodule must be committed inside that submodule;
- the root repo then records the updated submodule pointer;
- upstream updates should be reviewed before merging because local fork changes
  may exist.

Use:

```bash
git submodule status --recursive
git -C agno_fastomop status
git -C omcp status
git -C foem status
```

## Project-Owned Code

Project-owned code should live outside the FastOMOP submodules unless it is a
deliberate contribution to FastOMOP itself.

Use these locations:

```text
scripts/snuh_etl/   SNUH-specific preparation and vocabulary cleanup
causal/             causal cohort construction and future estimators
docs/               project documentation
docs/reproducible_causal_review.md
                    literature and public dataset review work package
notebooks/          exploratory analysis
```

This keeps the thesis/project work separate from upstream FastOMOP code.

## Data and Generated Files

The following should not be committed:

- DuckDB files;
- Athena vocabulary CSVs;
- generated backups;
- virtual environments;
- `.env` files;
- Python cache folders;
- generated HTML exports unless explicitly needed.

The `data/README.md` file is tracked so the expected local layout is visible on
GitHub without committing any data.

## Recommended Cleanup Rules

Keep:

- `causal/`;
- `docs/`;
- `document_design.md`;
- `scripts/snuh_etl/`;
- selected notebooks that support the research story;
- FastOMOP submodule pointers.

Do not keep:

- local virtual environments such as `fhir/`;
- empty cache-only folders;
- local database backups in Git;
- copied upstream repositories outside submodules.

## Naming Conventions

Use folder names to mark ownership:

- `snuh_etl` means dataset-specific preparation;
- `causal` means reusable causal workflow code;
- `docs` means human-facing project explanation;
- submodule folders keep their upstream names.
