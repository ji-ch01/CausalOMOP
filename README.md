# K-OMOP-CDM

Research repository for adapting FastOMOP to Korean OMOP CDM data and
prototyping a causal reasoning extension for OMOP-based real-world evidence.

The repository has two complementary roles:

- keep a working FastOMOP NL-to-SQL stack around OMOP data;
- develop a causal workflow that can turn clinical causal questions into
  auditable cohort definitions, covariates, estimands, and reports.

The SNUH synthetic dataset is kept in the project because it was useful for
testing the ETL and FastOMOP NL-to-SQL stack. It is no longer treated as the
primary dataset for causal inference.

## Current Scope

The project is centered on this causal research question:

> Does dapagliflozin, compared with sitagliptin, increase the 90-day risk of
> urinary or genital infection among patients with Type 2 Diabetes Mellitus?

The intended design is an active-comparator, new-user target trial emulation.
The current implemented causal code is still an early prototype: it can build
basic treatment/comparator cohorts and feasibility summaries, but it does not
yet run the full causal agent workflow or final effect estimation.

## Repository Map

```text
K-OMOP-CDM/
  README.md
  document_design.md              # long causal extension design draft

  docs/
    repo_map.md                   # what each folder is for
    datasets.md                   # local data policy and SNUH status
    fastomop_integration.md       # FastOMOP submodules and runtime notes
    causal_architecture.md        # proposed causal/NL2SQL architecture

  scripts/
    snuh_etl/                     # SNUH synthetic ETL and data-quality scripts

  causal/
    data_builder/cohort.py        # current causal cohort prototype

  notebooks/                      # exploratory analyses

  agno_fastomop/                  # FastOMOP NL-to-SQL agent submodule
  omcp/                           # OMOP MCP server submodule
  foem/                           # FastOMOP evaluation submodule

  data/                           # local DuckDB/vocabulary files, not committed
```

See [docs/repo_map.md](docs/repo_map.md) for the detailed ownership and Git
policy for each folder.

## FastOMOP Components

This repo currently keeps three FastOMOP-related dependencies as submodules:

| Path | Role |
|---|---|
| `agno_fastomop/` | NL-to-SQL agent workflow using semantic and database agents |
| `omcp/` | MCP server exposing OMOP database tools |
| `foem/` | evaluation and monitoring queries for FastOMOP |

For this project, FastOMOP is used first as the descriptive OMOP query layer.
The causal extension should sit close to the NL-to-SQL team because causal
questions still need OMOP concept mapping and SQL cohort extraction.

See [docs/fastomop_integration.md](docs/fastomop_integration.md).

## Data Policy

Local data files are intentionally excluded from Git:

- `data/*.duckdb`
- `data/backups/`
- `data/vocabulary/`
- `.env` files and local virtual environments

Expected local files include:

```text
data/synthetic_snuh_raw.duckdb
data/synthetic_snuh_curated.duckdb
data/vocabulary/CONCEPT.csv
data/vocabulary/CONCEPT_RELATIONSHIP.csv
data/vocabulary/CONCEPT_ANCESTOR.csv
```

The SNUH synthetic DuckDB is useful for:

- smoke-testing OMOP ETL;
- validating concept coverage;
- testing FastOMOP NL-to-SQL;
- prototyping cohort-builder mechanics.

It should not be presented as the main causal inference dataset. See
[docs/datasets.md](docs/datasets.md).

## Quick Start

Initialize submodules:

```bash
git submodule update --init --recursive
```

Install the lightweight root dependencies:

```bash
pip install -r requirements.txt
```

Run the SNUH ETL/data-quality scripts from the repository root. Use module
execution so imports resolve correctly:

```bash
python -m scripts.snuh_etl.load_vocabulary
python -m scripts.snuh_etl.fix_drugs
python -m scripts.snuh_etl.fix_measurements
python -m scripts.snuh_etl.remap_observations
python -m scripts.snuh_etl.extend_concept_ancestor
python -m scripts.snuh_etl.validate_vocabulary
```

The scripts default to:

```text
OMOP_DB_PATH=data/synthetic_snuh_curated.duckdb
OMOP_VOCAB_DIR=data/vocabulary
```

Override those environment variables when using another OMOP database or
vocabulary directory.

## Running FastOMOP

Docker support is kept at the root for convenience:

```bash
docker compose up --build
```

Check `docker-compose.yml` before running: the container `DB_PATH` must match
the DuckDB filename mounted under `/app/data`.

For local development, use the submodule instructions:

```bash
cd agno_fastomop
uv sync
uv run python -m agno_fastomop.run_agent
```

The agent stack launches `omcp` through the configured MCP command. See
[docs/fastomop_integration.md](docs/fastomop_integration.md) for details.

## Causal Work

The current implemented causal code lives in:

```text
causal/data_builder/cohort.py
```

It currently supports:

- untreated-control and active-comparator cohorts;
- treatment assignment `A`;
- outcome flag `Y`;
- index date and follow-up window;
- basic feasibility scans across candidate cohorts.

The planned architecture adds:

- causal intent routing inside the NL-to-SQL team;
- target trial specification;
- DAG/evidence support;
- OMOP concept-set mapping;
- covariate extraction;
- propensity score and IPW estimation;
- an auditable causal report.

See [docs/causal_architecture.md](docs/causal_architecture.md) and
[document_design.md](document_design.md).

## Notebooks

The notebooks are exploratory analysis artifacts, not production code:

| Notebook | Purpose |
|---|---|
| `notebooks/omop_analysis.ipynb` | general OMOP/SNUH readiness check |
| `notebooks/t2dm_analysis.ipynb` | Type 2 Diabetes treatment and outcome exploration |
| `notebooks/depressive_disorder_analysis.ipynb` | depressive disorder exploratory analysis |

See [notebooks/README.md](notebooks/README.md).

## Development Notes

- Treat `agno_fastomop/`, `omcp/`, and `foem/` as submodules. Changes inside
  them belong to their own Git history.
- Keep local `.duckdb`, Athena vocabulary CSVs, backups, `.env`, and virtual
  environments out of Git.
- Keep SNUH-specific ETL under `scripts/snuh_etl/`.
- Keep causal reusable code under `causal/`.
- Keep long design discussion in `document_design.md` and short operational
  guidance in `docs/`.

## Current Next Steps

- Implement explicit urinary/genital infection concept sets.
- Update cohort building to use concept sets and seed-plus-descendant logic.
- Add baseline covariate extraction around the index date.
- Add propensity score/IPW estimation and balance diagnostics.
- Wire causal intent routing into the FastOMOP NL-to-SQL workflow.
