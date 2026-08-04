# CausalOMOP

CausalOMOP is a research prototype for designing and auditing causal studies
over observational health data represented in OMOP CDM. It turns a clinical
question into reviewable artifacts: a target-trial draft, an evidence-grounded
causal DAG, an identification assessment, an OMOP variable registry, governed
SQL, feasibility diagnostics, and an execution-readiness decision.

The repository does **not** yet estimate treatment effects. No ATE, ATT, CATE,
g-computation, TMLE, or clinical conclusion should be inferred from its current
outputs.

## Documentation

Start with the [documentation index](docs/index.md). The authoritative pages
are:

- [causal package and implementation status](causal/README.md);
- [causal roadmap and remaining scientific gaps](docs/causal_roadmap.md);
- [datasets and data-handling policy](docs/datasets.md);
- [FastOMOP/OMCP integration and trust boundaries](docs/fastomop_integration.md);
- [LLM, literature, knowledge-graph, and critique workflow](docs/llm_hypothesis_integration.md);
- [current architecture diagrams](docs/project_architecture_diagrams.md);
- [responsible demonstration guide](docs/responsible_demo_guide.md).

`document_design.md`, `docs/causal_architecture.md`, and
`docs/reproducible_causal_review.md` are retained as short historical records;
they are not current specifications.

## What is implemented

| Area | Current status |
|---|---|
| Natural-language question intake | LLM-based causal/descriptive/clarification gate in the CLI and Streamlit UI; descriptive questions are redirected to FastOMOP, but automatic cross-application routing is not implemented |
| Target-trial draft | Strictly parsed LLM or manually supplied protocol; human approval remains mandatory |
| Evidence grounding | PubMed and OpenAlex retrieval, optional OptimusKG relations, deterministic citation allow-list |
| DAG review | Role/timing validation, cycle checks, bounded critique/revision, optional multi-candidate tournament, backdoor identification, minimum adjustment sets, SHD/SID audit metrics |
| OMOP operationalization | Human-reviewed concept mapping, versioned variable registry, mapping diagnostics, windows and aggregation rules |
| Data access | BigQuery or local DuckDB, with governed OMCP execution as the default causal path |
| Study construction | Aggregate feasibility and follow-up audits, plus governed person-level analytic-cohort SQL generation |
| Estimation and final causal report | Not implemented |

The current workflow still uses one combined `protocol` review gate instead of
separate target-trial, DAG, and identification gates. The general
`CausalStudySpecification` contracts exist in `causal/roadmap.py`, but the
protocol and estimand contracts are not yet the persistent workflow's single
source of truth.

## Repository layout

```text
CausalOMOP/
  causal/                         project-owned causal workflow
  docs/                           maintained documentation and audit notes
  studies/albumin_sepsis_28d/     first literature-backed validation case
  scripts/bigquery_etl/           BigQuery-to-DuckDB export utility
  scripts/snuh_etl/               SYN-SNUH preparation and validation
  notebooks/                      exploratory analyses only
  data/                           local, ignored datasets and vocabularies
  artifacts/runs/                 generated run artifacts (may be sensitive)
  agno_fastomop/                  FastOMOP Git submodule
  omcp/                           OMCP Git submodule
  foem/                           FOEM Git submodule
  external/ohdsi_mimic/           OHDSI MIMIC ETL Git submodule
```

See [docs/repo_map.md](docs/repo_map.md) for ownership and Git rules.

## External projects

The integration repositories are pinned as Git submodules. Their own READMEs
remain authoritative for installation and upstream behavior:

| Local path | Official repository |
|---|---|
| `agno_fastomop/` | [ji-ch01/agno_fastomop](https://github.com/ji-ch01/agno_fastomop) |
| `omcp/` | [ji-ch01/omcp](https://github.com/ji-ch01/omcp) |
| `foem/` | [ji-ch01/foem](https://github.com/ji-ch01/foem) |
| `external/ohdsi_mimic/` | [OHDSI/MIMIC](https://github.com/OHDSI/MIMIC) |

Initialize them with:

```bash
git submodule update --init --recursive
```

## Installation

Python 3.12 is used by CI. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install pytest
```

Install only the optional groups needed for the task:

```powershell
pip install -r requirements-bigquery.txt
pip install -r requirements-mcp.txt
pip install -r requirements-llm.txt
pip install -r requirements-ui.txt
pip install -r requirements-knowledge-graph.txt
pip install -r requirements-causal-discovery.txt
```

Copy `sample.env` to `.env` and replace placeholders locally:

```powershell
Copy-Item sample.env .env
```

Never commit `.env`, credentials, local databases, vocabulary exports, or
patient-level artifacts.

## Configuration and tests

The deterministic suite does not require a live cloud service:

```powershell
python -m pytest tests -q
python -m causal.cli config --json
```

With BigQuery dependencies and Application Default Credentials configured:

```powershell
python -m causal.cli preflight --count-persons
```

Dataset roles are configured through `CAUSALOMOP_BQ_PROJECT` and the four
`CAUSALOMOP_BQ_*_DATASET` variables documented in
[docs/datasets.md](docs/datasets.md).

## Start a causal workflow

From a reviewed protocol:

```powershell
$state = python -m causal.cli workflow start studies\albumin_sepsis_28d\protocol.json |
  ConvertFrom-Json
$runId = $state.run_id

python -m causal.cli workflow hypothesis $runId
python -m causal.cli workflow status $runId
```

From a natural-language causal question:

```powershell
python -m causal.cli workflow start-question "Among adult ICU patients with sepsis, what is the effect of albumin plus crystalloids versus crystalloids alone on 28-day mortality?"
```

This path first classifies the question. A descriptive question is refused with
a pointer to FastOMOP; an ambiguous question returns a clarification request.
`--skip-question-gate` exists for controlled testing, not as a scientific
approval mechanism.

Useful optional flags are `--skip-literature`, `--skip-knowledge-graph`,
`--skip-critique`, `--max-critique-rounds`, and `--tournament-size`. Every raw
LLM response is persisted in `artifacts/runs/<run_id>/llm_outputs.json`; do not
put patient-level information in prompts.

## Review, mapping, and feasibility

The CLI exposes explicit review actions. A typical exploratory sequence is:

```powershell
python -m causal.cli workflow gate $runId protocol approved --reviewer "reviewer-id"
python -m causal.cli workflow advance $runId
python -m causal.cli workflow plan $runId --with-mcp

# Review protocol concept sets and DAG covariates with the dedicated
# concept-set-map / covariate-* commands or the Streamlit interface.

python -m causal.cli workflow registry-build $runId
python -m causal.cli workflow gate $runId clinical_mapping provisional `
  --reviewer "reviewer-id" --comment "Aggregate exploration only"
python -m causal.cli workflow advance $runId
python -m causal.cli workflow feasibility $runId --transport mcp
python -m causal.cli workflow execution-readiness $runId
python -m causal.cli workflow cohort-plan $runId
python -m causal.cli workflow cohort-profile $runId
python -m causal.cli workflow followup-audit $runId
python -m causal.cli workflow mapping-audit $runId <variable_id>
python -m causal.cli workflow demo-report $runId
```

Use `python -m causal.cli workflow --help` for the full command list. The
direct BigQuery feasibility transport remains available for equivalence checks;
OMCP is the default governed path.

The review interface can be started with:

```powershell
streamlit run causal/ui/streamlit_app.py
```

## Data

The primary research dataset is a credentialed MIMIC-IV 2.2 conversion to OMOP
CDM 5.3.1. Neither source data nor converted patient-level tables are included.
Access is governed by [PhysioNet's MIMIC-IV terms](https://physionet.org/content/mimiciv/2.2/),
and conversion follows [OHDSI/MIMIC](https://github.com/OHDSI/MIMIC).

SYN-SNUH remains a local development dataset for ETL, vocabulary, FastOMOP, and
smoke testing. It is not the primary source of causal evidence.

For lower-latency local development, export the configured BigQuery OMOP
dataset to an ignored DuckDB file:

```powershell
python -m scripts.bigquery_etl.export_duckdb data/mimiciv_omop.duckdb
```

The export inherits the source dataset's access restrictions. See
[docs/datasets.md](docs/datasets.md) before moving or sharing it.

## Scientific boundary

The current outputs support architecture review, protocol review, concept
mapping, DAG inspection, identification checks, feasibility, and diagnostics.
They do not support a claim that a treatment causes an outcome. The next major
scientific work is to wire the complete causal-study contracts into the
persistent gates, validate study phenotypes and covariates, define the observed
data estimand, and implement prespecified estimators with diagnostics and
sensitivity analyses.
