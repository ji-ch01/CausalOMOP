# CausalOMOP

Research repository for building an auditable causal-inference workflow over
observational health data represented in OMOP CDM.

The repository has two complementary roles:

- keep a working FastOMOP NL-to-SQL stack around OMOP data;
- transform a clinical causal question into a target trial, counterfactual
  estimand, reviewed causal DAG, identification strategy, OMOP
  operationalization, statistical analysis and qualified report.

The SNUH synthetic dataset is kept because it is useful for testing ETL,
FastOMOP, OMOP concept access, and early workflow mechanics. It is not treated
as the primary validation dataset for causal inference.

## Causal Roadmap

The scientific design comes before database extraction. OMOP determines whether
required causal variables can be operationalized; it must not determine which
variables are allowed to exist in the causal model.

Target workflow:

```text
User question
  -> 1. clarify the causal question
  -> 2. specify the target trial
  -> 3. define the counterfactual causal estimand
  -> 4. construct and review the causal DAG
  -> 5. identify the effect and an admissible adjustment set
  -> 6. map required variables to OMOP concepts, tables and time windows
  -> 7. audit feasibility, missingness, positivity and follow-up
  -> 8. define the observed-data statistical estimand
  -> 9. run prespecified estimators
  -> 10. run diagnostics and sensitivity analyses
  -> 11. produce an auditable, assumption-qualified report
```

This follows the Causal Roadmap and target-trial grounding recommended for
medical causal co-pilots in [Alaa et al. (2024)](https://arxiv.org/abs/2407.19118).
The LLM assists with structured proposals and critiques; deterministic code and
human gates decide whether artifacts are coherent, identified and approved. In
particular, Alaa et al.'s emphasis on *grounding* an LLM co-pilot in external,
verifiable evidence rather than its own claims motivated the current
Hypothesis Agent design: candidate DAG edges are drafted only from PubMed
abstracts actually retrieved for the question, and any PMID the LLM cites that
was not retrieved is rejected deterministically. See
[docs/llm_hypothesis_integration.md](docs/llm_hypothesis_integration.md) for
the implementation.

The primary methodological objective remains heterogeneous treatment-effect
estimation with OMOP CDM data. Planned estimators include g-computation and TMLE
with SuperLearner-style nuisance models, after identification and data-support
checks pass.

## Implementation Status

| Layer | Status |
|---|---|
| Causal question, target trial and causal estimand contracts | Implemented |
| General causal DAG with roles and temporal ordering | Implemented |
| Cycle detection and backdoor adjustment assessment | Implemented |
| Minimum adjustment-set enumeration for reviewable DAGs | Implemented |
| LLM question-to-protocol prototype, PubMed grounding | Implemented as an initial prototype |
| OMCP vocabulary candidates and aggregate feasibility | Implemented as governed prototype |
| Causal-roadmap workflow gates and LLM integration | Not yet wired |
| General DAG-to-OMOP variable mapping | Not yet implemented |
| Statistical estimand, estimation and sensitivity analysis | Not yet implemented |

The existing filesystem workflow remains available for technical testing, but
its restricted `candidate_dag` is not the final causal pipeline and cannot by
itself establish identification. The normative design is documented in
[docs/causal_roadmap.md](docs/causal_roadmap.md).

The next validation step is to identify a public dataset and a causal question
that has already been studied in the scientific literature, so the pipeline can
later be compared against published assumptions, variables, and expected
results.

## Repository Map

```text
CausalOMOP/
  README.md
  document_design.md              # long causal extension design draft

  docs/
    repo_map.md                   # what each folder is for
    datasets.md                   # local data policy and SNUH status
    fastomop_integration.md       # FastOMOP submodules and runtime notes
    causal_roadmap.md             # normative causal pipeline and integration plan
    causal_architecture.md        # earlier architecture, retained for history
    reproducible_causal_review.md # next literature and dataset review package

  scripts/
    snuh_etl/                     # SNUH synthetic ETL and data-quality scripts

  causal/
    roadmap.py                    # causal contracts and DAG identification
    agents/                       # initial prototype agents
    literature/                   # PubMed retrieval and evidence grounding
    llm/                          # dev/cloud and production/local LLM adapters
    orchestrator/                 # persistent governed workflow prototype
    data_builder/                 # analytic-cohort contracts and SQL builders
    prompts/                      # hypothesis and literature-DAG prompt templates

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
The causal extension should reuse this NL-to-SQL / SQL execution layer as its
data-access layer. Causal planning and estimation are separate responsibilities,
but they should not bypass OMOP concept mapping, controlled SQL generation, or
schema-aware extraction.

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

MIMIC-IV v2.2 is a credentialed-access dataset available through
https://physionet.org/content/mimiciv/2.2/. Its source data and converted OMOP
tables must remain in an access-controlled environment and are not distributed
with this repository.

## Installation and Tests

Create or activate the project virtual environment, install dependencies and
run the deterministic suite without contacting a cloud service:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m causal.cli config --json
```

LLM, MCP and BigQuery dependencies remain separated:

```powershell
pip install -r requirements-llm.txt
pip install -r requirements-mcp.txt
pip install -r requirements-bigquery.txt
```

### Local configuration (`.env`)

Copy the template once and fill in only what the workflow you're running needs;
`.env` is git-ignored and must never be committed:

```powershell
Copy-Item sample.env .env
```

| Group | Keys | Notes |
|---|---|---|
| BigQuery datasets | `CAUSALOMOP_BQ_PROJECT`, `CAUSALOMOP_BQ_OMOP_DATASET`, `CAUSALOMOP_BQ_VOCAB_DATASET`, `CAUSALOMOP_BQ_ETL_DATASET`, `CAUSALOMOP_BQ_METRICS_DATASET` | required for `preflight`, `workflow feasibility`, `cohort-plan`, `cohort-profile`, `followup-audit` |
| LLM provider | `OPENROUTER_API_KEY`, `CAUSALOMOP_LLM_MODEL`, `CAUSALOMOP_ENV`, `LOCAL_LLM_BASE_URL`, `LOCAL_LLM_MODEL`, `LOCAL_LLM_API_KEY` | `CAUSALOMOP_ENV=dev` (default) uses OpenRouter; `prod` requires the local provider and refuses to start if a cloud key is still set. See [docs/llm_hypothesis_integration.md](docs/llm_hypothesis_integration.md) |
| Literature grounding | `NCBI_API_KEY` | optional; raises the PubMed E-utilities rate limit used by the DAG-drafting stage |

## MIMIC-IV OMOP Setup

The MIMIC-IV to OMOP CDM conversion follows the OHDSI repository:
https://github.com/OHDSI/MIMIC.

Development currently uses the study *Step-by-step causal analysis of EHRs to
ground decision-making* (https://doi.org/10.1371/journal.pdig.0000721) as a
reference for the causal question and study specification.

Resolve the production BigQuery dataset configuration without contacting GCP:

```bash
python -m causal.cli config --json
```

Install the BigQuery runtime, configure Application Default Credentials, then
validate access and the minimum OMOP table contract:

```bash
pip install -r requirements-bigquery.txt
python -m causal.cli preflight --count-persons
```

The private project and dataset identifiers are configured locally through
environment variables. They must never be committed. Dataset roles are
documented using generic names in [docs/datasets.md](docs/datasets.md).

The first technical validation case is albumin plus crystalloids versus
crystalloids alone for 28-day mortality in ICU patients with sepsis. Its draft
target-trial protocol and concept-audit instructions live in
[`studies/albumin_sepsis_28d/`](studies/albumin_sepsis_28d/).

### Current technical workflow

Start and inspect the existing persistent exploratory workflow. These commands
test orchestration, governance and OMOP access; they do not yet exercise the
complete causal-roadmap identification pipeline:

```powershell
$state = python -m causal.cli workflow start studies\albumin_sepsis_28d\protocol.json |
    ConvertFrom-Json
$runId = $state.run_id

python -m causal.cli workflow hypothesis $runId
python -m causal.cli workflow gate $runId protocol approved `
    --reviewer "technical-reviewer"
python -m causal.cli workflow advance $runId

python -m causal.cli workflow plan $runId
python -m causal.cli workflow gate $runId clinical_mapping provisional `
    --reviewer "technical-reviewer" `
    --comment "Exploratory feasibility only"
python -m causal.cli workflow advance $runId

python -m causal.cli workflow feasibility $runId --transport mcp
python -m causal.cli workflow advance $runId
python -m causal.cli workflow execution-readiness $runId
python -m causal.cli workflow cohort-plan $runId
python -m causal.cli workflow cohort-profile $runId
python -m causal.cli workflow followup-audit $runId
python -m causal.cli workflow demo-report $runId
```

Use `workflow plan $runId --with-mcp` instead of `workflow plan $runId` to
retrieve review-only vocabulary candidates. This never approves a concept set.

Feasibility uses the shared OMCP server by default. During migration, the old
direct BigQuery path remains available only for equivalence checks:

```powershell
pip install -r requirements-mcp.txt
python -m causal.cli workflow feasibility <run_id> --transport mcp
python -m causal.cli workflow feasibility <run_id> --transport direct
```

Create a governed draft from a natural-language clinical question
through OpenRouter. The default development model is
`openai/gpt-oss-120b`:

```powershell
pip install -r requirements-llm.txt
python -m causal.cli workflow start-question "Among adult ICU patients with sepsis, what is the effect of albumin plus crystalloids versus crystalloids alone on 28-day mortality?"
```

The generated run remains at the protocol-review gate. The LLM cannot approve
the target trial, causal graph, OMOP mappings or causal execution.

Each exact LLM response is displayed in the terminal on stderr. Successful runs
also persist the unmodified responses in:

```text
artifacts/runs/<run_id>/llm_outputs.json
```

Outputs are tagged as `target_trial_draft` and, when PubMed returns literature,
`literature_dag_draft`. Raw output can contain the submitted clinical question;
never send patient-level data to the cloud LLM and protect run artifacts as
sensitive research material.

Use `--skip-literature` to test only the first LLM call:

```powershell
python -m causal.cli workflow start-question "<causal question>" --skip-literature
```

An exploratory run may pass `clinical_mapping` with a `provisional` decision;
a final run requires full approval.

See [docs/responsible_demo_guide.md](docs/responsible_demo_guide.md) for a
step-by-step demonstration script and the boundary between technical findings
and causal results.

The current natural-language prototype is documented in
[docs/llm_hypothesis_integration.md](docs/llm_hypothesis_integration.md). LLM
output is treated as an unapproved draft and must pass the same deterministic
contract and human gates as a manually authored protocol.

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

The causal foundation lives in:

```text
causal/roadmap.py
```

It currently provides:

- structured causal question, target-trial and estimand contracts;
- a general DAG containing observed or unobserved exposure, outcome,
  confounder, mediator, collider, prognostic, effect-modifier and selection
  nodes;
- node temporal ordering, edge provenance and cycle detection;
- deterministic validation of proposed adjustment sets;
- d-separation in the exposure backdoor graph;
- bounded enumeration of minimum-cardinality adjustment sets;
- explicit failure when the proposed effect is not identified through observed
  backdoor adjustment.

The existing cohort builders and initial agents remain useful infrastructure,
but must be integrated behind the causal-roadmap gates. This integration adds:

- causal intent routing inside the NL-to-SQL team;
- separate human review of the question/target trial, DAG and identification;
- evidence-grounded construction of the general DAG;
- a DAG-node-to-OMOP variable registry with concepts, tables, fields, windows,
  aggregation and missingness rules;
- controlled SQL/OMOP extraction only after identification;
- leakage-safe cohort and covariate extraction;
- CATE-oriented causal estimation with TMLE/SuperLearner and g-computation;
- diagnostics for positivity, overlap, balance, model fit and missingness;
- negative controls and quantitative sensitivity analyses where appropriate;
- an auditable causal report.

See [docs/causal_roadmap.md](docs/causal_roadmap.md) for the normative
pipeline and integration boundary. The earlier design discussions remain in
[docs/causal_architecture.md](docs/causal_architecture.md) and
[document_design.md](document_design.md). See
[docs/reproducible_causal_review.md](docs/reproducible_causal_review.md) for the
next literature and dataset review work package.

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

1. Wire `CausalStudySpecification` into the persistent workflow.
2. Add separate gates for question/target-trial review, DAG review and
   identification review.
3. Replace the restricted prototype `candidate_dag` with the general DAG
   contract while retaining PMID and expert-judgment provenance.
4. Implement the OMOP variable registry from approved DAG nodes instead of
   hard-coded phenotype names.
5. Audit whether MIMIC-IV OMOP supports the required baseline variables,
   treatment strategies, outcome, follow-up and positivity assumptions.
6. Define the observed-data statistical estimand and analysis plan.
7. Implement g-computation first, then TMLE and prespecified CATE analyses with
   appropriate diagnostics and sensitivity analyses.
8. Wire descriptive-versus-causal routing into the user-facing orchestrator and
   produce the final auditable report package.
