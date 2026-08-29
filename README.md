# CausalOMOP

CausalOMOP is a research prototype for designing and auditing causal studies
over observational health data represented in OMOP CDM. It turns a clinical
question into reviewable artifacts: a target-trial draft, an evidence-grounded
causal DAG, an identification assessment, an OMOP variable registry, governed
SQL, feasibility diagnostics, and an execution-readiness decision.

The repository can now estimate a prespecified, transparent effect,
risk-difference scale, via three reviewed estimators: g-computation/
standardization or TMLE for the population ATE, or a causal forest for a
prespecified-subgroup CATE (see [causal/README.md](causal/README.md)'s
Current Capabilities), plus E-value sensitivity analysis and negative
controls as diagnostics. No clinical conclusion should be inferred from its
current outputs (see [docs/causal_roadmap.md](docs/causal_roadmap.md)).

## Documentation

Start with the [documentation index](docs/index.md). The authoritative pages
are:

- [causal package and implementation status](causal/README.md);
- [causal roadmap and remaining scientific gaps](docs/causal_roadmap.md);
- [datasets and data-handling policy](docs/datasets.md);
- [vocabulary mapping coverage, a source-data limitation](docs/vocabulary_mapping_coverage.md);
- [FastOMOP/OMCP integration and trust boundaries](docs/fastomop_integration.md);
- [LLM, literature, knowledge-graph, and critique workflow](docs/llm_hypothesis_integration.md);
- [current architecture diagrams](docs/project_architecture_diagrams.md);
- [responsible demonstration guide](docs/responsible_demo_guide.md).

## What is implemented

| Area | Current status |
|---|---|
| Natural-language question intake | LLM-based causal/descriptive/clarification gate in the CLI and in the Streamlit UI's conversation panel (multi-round clarification, persisted per run as `conversation.jsonl`); descriptive questions are routed to FastOMOP automatically (`causal.fastomop_bridge`) and answered inline |
| Target-trial draft | Strictly parsed LLM or manually supplied protocol; human approval remains mandatory |
| Evidence grounding | PubMed and OpenAlex retrieval, optional OptimusKG relations, deterministic citation allow-list |
| DAG review | Role/timing validation, cycle checks, bounded critique/revision, optional multi-candidate tournament, backdoor identification, minimum adjustment sets, SHD/SID audit metrics |
| OMOP operationalization | Human-reviewed concept mapping, versioned variable registry, mapping diagnostics, windows and aggregation rules |
| Data access | BigQuery or local DuckDB, with governed OMCP execution as the default causal path |
| Study construction | Aggregate feasibility and follow-up audits, plus governed person-level analytic-cohort SQL generation and materialization into a local, non-versioned dataset |
| Estimation and final causal report | G-computation or TMLE ATE estimator, or a causal-forest CATE estimator for a prespecified effect-modifier subgroup family (reviewer's choice per run), SMD covariate-balance and calibration diagnostics, E-value quantitative sensitivity analysis, negative controls (LLM-proposed, human-approved), and an auditable `causal_report.md` gated by human `results` review, with an optional advisory-only LLM narrative (CATE subgroup results are reported as a deterministic table only, no LLM narrative yet) |

The workflow now has separate `target_trial`, `dag_review`, and
`identification_estimand` gates (see [docs/causal_roadmap.md](docs/causal_roadmap.md)'s
governance boundary) instead of one combined `protocol` gate. Approving
`identification_estimand` builds and persists the full `CausalStudySpecification`
contract (`causal/roadmap.py`) as `study_specification.json` -- the persistent
workflow's single validated source for the question, target trial, estimand,
and DAG together.

## References

CausalOMOP is a general-purpose causal-inference tool: the pipeline, agents,
and SQL builders are protocol-driven and are not specific to any one disease,
drug, or outcome (see [Repository layout](#repository-layout) and
[`causal/queries/feasibility.py`](causal/queries/feasibility.py), whose
concept-set-driven builders resolve population/treatment/comparator/outcome
from whatever protocol they are given). Its design choices are grounded in
established causal-inference and biomedical-informatics literature rather
than invented from scratch:

- **Causal DAGs and the backdoor criterion** -- Pearl, J. (1995). Causal
  diagrams for empirical research. *Biometrika*, 82(4), 669-688.
  [doi:10.1093/biomet/82.4.669](https://doi.org/10.1093/biomet/82.4.669).
  Implemented by the d-separation and minimum-adjustment-set logic in
  `causal/roadmap.py`.
- **Target trial emulation** -- Hernán, M.A. & Robins, J.M. (2016). Using Big
  Data to Emulate a Target Trial When a Randomized Trial Is Not Available.
  *American Journal of Epidemiology*, 183(8), 758-764.
  [doi:10.1093/aje/kwv254](https://doi.org/10.1093/aje/kwv254). Every study
  protocol (`protocol.json`) is a target-trial specification in this sense,
  not a bespoke, hand-tuned query.
- **The Causal Roadmap** -- Petersen, M.L. & van der Laan, M.J. (2014). Causal
  Models and Learning from Data: Integrating Causal Modeling and Statistical
  Estimation. *Epidemiology*, 25(3), 418-426. The 11-step sequence in
  [docs/causal_roadmap.md](docs/causal_roadmap.md) follows this roadmap's
  separation of causal model, causal question, identification, statistical
  estimand, and estimation.
- **SHD/SID for DAG structural comparison** -- Peters, J. & Bühlmann, P.
  (2015). Structural Intervention Distance (SID) for Evaluating Causal
  Graphs. *Neural Computation*, 27(3), 771-799.
  [arXiv:1306.1043](https://arxiv.org/abs/1306.1043) (SHD cited therein to
  Acid & de Campos 2003 and Tsamardinos et al. 2006). Implemented in
  [`causal/agents/dag_metrics.py`](causal/agents/dag_metrics.py) and used
  both for tournament deduplication/merge and for comparing a generated DAG
  against a study's reference DAG.
- **G-computation / standardization** -- Hernan, M.A. & Robins, J.M. (2020).
  *Causal Inference: What If*. Boca Raton: Chapman & Hall/CRC. Chapter 13
  (Standardization and the parametric g-formula). Implemented as the ATE
  baseline in [`causal/agents/estimation.py`](causal/agents/estimation.py)
  (`EstimationAgent`): an unpenalized logistic outcome-regression model,
  standardized under forced treatment/comparator, with a nonparametric
  bootstrap confidence interval.
- **SMD covariate-balance diagnostics** -- Austin, P.C. (2009). Balance
  diagnostics for comparing the distribution of baseline covariates between
  treatment groups in propensity-score matched samples. *Statistics in
  Medicine*, 28(25), 3083-3107.
  [doi:10.1002/sim.3697](https://doi.org/10.1002/sim.3697). Implemented in
  [`causal/agents/diagnostics.py`](causal/agents/diagnostics.py)
  (`DiagnosticsAgent`), `|SMD| > 0.1` as the standard flag threshold.
- **TMLE** -- Van der Laan, M.J. & Rose, S. (2011). *Targeted Learning:
  Causal Inference for Observational and Experimental Data*. New York:
  Springer. Doubly robust sibling estimator to the g-computation baseline
  above, for the identical ATE, risk-difference-scale functional. Implemented
  in [`causal/agents/tmle.py`](causal/agents/tmle.py) (`TMLEAgent`):
  `causal.nuisance.SuperLearner`-ensembled outcome/treatment nuisance models,
  a targeting step, and an efficient-influence-curve closed-form variance
  estimator.
- **CATE (conditional average treatment effect)** -- Athey, S. & Imbens, G.
  (2016). Recursive partitioning for heterogeneous causal effects. *PNAS*,
  113(27), 7353-7360. [doi:10.1073/pnas.1510489113](https://doi.org/10.1073/pnas.1510489113);
  Wager, S. & Athey, S. (2018). Estimation and Inference of Heterogeneous
  Treatment Effects using Random Forests. *JASA*, 113(523), 1228-1242.
  [doi:10.1080/01621459.2017.1319839](https://doi.org/10.1080/01621459.2017.1319839).
  The primary estimand target (see [causal/README.md](causal/README.md)'s
  Planned Direction). Implemented in
  [`causal/agents/cate.py`](causal/agents/cate.py) (`CATEAgent`, a causal
  forest via `econml.dml.CausalForestDML`) for a prespecified family of
  effect-modifier subgroups only -- an unprespecified search over every
  covariate for "the subgroup with the biggest effect" is the textbook way
  to manufacture a false positive. Multiple-testing correction across the
  prespecified family via Holm's (1979) sequentially rejective stepdown
  procedure (Scand J Stat, 6(2), 65-70), motivated by List, Shaikh & Xu
  (2019). Multiple hypothesis testing in experimental economics. *Exp
  Econ*, 22(4), 773-793. [doi:10.1007/s10683-018-9597-5](https://doi.org/10.1007/s10683-018-9597-5).
- **E-value sensitivity analysis** -- VanderWeele, T.J. & Ding, P. (2017).
  Sensitivity Analysis in Observational Research: Introducing the E-Value.
  *Annals of Internal Medicine*, 167(4), 268-274.
  [doi:10.7326/M16-2607](https://doi.org/10.7326/M16-2607). The minimum
  risk-ratio-scale strength an unmeasured confounder would need to fully
  explain away the observed association. Implemented in
  [`causal/sensitivity.py`](causal/sensitivity.py) (`e_value`); both
  implemented estimators report a risk ratio alongside their primary
  risk-difference estimate for this.
- **Grounding an LLM causal co-pilot in verifiable evidence** -- Alaa, A. et
  al. (2024). [arXiv:2407.19118](https://arxiv.org/abs/2407.19118). Shaped
  the Hypothesis Agent's evidence-grounded drafting and citation-allow-list
  design (`causal/agents/hypothesis.py`).
- **OptimusKG** -- Vittor et al. OptimusKG: Unifying biomedical knowledge in
  a modern multimodal graph. [arXiv:2604.27269](https://arxiv.org/pdf/2604.27269)
  (Zitnik Lab, Harvard Medical School;
  [github.com/mims-harvard/OptimusKG](https://github.com/mims-harvard/OptimusKG)).
  Optional knowledge-graph evidence source corroborating DAG edges alongside
  PubMed/OpenAlex (`causal/knowledge_graph/`).
- **CohortDiagnostics-style mapping and index-event checks** -- Rao GA,
  Shoaibi A, Makadia R, et al. (2025). CohortDiagnostics: Phenotype
  evaluation across a network of observational data sources using
  population-level characterization. *PLOS ONE*, 20(1), e0310634.
  [doi:10.1371/journal.pone.0310634](https://doi.org/10.1371/journal.pone.0310634).
  Adapted in `causal/mapping/audit.py` for orphan-concept, index-event, and
  incidence-rate diagnostics, and in
  [docs/vocabulary_mapping_coverage.md](docs/vocabulary_mapping_coverage.md)
  for the source-vocabulary coverage check.

### Validation strategy: comparison, not scope

CausalOMOP does not restrict what it can study, and does not search for a
matching published study at runtime. Rigor instead comes from picking a
small number of causal questions that already have a published,
literature-grounded analysis with a known expert DAG or concept mapping, and
running the actual pipeline against each one so its output can be compared
against that published ground truth (SHD/SID for DAG structure, recall@k/MRR
for concept mapping -- see [docs/causal_roadmap.md](docs/causal_roadmap.md)).
These studies are worked examples used to evaluate and demonstrate pipeline
effectiveness, not a constraint on what questions the pipeline supports:

- **Worked validation case** --
  [studies/albumin_sepsis_28d](studies/albumin_sepsis_28d/README.md)
  reproduces the albumin-plus-crystalloids-vs-crystalloids, ICU sepsis,
  28-day-mortality question and its MIMIC-IV OMOP grounding from *Step-by-
  step causal analysis of EHRs to ground decision-making*,
  [doi:10.1371/journal.pdig.0000721](https://doi.org/10.1371/journal.pdig.0000721).
- **Second candidate comparison case** -- Wang, X., Zhang, K., Wu, Y., Zhang,
  Z., Li, Y., Han, R., & Liang, F. (2026). Target trial emulation of early
  acetaminophen use and mortality in critically ill stroke patients.
  *Scientific Reports*.
  [doi:10.1038/s41598-026-46381-5](https://doi.org/10.1038/s41598-026-46381-5).
  Also a target-trial emulation on MIMIC-IV/eICU ICU data; identified as a
  second comparison case for the causal-question and identification process,
  not yet built into a `studies/` package.

## Repository layout

```text
CausalOMOP/
  causal/                         project-owned causal workflow
  docs/                           maintained documentation and audit notes
  studies/albumin_sepsis_28d/     first literature-comparison validation example
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

The integration repositories are pinned as Git submodules. Each `url` in
`.gitmodules` is a `ji-ch01` fork that carries the pinned commit and keeps
the official repository as its `upstream` remote; the official projects'
own READMEs remain authoritative for installation and upstream behavior.
See the [FastOMOP organization](https://github.com/fastomop) for the
official upstream projects:

| Local path | Pinned fork (`.gitmodules`) | Official upstream |
|---|---|---|
| `agno_fastomop/` | `ji-ch01/agno_fastomop` | [fastomop/agno_fastomop](https://github.com/fastomop/agno_fastomop) |
| `omcp/` | `ji-ch01/omcp` | [fastomop/omcp](https://github.com/fastomop/omcp) |
| `foem/` | `ji-ch01/foem` | [fastomop/foem](https://github.com/fastomop/foem) |
| `external/ohdsi_mimic/` | `ji-ch01/MIMIC` | [OHDSI/MIMIC](https://github.com/OHDSI/MIMIC) |

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
pip install -r requirements-estimation.txt   # g-computation, TMLE, CATE, SuperLearner
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

This path first classifies the question. A descriptive question is answered
inline through FastOMOP instead of drafting a causal study; an ambiguous
question returns a clarification request.
`--skip-question-gate` exists for controlled testing, not as a scientific
approval mechanism.

Useful optional flags are `--skip-literature`, `--skip-knowledge-graph`,
`--skip-critique`, `--max-critique-rounds`, and `--tournament-size`. Every raw
LLM response is persisted in `artifacts/runs/<run_id>/llm_outputs.json`; do not
put patient-level information in prompts.

## Review, mapping, and feasibility

The CLI exposes explicit review actions. A typical exploratory sequence is:

```powershell
python -m causal.cli workflow gate $runId target_trial approved --reviewer "reviewer-id"
python -m causal.cli workflow advance $runId
python -m causal.cli workflow hypothesis $runId
python -m causal.cli workflow gate $runId dag_review approved --reviewer "reviewer-id"
python -m causal.cli workflow advance $runId
python -m causal.cli workflow gate $runId identification_estimand approved --reviewer "reviewer-id"
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
python -m causal.cli workflow positivity-audit $runId
python -m causal.cli workflow covariate-missingness-audit $runId
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
scientific work is to validate study phenotypes and covariates against the
real dataset, define the observed data estimand and missing-data/censoring
strategy, and implement prespecified estimators with diagnostics and
sensitivity analyses -- see [docs/causal_roadmap.md](docs/causal_roadmap.md)'s
remaining scientific milestones.
