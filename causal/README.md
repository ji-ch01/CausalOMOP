# Causal Package

Governed, protocol-driven causal-inference workflow over any longitudinal
OMOP CDM dataset. Nothing in this package hardcodes a disease, drug, or
outcome; every population/treatment/comparator/outcome/covariate is resolved
from the reviewed protocol at run time (`causal/queries/feasibility.py`,
`causal/data_builder/analytic.py`). The current validation environment is a
credentialed MIMIC-IV conversion to OMOP CDM 5.3.1 (see `docs/datasets.md`),
used to compare pipeline output against published, literature-grounded case
studies -- see `README.md`'s References section -- not because the pipeline
is scoped to MIMIC-IV or to those studies' clinical questions. The early
SNUH-era prototype (`data_builder/cohort.py`, string-matching cohort
definitions) has been removed; it was superseded by the pipeline below and
was never on any live code path.

## Architecture

```text
protocol.json (causal question, target-trial protocol, first-pass covariates)
  -> causal.agents.hypothesis.HypothesisAgent      (DAG draft, identification)
  -> human review/approval gates                    (causal.orchestrator)
  -> causal.mapping.registry / causal.mapping.audit  (OMOP concept mapping + diagnostics)
  -> causal.agents.data_science.DataScienceAgent     (analysis_plan.json)
  -> causal.agents.execution.ExecutionAgent          (execution_readiness.json)
  -> causal.queries.feasibility / causal.data_builder.analytic
       (feasibility SQL, person-level analytic cohort SQL)
  -> causal.mcp_client.OMCPClient -> omcp            (governed query execution)
```

`causal.orchestrator.workflow.WorkflowManager` drives all of the above
through an explicit phase/approval-gate state machine
(`causal.orchestrator.state`) persisted per run under `artifacts/runs/`.
`causal.cli` and `causal.ui.streamlit_app` are the two entry points that
call it.

## Current Capabilities

- literature-grounded causal DAG drafting and backdoor identification;
- a validated `CausalStudySpecification` (question, target trial, estimand,
  DAG) built and persisted on `identification_estimand` gate approval;
- human-in-the-loop concept mapping and analysis-plan review, with gates
  that cannot be approved while the plan is incomplete and that reset once
  approved content is later changed;
- aggregate-only feasibility counts, mapping-audit diagnostics, univariate
  positivity screening, and per-covariate missingness audits;
- person-level analytic cohort SQL generation (`A`, `Y`, `index_datetime`,
  covariates, and an optional reviewer-approved `censored` column) for OMOP
  CDM, portable across BigQuery and DuckDB;
- governed, read-only query execution via OMCP, with SQL-safety validation
  before anything reaches a database.

## Current Limitations

- no TMLE, SuperLearner-style nuisance modeling, g-computation, or CATE
  estimator is implemented yet -- this package stops at cohort construction
  and readiness gating, deliberately, before causal estimation.
- source OMOP vocabulary mapping completeness is checked and enforced per
  concept set before a cohort query runs, but not guaranteed by this
  package -- an unmapped concept blocks with an explicit error rather than
  being silently undercounted; see
  [docs/vocabulary_mapping_coverage.md](../docs/vocabulary_mapping_coverage.md).

## Planned Direction

The primary methodological target is CATE inference with OMOP CDM data.
Planned estimators include TMLE with SuperLearner-style nuisance models and
g-computation. Average treatment-effect summaries may still be reported as
secondary outputs for comparison with published studies.

## Entry points

```powershell
python -m causal.cli --help
python -m causal.cli workflow --help
streamlit run causal/ui/streamlit_app.py
```

The natural-language entry point includes a causal-question classifier before
target-trial drafting. It refuses descriptive questions with a FastOMOP
redirect and requests clarification when the intervention or outcome is too
ambiguous. The gate is LLM-based and may be skipped explicitly for controlled
testing; it is not a replacement for human protocol review.

## Important implementation boundary

`causal.roadmap.CausalStudySpecification` defines the question, target-trial,
estimand, and DAG contracts. `causal.agents.dag_identification.
build_study_specification` wires them into the persistent orchestrator: the
`identification_estimand` gate cannot be approved unless it constructs
without error, and the result is persisted as `study_specification.json`.
The scientific review decisions are split across three gates
(`target_trial`, `dag_review`, `identification_estimand`) instead of one
combined `protocol` gate. `causal.mapping.registry.OMOPVariableRegistry` is
still not the sole source every query builder resolves population/treatment/
comparator/outcome concepts from -- only covariates are registry-backed
today.

For maintained documentation, see `docs/index.md`,
`docs/causal_roadmap.md`, and `docs/fastomop_integration.md`.
