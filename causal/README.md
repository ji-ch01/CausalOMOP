# Causal Package

Governed causal-inference workflow over MIMIC-IV OMOP CDM data (see
`docs/datasets.md`). The early SNUH-era prototype (`data_builder/cohort.py`,
string-matching cohort definitions) has been removed; it was superseded by
the pipeline below and was never on any live code path.

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
- human-in-the-loop concept mapping and analysis-plan review, with gates
  that cannot be approved while the plan is incomplete and that reset once
  approved content is later changed;
- aggregate-only feasibility counts and mapping-audit diagnostics;
- person-level analytic cohort SQL generation (`A`, `Y`, `index_datetime`,
  covariates) for OMOP CDM, portable across BigQuery and DuckDB;
- governed, read-only query execution via OMCP, with SQL-safety validation
  before anything reaches a database.

## Current Limitations

- no TMLE, SuperLearner-style nuisance modeling, g-computation, or CATE
  estimator is implemented yet -- this package stops at cohort construction
  and readiness gating, deliberately, before causal estimation.

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

`causal.roadmap.CausalStudySpecification` defines the intended question,
target-trial, estimand, and DAG contracts. The general DAG and backdoor checks
are wired into the workflow, but the protocol and estimand contracts are not
yet the persistent orchestrator's single validated state. The current
`protocol` gate therefore combines several scientific review decisions that
should eventually be separated.

For maintained documentation, see `docs/index.md`,
`docs/causal_roadmap.md`, and `docs/fastomop_integration.md`. The old
`document_design.md` and `docs/causal_architecture.md` files are historical
records only.
