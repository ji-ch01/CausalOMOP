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
  -> human `statistical_estimand` gate                (roadmap.StatisticalEstimand, statistical_estimand.json)
  -> causal.agents.execution.ExecutionAgent          (execution_readiness.json)
  -> causal.queries.feasibility / causal.data_builder.analytic
       (feasibility SQL, person-level analytic cohort SQL)
  -> causal.mcp_client.OMCPClient -> omcp            (governed query execution)
  -> causal.data_builder.materialize                 (analytic_dataset.parquet, dataset_manifest.json)
  -> causal.agents.estimation.EstimationAgent        (g-computation ATE, estimation.json)
  -> causal.agents.diagnostics.DiagnosticsAgent      (SMD balance, diagnostics.json)
  -> causal.agents.results_narrative (optional, advisory-only LLM narrative + structured claims)
  -> causal.agents.results_faithfulness              (deterministic fact-check, no second LLM)
  -> human review of the narrative                    (WorkflowManager.review_results_narrative)
  -> human `results` gate                            (causal.orchestrator, independent of the narrative)
  -> causal.orchestrator.workflow.build_causal_report (causal_report.md)
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
- a separate, persisted observed-data statistical estimand
  (`causal.roadmap.StatisticalEstimand`: identifying functional, outcome
  model, the adjustment set as actually built into `cohort.sql`, and an
  explicit reviewer-set missing-data/censoring strategy), gated by its own
  `statistical_estimand` review before any patient-level row is
  materialized -- the analytic cohort's per-row `censored` column, unused by
  anything before this gate existed, is now excluded from the estimation
  sample when the reviewed strategy says so. In the Streamlit UI those three
  choices are settled through the estimand-dialogue companion
  (`causal/companion/companion.py`), which proposes
  `set_statistical_estimand_strategy` as a human-confirmed card behind a
  code-level allow-list (`causal/companion/actions.py`); approving the gate
  itself stays manual;
- aggregate-only feasibility counts, mapping-audit diagnostics, univariate
  positivity screening, and per-covariate missingness audits;
- person-level analytic cohort SQL generation (`A`, `Y`, `index_datetime`,
  covariates, and an optional reviewer-approved `censored` column) for OMOP
  CDM, portable across BigQuery and DuckDB;
- governed, read-only query execution via OMCP, with SQL-safety validation
  before anything reaches a database;
- governed materialization of that person-level cohort into a local
  `analytic_dataset.parquet` (`causal.data_builder.materialize`), the one
  place patient-level rows are ever pulled into memory -- never delivered to
  an LLM;
- a transparent g-computation (standardization) baseline for the ATE,
  risk-difference scale, binary outcome, with a nonparametric bootstrap
  confidence interval (`causal.agents.estimation.EstimationAgent`, cited to
  Hernan & Robins *Causal Inference: What If* ch. 13);
- standardized mean difference (SMD) covariate-balance diagnostics (Austin
  2009, `causal.agents.diagnostics.DiagnosticsAgent`) and cross-validated
  outcome-model AUC/Brier calibration;
- a stabilized inverse-probability-of-censoring-weighted (IPCW) g-formula
  for single-interval administrative censoring, cited to Hernan & Robins
  ch. 12, alongside row-exclusion and an explicit waiver as reviewed
  `censoring_strategy` choices;
- SuperLearner nuisance-model ensembling (`causal.nuisance.SuperLearner`,
  cited to van der Laan, Polley & Hubbard 2007): a prespecified library of
  five sklearn-only candidates combined by cross-validated non-negative
  least squares;
- targeted maximum likelihood estimation (TMLE) for the same ATE, risk-
  difference-scale functional (`causal.agents.tmle.TMLEAgent`, cited to Van
  der Laan & Rose 2011): SuperLearner-ensembled outcome and treatment
  nuisance models, a targeting step, and an efficient-influence-curve
  closed-form variance estimator in place of a bootstrap -- doubly robust,
  consistent if either nuisance model is correctly specified rather than
  only both. A reviewer picks `plug_in_standardization` (the g-computation
  baseline) or `targeted_minimum_loss_based_estimation` per run via
  `StatisticalEstimand.estimation_procedure`;
- E-value quantitative sensitivity analysis (`causal.sensitivity.e_value`,
  cited to VanderWeele & Ding 2017): the minimum risk-ratio-scale strength
  an unmeasured confounder would need, for both the point estimate and the
  confidence-interval bound closest to the null, to fully explain away the
  observed association. Both estimators report a risk ratio alongside their
  primary risk-difference estimate for exactly this
  (`WorkflowManager.run_sensitivity_analysis`, `sensitivity_analysis.json`,
  included in `causal_report.md` when present);
- negative controls (`causal.queries.negative_controls`): a reviewed
  exposure or outcome believed to have no true causal effect, substituted
  in for the real one via `causal.queries.feasibility`'s existing cohort-
  building machinery -- no new SQL-generation logic. LLM-assisted candidate
  proposal (clinical label + rationale only, never a concept id) mirrors
  `causal.agents.dag_critique`'s bounded prompt/parse pattern and can never
  auto-approve a candidate; a human maps concepts and approves/rejects in
  one step (`WorkflowManager.map_negative_control`, folded into the
  existing `clinical_mapping` gate). `WorkflowManager.run_negative_control_
  estimation` fits whichever estimator was reviewed for the real study
  against the substituted cohort -- a diagnostic `causal_report.md`
  surfaces, never a blocking gate;
- an auditable causal report (`causal_report.md`) linking the estimate to
  the protocol/estimand contract, diagnostics, assumptions, the full
  approval-gate history, and each artifact's provenance, gated by a human
  `results` decision; with an optional, gracefully-degrading LLM narrative
  section that only ever reads already-computed JSON results, never the
  patient-level dataset, is fed a deterministic `narrative_display` block
  (`causal.agents.narrative_display`: every figure pre-rounded for
  communication, plus events-per-1000 / relative-risk-change / number-
  needed-to-treat restatements) so its prose reads clinically without ever
  rounding a number itself, must return typed/source-cited `claims`
  alongside its prose, is deterministically fact-checked by `causal.agents.
  results_faithfulness` (no second LLM), and is included in the report only
  once a human reviewer approves it with no failed check
  (`WorkflowManager.review_results_narrative`) -- a decision entirely
  separate from the mandatory `results` gate. In the Streamlit UI that
  narrative is normally drafted through the results-interpretation dialogue
  companion (`WorkflowManager.run_results_companion_turn`), the
  results-review counterpart of the estimand dialogue; "Draft in one step"
  is the no-conversation fallback and both paths write the same verified,
  human-approved artifact;
- CATE (conditional average treatment effect) for a prespecified family of
  effect-modifier subgroups, cited to Athey & Imbens 2016 and Wager & Athey
  2018 (`causal.agents.cate.CATEAgent`, a causal forest via `econml.dml.
  CausalForestDML`). Multiple-testing correction across the prespecified
  family via Holm's (1979) stepdown procedure, motivated by List, Shaikh &
  Xu 2019; exploratory (non-prespecified) subgroups are reported separately
  and never family-wise-error controlled. `StatisticalEstimand.
  population_summary` (`"ATE"`/`"ATT"`/`"CATE"`) gates whether a run may
  call `WorkflowManager.run_cate` at all -- CATE is never the default, a
  study must explicitly declare it as the primary estimand. Reported as a
  deterministic markdown table only (`build_causal_report`) -- no LLM
  narrative for subgroup results yet.

Also implemented, but deliberately outside the governed pipeline above --
an offline maintainer tool, not a workflow step any clinical study run
touches:

- an outcome-blind simulation harness (`causal.simulation`, cited to
  Rudolph, Fox & Naimi 2021) validating `EstimationAgent`/`TMLEAgent`'s
  bias, variance calibration, and CI coverage against three closed-form
  data-generating processes with an exactly-known true ATE, including one
  whose outcome mechanism a logistic-in-the-raw-covariates model
  structurally cannot represent (`tests/test_simulation.py`, opt-in via
  `pytest -m slow`).

## Current Limitations

- CATE subgroup results have no LLM narrative interpretation yet
  (deterministic table only) -- `results_faithfulness`'s null-crossing
  logic is built around one scalar estimate, and stretching it to N
  subgroup results without redesigning it would be exactly the kind of
  bolted-on half-extension this project avoids. TMLE also does not support
  `ipcw_administrative_censoring` (`StatisticalEstimand` rejects the pair at
  `statistical_estimand` gate-approval time, and `TMLEAgent.run` raises the
  same `ValueError` if reached): censored-data TMLE needs a materially
  different efficient-influence-function construction (Van der Laan & Rose
  2011, ch. 5), not a reuse of `EstimationAgent`'s stabilized-weight IPCW.
- IPCW censoring weighting (g-computation only) covers a single fixed-
  horizon administrative-censoring indicator, not time-varying censoring;
  multiple imputation for missing covariates is not implemented
  (`docs/causal_roadmap.md` milestone 3).
- source OMOP vocabulary mapping completeness is checked and enforced per
  concept set before a cohort query runs, but not guaranteed by this
  package -- an unmapped concept blocks with an explicit error rather than
  being silently undercounted; see
  [docs/vocabulary_mapping_coverage.md](../docs/vocabulary_mapping_coverage.md).

## Planned Direction

The primary methodological target, CATE inference with OMOP CDM data, is
now implemented (`causal.agents.cate.CATEAgent`) for a prespecified
effect-modifier subgroup family. Average treatment-effect summaries from
either ATE estimator (g-computation or TMLE) are also reported as secondary
outputs for comparison with published studies. An LLM narrative
interpretation of CATE subgroup results remains future work.

## Entry points

```powershell
python -m causal.cli --help
python -m causal.cli workflow --help
streamlit run causal/ui/streamlit_app.py
```

The natural-language entry point includes a causal-question classifier before
target-trial drafting. A descriptive question is routed to FastOMOP
(`causal.fastomop_bridge`) and its answer returned inline instead of drafting
a causal study; a question whose intervention or outcome is too ambiguous
returns a clarification request. The gate is LLM-based and may be skipped
explicitly for controlled testing; it is not a replacement for human protocol
review. In the Streamlit
UI this intake is a conversation panel (`causal/companion/`): clarification
is answerable over multiple rounds and the exchange is persisted per run as
`conversation.jsonl`, an advisory record with no gate authority. That same
left panel becomes an active dialogue at the two points a run most needs
one -- the `statistical_estimand` gate (estimand dialogue) and results
review (results-interpretation dialogue) -- each a read-only companion turn
whose only outcome is a human-confirmed card. The estimand dialogue also
gets a deterministic **decision memory**
(`causal/companion/decision_memory.py`) -- the reviewer's already-recorded
covariate exclusions and decisions, distilled with no LLM -- so it does not
re-propose something the reviewer already ruled out. Inspect it with
`workflow decision-memory <run_id>`; `causal-report` freezes it as
`decision_memory.json`. An opt-in **reviewer memory**
(`causal/companion/reviewer_memory.py`, `artifacts/reviewers/<slug>/`)
carries the same kind of decisions across studies when a human clicks "Add
this run's decisions to my reviewer memory" (or `workflow remember-run`).

## Important implementation boundary

`causal.roadmap.CausalStudySpecification` defines the question, target-trial,
estimand, and DAG contracts. `causal.agents.dag_identification.
build_study_specification` wires them into the persistent orchestrator: the
`identification_estimand` gate cannot be approved unless it constructs
without error, and the result is persisted as `study_specification.json`.
The scientific review decisions are split across three gates
(`target_trial`, `dag_review`, `identification_estimand`) instead of one
combined `protocol` gate. `causal.mapping.registry.OMOPVariableRegistry` is
now the sole source every query builder resolves population/treatment/
comparator/outcome concepts from, the same as covariates
(`OMOPVariableRegistry.resolve_role`, `_load_concept_specs`).

For maintained documentation, see `docs/index.md`,
`docs/causal_roadmap.md`, and `docs/fastomop_integration.md`.
