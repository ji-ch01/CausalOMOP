# CausalOMOP Causal Roadmap

This is the normative scientific sequence for CausalOMOP and the reference for
what the current prototype may claim.

## Product objective

CausalOMOP should produce an auditable causal study from longitudinal data in
OMOP CDM. A successful SQL query or an acyclic graph is not, by itself, a
causal analysis. The question, target trial, counterfactual estimand, causal
model, identification assumptions, data support, statistical estimand,
estimator, diagnostics, and interpretation must remain distinguishable.

## Required order

```text
Clinical question
  -> 1. classify and clarify the causal question
  -> 2. specify the target trial
  -> 3. define the counterfactual estimand
  -> 4. construct and clinically review the causal DAG
  -> 5. assess identification and select an admissible adjustment set
  -> 6. map required variables to OMOP concepts, fields, and windows
  -> 7. audit availability, missingness, follow-up, and positivity
  -> 8. define the observed-data statistical estimand
  -> 9. prespecify and run estimators
  -> 10. run diagnostics and sensitivity analyses
  -> 11. report estimates without exceeding their assumptions
```

OMOP operationalization begins only after the scientific variables have been
defined. A clinically important but unobserved confounder stays in the DAG and
becomes an explicit limitation; it must not disappear because no convenient
OMOP concept exists.

## Validation strategy

CausalOMOP is general-purpose by design: every stage above is driven by the
reviewed protocol, not by a hardcoded disease, drug, or outcome, and the
pipeline never searches for a matching published study at run time. Rigor is
instead demonstrated by hand-picking a small number of causal questions that
already have a published, literature-grounded analysis with a known expert
DAG or concept mapping, then running the actual pipeline against each one so
its output can be compared to that published ground truth -- SHD/SID for DAG
structure (`causal/agents/dag_metrics.py`), recall@k/MRR for concept mapping.
Three scaffolded `studies/` packages carry this comparison --
[ppi_vs_h2ra_sup_mortality](../studies/ppi_vs_h2ra_sup_mortality/README.md)
(Q1, PEPTIC), [corticosteroids_severe_cap_mortality](../studies/corticosteroids_severe_cap_mortality/README.md)
(Q8, CAPE COD), and [vanco_piptazo_vs_cefepime_aki](../studies/vanco_piptazo_vs_cefepime_aki/README.md)
(Q5, ACORN) -- each a draft pending human review, not a description of what
the pipeline is built to study. The selection criteria, the reference-anchored
shortlist, and the ten-question operational set are in
[docs/validation_study_candidates.md](validation_study_candidates.md) and
[docs/validation_question_bank.md](validation_question_bank.md). A sepsis
target trial whose exposure and time zero are intensive-care infusions
(albumin plus crystalloids vs crystalloids, Doutreligne et al. 2025) is
future work, blocked on an `inputevents` ETL extension (see
[docs/vocabulary_mapping_coverage.md](vocabulary_mapping_coverage.md)). See
[README.md's References](../README.md#references) for the full literature
grounding.

## Current implementation map

| Roadmap stage | Implementation |
|---|---|
| 1. Question classification | Implemented as an LLM-based `causal`, `descriptive`, or `needs_clarification` gate in CLI and Streamlit. It can be skipped explicitly; a `descriptive` result automatically invokes FastOMOP (`causal.fastomop_bridge`) and returns its answer instead of drafting a causal study. |
| 2. Target-trial draft | Implemented for manual protocols and strictly parsed LLM drafts; human protocol approval is required. |
| 3. Counterfactual estimand | Implemented: `causal.agents.dag_identification.build_study_specification` constructs and validates the full `CausalQuestion`/`TargetTrialProtocol`/`CausalEstimand` contract from `protocol.json` + `hypothesis.json`, persisted as `study_specification.json` when the `identification_estimand` gate is approved. `protocol["target_trial"]` carries the complete field set (eligibility criteria; assignment, censoring, and analysis strategy; effect measure; treatment/comparator contrast values), required by both the LLM drafting prompt (`causal/prompts/hypothesis.md`) and the `target_trial` gate. |
| 4. Causal DAG | Implemented: literature/KG drafting, typed roles and timing, bounded critique/revision, optional tournament, human covariate review. |
| 5. Identification | Implemented for backdoor adjustment: cycle/temporal checks, d-separation, minimum adjustment sets, explicit failure. |
| 6. OMOP operationalization | Implemented as a first governed version: concept-set and covariate review actions, `OMOPVariableRegistry`, windows, aggregations, and deterministic SQL adapters. |
| 7. Data support | Extended: aggregate feasibility, mapping audits, cohort profile, and follow-up observability, plus univariate treatment-arm overlap (`positivity_audit_sql` -- one covariate at a time, not full joint-adjustment-set stratification), per-covariate missingness (`covariate_missingness_sql`, generalized beyond the three hardcoded demographics), and a reviewer-approvable per-row administrative censoring definition (`WorkflowManager.approve_censoring_definition`, reusing `followup_audit_sql`'s own aggregate logic per person). None of these is a full positivity/missing-data-mechanism/censoring-model validation in the statistical sense -- see each function's own docstring for what it does and does not claim. |
| 8. Statistical estimand | Implemented: `roadmap.StatisticalEstimand` is a separate, persisted contract from the counterfactual estimand (`CausalEstimand`) -- identifying functional, outcome-model specification, the adjustment set as actually built into `cohort.sql` (`cohort_manifest.json`'s `baseline_covariate` columns, not re-derived), and an explicit, reviewer-set missing-data and censoring strategy (`WorkflowManager.set_statistical_estimand_strategy`), persisted as `statistical_estimand.json` when the `statistical_estimand` gate is approved (`estimand_review` phase, between `analysis_plan` and `execution`). The analytic cohort's per-row `censored` column, previously unused by anything, is now read by `EstimationAgent` when `censoring_strategy="exclude_censored_before_analysis"`; `"ignore_censoring_administrative_waiver"` is the explicit, reviewed alternative when a reviewer judges it unused. Neither strategy is IPCW or another censoring-weighting method -- see milestone 3 below for what remains open. In the Streamlit UI the three choices are settled through the estimand-dialogue companion (`causal.companion.companion`), which proposes `set_statistical_estimand_strategy` as a human-confirmed card; the CLI `estimand-strategy` command and the UI's "Fill the form directly" fallback are the same call. |
| 9. Estimation | Implemented for the ATE, risk-difference scale, binary outcome, via a reviewer's choice of estimator: `causal.agents.estimation.EstimationAgent` (transparent g-computation/standardization, an unpenalized logistic outcome-regression model, nonparametric bootstrap CI, cited to Hernan & Robins *Causal Inference: What If* ch. 13) or `causal.agents.tmle.TMLEAgent` (TMLE, `SuperLearner`-ensembled nuisance models, doubly robust). Also implemented: `causal.agents.cate.CATEAgent` for the CATE, a prespecified effect-modifier subgroup family via a causal forest. `WorkflowManager.materialize_analytic_dataset` is the one code path that pulls patient-level rows into memory (via OMCP's already-existing, previously-unused `cohort_build` governed operation) to fit any of them. See milestone 4 for full detail and what remains (an LLM narrative for CATE subgroup results). |
| 10. Diagnostics and sensitivity | Extended beyond pre-estimation data diagnostics: standardized mean difference (SMD) covariate balance per arm (`causal.agents.diagnostics.DiagnosticsAgent`, Austin 2009, `\|SMD\| > 0.1` flag), in-sample outcome-model AUC/Brier calibration, E-value quantitative sensitivity analysis (`causal.sensitivity.e_value`, VanderWeele & Ding 2017), and negative controls (`causal.queries.negative_controls`) are implemented. Joint/multivariate covariate balance is not (see milestone 6). |
| 11. Causal report | Implemented: `WorkflowManager.build_causal_report` writes `causal_report.md`, linking the estimate, diagnostics, assumptions/limitations, approval-gate history, and per-artifact provenance -- the `results` gate (`REQUIRED_GATES["results_review"]`, previously defined but never reached by any run) is now the human sign-off before it, entirely independent of the optional narrative below. An optional, gracefully-degrading LLM narrative section (`causal.agents.results_narrative`) may interpret the already-computed numbers for a clinical reviewer; it is fed only JSON-safe artifacts, never the patient-level dataset, must return structured `claims` alongside its prose, and is deterministically fact-checked by `causal.agents.results_faithfulness.verify_narrative` (no second LLM) before a human reviewer (`WorkflowManager.review_results_narrative`) can approve or exclude it -- only an `approved` narrative with no `failed` check is ever included in `causal_report.md`; everything else renders as "not requested" or "excluded". `demo_report.md` (`build_demo_report`) is unchanged and remains the earlier pre-estimation feasibility snapshot. |

## Core contracts

`causal/roadmap.py` defines `CausalQuestion`, `TargetTrialProtocol`,
`CausalEstimand`, `CausalDAG`, `CausalStudySpecification`, and
`StatisticalEstimand`. The question, target-trial, and estimand contracts
are instantiated as one persistent workflow object:
`causal.agents.dag_identification.build_study_specification` builds and
validates the full `CausalStudySpecification` from `protocol.json` +
`hypothesis.json`, the `identification_estimand` gate cannot be approved
unless it constructs without error, and the result is persisted as
`study_specification.json`. The single combined `protocol` gate is gone --
scientific review is split across `target_trial`, `dag_review`, and
`identification_estimand`.

When the LLM draft of the study design is wrong in a bounded way, the
reviewer does not have to regenerate the whole run (which discards every
covariate decision and mapping already made). Scoped in-place corrections
exist, each recording a `before`/`after` in `state.history`, appending to
`protocol.json`'s `correction_log`, and re-opening the gates that depended on
the changed content:

- `set_target_trial_estimand` (`primary_estimand`/`secondary_estimands`) and
  `set_target_trial_timing` (`time_zero`/`grace_period_hours`/
  `follow_up_days`, CLI `workflow target-trial-timing`, also re-opening
  `analysis_plan` and `statistical_estimand` because `follow_up_days` feeds
  the cohort SQL and the censoring definition). Both are refused once
  `target_trial` is `approved` -- reject it first.
- `revise_candidate_dag_from_points` (CLI `workflow revise-dag`): a reviewer's
  free-text DAG correction points ("add baseline creatinine as a confounder",
  "vasopressor use is a mediator", "remove ICU LOS, it is a collider"). An LLM
  *formats* each point into an `add`/`reclassify`/`remove` operation
  (`causal.agents.dag_point_revision`) -- the reviewer is the grounding, so
  added variables are expert-asserted (`pmid=""`), the same as
  `review_covariate`; this is not the literature-grounded
  `_critique_and_revise_dag` loop and skips its citation allow-list. The
  revised graph is still validated by `HypothesisAgent` before anything is
  written, and `dag_review`/`clinical_mapping`/`analysis_plan` re-open.

The free-text population/treatment/comparator/outcome fields have no in-place
editor: they also seed the literature search and `first_pass_covariates`, so
correcting them stays a regeneration -- but the accumulated `correction_log`
is replayed into that regeneration prompt
(`UserQuestionHypothesisService.generate`'s `prior_corrections`) so a fresh
draft is less likely to repeat a mistake a reviewer already fixed.

During `target_trial_review` and `dag_review` the left-column companion
becomes a **study-design dialogue** (`run_design_companion_turn`,
`causal/prompts/design_dialogue.md`): the reviewer describes a mistake, the
companion proposes one of the three corrections above as a card the reviewer
confirms -- same allow-list mechanism (`causal.companion.actions`) and the
same "companion never approves a gate" boundary as the estimand dialogue.

`StatisticalEstimand` is deliberately a separate contract from
`CausalEstimand`, not a few extra fields on it: `CausalEstimand` (stage 3)
fixes the counterfactual contrast and is meaningful before any OMOP data
exists; `StatisticalEstimand` (stage 8) fixes how that contrast is actually
estimated from the observed, incomplete, and administratively censored data
this pipeline extracts -- persisted separately as `statistical_estimand.json`
once its own `statistical_estimand` gate is approved, not folded into
`study_specification.json`.

The deterministic DAG layer rejects unknown nodes, duplicate identifiers,
cycles, self-loops, temporal contradictions, and inadmissible adjustment for
exposure, outcome, unobserved variables, mediators, colliders, selection
variables, or treatment descendants. It assesses d-separation in the exposure
backdoor graph and enumerates minimum-cardinality adjustment sets for bounded,
reviewable graphs.

## Evidence and LLM boundary

- The question gate, the drafting agents, the study-design dialogue companion
  (`causal.companion.companion`, at `target_trial_review` / `dag_review`), the
  estimand-dialogue companion (same module, at the `statistical_estimand`
  gate), and the results-interpretation dialogue companion (same module, at
  `results_review` / `report`) are advisory LLM steps. They cannot approve a
  protocol, a DAG, an OMOP mapping, a statistical estimand, execution, or the
  `results` gate. The design and estimand companions' possible writes
  (`set_target_trial_timing`, `set_target_trial_estimand`,
  `revise_candidate_dag_from_points`, `set_statistical_estimand_strategy`) are
  each an explicitly human-confirmed card gated by a code-level allow-list
  (`causal.companion.actions`, which also rejects an injected `reviewer` /
  `run_id`); approving the gate itself stays manual. The results companion's
  only outcome is a drafted narrative the reviewer explicitly saves
  (`save_results_narrative_draft`), which then goes through the same
  deterministic `verify_narrative` and the same human
  `review_results_narrative` approval as a one-shot narrative.
- The estimand companion is additionally fed a **decision memory**
  (`causal.companion.decision_memory`): a deterministic, no-LLM digest of the
  reviewer's already-recorded exclusions (`protocol.json`'s
  `rejected_covariates`) and decisions (`state.history` entries, gate
  invalidations, and any rejected gate with its comment), so it will not
  re-propose a covariate the reviewer threw out or re-open a settled choice.
  It changes what the companion is told, never what it can do.
  `build_causal_report` freezes it as a `decision_memory.json` snapshot and
  lists it in the report's Provenance table. An opt-in **reviewer memory**
  (`causal.companion.reviewer_memory`, `artifacts/reviewers/<slug>/`)
  carries the same class of decisions *across* studies when a human asks;
  it is rendered as `REVIEWER CONVENTIONS`, still advisory, still no
  patient data.
- PubMed and OpenAlex provide literature records; OptimusKG is an optional
  relation source. Retrieved identifiers form a deterministic allow-list for
  citations emitted by the model.
- The critique/revision loop is bounded and auditable. Its output remains a
  candidate DAG.
- SHD and SID quantify structural differences; they do not validate clinical
  correctness.
- Human reviewers remain responsible for clinical completeness, causal roles,
  temporal interpretation, mappings, and gate decisions.
- The optional results narrative (`causal.agents.results_narrative`) is the
  same kind of advisory step: it reads only the already-computed estimate,
  diagnostics, and gate history (never the patient-level dataset), must
  quote every number verbatim from that input, must break its own prose
  into structured `claims` (typed, each citing the JSON path it is based
  on), and its output is always wrapped in a fixed, code-appended boundary
  statement the model's own text cannot override. So that the narrative
  reads like a clinician's write-up rather than a string of raw floats, the
  input carries a deterministic `estimation.narrative_display` block
  (`causal.agents.narrative_display`): every reported quantity rounded to a
  communication precision, plus the derived restatements a clinician
  reasons with (events per 1000 patients, relative risk change, number
  needed to treat) and a plain-language sentence per topic. It is pure
  arithmetic on numbers already in `estimation.json` /
  `diagnostics.json` / `sensitivity_analysis.json` -- so every rounded value
  is still a verbatim token `verify_narrative` accepts, and `estimation.
  json` keeps full precision for audit. `build_causal_report` renders its
  "Causal effect estimate" section from the same block. Before any human ever
  sees it, `causal.agents.results_faithfulness.verify_narrative` -- plain
  deterministic code, not a second LLM -- checks every number, the effect
  direction, the null-crossing interpretation, diagnostic-warning coverage,
  assumption/limitation coverage, and every claim's source-field
  provenance. A reviewer (`WorkflowManager.review_results_narrative`) can
  then approve the narrative only when no check `failed`, or exclude it
  outright; either way this is a separate decision from the mandatory
  `results` gate, which the narrative can never approve, invalidate, or
  otherwise influence. `causal_report.md` includes the narrative's text
  only once it is `approved` with a clean verification -- an unrequested,
  still-`pending`, or `excluded` narrative renders as a plain status line
  instead. In the Streamlit UI this narrative is normally drafted through
  the results-interpretation dialogue (the companion walks the reviewer
  through the estimate one topic at a time, then drafts the
  `{ narrative, claims }`); the one-shot `run_results_narrative` remains the
  no-conversation fallback and `results_narrative.json` records which path
  produced it (`source`).

See [llm_hypothesis_integration.md](llm_hypothesis_integration.md) and the
[DAG generation reference](dag_generation_audit.md).

## OMOP variable registry

`causal/mapping/registry.py` separates:

- lookup candidates from approved root concepts;
- mapping approval from observed availability;
- a stored review decision from derived SQL executability;
- phenotype definitions from baseline covariate definitions;
- concept inclusion from descendant and exclusion rules;
- temporal windows, aggregation, missingness, provenance, and reviewer data.

The workflow builds `registry.json` from `protocol.json`. Every query builder
now resolves through the registry: covariate SQL adapters consume
registry-shaped definitions, and population/treatment/comparator/outcome
query specs resolve by role through `OMOPVariableRegistry.resolve_role`
(`_load_concept_specs` is a thin wrapper over it). The `concept_sets` entry
on `protocol.json` remains the one place a reviewer records an approval; the
separate `query_concept_sets` store that a reviewer had to map independently,
with nothing checking the two agreed, is gone.

`workflow mapping-audit` executes aggregate-only diagnostics through OMCP and
writes `mapping_audit_<variable_id>.json`. Audit findings do not silently
change the mapping; a reviewer must act through `concept-set-map` or the
`covariate-*` review commands.

## Governance boundary

The persistent workflow has seven named gates: `target_trial`, `dag_review`,
`identification_estimand`, `clinical_mapping`, `analysis_plan`,
`statistical_estimand`, and `results` -- each tied to its own phase
(`target_trial_review`, `dag_review`, `identification_review`,
`concept_mapping`, `analysis_plan`, `estimand_review`, `results_review`). A
final-mode run requires approval; exploratory provisional mapping can
authorize only the limited operations encoded by the Execution Agent and
OMCP context.

This implements the split an earlier version of this section proposed as
future work, into separate reviews for:

1. question and target trial (`target_trial` gate: `CausalQuestion`/
   `TargetTrialProtocol` must construct from `protocol.json` alone);
2. causal DAG and clinical completeness (`dag_review` gate: every major
   confounder resolved);
3. identification and estimand (`identification_estimand` gate: backdoor
   identification holds *and* the full `CausalStudySpecification` --
   question, target trial, estimand, DAG -- validates as one internally
   consistent contract; persists `study_specification.json` on approval);
4. OMOP operationalization (`clinical_mapping` gate, unchanged);
5. the observed-data statistical estimand (`statistical_estimand` gate,
   `estimand_review` phase between `analysis_plan` and `execution`):
   `roadmap.StatisticalEstimand` -- identifying functional, outcome model,
   the adjustment set as actually built into `cohort.sql`
   (`cohort_manifest.json`), and the reviewer's missing-data/censoring
   strategy (`WorkflowManager.set_statistical_estimand_strategy`) -- must
   construct and persists `statistical_estimand.json` on approval; blocked
   if `study_specification.json`'s declared `effect_measure` is not
   `risk_difference` (the only scale `EstimationAgent` implements) or if
   `censoring_strategy="exclude_censored_before_analysis"` is chosen before
   the per-row `censored` column has itself been approved
   (`approve_censoring_definition`).

OMCP is the default governed execution transport. The direct BigQuery
feasibility path is retained for equivalence testing, not as the preferred
causal workflow. Aggregate diagnostic paths rely on OMCP's SQL and causal
authorization checks; the local `QueryValidationAgent` is not a full SQL
parser and is not applied uniformly to those aggregate paths.

## Remaining scientific milestones

1. Done: `OMOPVariableRegistry` is now the sole operational source for all
   four query roles, not just covariates. `_load_concept_specs` is a thin
   wrapper around `OMOPVariableRegistry.from_protocol` +
   `OMOPVariableRegistry.resolve_role` (population/treatment/comparator/
   outcome resolve the same way `_load_covariate_specs` already resolved
   covariates), with two additive registry functions covering the two gaps
   that previously blocked this: `fixed_domain_to_concept_set_spec` for the
   `death`/`standard` fixed-domain roles `build_extraction_plan` cannot
   handle (it requires a concept-based domain and `root_concept_ids`), and a
   widened `variable_from_concept_set` domain check tolerating
   `domain == ""` for the `exposure_contrast="vs_unexposed"` comparator's
   genuinely domain-less state. Reviewer controls for descendant
   inclusion/exclusion remain finished as before this item landed.
2. Validate the MIMIC-IV Sepsis-3 population, time zero, exposure strategies,
   censoring, outcome, baseline covariates, missingness, and positivity --
   the diagnostics now exist (`positivity_audit_sql`,
   `covariate_missingness_sql`, `approve_censoring_definition`); running
   them against the real dataset and acting on the results does not.
3. Done for exclusion/waiver: `roadmap.StatisticalEstimand` and the
   `statistical_estimand` gate formalize the observed-data estimand and
   require a reviewer to explicitly pick a missing-data strategy
   (`complete_case_analysis`, the only one implemented) and a censoring
   strategy (`exclude_censored_before_analysis`, which now actually drops
   `censored=1` rows in `EstimationAgent`, or
   `ignore_censoring_administrative_waiver`, an explicit waiver rather than
   silent non-use, or `ipcw_administrative_censoring`, a stabilized
   inverse-probability-of-censoring-weighted g-formula (`causal.agents.
   estimation.EstimationAgent._fit_and_standardize_ipcw`) -- unlike
   exclusion, this keeps the estimate over the original eligible
   population instead of changing who it is over. Still not implemented:
   multiple imputation for missing covariates, and censoring weighting for
   anything beyond a single fixed-horizon administrative-censoring
   indicator (e.g. time-varying censoring).
4. Done for the baseline: a transparent g-computation estimator
   (`causal.agents.estimation.EstimationAgent`) is implemented for the ATE,
   risk-difference scale, binary outcome, with a nonparametric bootstrap
   confidence interval. Done for SuperLearner nuisance-model infrastructure:
   `causal.nuisance.SuperLearner` (a prespecified candidate library
   combined by cross-validated non-negative least squares, van der Laan,
   Polley & Hubbard 2007). Done for TMLE: `causal.agents.tmle.TMLEAgent`
   (Van der Laan & Rose 2011) pairs a `SuperLearner`-ensembled outcome and
   treatment nuisance fit with a targeting step and an efficient-influence-
   curve closed-form variance estimator instead of a bootstrap -- doubly
   robust (consistent if either nuisance model is correctly specified, not
   only both), unlike the plug-in baseline. Both estimators target the same
   `g_formula_standardization` functional; `StatisticalEstimand.
   estimation_procedure` (schema_version 1.1) records which one a reviewer
   chose for a given run, and `WorkflowManager.run_estimation` dispatches
   accordingly. Scoped out of TMLE v1, and not silently half-implemented:
   `ipcw_administrative_censoring` (`TMLEAgent.run` raises `ValueError` --
   censored-data TMLE needs a materially different efficient-influence-
   function construction, Van der Laan & Rose 2011 ch. 5, not a reuse of
   `EstimationAgent`'s stabilized-weight IPCW).

   Done for CATE: `causal.agents.cate.CATEAgent` (Athey & Imbens 2016,
   Wager & Athey 2018) fits `econml.dml.CausalForestDML` over
   `SuperLearner`-backed nuisance models (`SuperLearnerClassifier`/
   `SuperLearnerRegressor`, new sklearn-compatible adapters in `causal.
   nuisance` -- `SuperLearner` itself stays deliberately non-sklearn-
   compatible), for a prespecified family of `causal_role="effect_modifier"`
   subgroup covariates resolved from `analysis_plan.json` (never
   `hypothesis.json`'s free-text, ungated `candidate_effect_modifiers`).
   Multiple-testing correction across the family uses Holm's (1979)
   sequentially rejective stepdown procedure -- not the originally-planned
   resampling-based stepdown (List, Shaikh & Xu 2019's headline method):
   bootstrapping the causal forest's already-fitted per-unit point
   estimates (cheap) was tried first and found to understate the null
   distribution's spread (it captures only resampling noise of averaging
   fixed numbers, not genuine forest estimation uncertainty), producing
   spuriously significant adjusted p-values -- caught by inspecting output
   before shipping it, documented in `causal.agents.cate._holm_stepdown_
   fwer_correction`'s own docstring. New `StatisticalEstimand.
   population_summary` (schema_version 1.2, `"ATE"`/`"ATT"`/`"CATE"`) gates
   `WorkflowManager.run_cate` -- CATE is never the default. `causal.agents.
   cate`/`econml` are lazily imported inside `run_cate` only (not at
   `causal.orchestrator.workflow` module scope): `econml` alone costs ~5s
   to import (numba/lightgbm/statsmodels transitively), which would
   otherwise tax nearly every test in the suite. Deliberately not
   implemented alongside v1: an LLM narrative for subgroup results
   (`build_causal_report`'s CATE section is a deterministic table only).
5. Partially done: standardized mean difference (SMD) covariate balance
   (`causal.agents.diagnostics.DiagnosticsAgent`) and cross-validated
   outcome-model AUC/Brier calibration (`causal.agents.estimation.
   EstimationAgent._cross_validated_outcome_diagnostics`, falling back to
   in-sample only when too few outcome events exist to stratify into
   folds) are implemented. Joint/multivariate covariate balance remains
   future work.
6. Done for E-values: `causal.sensitivity.e_value` (VanderWeele & Ding 2017)
   computes the minimum unmeasured-confounder strength, on the risk-ratio
   scale, that would fully explain away the observed association -- for
   both the point estimate and the confidence-interval bound closest to the
   null. Both estimators (`GComputationEstimate`/`TMLEEstimate`) now report
   a risk ratio alongside their primary risk-difference estimate so the
   E-value has one to work with (g-computation: the same bootstrap
   resamples, tracked for the ratio too, not a second model;
   TMLE: a delta-method Wald CI on `log(psi1/psi0)` from the per-arm
   targeted means, no second fluctuation). `WorkflowManager.
   run_sensitivity_analysis` writes `sensitivity_analysis.json`, callable
   any time after `run_estimation`; `build_causal_report` includes an
   E-value section when it exists.

   Done for negative controls: `causal.queries.negative_controls` reuses
   `causal.queries.feasibility`'s existing cohort-building machinery (no
   new SQL-generation logic) via a substituted `ConceptSetSpec` --
   `substitute_spec` swaps a reviewed negative control in for `treatment`
   or `outcome`, keeping population/comparator/covariates/censoring
   identical to the real study. Candidate proposal mirrors `causal.agents.
   dag_critique`'s bounded prompt/parse pattern (`WorkflowManager.
   propose_negative_controls`, no orchestration duplicated): an LLM
   proposes a clinical label and rationale only, never a concept id, and
   `parse_negative_control_response` cannot return an approved candidate
   regardless of what the raw response claims. `WorkflowManager.
   map_negative_control` is the one place a candidate is actually mapped
   to OMOP concepts and approved or rejected, folded into the existing
   `clinical_mapping` gate rather than a new one. `WorkflowManager.
   run_negative_control_estimation` fits whichever estimator was reviewed
   for the real study against the substituted cohort and writes
   `negative_control_estimation_<control_id>.json` -- a diagnostic
   `build_causal_report` surfaces, never a blocking gate. The
   materialized negative-control cohort is never written to disk, unlike
   the real study's `analytic_dataset.parquet`.
7. Done: `WorkflowManager.build_causal_report` writes `causal_report.md`,
   linking the estimate to the protocol/estimand contract
   (`study_specification.json`), the analytic-dataset manifest, diagnostics,
   every upstream warning/assumption, the full approval-gate history, and
   each artifact's own provenance metadata.

Milestone 2 remains open, and milestone 5 is partial (joint/multivariate
covariate balance remains future work). Until milestone 2 is complete, the
g-computation, TMLE, and CATE estimates this pipeline can now produce are a
transparent, prespecified baseline under stated (not yet fully validated)
assumptions -- not a validated clinical causal-effect engine.
