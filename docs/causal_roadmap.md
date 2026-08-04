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

## Current implementation map

| Roadmap stage | Implementation |
|---|---|
| 1. Question classification | Implemented as an LLM-based `causal`, `descriptive`, or `needs_clarification` gate in CLI and Streamlit. It can be skipped explicitly and does not automatically invoke FastOMOP. |
| 2. Target-trial draft | Implemented for manual protocols and strictly parsed LLM drafts; human protocol approval is required. |
| 3. Counterfactual estimand | Represented in protocol fields and general contracts, but the persistent workflow does not yet validate the full `CausalEstimand` contract. |
| 4. Causal DAG | Implemented: literature/KG drafting, typed roles and timing, bounded critique/revision, optional tournament, human covariate review. |
| 5. Identification | Implemented for backdoor adjustment: cycle/temporal checks, d-separation, minimum adjustment sets, explicit failure. |
| 6. OMOP operationalization | Implemented as a first governed version: concept-set and covariate review actions, `OMOPVariableRegistry`, windows, aggregations, and deterministic SQL adapters. |
| 7. Data support | Partly implemented: aggregate feasibility, mapping audits, cohort profile, and follow-up observability. Full positivity, missing-data, and censoring validation remain incomplete. |
| 8. Statistical estimand | Not implemented as an executable contract. |
| 9. Estimation | Not implemented. |
| 10. Diagnostics and sensitivity | Only pre-estimation data diagnostics exist; estimator diagnostics and causal sensitivity analyses are not implemented. |
| 11. Causal report | Not implemented. `demo_report.md` is explicitly a technical/feasibility report. |

## Core contracts

`causal/roadmap.py` defines `CausalQuestion`, `TargetTrialProtocol`,
`CausalEstimand`, `CausalDAG`, and `CausalStudySpecification`. The general DAG
logic is used by the live workflow. The question, protocol, and estimand
contracts are not yet instantiated as one persistent workflow object, so the
current `protocol` human gate carries more responsibility than the target
architecture intends.

The deterministic DAG layer rejects unknown nodes, duplicate identifiers,
cycles, self-loops, temporal contradictions, and inadmissible adjustment for
exposure, outcome, unobserved variables, mediators, colliders, selection
variables, or treatment descendants. It assesses d-separation in the exposure
backdoor graph and enumerates minimum-cardinality adjustment sets for bounded,
reviewable graphs.

## Evidence and LLM boundary

- The question gate and drafting agents are advisory LLM steps. They cannot
  approve a protocol, a DAG, an OMOP mapping, or execution.
- PubMed and OpenAlex provide literature records; OptimusKG is an optional
  relation source. Retrieved identifiers form a deterministic allow-list for
  citations emitted by the model.
- The critique/revision loop is bounded and auditable. Its output remains a
  candidate DAG.
- SHD and SID quantify structural differences; they do not validate clinical
  correctness.
- Human reviewers remain responsible for clinical completeness, causal roles,
  temporal interpretation, mappings, and gate decisions.

See [llm_hypothesis_integration.md](llm_hypothesis_integration.md) and the
dated [DAG audit](dag_generation_audit.md).

## OMOP variable registry

`causal/mapping/registry.py` separates:

- lookup candidates from approved root concepts;
- mapping approval from observed availability;
- a stored review decision from derived SQL executability;
- phenotype definitions from baseline covariate definitions;
- concept inclusion from descendant and exclusion rules;
- temporal windows, aggregation, missingness, provenance, and reviewer data.

The workflow builds `registry.json` from `protocol.json`. Covariate SQL adapters
consume registry-shaped definitions, while population/treatment/comparator/
outcome query specs still resolve directly from the reviewed `concept_sets`
store in `protocol.json`. This is one human-reviewed store, but the registry is
not yet the sole persisted source used by every query builder.

`workflow mapping-audit` executes aggregate-only diagnostics through OMCP and
writes `mapping_audit_<variable_id>.json`. Audit findings do not silently
change the mapping; a reviewer must act through `concept-set-map` or the
`covariate-*` review commands.

## Governance boundary

The persistent workflow currently has four named gates: `protocol`,
`clinical_mapping`, `analysis_plan`, and `results`. A final-mode run requires
approval; exploratory provisional mapping can authorize only the limited
operations encoded by the Execution Agent and OMCP context.

The target architecture should split the combined protocol decision into
separate reviews for:

1. question and target trial;
2. causal DAG and clinical completeness;
3. identification and estimand;
4. OMOP operationalization.

OMCP is the default governed execution transport. The direct BigQuery
feasibility path is retained for equivalence testing, not as the preferred
causal workflow. Aggregate diagnostic paths rely on OMCP's SQL and causal
authorization checks; the local `QueryValidationAgent` is not a full SQL
parser and is not applied uniformly to those aggregate paths.

## Remaining scientific milestones

1. Make `CausalStudySpecification` the persistent pre-OMOP contract and split
   the combined protocol gate.
2. Finish reviewer controls for descendant inclusion/exclusion and make the
   registry the operational source used consistently by every builder.
3. Validate the MIMIC-IV Sepsis-3 population, time zero, exposure strategies,
   censoring, outcome, baseline covariates, missingness, and positivity.
4. Define the observed-data estimand and missing-data/censoring strategy.
5. Implement a transparent g-computation baseline, then prespecified TMLE and
   CATE analyses with reproducible nuisance-model specifications.
6. Add balance, overlap, positivity, calibration, and model diagnostics.
7. Add negative controls and quantitative sensitivity analyses where the study
   design supports them.
8. Produce a final report that links estimates to protocol versions, registry
   versions, code versions, diagnostics, assumptions, and human decisions.

Until these milestones are complete, CausalOMOP is a governed design and
feasibility prototype, not a clinical causal-effect engine.
