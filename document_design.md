# FastOMOP Causal Reasoning Extension
## Design Document (Draft v1)

---

# Goal

This document summarizes the proposed architecture for extending **FastOMOP** with
causal reasoning capabilities.

The objective is **not** simply to generate Directed Acyclic Graphs (DAGs),
but to transform natural-language causal questions into complete,
auditable Real-World Evidence (RWE) workflows built on OMOP CDM.

The DAG is only one component of this workflow.

## Current Project Alignment

The primary validation use case for this extension is now:

> Does dapagliflozin, compared with sitagliptin, increase the 90-day risk of
> urinary or genital infection in Type 2 Diabetes Mellitus patients?

This side-effect endpoint is better aligned with the supervisors' original
question than the earlier 365-day mortality endpoint. The mortality cohort
remains useful as a feasibility/demo candidate, but it is not the preferred
primary causal endpoint because mortality events are sparse for the selected
T2DM active-comparator study.

Current implemented code status:

- ETL/data-quality scripts rebuild `data/synthetic_snuh_curated.duckdb` from a
  preserved raw copy and Athena vocabulary files.
- Current validation snapshot: conditions 100% mapped, measurements 100%
  mapped, drug exposures 97.3% mapped to standard Drug concepts, observations
  89.3% mapped after HSDN/MeSH remapping.
- Remaining unmapped concepts are intentionally left unmapped when no reliable
  standard concept exists, rather than silently forcing incorrect mappings.
- `causal/data_builder/cohort.py` implements cohort construction and feasibility
  scans, including active-comparator logic and condition outcomes.
- The full causal agent team is not yet implemented. Only the cohort-builder
  portion of the future Execution Agent exists today.
- FastOMOP still follows the existing descriptive NL→SQL workflow by default;
  causal/descriptive routing remains a planned integration step.

---

# Motivation

FastOMOP currently provides a governed multi-agent architecture capable of
answering descriptive questions through NL→SQL translation.

Example:

> How many Type 2 Diabetes patients received metformin?

↓

Natural language

↓

Semantic normalization

↓

SQL generation

↓

DuckDB OMOP

↓

Answer

This architecture is ideal for descriptive epidemiology.

However, it cannot yet answer causal questions such as

> Does dapagliflozin increase the risk of urinary or genital infection
compared with sitagliptin?

because answering such questions requires:

- target trial specification
- cohort definition
- confounder identification
- causal assumptions
- adjustment strategy
- treatment effect estimation

rather than SQL alone.

---

# Proposed Contribution

The proposed contribution is **not another standalone DAG generator.**

Instead, the proposal extends FastOMOP with a new **Causal Reasoning Agent-Team**
that inherits FastOMOP's existing

- orchestration
- governance
- observability

layers.

The new team transforms

Natural-language causal question

↓

Target Trial Specification

↓

Conceptual DAG

↓

OMOP covariate extraction

↓

Treatment effect estimation

↓

Auditable causal report

---

# Overall Architecture

                    User Question
                           │
                           ▼
                 FastOMOP Orchestrator
                           │
      ┌────────────────────┴─────────────────────┐
      │                                          │
      ▼                                          ▼
 NL2SQL Agent-Team                   Causal Reasoning Agent-Team
 Existing capability                  Proposed extension
      │                                          │
      ▼                                          ▼
 SQL Answer                          Causal Analysis Workflow

All agents automatically inherit:

- Governance Layer
- Observability Layer
- MCP tool access
- Prompt versioning
- Audit logging

---

# Orchestrator Routing: Current Gap and Implementation Plan

The diagram above assumes the FastOMOP Orchestrator already routes a
user question to either agent-team. **It does not, today.**

Current state (as implemented in `agno_fastomop`):

- `workflows/omop_workflow.py` exposes `run_omop_query()`, which runs
  a **fixed, linear** `Workflow`: Semantic Extraction → SQL
  Generation and Execution. There is no branching — every question,
  descriptive or causal, enters the same pipeline.
- `agents/supervisor.py` implements an agent-with-tools pattern
  (an `Agent` whose `tools=[semantic_agent, database_agent]` lets an
  LLM decide which sub-agent to call) — the mechanism an orchestrator
  would need. **It is not wired into any entrypoint.**
  `run_agent.py` and `web_interface.py` both call
  `initialize_workflow()` / `run_omop_query()` directly and never
  import the supervisor.
- `prompts/supervisor.txt` only contains classification logic for
  descriptive/comparative query patterns (simple count, AND/OR,
  temporal, comparison). It has no notion of a causal question.

So there is currently no point in the live code path where a causal
question could be detected and redirected — the orchestrator in the
architecture diagram does not exist yet as a running component.

## Recommended Mechanism

Reuse the agent-with-tools pattern already sketched in
`supervisor.py`, and make it the actual entrypoint:

1. `run_agent.py` / `web_interface.py` call the supervisor agent
   instead of `run_omop_query()` directly.
2. The supervisor is given two tools instead of two sub-agents:
   - `nl2sql_tool` — wraps the existing `run_omop_query()` pipeline
   - `causal_tool` — wraps a new `run_causal_query()` entrypoint
     driving the Causal Planner → DAG & Evidence Agent → Causal
     Execution Agent pipeline (not yet built; today only the cohort
     builder piece of Agent 3 exists, as a standalone script in
     `causal/data_builder/cohort.py`, not agent-orchestrated)
3. `supervisor.txt` is rewritten to classify intent (descriptive vs.
   causal) before deciding which tool to call, following the same
   example-driven style already used for comparative queries — with
   added examples of causal phrasing ("does X increase the risk of
   Y", "what is the effect of X on Y", "compared with Z").

This keeps NL2SQL untouched and adds causal routing as a thin layer
on top, consistent with the "does not modify the three foundational
layers" constraint stated earlier in this document.

## Trade-off

Routing is done by LLM judgment (intent classification inside the
supervisor's own reasoning), not by a deterministic rule. This is
consistent with the rest of the architecture, which already accepts
after-the-fact governance validation (positivity, no-leakage) rather
than upfront hard rules — a misclassification is not catastrophic,
since the causal tool can still reject or redirect a question that
turns out to be descriptive. The alternative (a separate deterministic
classifier ahead of the orchestrator) would be more predictable but
adds another component that itself needs governance and maintenance.

---

# FastOMOP Infrastructure

The extension does **not** modify the three foundational layers.

These remain exactly as described in the FastOMOP paper.

## 1. Orchestration

Responsible for:

- intent routing
- workflow planning
- context preservation
- delegation between agent teams

---

## 2. Governance

Responsible for:

- SQL validation
- table whitelist
- dangerous operation blocking
- PHI protection
- safety constraints
- deterministic validation

The causal extension should integrate additional governance rules such as

- no post-treatment leakage
- adjustment variables must precede index date
- forbidden mediator adjustment
- positivity checks

---

## 3. Observability

Responsible for

- trace logging
- prompt versioning
- intermediate outputs
- benchmarking
- debugging

The causal extension should additionally log

- DAG versions
- causal assumptions
- evidence sources
- adjustment sets
- concept mappings
- vocabulary version (Athena release) used for each concept mapping
- concept hierarchy coverage per DAG node (see Agent 3 — Vocabulary
  Coverage Check)

---

# Causal Reasoning Agent-Team

Instead of introducing many independent agents,
the architecture groups causal reasoning into three specialised agents.

---

## Agent 1 — Causal Planner

Purpose

Convert a natural-language question into a structured causal study specification.

Example

Input

Does dapagliflozin increase urinary or genital infection risk compared with sitagliptin?

Output

Population

Type 2 Diabetes Mellitus

Treatment

dapagliflozin

Comparator

sitagliptin

Outcome

urinary/genital infection

Follow-up

90 days

Target trial

active comparator
new-user design

Estimand

ATE
Risk Difference
Risk Ratio

Design Selection Rule

The Planner must prefer an **active-comparator new-user design**
whenever a clinically plausible comparator drug exists.

Untreated-control designs (index date = first diagnosis rather than
first exposure) are permitted only as a fallback, because they
compare a treated arm anchored to an event (drug initiation) against
a control arm anchored to a different kind of event (diagnosis).
This asymmetry is a known source of immortal-time bias and
confounding by indication.

When untreated controls are unavoidable, the Planner must:

- impose a grace period during which the control arm remains at risk
  of becoming treated
- flag the design as higher bias-risk in the target trial specification
- require this flag to be carried into the final governance report

Each estimand must be bound to a compatible estimator (see
Agent 3 — Estimand/Estimator Contract). ATE requires IPW or
standardization; ATT requires matching. The Planner's chosen
estimand constrains which estimators Agent 3 is allowed to use.

Responsibilities

- identify population
- identify treatment
- identify comparator
- identify outcome
- identify follow-up
- identify estimand
- define target trial specification
- select design (active comparator vs untreated control) per the
  Design Selection Rule above
- flag competing-risk exposure when the outcome is not death itself
  and mortality can occur within the follow-up window

---

## Agent 2 — DAG & Evidence Agent

Purpose

Construct a **conceptual DAG**
supported by retrieved clinical evidence.

This agent DOES NOT create OMOP variables.

It creates causal concepts.

Example

Age

Sex

Diabetes severity

Cardio-renal disease

Prior infection

Healthcare utilization

BMI (unobserved)

Post-treatment HbA1c (mediator)

Treatment switch (post-treatment)

For each node

identify

- confounder
- mediator
- collider
- outcome predictor
- unobserved confounder
- measurement window (must be pre-index unless explicitly marked
  mediator/post-treatment)

Nodes with an ambiguous measurement window (e.g. "Healthcare
utilization", which can be either a pre-index confounder or a
post-index mediator/collider depending on when it is measured) must
not be left unresolved. The DAG Agent must force an explicit
pre-index vs post-index decision at this stage, rather than deferring
it to Agent 3's validation step.

Evidence retrieval

Sources may include

- ADA guidelines
- KDIGO
- drug labels
- OHDSI
- LEGEND studies
- review papers

Each edge should include

- rationale
- confidence
- evidence source

Evidence Verification

Because evidence is retrieved by an LLM, cited sources are not
guaranteed to exist or to support the claim attached to them. Before
a DAG can be versioned and handed to Agent 3, each edge's evidence
source must pass a verification step:

- source is resolvable (real guideline / label / study, not fabricated)
- cited claim is actually supported by the source content
- edges with low-confidence or unverifiable evidence are flagged for
  human review rather than silently accepted

A DAG version is only eligible for execution once all edges are
either verified or explicitly marked as expert judgment without
literature support. This distinction must be preserved in
`evidence_table.csv`.

Validation

The same agent also performs first-level DAG validation

Checks

- DAG is acyclic
- treatment precedes outcome
- mediators identified
- colliders identified
- post-index variables flagged
- unobserved confounders documented

Outputs

candidate_dag.json

dagitty.txt

adjustment_set.json

evidence_table.csv

---

## Agent 3 — Causal Execution Agent

Purpose

Translate the conceptual DAG into an executable OMOP study.

Responsibilities

Map DAG concepts

↓

OMOP concepts

↓

Covariate extraction

↓

Cohort construction

↓

Propensity score

↓

IPW

↓

ATE estimation

↓

Report

Example

Concept

Cardio-renal disease

↓

condition_occurrence

↓

SNOMED descendants

↓

baseline covariates

Another example

Prior diabetes medications

↓

drug_exposure

↓

RxNorm descendants

↓

binary features

Vocabulary Coverage Check

Automatic concept expansion via `concept_ancestor` assumes the
source vocabulary has a complete hierarchy. This does not hold
uniformly: locally-sourced vocabularies (e.g. KDC, KNHIS codes used
alongside RxNorm/SNOMED in this project) may have little or no
`concept_ancestor` coverage, so descendant-based expansion will
silently under-extract covariates for those concepts.

In the current curated database, drug concepts are mostly mapped to standard
RxNorm/RxNorm Extension ingredients. This means mapped drugs are usable, but
they often have little non-self parent hierarchy. The project therefore keeps
Athena's `concept_ancestor` table intact and creates a local
`concept_ancestor_augmented` view for analyses that need local additions. Drug
retrieval must use a seed-concept-plus-descendants pattern:

1. identify standard Drug seed concepts by exact/controlled name matching;
2. include the seed concepts themselves;
3. include descendants from `concept_ancestor_augmented` when available;
4. report both self-inclusive coverage and non-self parent coverage.

For drug classes such as SGLT2 inhibitors or DPP-4 inhibitors, hierarchy alone
is insufficient because ATC is not loaded. These classes should be represented
as explicit concept sets or data-driven enriched drug lists, not inferred from
`concept_ancestor` alone.

Before extracting covariates for a DAG node, the Execution Agent
must:

- compute the fraction of relevant source concepts covered by
  `concept_ancestor_augmented` when present, with separate reporting for
  self-inclusive coverage and non-self parent coverage
- warn and log when coverage falls below a defined threshold
- fall back to name/synonym string matching, or a manually curated
  concept set, when hierarchy coverage is insufficient
- report per-node coverage in the governance report so a low-coverage
  covariate is never silently treated as fully extracted

Estimand / Estimator Contract

The estimator used must match the estimand specified by the Causal
Planner. IPW and standardization target the ATE; matching targets
the ATT. The Execution Agent must validate that the requested
estimand and the chosen estimation method are compatible, and must
reject (or explicitly relabel) an analysis that mixes them, e.g.
reporting a matched estimate as an ATE.

Competing Risks

When the outcome is not death itself, death occurring within the
follow-up window is a competing event, not ordinary censoring. The
Execution Agent must not treat "no event, but the patient died
before it could occur" the same as "no event, patient remained
event-free and alive." At minimum, the method used (cause-specific
hazard vs. subdistribution/Fine-Gray) must be stated, and the choice
must be reported alongside the ATE.

Outputs

Patient-level dataset

person_id

A

Y

covariates

index_date

follow_up_days

Then

Propensity Score

↓

Balance diagnostics

↓

ATE

↓

Sensitivity analysis

---

# Conceptual DAG Philosophy

The DAG is NOT intended to reproduce every OMOP variable.

Instead

the DAG contains

clinical concepts

such as

Cardio-renal disease

rather than

- CKD stage 3
- CKD stage 4
- diabetic nephropathy
- renal insufficiency

The translation into OMOP occurs afterwards.

---

# Why Concept-Level DAGs?

A patient database may contain

hundreds

or

thousands

of possible variables.

Drawing them manually is impossible.

Instead

the DAG should represent

high-level causal concepts

that can later be expanded automatically
through OMOP vocabularies.

Example

Cardio-renal disease

↓

SNOMED hierarchy

↓

all descendants

↓

automatic feature extraction

---

# Relationship with OHDSI

The architecture is inspired by OHDSI's philosophy.

OHDSI generally performs

large-scale propensity adjustment

using many pre-index covariates.

The DAG therefore serves to

justify

which families of variables

should be extracted,

rather than manually selecting every variable.

---

# Clinical Question Workflow

Example

User asks

Does dapagliflozin increase urinary or genital infection risk?

↓

Planner

↓

Population

Treatment

Comparator

Outcome

↓

DAG Agent

↓

Confounders

Mediators

Colliders

↓

Execution Agent

↓

OMOP covariates

↓

PS/IPW

↓

Treatment Effect

---

# Example Candidate DAG

Treatment

↓

Urinary/genital infection

Confounders

Age

Sex

Diabetes severity

Previous urinary/genital infection

Kidney disease

Healthcare utilization

Prior diabetes medications

Calendar time

BMI (unobserved)

Mediator

Treatment

↓

Glycosuria

↓

UTI

Post-treatment variables

Treatment switch

Post-treatment HbA1c

Follow-up visits

should not be adjusted for
when estimating the total effect.

For execution, the urinary/genital infection endpoint should be implemented as
a concept set rather than a single string match. The initial concept-set target
should include urinary tract infection, cystitis, pyelonephritis, genital
candidiasis or vulvovaginal candidiasis, balanitis, and related genital
infection terms present in the curated vocabulary/data.

---

# Governance Extensions

The causal module should inherit FastOMOP governance
while adding causal-specific validation.

Examples

✓ all covariates measured before index date

✓ no mediator adjustment

✓ no post-treatment leakage

✓ positivity check — must include propensity-score distribution
overlap across treatment arms, not only the crude overall treated
percentage; a global 10–90% split can still hide non-overlapping
regions once covariates are conditioned on

✓ target trial consistency

✓ follow-up window validation

✓ estimand/estimator consistency (see Agent 3 — Estimand/Estimator
Contract)

✓ competing-risk handling declared for non-mortality outcomes

✓ vocabulary version pinned and logged for every concept mapping,
so covariate extraction is reproducible against a fixed Athena
release

✓ concept hierarchy coverage checked before automatic covariate
expansion (see Agent 3 — Vocabulary Coverage Check)

✓ forbidden SQL operations remain enforced by MCP

---

# Future Improvements

Negative control outcomes/exposures are not deferred to future work.
Without empirical calibration (in the LEGEND/OHDSI sense — testing
the adjustment method against outcomes with no plausible causal
effect), a point estimate can look precise while remaining
systematically biased, which undermines the "auditable" claim of the
whole workflow. At minimum, the Execution Agent's Sensitivity
Analysis step should run a small negative-control outcome set
alongside the primary ATE estimate. A full standalone Negative
Control Agent-Team can still come later.

Potential future agents

Study Design Agent

Sensitivity Analysis Agent (expanded)

HTE Agent

Counterfactual Explanation Agent

These can later become independent agent-teams.

---

# Expected Outputs

The causal workflow should produce

Target Trial Specification

Conceptual DAG

Evidence Table

Adjustment Set

OMOP Cohort

Covariate Matrix

Balance Diagnostics

ATE Estimate

Sensitivity Analysis

Governance Report

Complete Audit Trail

---

# Research Contribution

The contribution is **not**

"automatic DAG generation."

The contribution is

"A governed causal reasoning workflow that transforms natural-language causal questions into auditable OMOP-based treatment effect analyses."

The DAG is one intermediate representation within this workflow.

---

# Proposed Title

FastOMOP Causal Reasoning Extension

Subtitle

From Natural-Language Causal Questions to Source-Grounded DAGs and OMOP-based Real-World Evidence Generation
