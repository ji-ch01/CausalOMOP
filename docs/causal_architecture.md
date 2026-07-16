# Causal Architecture

> This file describes the earlier prototype. The clarified normative architecture is
> [CausalOMOP Causal Roadmap](causal_roadmap.md). In particular, causal design and
> identification must precede OMOP extraction; the database must not determine
> the scientific DAG or adjustment set.

The causal extension should transform a natural-language causal question into an
auditable OMOP-based study workflow. SQL is necessary, but it is not sufficient:
the system also needs target trial specification, literature-grounded causal
assumptions, estimation, diagnostics, and reporting.

## Working Principle

The Causal Agent Team should not be designed as a completely separate module
from the existing NL-to-SQL system. It should be connected to the NL-to-SQL /
SQL execution layer, because causal analyses still require structured
information from OMOP CDM through SQL queries.

The NL-to-SQL layer can serve as the data-access layer for causal work:

- construct candidate cohorts;
- extract baseline and follow-up variables;
- check OMOP concepts and concept-set coverage;
- run feasibility and positivity checks;
- generate the analytic datasets required for causal inference.

```text
User question
  -> descriptive vs causal routing
  -> Causal question and Target Trial specification
  -> causal estimand and literature-grounded DAG
  -> deterministic identification and human review
  -> controlled OMOP mapping and feasibility extraction
  -> causal estimation
  -> auditable report
```

## Current Implemented State

Implemented:

- `causal/data_builder/cohort.py`;
- active-comparator and untreated-control cohort construction;
- `A`, `Y`, `index_date`, follow-up, and censoring fields;
- feasibility scans for candidate cohorts.

Not implemented yet:

- causal intent routing;
- causal planner agent;
- DAG/evidence agent;
- concept-set based outcome definitions;
- covariate extraction;
- CATE estimation;
- TMLE with SuperLearner-style nuisance modeling;
- g-computation;
- final causal report generation.

## Validation Strategy

The project is no longer centered on a single hard-coded causal question. A
previous working example was:

> Does dapagliflozin, compared with sitagliptin, increase the 90-day risk of
> urinary or genital infection among patients with Type 2 Diabetes Mellitus?

This remains a useful design example, but the next validation work should
identify a more suitable public dataset, ideally one already used in a published
causal study. The objective is to reuse or approximate an existing causal
question and later compare pipeline output with published results.

The next review should cover:

- reproducible causal studies;
- public datasets suitable for causal reasoning;
- causal questions testable with those datasets;
- required variables and OMOP compatibility;
- expected literature results for future validation.

## Proposed Data Model

The causal planner should produce a structured object before final cohort and
covariate extraction:

```json
{
  "question_type": "causal",
  "source_question": "...",
  "population": "...",
  "treatment": "...",
  "comparator": "...",
  "outcome": "...",
  "follow_up_days": 90,
  "design": "active_comparator_new_user",
  "primary_estimand": "CATE",
  "secondary_estimands": ["ATE", "ATT"],
  "candidate_effect_modifiers": [
    "age",
    "sex",
    "baseline renal function",
    "baseline disease severity"
  ],
  "estimators": [
    "TMLE with SuperLearner-style nuisance models",
    "g-computation"
  ],
  "baseline_window_days": 365,
  "washout_days": 30,
  "covariates": [
    "age",
    "sex",
    "prior urinary/genital infection",
    "renal disease",
    "diabetes severity",
    "prior diabetes medications",
    "healthcare utilization"
  ]
}
```

This should be mapped to OMOP concept sets and SQL only after the design,
estimand, candidate effect modifiers, time windows, and measurement rules are
explicit.

## Suggested Agent Roles

### Causal Planner

Converts the user question into a target trial specification:

- population;
- treatment;
- comparator;
- outcome;
- index date;
- eligibility;
- follow-up;
- estimand;
- design choice;
- known limitations.

### DAG and Evidence Agent

Defines the causal adjustment logic from verifiable evidence. The DAG should be
generated from existing scientific literature, such as PubMed-indexed articles,
clinical guidelines, drug labels, or other reliable biomedical sources, with an
LLM helping retrieve, summarize, and organize the evidence.

Responsibilities:

- confounders;
- mediators;
- colliders;
- unobserved variables;
- pre-index versus post-index measurement windows;
- evidence or expert-judgment justification;
- source verification and citation tracking;
- documented adjustment-set proposal.

The agent must distinguish evidence-supported edges from expert assumptions.
Unverified sources or unsupported claims should be flagged for human review
instead of silently entering the executable adjustment set.

### Causal Execution Agent

Turns the study specification into executable OMOP work:

- concept-set mapping;
- cohort SQL;
- covariate SQL;
- feasibility checks;
- nuisance outcome and treatment models;
- CATE estimation;
- TMLE/SuperLearner and g-computation;
- diagnostics;
- report.

## Output Table Contract

The cohort/covariate builder should eventually produce a table shaped like:

```text
person_id
A
Y
index_date
follow_up_days
censored
age
sex
baseline_hba1c
baseline_egfr
prior_infection
prior_medication_count
healthcare_utilization
effect_modifier_*
...
```

Rules:

- `A` is treatment assignment;
- `Y` is outcome occurrence during follow-up;
- covariates must be measured before index date unless explicitly modeled as
  mediators or post-treatment variables;
- post-index variables must not enter the confounder adjustment set.
- candidate effect modifiers must be pre-index or baseline variables unless the
  estimand explicitly targets a post-baseline dynamic regime.

## Estimation Objective

The primary objective is inference of conditional average treatment effects
(CATE), not only a single population-average treatment effect. The workflow
should support:

- CATE estimation for predefined clinically meaningful subgroups;
- model-based CATE estimation across baseline covariates;
- TMLE with SuperLearner-style nuisance modeling for outcome and treatment
  mechanisms;
- g-computation as a transparent complementary estimator;
- ATE or ATT summaries as secondary estimands when useful for comparison with
  published studies.

## Why SNUH Is Not the Primary Causal Dataset

The SYN-SNUH synthetic data is useful for pipeline testing, but it has known
limitations for causal inference:

- synthetic temporal artifacts;
- missing or insufficiently detailed BMI/body-size information;
- limited adverse-event signal;
- incomplete observation mapping;
- limited vital-sign coverage;
- possible prevalent-user patterns.

It can be used to test mechanics, but causal conclusions should require a more
appropriate OMOP dataset.

## Integration Point

The practical integration point is a clinical query entrypoint such as:

```text
run_clinical_query(user_query)
  |
  +-- descriptive -> run_omop_query(user_query)
  |
  +-- causal -> run_causal_omop_query(user_query)
```

`run_causal_omop_query` should reuse the existing semantic and database
capabilities instead of bypassing them. The causal layer should request
controlled OMOP extractions and then perform causal planning, DAG construction,
estimation, and reporting on top of those extracted datasets.
