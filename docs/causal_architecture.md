# Causal Architecture

The causal extension should transform a natural-language causal question into an
auditable OMOP-based study workflow. SQL is necessary, but it is not sufficient:
the system needs a target trial specification before cohort SQL is generated.

## Working Principle

Use FastOMOP as the OMOP access and NL-to-SQL foundation, then add a causal mode
inside or next to the NL-to-SQL team.

```text
User causal question
  |
  v
Intent router
  |
  v
Causal Planner
  |
  v
CausalStudySpec
  |
  v
OMOP concept mapping
  |
  v
Cohort and covariate SQL
  |
  v
Estimator and diagnostics
  |
  v
Causal report
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
- propensity score modeling;
- IPW/standardization/matching;
- final causal report generation.

## Target Causal Question

Primary planned question:

> Does dapagliflozin, compared with sitagliptin, increase the 90-day risk of
> urinary or genital infection among patients with Type 2 Diabetes Mellitus?

Suggested design:

- population: patients with Type 2 Diabetes Mellitus;
- treatment: dapagliflozin;
- comparator: sitagliptin;
- design: active-comparator new-user;
- index date: first exposure to treatment or comparator;
- follow-up: 90 days;
- outcome: urinary or genital infection concept set;
- estimand: ATE or ATT, chosen explicitly before estimation.

## Proposed Data Model

The causal planner should produce a structured object before SQL generation:

```json
{
  "question_type": "causal",
  "population": "Type 2 Diabetes Mellitus",
  "treatment": "dapagliflozin",
  "comparator": "sitagliptin",
  "outcome": "urinary or genital infection",
  "follow_up_days": 90,
  "design": "active_comparator_new_user",
  "estimand": "ATE",
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

This should be mapped to OMOP concept sets and SQL only after the design is
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

Defines the causal adjustment logic:

- confounders;
- mediators;
- colliders;
- unobserved variables;
- pre-index versus post-index measurement windows;
- evidence or expert-judgment justification.

### Causal Execution Agent

Turns the study specification into executable OMOP work:

- concept-set mapping;
- cohort SQL;
- covariate SQL;
- feasibility checks;
- treatment model;
- effect estimation;
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
...
```

Rules:

- `A` is treatment assignment;
- `Y` is outcome occurrence during follow-up;
- covariates must be measured before index date unless explicitly modeled as
  mediators or post-treatment variables;
- post-index variables must not enter the confounder adjustment set.

## Why SNUH Is Not the Primary Causal Dataset

The SNUH synthetic data is useful for pipeline testing, but it has known
limitations for causal inference:

- synthetic temporal artifacts;
- missing BMI;
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
capabilities instead of bypassing them.

