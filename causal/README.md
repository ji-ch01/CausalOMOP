# Causal Prototype

This folder contains project-owned causal workflow code.

Current implemented code:

```text
causal/data_builder/cohort.py
```

The cohort builder creates patient-level causal datasets with:

```text
person_id
A
Y
index_date
follow_up_days
censored
```

## Current Capabilities

- untreated-control cohorts;
- active-comparator cohorts;
- condition and mortality outcomes;
- basic feasibility scans;
- reusable `CohortDefinition` objects.

## Current Limitations

- concept matching is still string-based;
- explicit outcome concept sets are not implemented yet;
- covariate extraction is not implemented yet;
- no propensity score model or effect estimator is implemented yet;
- SNUH synthetic data is not the final causal inference dataset.

## Planned Direction

The causal workflow should become:

```text
CausalStudySpec
  -> OMOP concept sets
  -> cohort SQL
  -> baseline covariates
  -> estimator
  -> diagnostics
  -> report
```

See `docs/causal_architecture.md` and `document_design.md`.

