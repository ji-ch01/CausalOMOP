# Reproducible Causal Study Review — Completed Work Package

This document originally proposed selecting a public or accessible validation
dataset and a literature-backed causal question. That selection step is
complete:

- primary dataset: credentialed MIMIC-IV 2.2 converted to OMOP CDM 5.3.1;
- conversion reference: [OHDSI/MIMIC](https://github.com/OHDSI/MIMIC);
- first validation question: albumin plus crystalloids versus crystalloids
  alone for 28-day mortality in adults with ICU sepsis;
- study package: [studies/albumin_sepsis_28d](../studies/albumin_sepsis_28d/README.md).

The remaining work is validation, not dataset discovery: reproduce the source
study's Sepsis-3 population and time zero, approve standard OMOP phenotypes,
audit covariate availability and follow-up, assess positivity, then implement
and benchmark the prespecified estimators.

See [datasets.md](datasets.md) for access and data policy, and
[causal_roadmap.md](causal_roadmap.md) for the current scientific sequence.
