# Albumin for sepsis: OMOP replication

This study is the first end-to-end literature-comparison example for
CausalOMOP, not a description of what CausalOMOP is built to study --
CausalOMOP is a general-purpose, protocol-driven tool for any causal question
over OMOP CDM data (see [README.md's References](../../README.md#references)).
It translates the published MIMIC-IV v2.2 analysis (DOI
`10.1371/journal.pdig.0000721`) into an OMOP CDM workflow so the pipeline's
DAG, concept mapping, and feasibility output can be compared against that
published study's design and results.

The protocol is intentionally marked as a draft. Candidate concepts found by
name are evidence for review, not executable concept sets. Approve seed concepts
and descendant rules before cohort SQL is generated.

Reference artifacts in this directory:

- `protocol.json`: versioned study draft;
- `reference_dag.json`: published/reference DAG used for structural metrics;
- `reference_concept_mappings.json`: reference mapping benchmark;
- `icu_time_zero.json`: time-zero audit notes;
- `concept_audit.md`: dated feasibility snapshot, not a current causal result.

The active blockers are a reproducible Sepsis-3 population, validated ICU time
zero, approved albumin/crystalloid phenotypes, baseline covariate coverage,
follow-up/censoring rules, and the absence of an implemented estimator.

## Concept audit

Run from the repository root with the virtual environment active:

```powershell
python -m causal.cli concepts "sepsis" --limit 20
python -m causal.cli concepts "septic shock" --limit 20
python -m causal.cli concepts "albumin" --limit 30
python -m causal.cli concepts "sodium chloride" --limit 30
python -m causal.cli concepts "lactated ringer" --limit 30
python -m causal.cli concept-usage condition "sepsis" --limit 30
python -m causal.cli concept-usage drug "albumin" --limit 30
python -m causal.cli concept-usage drug "sodium chloride" --limit 30
python -m causal.cli source-usage drug "albumin" --limit 30
python -m causal.cli source-usage drug "ringer" --limit 30
```

The next gate is human approval of concept sets. Only after that gate should the
pipeline count the eligible population and treatment arms.

For the full governed command sequence, use the
[responsible demo guide](../../docs/responsible_demo_guide.md) and
`python -m causal.cli workflow --help`.
