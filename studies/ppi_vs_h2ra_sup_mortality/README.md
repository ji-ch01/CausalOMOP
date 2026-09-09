# PPI vs H2RA for stress-ulcer prophylaxis: OMOP target-trial emulation

Question-bank **Q1** (see [`docs/validation_question_bank.md`](../../docs/validation_question_bank.md);
= candidate **B1** in [`docs/validation_study_candidates.md`](../../docs/validation_study_candidates.md)).

Among adult ICU patients on invasive mechanical ventilation, this study
emulates the effect of **proton pump inhibitor** versus **histamine-2 receptor
antagonist** stress-ulcer prophylaxis on **in-hospital all-cause mortality**,
anchored to the PEPTIC randomized trial (JAMA 2020;323(7):616-626,
`doi:10.1001/jama.2019.22190`).

CausalOMOP is a general-purpose, protocol-driven tool; this is a
literature-comparison validation case, not a description of its scope. The
protocol is a **draft**: candidate concepts found by name are evidence for
review, not executable concept sets.

## Why this is the lowest-risk case to run first

- Active-comparator new-user design: both arms share the same indication, so
  confounding by indication is controlled by construction and positivity is
  generally satisfied.
- Both exposures are common, well-mapped oral/IV prescriptions; the population
  (invasive mechanical ventilation) is a staged procedure. Nothing here
  depends on the `inputevents` ETL gap.
- Large N (~15,000-20,000 ventilated adult first ICU stays).
- The expected answer is a **near-null** (PEPTIC in-hospital mortality 18.3 %
  vs 17.5 %, P=.054). Recovering a confidence interval that covers 0 is a
  clean **specificity** demonstration: the pipeline does not manufacture an
  effect.

## Files

- `protocol.json` - versioned study draft (target trial, eligibility, estimand,
  concept-set candidates with resolved `concept_ids`, `feasibility_checklist`
  against F1-F7, `published_benchmarks`, `_feasibility_snapshot_2026_09_03`).
- `reference_dag.json` - reference adjustment set for SHD/SID comparison
  against a generated candidate DAG. **Constructed from PEPTIC's covariates +
  clinical reasoning, not transcribed from a published DAG figure** (PEPTIC
  publishes none) - see its `source.description`.
- `concept_audit.md` - dated (2026-09-03) feasibility snapshot: resolved
  concept candidates with row/person counts, the ventilation-undercount
  finding, and crude arm sizes. **Not** a human-approved mapping.

## Open work before a run

1. **Human approval** of the four concept sets. The `concept_ids` in
   `protocol.json` are resolved candidates (`candidate_resolved_pending_human_review`),
   not approvals - confirm them via `workflow concept-set-map` or the Streamlit
   mapping tab. Counts are in `concept_audit.md`.
2. **Decide the ventilation seed.** Invasive mechanical ventilation is present
   only as billing procedure codes (~14,662 persons; true population
   ~25k-30k - the `procedureevents` itemid does not map to a standard concept
   in this build). Use the billing codes as primary and record the undercount
   as an ETL-coverage limitation; prespecify a SNOMED `4230167` sensitivity
   analysis.
3. Generate and clinically review a candidate DAG; compute SHD/SID vs
   `reference_dag.json` (`causal/agents/dag_metrics.py`).
4. Confirm the 90-day new-user look-back and the "exactly one prophylaxis
   class as the first agent" rule with governed feasibility counts.
5. Decide `care_site` adjustment vs stratification, and whether the
   per-protocol (censor-at-switch) sensitivity analysis runs alongside the
   ITT-analog primary.

## Known risks

- **Ventilation undercount** (see step 2) - the main data limitation; report it.
- **Crossover.** PEPTIC had substantial crossover between classes; the primary
  is an intention-to-treat analog. Report the per-protocol sensitivity
  analysis next to it.
- **Upper-GI-bleeding secondary** has low PPV from a condition code; it is
  exploratory only and must be paired with a transfusion or endoscopy
  procedure.
- **Site-level prescribing culture** is only partially captured by the
  `care_site` label (`reference_dag.json` marks the rest unobserved).
- **`care_site`, `calendar_year`, and `admission_type` are not currently
  extractable as covariates.** A live end-to-end run found the
  covariate-extraction system only supports `condition`/`drug`/`procedure`/
  `measurement`/`observation` domains, with no `visit_occurrence`-derived
  covariate path yet (see
  [`docs/eligibility_refinements_design.md`](../../docs/eligibility_refinements_design.md)
  section 4). This is an engineering-coverage gap, not unmeasured data:
  these three are recorded in the OMOP build but must be dropped from the
  adjustment set until the covariate system is extended. `care_site` is
  PEPTIC's own randomization unit and `calendar_year` tracks the PPI-uptake
  trend across MIMIC-IV's window (see `docs/validation_question_bank.md`'s
  Q1 entry) -- losing them is a real, not cosmetic, reduction in confounding
  control and must be stated in any report built from a run missing them.

For the governed command sequence, see
[`docs/responsible_demo_guide.md`](../../docs/responsible_demo_guide.md) and
`python -m causal.cli workflow --help`.
