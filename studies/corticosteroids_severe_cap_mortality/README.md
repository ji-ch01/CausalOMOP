# Corticosteroids in severe community-acquired pneumonia: OMOP target-trial emulation

Question-bank **Q8** (see [`docs/validation_question_bank.md`](../../docs/validation_question_bank.md)).

Among adults admitted to the ICU with severe community-acquired pneumonia,
this study emulates the effect of **starting a systemic corticosteroid within
48 hours of admission** versus **no systemic corticosteroid** on **in-hospital
all-cause mortality**, anchored to the CAPE COD randomized trial
(NEJM 2023;388(21):1931-1941, `doi:10.1056/NEJMoa2215145`) and a 7-trial
meta-analysis (Crit Care 2023;27:274, `doi:10.1186/s13054-023-04561-z`).

CausalOMOP is a general-purpose, protocol-driven tool; this is a
literature-comparison validation case. The protocol is a **draft**: candidate
concepts found by name are evidence for review, not executable concept sets.

## Why this case is in the bank

This is the bank's clearest **sensitivity** test. The RCTs and the
meta-analysis agree on a real mortality benefit in the severe stratum
(CAPE COD 28-day mortality 6.2 % vs 11.9 %; meta-analysis RR 0.61). If the
pipeline returns a null here while recovering the near-null in Q1
(`../ppi_vs_h2ra_sup_mortality/`) and the confounded positive in Q5
(`../vanco_piptazo_vs_cefepime_aki/`), the triad shows the system is
calibrated in both directions rather than merely conservative.

New hyperglycaemia within 7 days is built in as a **positive control** - a
real, expected corticosteroid effect that must come out clearly positive.

## Files

- `protocol.json` - versioned study draft (target trial, severe-CAP
  phenotype, estimand, concept-set candidates with resolved `concept_ids`,
  `feasibility_checklist`, `published_benchmarks`,
  `_feasibility_snapshot_2026_09_03`).
- `reference_dag.json` - reference adjustment set for SHD/SID comparison.
  **Constructed from the CAP corticosteroid trials' covariates + clinical
  reasoning, not transcribed from a published DAG figure** - see its
  `source.description`.
- `concept_audit.md` - dated (2026-09-03) feasibility snapshot: resolved
  concept candidates with counts, the absent CAP-specific code, crude cohort
  sizes. **Not** a human-approved mapping.

## Open work before a run

1. **Human approval** of the `systemic_corticosteroid` concept set (resolved
   candidates in `protocol.json`; **route restriction to oral/IV/IM is
   essential** - the ingredients include inhaled/topical/ophthalmic) and the
   severe-CAP phenotype (`Pneumonia` 255848 + descendants + concurrent
   antibiotic + severity marker, minus HAP/VAP/aspiration-only codes). There
   is **no CAP-specific code** in this build - see `concept_audit.md`.
2. **A positive-predictive-value check on the pneumonia phenotype** before
   committing - this is the main feasibility risk (F5).
3. Confirm the non-CAP steroid-indication exclusions (COPD/asthma
   exacerbation primary, maintenance steroids, adrenal insufficiency,
   transplant, steroid-containing chemotherapy).
4. Generate and clinically review a candidate DAG; compute SHD/SID vs
   `reference_dag.json`.
5. Decide whether the COPD-relaxed-exclusion and per-protocol (>=3-day
   steroid) sensitivity analyses run alongside the primary.

## Known risks

- **Confounding by indication runs toward the null or toward harm** here
  (steroids given to the sicker patient), so a recovered protective effect is
  a conservative demonstration.
- **Influenza pneumonia** flips the expected direction; identified pathogen is
  a covariate and often only partially known at time zero.
- **Radiographic extent** of infiltrates drives the severity impression and
  is not in the OMOP tables (`reference_dag.json` marks it unobserved).

For the governed command sequence, see
[`docs/responsible_demo_guide.md`](../../docs/responsible_demo_guide.md) and
`python -m causal.cli workflow --help`.
