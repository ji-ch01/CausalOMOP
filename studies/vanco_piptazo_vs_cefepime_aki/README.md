# Vancomycin + piperacillin-tazobactam vs vancomycin + cefepime, and AKI: OMOP target-trial emulation

Question-bank **Q5** - the **headline diagnostic case**
(see [`docs/validation_question_bank.md`](../../docs/validation_question_bank.md)).

Among adult ICU patients started on empiric vancomycin plus an antipseudomonal
beta-lactam, this study emulates the effect of **piperacillin-tazobactam
versus cefepime** as the beta-lactam partner on **KDIGO stage 2 or higher
acute kidney injury within 7 days**.

- Randomized truth: **ACORN** (JAMA 2023;330(16):1557-1567,
  `doi:10.1001/jama.2023.20583`) - highest AKI stage or death by day 14,
  OR 0.95 (95 % CI 0.80-1.13). **No AKI excess with piperacillin-tazobactam.**
- Observational literature: consistently **elevated** AKI risk with the
  combination - OR 2.68 (1.83-3.91) vs vancomycin + cefepime (Luther, Crit
  Care Med 2018, `doi:10.1097/CCM.0000000000002769`); OR 2.55 (2.00-3.28) in a
  2025 network meta-analysis (`doi:10.1093/jac/dkae395`).

CausalOMOP is a general-purpose, protocol-driven tool; this is a
literature-comparison validation case. The protocol is a **draft**: candidate
concepts found by name are evidence for review, not executable concept sets.

## Why this is the case that most directly supports the paper's claim

The paper's central argument is that CausalOMOP is **auditable and honest**:
it halts on real barriers and surfaces the fragility of an estimate rather
than presenting fluent prose as a valid analysis.

Here the pipeline is **expected to produce a "positive" result** - an elevated
AKI risk difference on the piperacillin-tazobactam arm, because that
association is real *in observational data*. It is driven by two mechanisms,
neither of which a randomized trial supports as a causal kidney-injury effect:

1. **Residual confounding / channelling** - sicker patients, intra-abdominal
   sources, and those already on nephrotoxins are preferentially given
   piperacillin-tazobactam.
2. **Creatinine pseudotoxicity** - piperacillin-tazobactam inhibits tubular
   secretion of creatinine, raising *measured* serum creatinine without a fall
   in true GFR. This is a direct exposure -> measured-outcome path that **no
   adjustment set can close** (`reference_dag.json` `source.construction_notes`
   explains why it is not encoded as a mediator).

The test is whether CausalOMOP's **E-value, negative controls, and
covariate-balance diagnostics** correctly present this estimate as an
association a randomized trial (ACORN) does not support - and whether the
`causal_report.md` names both mechanisms. Success is **not** a null estimate.

## Files

- `protocol.json` - versioned study draft (target trial, KDIGO phenotype,
  estimand, concept-set candidates with resolved `concept_ids`,
  `feasibility_checklist`, `published_benchmarks` with both the RCT and
  observational numbers, `_feasibility_snapshot_2026_09_03`).
- `reference_dag.json` - reference adjustment set for SHD/SID comparison.
  **Constructed from the observational emulations' covariates + clinical
  reasoning, not transcribed from a published DAG figure.**
- `concept_audit.md` - dated (2026-09-03) feasibility snapshot: resolved
  concept candidates with counts (all four well mapped; creatinine 3.7 M
  rows), the pip-tazo-is-two-ingredients note, crude arm sizes (~7.7k / ~10.7k
  ever-exposure). **Not** a human-approved mapping.

## Open work before a run

1. **Human approval** of the concept sets (resolved candidates in
   `protocol.json`): `vancomycin` (1707687, **IV only** - exclude oral),
   `piperacillin_tazobactam` (1746114 - see the two-ingredient note in
   `concept_audit.md`), `cefepime` (1748975), and the
   `kdigo_acute_kidney_injury` derived phenotype (creatinine 3016723, baseline
   rule, staging thresholds).
2. Confirm the `unit_concept_id` on creatinine (expect mg/dL) and the
   baseline-renal eligibility rule against the data.
3. Map the nephrotoxin co-exposure covariate cluster (7 drug groups) and the
   neutropenia / intra-abdominal-source covariates.
4. Generate and clinically review a candidate DAG; compute SHD/SID vs
   `reference_dag.json`.
5. Propose and approve negative-control outcomes (e.g. outcomes with no
   plausible link to beta-lactam choice) via `causal/queries/negative_controls.py`,
   and run empirical calibration of the estimate against them.

## Known risks

- **The endpoint is mechanism-contaminated** by pseudotoxicity, and a
  cystatin-C sensitivity analysis is not possible on MIMIC. State this
  plainly - it is part of the lesson, not a defect to hide.
- **Channelling by infection source** is only partially observable.
- **Institutional antibiogram / stewardship pathway** drives the partner
  choice and is unobserved (`reference_dag.json`).

For the governed command sequence, see
[`docs/responsible_demo_guide.md`](../../docs/responsible_demo_guide.md) and
`python -m causal.cli workflow --help`.
