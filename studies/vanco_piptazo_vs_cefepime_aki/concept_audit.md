# Concept audit — 2026-09-03

Dataset: `data/mimiciv_omop.duckdb` (MIMIC-IV v2.2 → OMOP CDM 5.3.1, OHDSI/MIMIC
ETL; local DuckDB export dated 2026-08-26).

> Dated feasibility snapshot. Counts reflect the dataset version and the
> provisional phenotype rules used on 2026-09-03. Re-run the governed audits
> for current values. Concept candidates below are **not yet human-approved**
> (`protocol.json` marks them `candidate_resolved_pending_human_review`).

## Drugs (`drug_exposure`, standard RxNorm ingredients)

| Drug | proposed concept_id | rows (+ descendants) | persons | name-matched `concept_id = 0` |
|---|---:|---:|---:|---:|
| vancomycin | 1707687 | 196,777 | 44,542 | 124 |
| piperacillin | 1746114 | 43,940 | 15,613 | 4 |
| tazobactam | 1741122 | 44,057 | 15,624 | 4 |
| cefepime | 1748975 | 61,104 | 17,913 | 10 |

Coverage is excellent for all four.

### piperacillin-tazobactam as a single exposure

OMOP/RxNorm keeps **piperacillin and tazobactam as two ingredients**; there is
no single standard "piperacillin/tazobactam" concept in this build (only
branded drugs such as Zosyn). The near-identical piperacillin (15,613) and
tazobactam (15,624) person counts confirm that essentially all piperacillin in
MIMIC is piperacillin-tazobactam.

**Proposed definition:** the piperacillin-tazobactam arm = `drug_exposure` rows
descending from piperacillin (`1746114`). Piperacillin monotherapy is
negligible in the 2008–2019 window. A stricter alternative = clinical/branded
drug concepts whose name contains both "piperacillin" and "tazobactam".

**Route restriction:** exclude oral vancomycin (Clostridioides difficile
treatment, no nephrotoxicity signal) via `route_concept_id` or the oral
clinical-drug descendants.

## Outcome — serum creatinine (`measurement`, for the KDIGO phenotype)

| concept_id | name | rows | persons |
|---|---|---:|---:|
| 3016723 | Creatinine [Mass/volume] in Serum or Plasma | 3,708,634 | 239,896 |
| 3051825 | Creatinine [Mass/volume] in Blood | 4,327 | 3,362 |
| 3007760 | Creatinine [Mass/volume] in Arterial blood | 0 | 0 |

`3016723` is **the** serum-creatinine concept in this build (3.7 M rows). The
KDIGO phenotype is derived from it:

- baseline = lowest serum creatinine in the 7 days before / at time zero;
- KDIGO stage 2 = ≥ 2.0× baseline; stage 3 = ≥ 3.0× baseline, or ≥ 4.0 mg/dL
  with an acute rise, or new dialysis;
- confirm `unit_concept_id` (expect mg/dL) before applying the 4.0 threshold;
- urine-output KDIGO criteria added only if `outputevents`-derived measurements
  are confirmed present on this build.
- Resolve renal-replacement-therapy procedure concepts at mapping time for the
  stage-3 dialysis criterion and the MAKE secondary.

## Reminder: the endpoint is mechanism-contaminated

Piperacillin-tazobactam inhibits tubular secretion of creatinine, so it raises
*measured* serum creatinine without a fall in true GFR (pseudotoxicity). This
is a direct exposure → measured-outcome path that **no adjustment set closes**,
and a cystatin-C sensitivity analysis is not possible on MIMIC. This is
intentional for this validation case — see `protocol.json` `design_constraints`
and `reference_dag.json` `source.construction_notes`.

## Crude feasibility (before co-start / new-user / baseline-renal rules)

Among the 44,542 vancomycin persons, ever-exposure to the beta-lactam partners:

| Group | persons |
|---|---:|
| vancomycin + piperacillin-tazobactam only (ever) | 7,684 |
| vancomycin + cefepime only (ever) | 10,722 |
| both beta-lactams (ever) | 5,877 |

After the "co-started within 24 h of the first vancomycin dose" + new-user +
baseline eGFR ≥ 30 + "not already KDIGO stage 2 at time zero" rules, the
analytic cohort is expected to be roughly **3,000–6,000 per arm** — good power
for the observational association and the calibration analysis.

## Gate decision

Feasible — the strongest of the three question-bank scaffolds on data grounds.
Before cohort construction:

1. human approval of the vancomycin (IV only), piperacillin-tazobactam,
   cefepime, and creatinine concept sets and the KDIGO derivation rule;
2. map the nephrotoxin co-exposure covariate cluster (contrast, NSAIDs,
   aminoglycosides, amphotericin B, loop diuretics, ACEi/ARB, calcineurin
   inhibitors) and the neutropenia / intra-abdominal-source covariates;
3. generate a candidate DAG, review it, compute SHD/SID vs `reference_dag.json`;
4. propose and approve negative-control outcomes; run empirical calibration;
5. governed feasibility for the co-start, new-user, baseline-renal cohort.
