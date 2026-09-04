# Concept audit — 2026-09-03

Dataset: `data/mimiciv_omop.duckdb` (MIMIC-IV v2.2 → OMOP CDM 5.3.1, OHDSI/MIMIC
ETL; local DuckDB export dated 2026-08-26).

> Dated feasibility snapshot. Counts reflect the dataset version and the
> provisional phenotype rules used on 2026-09-03. Re-run the governed audits
> for current values. Concept candidates below are **not yet human-approved**
> (`protocol.json` marks them `candidate_resolved_pending_human_review`).

## Exposure — systemic corticosteroids (`drug_exposure`, standard RxNorm ingredients)

Proposed seed set `[975125, 1506270, 1551099, 1550557, 1518254]`:

| Ingredient | concept_id | rows (+ descendants) | persons | name-matched `concept_id = 0` |
|---|---:|---:|---:|---:|
| prednisone | 1551099 | 76,032 | 15,833 | 9 |
| dexamethasone | 1518254 | 42,400 | 9,424 | 11 |
| methylprednisolone | 1506270 | 27,745 | 8,283 | 131 |
| hydrocortisone | 975125 | 23,526 | 8,550 | 221 |
| prednisolone | 1550557 | 4,053 | 1,707 | 153 |

Coverage is good. The small name-matched-unmapped counts (prednisolone 153,
hydrocortisone 221) warrant a source-value spot check but are not a blocker.

**Route restriction is essential.** These ingredients include inhaled, topical,
ophthalmic, intra-articular, and rectal formulations. Restrict to
oral / IV / IM via `route_concept_id` or the systemic clinical-drug
descendants. Fludrocortisone is excluded from the primary (mineralocorticoid).

**Indication contamination.** Dexamethasone especially is heavily used for
antiemesis, cerebral oedema, and airway/COPD indications. The non-CAP
steroid-indication exclusions in `eligibility_criteria` carry most of the
weight for this study.

## Population — community-acquired pneumonia (`condition_occurrence`)

| concept_id | name | rows (+ descendants) | persons |
|---|---|---:|---:|
| 255848 | Pneumonia | 24,798 | 17,801 |
| 257315 | Bacterial pneumonia | 4,801 | 3,913 |
| 4133224 | Lobar pneumonia | 116 | 116 |
| 4293463 | Community acquired pneumonia | **0** | **0** |

**There is no CAP-specific code in this build.** MIMIC ICD coding does not
distinguish community- from hospital-acquired pneumonia. The phenotype must be
`Pneumonia` (255848) + descendants, tightened by:

- a concurrent systemic antibiotic course started ≤ 48 h of admission;
- exclusion of "Ventilator associated pneumonia", "Hospital acquired
  pneumonia", and aspiration-pneumonia-only descendant codes;
- optionally a chest-imaging procedure ≤ 48 h (high-specificity sensitivity
  analysis).

**A PPV check on the resulting phenotype is required before this study is
committed (feasibility checklist F5).**

## Positive-control and secondary outcomes

- **Hyperglycaemia** (positive control): serum glucose `measurement` ≥ 180 mg/dL
  or a new insulin order. Glucose is among the best-covered `measurement`
  concepts (verify the exact concept_id at mapping time).
- **Invasive mechanical ventilation after time zero**: same billing-code
  limitation as the Q1 study (`../ppi_vs_h2ra_sup_mortality/concept_audit.md`) —
  ~14,662 billing-identified invasively-ventilated persons dataset-wide.

## Crude feasibility (before severity / new-user / exclusion rules)

| Metric | persons |
|---|---:|
| any pneumonia code | 17,801 |
| pneumonia + any systemic steroid (ever) | 7,980 |

After restricting to **severe** (ICU admission, or vasopressor / invasive
ventilation ≤ 48 h), **new-user** (no steroid in the prior 30 days), and
excluding non-CAP steroid indications and HAP/VAP/aspiration codes, the
eligible cohort is expected to be roughly **3,000–6,000**, of whom
**~1,000–2,500** are steroid-treated within 48 h. The exclusions bite hard
here; confirm the windowed counts with governed feasibility before relying on
power.

## Gate decision

Feasible, with the CAP-phenotype PPV as the open question. Before cohort
construction:

1. human approval of the corticosteroid concept set (with the route
   restriction) and the pneumonia phenotype (with the antibiotic tightening
   and the HAP/VAP/aspiration exclusions);
2. a PPV check on the pneumonia phenotype;
3. generate a candidate DAG, review it, compute SHD/SID vs `reference_dag.json`;
4. governed feasibility for the severe, new-user, exclusion-applied cohort.
