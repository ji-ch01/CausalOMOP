# Concept audit — 2026-07-15

Dataset audited: restricted OMOP dataset configured at runtime.

> Dated feasibility snapshot. Counts below reflect the dataset version,
> mappings, and provisional phenotype rules used on 2026-07-15. Re-run the
> governed audits for current values; do not treat this file as a current
> cohort definition or causal result.

## Conditions

The following standard SNOMED concepts occur in `condition_occurrence`:

| Candidate | concept_id | Rows | Persons | Decision |
|---|---:|---:|---:|---|
| Sepsis | 132797 | 6,283 | 4,838 | candidate seed; descendants must be evaluated |
| Septic shock | 196236 | 2,992 | 2,719 | candidate effect-modifier seed |
| Postoperative septic shock | 4308715 | 50 | 50 | review inclusion as descendant/related concept |

Text search also found organism-specific sepsis concepts. The final population
must be built from approved seed concepts plus explicit descendant rules, not
from `LIKE '%sepsis%'`.

## Albumin

Albumin administrations are present in `drug_exposure`, but all audited albumin
source values currently map to `drug_concept_id = 0` (`No matching concept`).
Examples include:

| Source value | Rows | Persons |
|---|---:|---:|
| Albumin 5% (25g / 500mL) 500mL Bottle | 8,790 | 5,055 |
| Albumin 25% (12.5g / 50mL) 50 mL Bag | 14,840 | 4,562 |
| Albumin 5% (12.5g / 250mL) 250 mL Bag | 8,974 | 4,034 |
| Albumin 25% (12.5g / 50mL) 100 mL Bag | 12,376 | 2,681 |

Therefore a standard OMOP concept set cannot yet identify the exposure. The ETL
mapping should be repaired in the configured vocabulary / ETL datasets, or a
versioned source-value phenotype must be used temporarily and flagged as such.

## Crystalloids

Lactated Ringer's administrations are likewise present but unmapped:

| Source value | Rows | Persons |
|---|---:|---:|
| Lactated Ringers Floor Stock Bag | 119,049 | 42,667 |
| Lactated Ringers 1000ml Bag | 42,428 | 21,599 |

Some sodium-chloride products are standard-mapped, but the most frequent items
include flush syringes and 100 mL injections. They cannot all be treated as
resuscitation fluid without rules on formulation, route, volume, and timing.

## Gate decision

The study is **feasible at source-value level but not yet ready as a fully
standard-concept OMOP phenotype**. Before cohort construction:

1. approve the sepsis and septic-shock seed/descendant logic;
2. map the albumin source products to suitable standard RxNorm concepts;
3. define resuscitation crystalloids, excluding flushes, irrigation, nasal,
   ophthalmic, enema, and drug-carrier products;
4. validate ICU time zero and the 24-hour exposure window;
5. rerun counts and compare them with the published benchmark.

## Provisional feasibility run

Using the first adult ICU stay, the sepsis seed `132797` plus descendants, a
24-hour exposure window, and unvalidated source-value rules produced:

| Metric | Count |
|---|---:|
| Eligible persons | 3,048 |
| Albumin plus crystalloid | 416 |
| Crystalloid only | 2,496 |
| Neither recognized exposure | 130 |
| Deaths within 28 days | 793 |

These are feasibility counts only. The large difference from the published
cohort (approximately 18,421) is evidence that a diagnosis-based OMOP sepsis
phenotype does not reproduce the paper's derived Sepsis-3 population.
