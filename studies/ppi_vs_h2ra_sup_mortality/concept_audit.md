# Concept audit — 2026-09-03

Dataset: `data/mimiciv_omop.duckdb` (MIMIC-IV v2.2 → OMOP CDM 5.3.1, OHDSI/MIMIC
ETL; local DuckDB export dated 2026-08-26).

> Dated feasibility snapshot. Counts reflect the dataset version and the
> provisional phenotype rules used on 2026-09-03. Re-run the governed audits
> for current values. This is **not** a current cohort definition or a causal
> result, and the concept candidates below are **not yet human-approved** —
> `protocol.json` marks every entry `candidate_resolved_pending_human_review`.

## Drug exposures (`drug_exposure`, standard RxNorm ingredients)

### Proton pump inhibitors — proposed seed set `[948078, 923645, 904453, 929887, 911735, 19039926]`

| Ingredient | concept_id | rows (ingredient + descendants) | persons | name-matched `concept_id = 0` |
|---|---:|---:|---:|---:|
| pantoprazole | 948078 | 163,531 | 43,389 | 15 |
| omeprazole | 923645 | 87,663 | 36,251 | 57 |
| esomeprazole | 904453 | 1,442 | 956 | 34 |
| lansoprazole | 929887 | 9,891 | 5,965 | 35 |
| rabeprazole | 911735 | 19 | 14 | 18 |
| dexlansoprazole | 19039926 | 8 | 6 | 29 |

Coverage is effectively complete: fewer than 60 `concept_id = 0` rows name any
of these ingredients. Pantoprazole dominates (expected for a US ICU formulary).

### Histamine-2 receptor antagonists — proposed seed set `[953076, 961047, 997276, 950696]`

| Ingredient | concept_id | rows | persons | name-matched `concept_id = 0` |
|---|---:|---:|---:|---:|
| famotidine | 953076 | 52,084 | 27,330 | 4 |
| ranitidine | 961047 | 49,818 | 19,759 | 10 |
| cimetidine | 997276 | 191 | 126 | 0 |
| nizatidine | 950696 | 0 | 0 | 4 |

Ranitidine is retained (in routine use across the 2008–2019 data window;
withdrawn 2020). Nizatidine is absent — kept in the set only for completeness.

## Population phenotype — invasive mechanical ventilation (`procedure_occurrence`)

**This is the study's main feasibility limitation.**

Invasive mechanical ventilation in this OMOP build is carried only as **billing
procedure codes**:

| concept_id | name | vocab | rows | persons |
|---|---|---|---:|---:|
| 2008008 | Continuous invasive mechanical ventilation < 96 h | ICD9Proc | 7,036 | 6,358 |
| 2008009 | Continuous invasive mechanical ventilation ≥ 96 h | ICD9Proc | 4,064 | 3,623 |
| 2008007 | Continuous invasive mechanical ventilation, unspecified | ICD9Proc | 4 | 4 |
| 2788038 | Respiratory Ventilation, > 96 h | ICD10PCS | 2,350 | 2,049 |
| 2788037 | Respiratory Ventilation, 24–96 h | ICD10PCS | 2,863 | 2,581 |
| 2788036 | Respiratory Ventilation, < 24 h | ICD10PCS | 1,533 | 1,482 |

**Union: 14,662 distinct persons.**

Findings:

- SNOMED `37158404` "Invasive mechanical ventilation" has **0 rows** in this
  build — do not use it as the seed.
- SNOMED `4230167` "Artificial ventilation" + descendants = 9,388 persons, but
  that set includes non-invasive ventilation.
- `device_exposure` carries no ventilator (only "Therapeutic bed" and "Insulin
  pump").
- The true invasively-ventilated MIMIC-IV population is ≈ 25,000–30,000. The
  `procedureevents` "Invasive Ventilation" itemid and the chartevents
  ventilator-setting rows do **not** map to a standard ventilation procedure
  concept here. The billing-code cohort is real but skews toward longer /
  billed courses.

**Implication:** run Q1 on the billing-code cohort and state the undercount as
an ETL-coverage limitation (analogous to the unstaged `inputevents` infusion
gap). Prespecify a sensitivity analysis on SNOMED `4230167` + descendants.

## Secondary outcome — upper-GI bleeding (`condition_occurrence`)

| concept_id | name | rows (+desc) | persons |
|---|---|---:|---:|
| 192671 | Gastrointestinal hemorrhage | 11,877 | 8,558 |
| 4103703 | Melena | 1,726 | 1,528 |
| 26727 | Hematemesis | 1,601 | 1,380 |

Exploratory only. Pair with a red-cell transfusion or an upper-endoscopy
procedure to raise PPV.

## Crude feasibility (no windowing / new-user / single-first-agent rules yet)

Among the 14,662 billing-identified invasively-ventilated persons, ever-exposure:

| Group | persons |
|---|---:|
| PPI only (ever) | 4,040 |
| H2RA only (ever) | 3,958 |
| both (ever) | 5,743 |
| neither | 921 |

After the new-user + "first prophylaxis class within 48 h of ventilation" +
first-ICU-stay rules, the analytic cohort is expected to be roughly
**6,000–9,000** with a fairly even PPI/H2RA split — adequate power for a
near-null (PEPTIC).

## Gate decision

Feasible. Before cohort construction:

1. human approval of the PPI, H2RA, ventilation, and UGIB concept sets
   (`workflow concept-set-map`, or the Streamlit mapping tab);
2. decide the ventilation seed (billing codes as primary, SNOMED 4230167
   sensitivity) and record the undercount limitation;
3. generate a candidate DAG, review it, compute SHD/SID vs `reference_dag.json`;
4. run governed feasibility for the windowed, new-user cohort counts.
