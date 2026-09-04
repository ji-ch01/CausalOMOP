# Validation study candidates

CausalOMOP is a general-purpose, protocol-driven causal-inference tool; it is
not scoped to any one disease, drug, or outcome. Rigor is demonstrated by
hand-picking a small number of causal questions that already have a
published, literature-grounded analysis -- ideally one anchored to a
randomized trial -- and running the actual pipeline against each one so its
output can be compared to that published ground truth (SHD/SID for DAG
structure, recall@k/MRR for concept mapping, confidence-interval overlap for
the estimate). See [README.md's References](../README.md#references) and the
[causal roadmap](causal_roadmap.md)'s Validation strategy.

This document is the candidate landscape for that choice: which questions
have a defensible reference truth **and** are runnable end-to-end on the
current MIMIC-IV -> OMOP CDM 5.3.1 conversion (OHDSI/MIMIC ETL). It is a
planning reference, not a commitment. Three `studies/` packages are
scaffolded today --
[`ppi_vs_h2ra_sup_mortality`](../studies/ppi_vs_h2ra_sup_mortality/README.md),
[`corticosteroids_severe_cap_mortality`](../studies/corticosteroids_severe_cap_mortality/README.md),
and [`vanco_piptazo_vs_cefepime_aki`](../studies/vanco_piptazo_vs_cefepime_aki/README.md)
-- each a `protocol.json` + reference DAG draft pending human review.

For the *operational* set -- ten concrete questions, each written as a target
trial with a per-question ETL-feasibility check, expected result, and run
priority for the workshop paper's multi-case evaluation -- see
[`validation_question_bank.md`](validation_question_bank.md). B1 and B2 below
are carried into it as Q1 and Q2.

> Every citation below was checked against its publisher or PubMed landing
> page in August 2026. Entries flagged **[verify pages]** still need their
> volume/page numbers confirmed. Re-verify all of them against the DOI
> landing page before they are relied on in a writeup.

---

## How a validation question is selected

Two filters, applied in order. A candidate must clear both.

### Filter 1 -- the reference truth to compare against

| Tier | Reference | What the run tests |
|---|---|---|
| **1 -- RCT** | A randomized trial (or meta-analysis of RCTs) gives the effect. The pipeline must recover it from observational data. | Does the pipeline's interval cover the randomized estimate? Strongest signal. |
| **2 -- published EHR emulation** | A paper ran the same target-trial emulation and published its expert DAG, concept definitions, and estimate. | Stage-by-stage comparison to that paper's artifacts (SHD/SID, recall@k, CI overlap). |
| **3 -- control sets** | Exposure-outcome pairs with a known-null (negative) or known-strong (positive) effect. | Calibration across many pairs, not a single headline question. |

The best candidate combines Tier 1 and Tier 2: a published EHR emulation
that was itself benchmarked against a trial. Tier 3 is a complement to add to
whichever headline question is chosen (`causal/queries/negative_controls.py`).

### Filter 2 -- MIMIC-IV -> OMOP feasibility checklist

A single failure disqualifies the candidate on the current data.

1. **Exposure is a prescription drug** (`drug_exposure`, ~96.8% mapped) **or
   a procedure/condition with a real code** -- *not* an ICU infusion. The
   OHDSI/MIMIC ETL does not stage `inputevents`, so fluids, vasopressors,
   titrated sedation, IV insulin, and blood products are absent or
   order-only. See
   [`vocabulary_mapping_coverage.md`](vocabulary_mapping_coverage.md)'s
   "Infusion administrations" note for the verified ETL-coverage findings.
2. **Clean index date and a point / short-window treatment** (new-user
   design), not time-varying titration.
3. **Outcome reliably captured**: in-hospital / 28-day / 90-day mortality
   (best), a condition code with acceptable PPV, or a lab threshold.
4. **Simple population phenotype** -- definable from a diagnosis or
   demographic, not a derived severity score. Evidence: in a MIMIC-IV audit a
   diagnosis-code Sepsis-3 proxy gives roughly 3,000 patients where a
   reconstructed Sepsis-3 cohort gives roughly 18,000.
5. **Adequate N** -- at least a few hundred treated.
6. **A published expert DAG or adjustment set** exists for the structural
   comparison.
7. **Confounders measurable in MIMIC** (first-day labs and vitals,
   demographics). MIMIC is rich here; this rarely fails.

---

## Candidate summary

| # | Question | Tier-1 anchor | Filter 2 | Verdict |
|---|---|---|---|---|
| B1 | PPI vs H2RA for stress-ulcer prophylaxis in ventilated ICU adults -> mortality; GI bleeding (secondary) | PEPTIC | pass | **Recommended headline case** |
| B2 | Early IV acetaminophen in sepsis -> mortality / organ-support-free days | ASTER (also HEAT) | pass (sepsis phenotype caveat) | **Recommended second case** |
| B3 | Hydrocortisone in septic shock -> 90-day mortality | ADRENAL / APROCCHSS | warn | Feasible drug; contradictory RCTs, shock phenotype |
| B4 | Accelerated vs standard RRT initiation in AKI -> 90-day mortality | STARRT-AKI / AKIKI / ELAIN / IDEAL-ICU | warn | Procedure is in ETL; strategy-based exposure, KDIGO phenotype |
| B5 | Early dexmedetomidine sedation vs usual care -> 90-day mortality | SPICE III | warn | Sedation is infusion-titrated; `drug_exposure` has orders only |
| B6 | First-line antihypertensive class -> AMI / stroke / HF (LEGEND-HTN) | LEGEND-HTN reference HRs | **fail** | Ideal method target, wrong dataset (ICU, short follow-up) |
| -- | Albumin + crystalloids vs crystalloids in sepsis -> 28-day mortality | ALBIOS + pooled meta-analysis | **fail** | Future work, blocked on `inputevents` ETL scope |
| -- | Early acetaminophen in critically ill stroke -> mortality | none (no stroke RCT) | pass | In-repo candidate; weak reference truth -- prefer B2 |

---

## Candidate detail

### B1 -- PPI vs H2RA for stress-ulcer prophylaxis (recommended headline)

- **Tier 1:** PEPTIC Investigators / ANZICS CTG. *Effect of Stress Ulcer
  Prophylaxis With Proton Pump Inhibitors vs Histamine-2 Receptor Blockers on
  In-Hospital Mortality Among ICU Patients Receiving Invasive Mechanical
  Ventilation: The PEPTIC Randomized Clinical Trial.* JAMA 2020;323(7):616-626.
  `doi:10.1001/jama.2019.22190`. 26,982 patients, 50 ICUs; in-hospital
  mortality 18.3% (PPI) vs 17.5% (H2RB), P = .054.
- **Tier 1 supporting:** the *BMJ* Rapid Recommendations on stress-ulcer
  prophylaxis in the ICU and its linked network meta-analysis (2020).
  **[verify authors, venue, and pages]**
- **Tier 2:** *Comparison of proton pump inhibitors and histamine 2 receptor
  antagonists for stress ulcer prophylaxis in the intensive care unit.* Sci
  Rep 2021;11:18424. `doi:10.1038/s41598-021-98069-7` **[verify article no.]**.
  Retrospective ICU cohort with an explicit adjustment set.
- **Tier 2 supporting:** *Proton Pump Inhibitors Versus Histamine-2 Receptor
  Blockers for Stress Ulcer Prophylaxis on In-hospital Mortality Among
  Intensive Care Unit Patients Hospitalized for Major Adverse Cardiovascular
  and Cerebrovascular Events: Retrospective Cohort Study.* Clinical
  Therapeutics 2024 (PII `S0149-2918(24)00195-4`). **[verify full citation]**
- **Feasibility:** clears all seven. Both exposures are prescription drugs,
  well mapped. Active-comparator new-user design minimizes confounding by
  indication and generally satisfies positivity. Population = adults on
  invasive ventilation within 24 h of first ICU admission (derivable) or
  simply first ICU stay. Mortality outcome is reliable; "clinically important
  GI bleeding" is a condition code (lower PPV -- keep it secondary). Large N.
- **Pipeline stages exercised:** active-comparator DAG; concept mapping for
  two drug classes plus a bleeding outcome; feasibility and univariate
  positivity; g-computation and TMLE on a near-null effect; negative controls
  (PPI/H2RA share many from the OHDSI sets); E-value.

### B2 -- Early IV acetaminophen in sepsis (recommended second)

- **Tier 1:** Ware LB, et al. *Acetaminophen for Prevention and Treatment of
  Organ Dysfunction in Critically Ill Patients With Sepsis: The ASTER
  Randomized Clinical Trial.* JAMA 2024;332(5):390-400.
  `doi:10.1001/jama.2024.8772`. 447 patients; no difference in days alive and
  free of organ support; less ARDS (2.2% vs 8.5%); mortality 17% vs 22% (NS).
- **Tier 1 alternative anchor (fever / suspected infection):** Young P, et
  al. *Acetaminophen for Fever in Critically Ill Patients with Suspected
  Infection (HEAT).* N Engl J Med 2015;373(23):2215-2224.
  `doi:10.1056/NEJMoa1508375`. 700 patients; no difference in ICU-free days
  or 28/90-day mortality.
- **Tier 2:** *Early acetaminophen use and 90-day mortality in ICU patients
  with ischemic stroke.* Front Pharmacol 2025;16:1622440.
  `doi:10.3389/fphar.2025.1622440` **[verify volume]**. And *Association of
  early acetaminophen administration with mortality among critically ill
  patients with ischemic stroke.* BMC Neurol 2025. `doi:10.1186/s12883-025-04553-2`.
- **Feasibility:** acetaminophen on a q6h schedule is a `prescriptions` row
  -> `drug_exposure` (not `inputevents`); very common. Single exposure ->
  simpler concept mapping than B1. Outcome reliable. Caveat: the sepsis
  population needs a reproducible phenotype -- do not use a bare sepsis
  condition code (see Filter 2, item 4). A fever / suspected-infection
  population (HEAT) is an alternative that sidesteps Sepsis-3.
- **Why it is a strong validation signal:** the Tier-2 observational studies
  report a large protective effect (hazard ratios ~0.66-0.70) while the RCTs
  are null. If the pipeline on MIMIC reproduces the observational protective
  effect, that is an informative demonstration of confounding by indication.

### B3 -- Hydrocortisone in septic shock

- **Tier 1a:** Venkatesh B, et al. *Adjunctive Glucocorticoid Therapy in
  Patients with Septic Shock (ADRENAL).* N Engl J Med 2018;378(9):797-808.
  `doi:10.1056/NEJMoa1705835`. 90-day mortality OR 0.95 (0.82-1.10), P = .50.
- **Tier 1b:** Annane D, et al. *Hydrocortisone plus Fludrocortisone for
  Adults with Septic Shock (APROCCHSS).* N Engl J Med 2018;378(9):809-818.
  `doi:10.1056/NEJMoa1705716` **[verify pages]**. 90-day mortality lower with
  treatment.
- **Tier 1 synthesis:** Rochwerg B, et al. *Corticosteroids in Sepsis: An
  Updated Systematic Review and Meta-Analysis.* Crit Care Med
  2018;46(9):1411-1420. **[verify]**
- **Feasibility:** hydrocortisone is present in `drug_exposure` (~23,500
  rows). Mortality outcome reliable. Two limitations: the two pivotal RCTs
  disagree, so the reference truth is not a single number; and "septic shock"
  is a derived clinical state (Sepsis-3 + vasopressor + lactate), not a
  condition code -- Filter 4 is only partially met.

### B4 -- Timing of renal replacement therapy in AKI

- **Tier 1:** STARRT-AKI Investigators. *Timing of Initiation of
  Renal-Replacement Therapy in Acute Kidney Injury.* N Engl J Med
  2020;383(3):240-251. `doi:10.1056/NEJMoa2000741`.
- Gaudry S, et al. *Initiation Strategies for Renal-Replacement Therapy in
  the Intensive Care Unit (AKIKI).* N Engl J Med 2016;375(2):122-133.
  `doi:10.1056/NEJMoa1603017`.
- Zarbock A, et al. *Effect of Early vs Delayed Initiation of Renal
  Replacement Therapy on Mortality in Critically Ill Patients with Acute
  Kidney Injury: The ELAIN Randomized Clinical Trial.* JAMA
  2016;315(20):2190-2199. `doi:10.1001/jama.2016.5828` **[verify pages]**.
- Barbar SD, et al. *Timing of Renal-Replacement Therapy in Patients with
  Acute Kidney Injury and Sepsis (IDEAL-ICU).* N Engl J Med
  2018;379(15):1431-1442. `doi:10.1056/NEJMoa1803213` **[verify pages]**.
- **Feasibility:** RRT comes from `procedureevents`, which the OHDSI/MIMIC
  ETL *does* stage (-> `procedure_occurrence`), so unlike the infusion
  exposures it is recoverable. The RCT consensus is clear (no benefit of
  earlier initiation) -- a good target. Costs: the exposure is a temporal
  strategy ("initiate now vs wait"), and the KDIGO AKI phenotype plus the
  time-zero definition are heavier than a drug new-user design.

### B5 -- Early dexmedetomidine sedation

- **Tier 1:** Shehabi Y, et al. *Early Sedation with Dexmedetomidine in
  Critically Ill Patients (SPICE III).* N Engl J Med 2019;380(26):2506-2517.
  `doi:10.1056/NEJMoa1904710`. 90-day mortality 29.1% in both arms; age
  modifies the effect.
- **Feasibility:** propofol and midazolam are in `drug_exposure` (propofol
  ~53,600 rows), but sedation is infusion-titrated -- `drug_exposure` carries
  the order, not the rate or duration, and dose/timing are central to the
  question. Marginal on Filter 1-2.

### B6 -- LEGEND-Hypertension (method target, wrong dataset)

- Suchard MA, et al. *Comprehensive comparative effectiveness and safety of
  first-line antihypertensive drug classes: a systematic, multinational,
  large-scale analysis.* Lancet 2019;394(10211):1816-1826.
  `doi:10.1016/S0140-6736(19)32317-7`.
- **Why it matters:** this is the canonical OMOP causal-method validation
  target -- published reference hazard ratios across nine databases, a
  public protocol, and ~50 empirically calibrated negative controls.
- **Why it fails here:** MIMIC is an ICU database. New-user antihypertensive
  cohorts with months of follow-up for cardiovascular endpoints fail
  Filter 5 (N) and the follow-up requirement. Reconsider only if an OMOP
  claims or primary-care dataset becomes available.

---

## Method-evaluation control sets (Tier 3 -- add to whichever case is chosen)

- Ryan PB, Schuemie MJ, Welebob E, Duke J, Valentine S, Hartzema AG.
  *Defining a Reference Set to Support Methodological Research in Drug
  Safety.* Drug Saf 2013;36(Suppl 1):S33-S47. `doi:10.1007/s40264-013-0097-8`
  **[verify]**. The OMOP reference set: 165 positive + 234 negative
  drug-outcome pairs over acute liver injury, AKI, AMI, and GI bleeding.
- Schuemie MJ, Ryan PB, Pratt N, et al. *How Confident Are We About
  Observational Findings in Health Care: A Benchmark Study.* Harvard Data
  Science Review 2020;2(1). `doi:10.1162/99608f92.147cc28e` **[verify]**.
- *The Book of OHDSI*, Chapter 18 (Method Validity); the `MethodEvaluation`
  R package (`omopReferenceSet`, `ohdsiDevelopmentNegativeControls`).

## Cross-cutting method references

- Hernan MA, Robins JM. *Using Big Data to Emulate a Target Trial When a
  Randomized Trial Is Not Available.* Am J Epidemiol 2016;183(8):758-764.
  `doi:10.1093/aje/kwv254`. (Already cited: `hernan2016`.)
- Matthews AA, Danaei G, Islam N, Kurth T. *Target trial emulation: applying
  principles of randomised trials to observational studies.* BMJ
  2022;378:e071108. `doi:10.1136/bmj-2022-071108`.
- Franklin JM, Patorno E, Desai RJ, et al. *Emulating Randomized Clinical
  Trials With Nonrandomized Real-World Evidence Studies: First Results From
  the RCT-DUPLICATE Initiative.* Circulation 2021;143(10):1002-1013.
  `doi:10.1161/CIRCULATIONAHA.120.051718`.
- Wang SV, Schneeweiss S, et al. *Emulation of Randomized Clinical Trials
  With Nonrandomized Database Analyses: Results of 32 Clinical Trials.* JAMA
  2023;329(16):1376-1385. `doi:10.1001/jama.2023.4221` **[verify pages]**.
- Doutreligne M, et al. *Step-by-step causal analysis of EHRs to ground
  decision-making.* PLOS Digit Health 2025;4(2):e0000721.
  `doi:10.1371/journal.pdig.0000721`. (Already cited: `doutreligne2025`.)

---

## Status of documented-but-unbuilt candidates

| Case | Reference | Status |
|---|---|---|
| Albumin + crystalloids vs crystalloids, ICU sepsis, 28-day mortality | Doutreligne et al. 2025, `doi:10.1371/journal.pdig.0000721`; ALBIOS (Caironi P, et al. N Engl J Med 2014;370:1412-1421) and the pooled meta-analysis (Xu J-Y, et al. Crit Care 2014;18:702) as RCT anchors | **Future work, blocked on data not merit.** Albumin and crystalloids are ICU infusions -> outside the current OMOP ETL scope; Sepsis-3 is not reproducible from condition codes. Requires an `inputevents` ETL extension before it can run (see [`vocabulary_mapping_coverage.md`](vocabulary_mapping_coverage.md)). |
| Early acetaminophen in critically ill stroke | Wang X, Zhang K, Wu Y, et al. *Target trial emulation of early acetaminophen use and mortality in critically ill stroke patients.* Sci Rep 2026. `doi:10.1038/s41598-026-46381-5` | Runnable (acetaminophen is a prescription drug; ischemic stroke is a clean condition phenotype), but there is **no stroke-specific RCT** -- the reference truth is another observational emulation. Prefer B2, which shares the exposure and gains an RCT anchor. |

---

## Recommendation

1. **Headline case: B1 (PPI vs H2RA, anchored to PEPTIC).** It clears every
   checklist item, is the canonical pharmaco-epidemiology active-comparator
   new-user design the OHDSI methods were built for, uses two well-mapped
   exposures and a simple phenotype, and exercises every pipeline stage
   non-trivially.
2. **Second case: B2 (early acetaminophen in sepsis, anchored to ASTER; HEAT
   as the fever/infection alternative).** Lower concept-mapping effort,
   complements B1, and the observational-vs-RCT gap is itself an informative
   result.
3. **Keep the albumin/sepsis infusion case as documented future work** until
   an `inputevents` ETL extension lands; it stays blocked on data
   availability, not merit.
4. Add OHDSI negative controls (Tier 3) to whichever case is built first.

The research contribution is not rediscovering these effects -- they are
deliberately known. It is showing that the agentic pipeline reproduces, end
to end, a target-trial emulation whose randomized answer is established:
DAG, phenotypes, estimand, estimation, and diagnostics, on a real OMOP
conversion.
