# Validation question bank — ten ETL-feasible causal questions

Companion to [`validation_study_candidates.md`](validation_study_candidates.md).
That document is the *selection method* (two filters, a reference-anchored
shortlist, the citation landscape). This document is the *operational set*: ten
concrete causal questions, each specified as a target trial, each with a
published reference truth, each checked against what the current MIMIC-IV ->
OMOP CDM 5.3.1 conversion (OHDSI/MIMIC ETL) can actually support. It exists so
the workshop paper's Section 5 evaluation
([`../paper/main.tex`](../paper/main.tex)) has a pool of runnable cases rather
than one blocked case.

> Every trial citation below was checked against its PubMed or publisher
> landing page in September 2026. Effect sizes are transcribed from the
> abstract or a summary source and must be re-read from the primary text
> before they are used to score a run. Entries tagged **[verify pages]** still
> need volume/issue/page confirmation.

The research contribution is **not** rediscovering these effects. They are
deliberately known. It is showing that the gated pipeline reproduces, end to
end, target-trial emulations whose answers are established — and, just as
importantly, that its diagnostics flag the questions where an observational
answer is untrustworthy.

> **Scaffolded so far.** The three recommended questions have `studies/`
> packages with a `protocol.json`, a reference DAG, and a dated
> `concept_audit.md` (concept candidates resolved against the local DuckDB
> export with row/person counts):
> [`studies/ppi_vs_h2ra_sup_mortality/`](../studies/ppi_vs_h2ra_sup_mortality/README.md) (Q1),
> [`studies/corticosteroids_severe_cap_mortality/`](../studies/corticosteroids_severe_cap_mortality/README.md) (Q8),
> [`studies/vanco_piptazo_vs_cefepime_aki/`](../studies/vanco_piptazo_vs_cefepime_aki/README.md) (Q5).
> Each has an exploratory run created in `artifacts/runs/` and is a draft
> pending **human** concept-set approval and a clinically reviewed candidate
> DAG. The other seven questions are specified here only.

---

## 1. The ETL is the filter

The OHDSI/MIMIC conversion this project reads
([`vocabulary_mapping_coverage.md`](vocabulary_mapping_coverage.md)) constrains
which questions are answerable at all. Every question in this bank clears all
seven checks; a single failure disqualifies a candidate.

| # | Check | Consequence if failed |
|---|---|---|
| F1 | **Exposure is a prescription drug** (`drug_exposure`, ~96.8 % standard-mapped) **or a procedure with a real code** (`procedure_occurrence`, from `procedureevents`). *Not* an ICU infusion — `inputevents` is not staged, so IV fluids, albumin, blood products, titrated vasopressors, sedation infusions and IV insulin are absent or order-only. | Zero-patient or silently undercounted exposure arm. This is what puts infusion-exposure questions out of scope on the current ETL. |
| F2 | **"Ever exposed in a fixed window" is enough.** `drug_exposure` rows are prescription *orders* (`drug_type_concept_id = 32838`): order time, no dose, no rate, no infusion-start. | No dose-response, no infusion-precise time zero, no titration strategy. |
| F3 | **Clean index date, point / short-window treatment** (new-user or active-comparator new-user). | Immortal-time bias; the pipeline has no clone-censor-weight implementation. |
| F4 | **Outcome reliably captured**: in-hospital / 28-day / 90-day all-cause mortality (best, from `death`), a lab threshold (`measurement`), or a condition code with acceptable PPV. | Outcome misclassification swamps the effect. |
| F5 | **Population phenotype is a diagnosis, a demographic, a procedure, or a first-day-lab rule** — not a bare under-coded diagnosis. In a MIMIC-IV audit a sepsis condition code gives roughly 3,000 patients where a reconstructed Sepsis-3 cohort gives roughly 18,000. | Wrong denominator, selection bias. |
| F6 | **N >= a few hundred treated**, with 90-day mortality ascertainable (MIMIC-IV carries out-of-hospital death within one year of discharge). | Underpowered; CI uninformative. |
| F7 | **Confounders measurable in MIMIC** — first-day vitals and labs, demographics, comorbidity history, admission type. MIMIC is rich here; this rarely fails, and the reference DAG / adjustment set from the anchor trial or its emulations tells us what is needed. | Uncloseable backdoor path that is not even a stated limitation. |

Three exposure types survive this filter and the bank is built from them:
**prescription drug vs no drug**, **prescription drug vs prescription drug**
(active comparator), and **procedure vs no procedure / procedure timing**.

---

## 2. What the ten collectively exercise

The set is chosen so that, run together, it stresses every pipeline stage and —
critically for the paper's claim — covers the three outcomes a reviewer needs
to see:

| Property under test | Questions |
|---|---|
| Active-comparator new-user design | Q1, Q5, Q10 |
| Single-drug vs unexposed (new-user / continuation) | Q2, Q3, Q7, Q8, Q9 |
| Procedure exposure / procedure-timing design | Q4, Q6 |
| Derived clinical-state phenotype (not a code): Sepsis-3, septic shock, KDIGO AKI, suspected infection | Q2, Q3, Q4 |
| Condition-code phenotype with PPV risk | Q8, Q9 |
| **Specificity** — RCT says null, pipeline must *not* manufacture an effect | Q1, Q4, Q6, Q9 |
| **Sensitivity** — RCT / meta says real effect, pipeline must recover it | Q8, Q10, and the Q3 secondary (shock reversal) |
| **Diagnostic value** — strong confounded observational signal the pipeline's E-value / negative controls / balance diagnostics must flag as fragile | Q2, Q5, Q7 |
| Contested benchmark (two pivotal RCTs disagree) | Q3 |
| Positive-control side effect (steroid -> hyperglycaemia) | Q3, Q8 |
| Mapping-coverage gate stress test (instrument itemids, urine output, dose parsing) | Q4, Q9, Q10 |
| Immortal-time-bias handling (landmark / grace period) | Q4, Q6 |
| SHD/SID vs a published/derivable expert DAG | Q1, Q5 (OHDSI LEGEND-style), Q3, Q4 |

---

## 3. Summary table

| # | Question (exposure vs comparator, population -> outcome) | Anchor trial(s) | Anchor result | Design | What it tests |
|---|---|---|---|---|---|
| Q1 | PPI vs H2RA for stress-ulcer prophylaxis, ventilated ICU adults -> in-hospital / 90-day mortality | PEPTIC | Near-null (in-hosp 18.3 % vs 17.5 %, P=.054) | Active comparator | Specificity; near-null recovery |
| Q2 | Early acetaminophen vs none, ICU adults with suspected infection / fever -> 28-day mortality; ARDS | ASTER; HEAT | Null | Drug vs none | Confounding by indication (observational HR ~0.7–0.8) |
| Q3 | Hydrocortisone vs no corticosteroid, septic shock -> 90-day mortality; time to shock reversal | ADRENAL; APROCCHSS; NEJM Evidence patient-level MA | Split: mortality null (ADRENAL) vs benefit (APROCCHSS); shock reversal faster in both | Drug vs none, derived phenotype | Contested benchmark; positive-control side effect |
| Q4 | Accelerated vs standard RRT initiation, KDIGO 2–3 AKI -> 90-day mortality; RRT dependence | STARRT-AKI; AKIKI; IDEAL-ICU (ELAIN discordant) | Null for mortality; more RRT dependence with accelerated | Procedure timing | Specificity; immortal time; procedure exposure |
| Q5 | Vancomycin + piperacillin-tazobactam vs vancomycin + cefepime -> KDIGO AKI within 7 d | ACORN (RCT, cefepime vs pip-tazo); observational meta-analyses | RCT null (OR 0.95); observational OR ~2.5–2.7 | Active comparator | **Headline diagnostic case** — pipeline should reproduce the observational signal, then flag it |
| Q6 | Early (<=4 d) vs late (>=10 d) tracheostomy, adults ventilated & expected prolonged -> 30-/90-day mortality; ventilator-free days | TracMan | Null (30-d 30.8 % vs 31.5 %) | Procedure timing | Specificity; immortal-time landmark design |
| Q7 | Statin continuation vs discontinuation, prior statin users with sepsis -> 28-day mortality | SAILS; HARP-2; Kruger (ANZ-STATInS) | Null | Continuation (prevalent user) | Healthy-continuer bias (observational RR ~0.7–0.9) |
| Q8 | Systemic corticosteroid vs none, adults with severe CAP -> in-hospital / 30-day mortality; hyperglycaemia | CAPE COD; Crit Care 2023 MA (7 RCTs) | **Benefit** (28-d 6.2 % vs 11.9 %; MA RR 0.61) | Drug vs none, condition phenotype | **Sensitivity** — recover a real protective effect; positive-control side effect |
| Q9 | Haloperidol vs no antipsychotic (or vs quetiapine), ICU delirium -> in-hospital mortality; ICU LOS | MIND-USA; REDUCE; Hope-ICU | Null | Drug vs none / active comparator | Specificity; phenotype-risk stress test |
| Q10 | LMWH (dalteparin/enoxaparin) vs unfractionated heparin, ICU adults on pharmacologic VTE prophylaxis -> symptomatic PE; proximal DVT; major bleeding | PROTECT | PE lower with LMWH (1.3 % vs 2.3 %, HR 0.51); DVT & bleeding similar | Active comparator | Sensitivity (PE); competing benefit/harm; dose-parsing |

---

## 4. Question detail

Each entry gives the target trial, the reference truth with citations, the ETL
feasibility argument (against F1–F7), the expected pipeline result and why, and
the known risks. Confounder lists are the minimum the DAG review must resolve;
they are not adjustment sets.

### Q1 — PPI vs H2RA for stress-ulcer prophylaxis

*(= B1 in the candidates doc; carried here with a full run spec.)*

- **Target trial.** Population: adults, first ICU stay, receiving invasive
  mechanical ventilation within 24 h of ICU admission (ventilation is a staged
  procedure), no PPI or H2RA in a 90-day look-back. Treatment: first
  stress-ulcer-prophylaxis drug class ordered within 48 h — PPI (pantoprazole,
  omeprazole, esomeprazole, lansoprazole) vs H2RA (famotidine; ranitidine is
  in-era for MIMIC-IV's 2008–2019 window). Time zero: first qualifying order.
  Outcome: in-hospital all-cause mortality (primary), 90-day mortality;
  clinically important upper-GI bleeding as a lower-PPV secondary (condition
  code plus a transfusion or endoscopy proxy). Estimand: ATE, risk difference.
- **Reference truth.** PEPTIC Investigators / ANZICS CTG. *Effect of Stress
  Ulcer Prophylaxis With PPIs vs H2RBs on In-Hospital Mortality Among ICU
  Patients Receiving Invasive Mechanical Ventilation: The PEPTIC Randomized
  Clinical Trial.* **JAMA 2020;323(7):616–626.**
  `doi:10.1001/jama.2019.22190`. 26,982 patients, 50 ICUs, cluster crossover.
  In-hospital mortality 18.3 % (PPI) vs 17.5 % (H2RB), OR 1.05
  (95 % CI 1.00–1.10), P=.054. UGIB lower with PPI (1.3 % vs 1.8 %); *C.
  difficile* similar. Supporting emulation: *Sci Rep 2021;11:18424*
  `doi:10.1038/s41598-021-98069-7` **[verify article no.]** (retrospective ICU
  cohort, explicit adjustment set).
- **ETL feasibility** *(concept audit 2026-09-03, `studies/ppi_vs_h2ra_sup_mortality/concept_audit.md`)*.
  F1 ✓ both classes are well-mapped prescriptions (pantoprazole 43 k persons,
  famotidine 27 k, ranitidine 20 k; name-matched-unmapped negligible). F2 ✓
  "which class first" needs only order presence. F3 ✓ active-comparator
  new-user. F4 ✓ mortality. F5 ⚠️ **ventilation undercount** — invasive
  mechanical ventilation is present only as ICD-9-Proc / ICD-10-PCS billing
  codes (~14,662 distinct persons; SNOMED `37158404` has 0 rows, the
  `procedureevents` itemid does not map). The true ventilated population is
  ~25–30 k; the billing-code cohort skews to longer/billed courses. Run on it
  and document the undercount as an ETL-coverage limitation (like the
  `inputevents` gap); prespecify a SNOMED `4230167` "Artificial ventilation"
  sensitivity analysis. F6 ✓ crude split among the 14,662: ~4.0 k PPI-only,
  ~4.0 k H2RA-only, ~5.7 k both — analytic cohort ~6–9 k after the new-user /
  first-agent rules. F7 ✓ first-day severity, comorbidity, admission type.
- **Expected result & why.** Risk difference ~0, CI covering 0, possibly a
  small PPI-associated harm. The active-comparator design controls confounding
  by indication tightly. If the pipeline returns a large effect in either
  direction, that is a design or mapping failure to diagnose, not a finding.
- **Risks.** The ventilation undercount (above). UGIB PPV is low; keep it
  strictly secondary. Site-level prescribing culture (the PEPTIC randomisation
  unit) is a clustered confounder — `care_site` should enter the adjustment
  set or a sensitivity analysis.

### Q2 — Early acetaminophen vs none in suspected infection / fever

*(= B2 in the candidates doc; full run spec here.)*

- **Target trial.** Population: ICU adults with suspected infection —
  reconstructed as an antibiotic order plus a body-fluid culture within the
  Sepsis-3 timing window (the reconstructed Sepsis-3 phenotype; see Q3) —
  **or** a documented temperature >= 38.3 °C in the first 24 h
  (`measurement`, well covered). Treatment: any acetaminophen order (enteral or
  IV) within 48 h of the qualifying event vs none in that window. Landmark at
  48 h. Outcome: 28-day all-cause mortality (primary); ARDS as a condition-code
  secondary.
- **Reference truth.** ASTER: Ware LB et al. *Acetaminophen for Prevention and
  Treatment of Organ Dysfunction in Critically Ill Patients With Sepsis: The
  ASTER RCT.* **JAMA 2024;332(5):390–400.** `doi:10.1001/jama.2024.8772`. 447
  patients; no difference in days alive and free of organ support; less ARDS
  (2.2 % vs 8.5 %); mortality 17 % vs 22 % (NS). HEAT: Young P et al.
  *Acetaminophen for Fever in Critically Ill Patients with Suspected Infection.*
  **NEJM 2015;373(23):2215–2224.** `doi:10.1056/NEJMoa1508375`. 700 patients;
  no difference in ICU-free days or 28/90-day mortality. **Observational
  contrast:** MIMIC-based emulations report a large protective association —
  ischaemic stroke 90-day HR 0.73 (`doi:10.1038/s41598-026-46381-5`;
  `doi:10.1186/s12883-025-04553-2`), sepsis-associated AKI in-hospital HR 0.72
  (`doi:10.1186/s40001-025-02786-y`), while cirrhosis shows the opposite
  (28-day HR 1.70, `doi:10.1186/s12876-025-04113-5`).
- **ETL feasibility.** F1 ✓ acetaminophen is the single most common ICU
  prescription (scheduled q6h + PRN). F2 ✓ ever-ordered is the exposure. F3 ✓
  new-user in a 48 h window with a landmark. F4 ✓ mortality; ARDS PPV modest.
  F5 ✓ suspected-infection or fever rule, both reconstructable. F6 ✓ >20 k. F7 ✓
  temperature, SOFA, lactate, vasopressor order, source of infection.
- **Expected result & why.** The RCTs are null. The observational literature on
  the *same database* is strongly protective — a textbook confounding-by-
  indication pattern (acetaminophen is withheld from the sickest / liver-injured
  and given to the more stable febrile patient). If the pipeline reproduces the
  protective HR, the **E-value and the SMD balance table should show the
  estimate is fragile**; that demonstration is the point of including Q2.
- **Risks.** "Suspected infection" and "fever" define different populations;
  prespecify one as primary. PRN-only acetaminophen (given for one fever spike)
  vs scheduled is a dose-intensity distinction the order records blur (F2).

### Q3 — Hydrocortisone vs no corticosteroid in septic shock

- **Target trial.** Population: ICU adults meeting a reconstructed septic-shock
  state — Sepsis-3 positive **and** a vasopressor order **and** serum
  lactate >= 2 mmol/L, all before time zero (this is a derived state, **not**
  SNOMED 196236). Treatment:
  hydrocortisone order (any route) within 24 h of shock onset vs none within
  that window. Outcome: 90-day all-cause mortality (primary), 28-day mortality;
  time to vasopressor discontinuation as a secondary; new hyperglycaemia
  (glucose >= 180 mg/dL or a rise in insulin orders) as a **positive control**.
- **Reference truth.** ADRENAL: Venkatesh B et al. *Adjunctive Glucocorticoid
  Therapy in Patients with Septic Shock.* **NEJM 2018;378(9):797–808.**
  `doi:10.1056/NEJMoa1705835`. 3,800 patients; 90-day mortality 27.9 % vs
  28.8 %, OR 0.95 (0.82–1.10), P=.50; faster shock resolution, faster ICU
  discharge, faster cessation of ventilation; more hyperglycaemia. APROCCHSS:
  Annane D et al. *Hydrocortisone plus Fludrocortisone for Adults with Septic
  Shock.* **NEJM 2018;378(9):809–818.** `doi:10.1056/NEJMoa1705716`
  **[verify pages]**. 1,241 patients; 90-day mortality 43.0 % vs 49.1 %, RR
  0.88 (0.78–0.99), P=.03. Synthesis: Pirracchio R et al. *Patient-Level
  Meta-Analysis of Low-Dose Hydrocortisone in Adults with Septic Shock.* **NEJM
  Evidence 2023.** `doi:10.1056/EVIDoa2300034` **[verify vol/issue]**.
- **ETL feasibility.** F1 ✓ hydrocortisone orders present (~23.5 k rows). F2 ✓
  ever-ordered within the window. F3 ✓ short window from shock onset. F4 ✓
  mortality; hyperglycaemia from `measurement`. F5 ⚠️ derived shock phenotype —
  heavier, but built only from `measurement` + `drug_exposure` (no
  `inputevents`), and the vasopressor *order* is enough to mark "on a
  vasopressor" even without a rate. F6 ✓ shock cohort in the thousands. F7 ✓
  SOFA, lactate, baseline steroid use, aetiology.
- **Expected result & why.** Mortality risk difference near 0 with a wide CI —
  the two pivotal RCTs disagree, so "the CI covers both" is the honest target,
  and the pipeline's report must not collapse a contested benchmark into a
  single number. The **secondary** (faster shock reversal) is a real effect
  both RCTs agree on — a sensitivity check. Hyperglycaemia should come out
  clearly positive — if it does not, the covariate/outcome plumbing is wrong.
- **Risks.** Etomidate exposure (adrenal suppression) is an effect modifier
  ADRENAL excluded on — flag it. Confounding by indication is severe here
  (steroids given to refractory shock); the derived-phenotype timing must put
  lactate and vasopressor status strictly pre-treatment.

### Q4 — Accelerated vs standard RRT initiation in AKI

- **Target trial.** Population: ICU adults reaching KDIGO stage 2–3 AKI
  (serum-creatinine trajectory vs baseline, plus urine-output criteria where
  `outputevents`-derived measurements are available on this build), no
  pre-existing kidney failure on RRT. Treatment strategy: RRT initiated within
  12 h of the first stage-3 criterion (**accelerated**) vs no RRT unless a
  conventional indication appears (**standard**). Outcome: 90-day all-cause
  mortality (primary); RRT dependence at hospital discharge / 90 days.
- **Reference truth.** STARRT-AKI Investigators. *Timing of Initiation of
  Renal-Replacement Therapy in Acute Kidney Injury.* **NEJM
  2020;383(3):240–251.** `doi:10.1056/NEJMoa2000741`. 2,927 analysed, 168
  hospitals / 15 countries; 90-day mortality 43.9 % (accelerated) vs 43.7 %
  (standard), RR 1.00 (0.93–1.09); among survivors, RRT dependence at 90 days
  higher with accelerated (10.4 % vs 6.0 %). Concordant: AKIKI (Gaudry S et al.
  *NEJM 2016;375(2):122–133*, `doi:10.1056/NEJMoa1603017`); IDEAL-ICU (Barbar
  SD et al. *NEJM 2018;379(15):1431–1442*, `doi:10.1056/NEJMoa1803213`
  **[verify pages]**). Discordant single-centre: ELAIN (Zarbock A et al. *JAMA
  2016;315(20):2190–2199*, `doi:10.1001/jama.2016.5828` **[verify pages]**).
- **ETL feasibility.** F1 ✓ RRT is staged (`procedureevents` ->
  `procedure_occurrence`). F2 ✓ procedure datetime gives initiation time. F3 ⚠️
  this is a *time-varying strategy*, not a point exposure — the honest
  operationalisation is a grace-period / target-trial clone design, and the
  pipeline supports only a simplified "RRT within 12 h of stage 3 vs not" form.
  Run it as that simplification and **document the gap** — this is itself a
  useful limitation for the paper. F4 ✓ mortality; RRT dependence from
  procedure recurrence. F5 ⚠️ KDIGO phenotype from `measurement` — verify
  creatinine and urine-output coverage on this build first. F6 ✓ AKI 2–3 cohort
  ~15 k; RRT ~3–4 k. F7 ✓ baseline creatinine, fluid balance, vasopressors,
  ventilation, sepsis.
- **Expected result & why.** Null for mortality; a real *harm* signal for RRT
  dependence with accelerated initiation. Recovering the RRT-dependence effect
  while returning null mortality is a strong specificity+sensitivity combined
  test.
- **Risks.** Immortal time: "accelerated" requires surviving to RRT — landmark
  at the stage-3 timestamp. Confounding by indication is the dominant threat
  (sicker patients get earlier RRT).

### Q5 — Vancomycin + piperacillin-tazobactam vs vancomycin + cefepime -> AKI

**Headline diagnostic case.**

- **Target trial.** Population: ICU adults started on empiric vancomycin plus an
  antipseudomonal beta-lactam within 48 h of admission, baseline
  eGFR >= 60 / not on RRT. Treatment: vancomycin + piperacillin-tazobactam vs
  vancomycin + cefepime (active comparator, new-user of the combination). Time
  zero: first co-administration day. Outcome: KDIGO stage >= 2 AKI within 7 days
  (serum creatinine, from `measurement`); a MAKE-style composite (death, new
  RRT, failure of creatinine to return to <= 1.5x baseline) as secondary.
- **Reference truth.** ACORN: Qian ET et al. *Cefepime vs Piperacillin-
  Tazobactam in Adults Hospitalized With Acute Infection: The ACORN Randomized
  Clinical Trial.* **JAMA 2023;330(16):1557–1567.**
  `doi:10.1001/jama.2023.20583` (verified Sept 2026). 2,511 patients, 77.2 %
  on vancomycin; highest AKI stage or death by day 14 — OR 0.95 (0.80–1.13),
  P=.56 (**no AKI excess with pip-tazo**); cefepime caused more neurological
  dysfunction.
  **Observational contrast:** Luther MK et al. *Vancomycin Plus Piperacillin-
  Tazobactam and AKI in Adults: A Systematic Review and Meta-Analysis.* **Crit
  Care Med 2018;46(1):12–20.** `doi:10.1097/CCM.0000000000002769`. OR vs
  vancomycin + cefepime/carbapenem 2.68 (1.83–3.91). Network meta-analysis
  *J Antimicrob Chemother 2025;80(1):47* `doi:10.1093/jac/dkae395`
  **[verify pages]**: OR 2.55 (2.00–3.28) vs vancomycin + cefepime. Hammond DA
  et al. *Clin Infect Dis 2017;64(5):666–674* `doi:10.1093/cid/ciw811`.
- **ETL feasibility.** F1 ✓✓ all three drugs are common, well-mapped IV
  prescriptions. F2 ✓ combination presence in a window. F3 ✓ active-comparator
  new-user. F4 ✓ creatinine is among the best-covered `measurement` concepts.
  F5 ✓ structural (drug + baseline-renal rule). F6 ✓ each arm in the low
  thousands. F7 ✓ baseline creatinine, nephrotoxin co-exposure (contrast,
  NSAIDs, aminoglycosides), sepsis severity, hypotension.
- **Expected result & why.** The pipeline should **reproduce the observational
  AKI association (OR ~2–2.7)** because that association is real *in
  observational data* — it is driven by pip-tazo's inhibition of tubular
  creatinine secretion (pseudotoxicity) plus residual confounding, and ACORN
  shows the causal effect is ~null. The test is whether CausalOMOP's **E-value,
  negative controls, and balance diagnostics correctly present this as an
  association that a randomized trial does not support** — i.e. whether the
  system is honest about a case where it gets a "positive" result that is
  wrong. This is the single most informative question in the bank for the
  paper's central claim.
- **Risks.** Creatinine-based AKI is exactly the endpoint pseudotoxicity
  corrupts; a cystatin-C sensitivity analysis is not possible on MIMIC. State
  that the endpoint itself is mechanism-contaminated — that is part of the
  lesson.

### Q6 — Early vs late tracheostomy

- **Target trial.** Population: adults mechanically ventilated for < 4 days at
  ICU day 4, judged (proxy: still ventilated, no extubation order, not
  moribund) to need >= 7 more days. Treatment: tracheostomy performed on days
  1–4 (**early**) vs day 10 or later / not at all (**late**), from
  `procedure_occurrence` datetimes. Landmark at day 4. Outcome: 30-day and
  90-day all-cause mortality (primary); ventilator-free days to day 28;
  ICU length of stay.
- **Reference truth.** Young D et al. *Effect of Early vs Late Tracheostomy
  Placement on Survival in Patients Receiving Mechanical Ventilation: The
  TracMan Randomized Trial.* **JAMA 2013;309(20):2121–2129.**
  `doi:10.1001/jama.2013.5154`. 909 patients, 70 UK units; 30-day mortality
  30.8 % (early) vs 31.5 % (late), absolute difference 0.7 % (95 % CI −5.4 to
  6.7); 2-year mortality 51.0 % vs 53.7 %; less sedative exposure with early; no
  ICU-LOS or ventilation-duration benefit; ~53 % of the late group never
  received a tracheostomy. Supporting: Terragni PP et al. *JAMA
  2010;303(15):1483–1489*; Cochrane review Andriolo BN et al. 2015 (CD007271)
  **[verify ID]** — low-quality signal toward lower mortality and earlier
  discharge.
- **ETL feasibility.** F1 ✓ tracheostomy and mechanical ventilation are staged
  procedures. F2 ✓ procedure datetime gives timing. F3 ✓ landmark design at day
  4. F4 ✓ mortality; VFD approximated from ventilation procedure spans. F5 ✓
  "ventilated >= 4 days" is structural. F6 ⚠️ MIMIC has ~1.5–2 k tracheostomies —
  moderate power; the early arm is the smaller. F7 ✓ age, severity, reason for
  ventilation, neurological status (GCS in `measurement`).
- **Expected result & why.** Null for mortality — a specificity test on a
  procedure-timing question. The design's value is the **immortal-time trap**:
  a naive "early trach vs late trach" comparison without the day-4 landmark
  makes early trach look protective because late-trach patients had to survive
  longer to qualify. The pipeline should either handle this or flag it.
- **Risks.** "Expected prolonged ventilation" is a clinical judgement not in the
  data; the ventilated-at-day-4 proxy over-includes. The large never-
  tracheostomised late group (as in TracMan) must be an explicit arm, not
  dropped.

### Q7 — Statin continuation vs discontinuation in sepsis

- **Target trial.** Population: ICU adults with reconstructed Sepsis-3 who were
  on a statin before admission (statin on the pre-admission / admission
  medication reconciliation in `drug_exposure`). Treatment: statin re-ordered
  within 48 h of ICU admission (**continued**) vs not re-ordered in that window
  (**discontinued**). Outcome: 28-day all-cause mortality.
- **Reference truth.** SAILS: NHLBI ARDS Network (Truwit JD et al.).
  *Rosuvastatin for Sepsis-Associated Acute Respiratory Distress Syndrome.*
  **NEJM 2014;370(23):2191–2200.** `doi:10.1056/NEJMoa1401520`. 745 patients;
  no mortality or organ-failure benefit; stopped for futility; more hepatic and
  renal dysfunction. HARP-2: McAuley DF et al. *Simvastatin in the Acute
  Respiratory Distress Syndrome.* **NEJM 2014;371(18):1695–1703.**
  `doi:10.1056/NEJMoa1403285`. 540 patients; 28-day mortality 22.0 % vs 26.8 %,
  P=.23. Kruger PS et al. *A Multicenter RCT of Atorvastatin Therapy in
  Intensive Care Patients with Severe Sepsis (ANZ-STATInS).* **Am J Respir Crit
  Care Med 2013;187(7):743–750.** `doi:10.1164/rccm.201209-1718OC`.
  **RCT-vs-observational contrast:** Wan YD et al. *Effect of Statin Therapy on
  Mortality from Infection and Sepsis: A Meta-Analysis of Randomized and
  Observational Studies.* **Crit Care Med 2014;42(3):e211.**
  `doi:10.1097/CCM.0000000000000135` **[verify pages]** — observational studies
  show a mortality benefit that the pooled RCTs do not.
- **ETL feasibility.** F1 ✓ statins are among the most common prescriptions. F2
  ✓ re-ordered vs not in a window. F3 ✓ continuation (prevalent-user) design
  with a clean index. F4 ✓ mortality. F5 ⚠️ needs Sepsis-3 reconstruction +
  reliable pre-admission statin status — MIMIC's admission medication list
  supports the latter. F6 ✓ prior-statin sepsis cohort ~5–8 k. F7 ✓ reason for
  discontinuation is the confounder (shock, ileus, hepatic injury, comfort
  care) — all partially observable.
- **Expected result & why.** Null (matching the RCTs). Observational cohorts
  report protective associations (RR ~0.7–0.9) that are **healthy-continuer
  bias** — statins get held in the patients who are deteriorating or made
  comfort-focused. Same diagnostic role as Q2: the pipeline's balance table and
  E-value should expose the fragility.
- **Risks.** "Comfort care / withdrawal of life support" is the strongest
  confounder and is only partially coded; a sensitivity analysis excluding
  early deaths (< 48 h) and documented care-limitation orders is needed.

### Q8 — Systemic corticosteroids vs none in severe community-acquired pneumonia

**Sensitivity case — a real effect the pipeline must recover.**

- **Target trial.** Population: adults admitted with community-acquired
  pneumonia (condition code; PPV supported by a concurrent antibiotic course
  and, where available, chest imaging) meeting a severity marker (ICU admission,
  or vasopressor/ventilation within 24 h). Treatment: systemic corticosteroid
  (hydrocortisone, methylprednisolone, prednisone, dexamethasone) order within
  48 h of admission vs none in that window. Exclude chronic steroid users and
  patients with another steroid indication (COPD exacerbation, asthma,
  autoimmune). Outcome: in-hospital / 30-day all-cause mortality (primary);
  new hyperglycaemia as a **positive control**; time to clinical stability
  (vasopressor/oxygen weaning) as a secondary.
- **Reference truth.** CAPE COD: Dequin PF et al. *Hydrocortisone in Severe
  Community-Acquired Pneumonia.* **NEJM 2023;388(21):1931–1941.**
  `doi:10.1056/NEJMoa2215145`. 800 patients, 31 French ICUs; 28-day mortality
  6.2 % vs 11.9 %, absolute difference −5.6 (95 % CI −9.6 to −1.7); less
  intubation and less vasopressor initiation; more insulin use. Meta-analysis:
  *Efficacy and safety of adjunctive corticosteroids in severe CAP: a
  systematic review and meta-analysis of RCTs.* **Crit Care 2023;27:274.**
  `doi:10.1186/s13054-023-04561-z`. 7 RCTs, 1,689 patients; day-30 mortality RR
  0.61 (0.44–0.85); less mechanical ventilation RR 0.57. Supporting: Torres A
  et al. *JAMA 2015;313(7):677–686*; Blum CA et al. *Lancet
  2015;385(9977):1511–1518*. Counter-weight: ESCAPe (Meduri GU et al.
  *Intensive Care Med 2022;48:1009–1023*) — methylprednisolone, no 60-day
  mortality benefit.
- **ETL feasibility.** F1 ✓ all four steroids are well-mapped prescriptions. F2
  ✓ ever-ordered in a window. F3 ✓ new-user, clean admission index. F4 ✓
  mortality; hyperglycaemia from `measurement`. F5 ⚠️ CAP condition-code
  phenotype — PPV is the main risk; tighten with the antibiotic co-requirement
  and exclude aspiration/hospital-acquired codes. F6 ✓ severe-CAP cohort
  ~5–8 k. F7 ✓ severity (SOFA, PaO2/FiO2, lactate), age, comorbidity,
  vasopressor/ventilation status.
- **Expected result & why.** A genuine protective effect — risk difference
  roughly −3 to −6 percentage points in the severe stratum. This is the
  bank's clearest **sensitivity** test: if the pipeline returns null here while
  the RCTs and a 7-trial meta-analysis agree on benefit, it is too
  conservative. Hyperglycaemia must come out positive.
- **Risks.** Confounding by indication runs *toward the null or toward harm*
  here (steroids given to the sicker patient), so a recovered protective effect
  is a conservative demonstration. Steroid use for a non-CAP indication is the
  key exclusion.

### Q9 — Haloperidol vs no antipsychotic for ICU delirium

- **Target trial.** Population: ICU adults with delirium — CAM-ICU / ICDSC
  positive if those assessments are mapped on this build, else a delirium
  condition code plus an agitation/sedation context. Treatment: haloperidol
  order within 24 h of the first positive assessment vs no antipsychotic in that
  window (primary); haloperidol vs quetiapine as an active-comparator
  alternative. Outcome: in-hospital all-cause mortality (primary); ICU length
  of stay; delirium/coma-free days if the instrument is available.
- **Reference truth.** MIND-USA: Girard TD et al. *Haloperidol and Ziprasidone
  for Treatment of Delirium in Critical Illness.* **NEJM 2018;379(26):2506–2516.**
  `doi:10.1056/NEJMoa1808217`. 566 patients; days alive without delirium or
  coma 7.9 (haloperidol) vs 8.7 (ziprasidone) vs 8.5 (placebo), P=.26; no
  effect on survival, LOS, or safety. REDUCE: van den Boogaard M et al.
  *JAMA 2018;319(7):680–690* (prophylactic haloperidol, no survival benefit).
  Hope-ICU: Page VJ et al. *Lancet Respir Med 2013;1(7):515–523*.
- **ETL feasibility.** F1 ✓ haloperidol and quetiapine are well-mapped
  prescriptions. F2 ✓ ever-ordered in a window. F3 ✓ new-user from the delirium
  index. F4 ✓ mortality; LOS structural. F5 ⚠️⚠️ **the phenotype is the stress
  test** — CAM-ICU lives in `chartevents` and may arrive as unmapped
  `measurement` itemids (the 5.85 % gap), and the delirium condition code has
  poor sensitivity. This question deliberately exercises the mapping-coverage
  gate: if the instrument is unmapped, the pipeline should *say so* rather than
  silently fall back to a tiny code-based cohort. F6 ✓ haloperidol exposure is
  common (thousands). F7 ✓ sedation exposure, mechanical ventilation, age,
  baseline cognition, sepsis.
- **Expected result & why.** Null on mortality (matching the RCTs). The
  observational literature is mixed and includes *harm* signals (antipsychotics
  given to the most agitated / sickest) — another confounding-by-indication
  pattern the diagnostics should surface.
- **Risks.** Defining "delirium" by the receipt of haloperidol would be
  circular; the population must be phenotyped independently of the exposure. QTc
  is not reliably in `measurement`; drop the torsades safety endpoint unless
  coverage is confirmed.

### Q10 — LMWH vs unfractionated heparin for VTE prophylaxis

- **Target trial.** Population: ICU adults started on pharmacologic VTE
  prophylaxis within 48 h of admission, no therapeutic anticoagulation, no
  active bleeding, first ICU stay. Treatment: prophylactic-dose LMWH
  (dalteparin, enoxaparin) vs prophylactic-dose unfractionated heparin —
  distinguished by drug identity and dose/frequency in the prescription record.
  Outcome: symptomatic pulmonary embolism (condition code + CT-pulmonary-
  angiography procedure) as primary; proximal DVT (condition code + compression
  ultrasound); major bleeding (condition code + transfusion proxy);
  in-hospital mortality.
- **Reference truth.** PROTECT Investigators (Cook D et al.). *Dalteparin
  versus Unfractionated Heparin in Critically Ill Patients.* **NEJM
  2011;364(14):1305–1314.** `doi:10.1056/NEJMoa1014475`. 3,764 patients;
  dalteparin 5000 IU daily vs UFH 5000 IU twice daily. Proximal DVT 5.1 % vs
  5.8 %, HR 0.92 (0.68–1.23); **PE 1.3 % vs 2.3 %, HR 0.51 (0.30–0.88) — lower
  with dalteparin**; major bleeding and heparin-induced thrombocytopenia
  similar; mortality similar. Prophylaxis-vs-none context: Alhazzani W et al.
  *Heparin Thromboprophylaxis in Medical-Surgical Critically Ill Patients: A
  Systematic Review and Meta-Analysis of RCTs.* **Crit Care Med
  2013;41(9):2088–2098.** `doi:10.1097/CCM.0b013e31828cf104` — heparin lowers
  DVT and PE vs none, small bleeding increase, no mortality effect.
- **ETL feasibility.** F1 ✓ enoxaparin, dalteparin, heparin are common SC
  prescriptions. F2 ⚠️ prophylactic vs therapeutic intent must be read from the
  prescription **dose and frequency** — a concept-mapping / registry task, not
  just an ingredient match (this exercises the dose-aware side of the
  registry). F3 ✓ active-comparator new-user. F4 ⚠️ PE/DVT PPV depends on
  pairing the condition code with the imaging procedure; keep mortality as a
  robust co-primary. F5 ✓ structural. F6 ✓ prophylaxis is near-universal in
  ICU — large N, but the LMWH arm may be the minority in a 2008–2019 US ICU
  (UFH-dominant), so check arm balance. F7 ✓ renal function (drives LMWH
  avoidance — a confounder), weight, bleeding risk, surgery.
- **Expected result & why.** A real reduction in PE with LMWH (sensitivity
  test), null for DVT and bleeding. Recovering the PE effect while returning
  null for the other endpoints is a discriminating multi-outcome test.
- **Risks.** Renal-dysfunction-driven channelling (UFH preferred in renal
  failure) is the dominant confounder and correlates with mortality — the
  adjustment set must include baseline creatinine / eGFR. Prophylactic-dose
  ascertainment errors would contaminate both arms.

---

## 5. Cross-cutting: negative controls and calibration (Tier 3)

Add to whichever questions are run first, using
`causal/queries/negative_controls.py` (LLM-proposed label + rationale,
human-approved mapping, diagnostic not a gate):

- **Negative-control outcomes** for the drug questions — outcomes with no
  plausible causal link to the exposure (e.g. for Q1/Q5: ingrown nail, contact
  dermatitis, routine screening encounters). A non-null estimate on these
  calibrates residual confounding.
- **Negative-control exposure-outcome pairs** from the OMOP reference set —
  Ryan PB et al. *Defining a Reference Set to Support Methodological Research in
  Drug Safety.* **Drug Saf 2013;36(Suppl 1):S33–S47.**
  `doi:10.1007/s40264-013-0097-8` **[verify]**. 165 positive + 234 negative
  pairs over acute liver injury, AKI, AMI, and GI bleeding — several are
  runnable on this ETL (all prescription-drug exposures).
- **Empirical calibration** of p-values / CIs against the negative-control
  distribution (Schuemie MJ et al., *The Book of OHDSI* ch. 18;
  `MethodEvaluation` R package) is the OHDSI-standard way to report what the
  calibrated effect looks like — worth one paragraph in the paper even if only
  applied to Q5.

## 6. Prioritisation for the workshop paper

The paper needs "at least one clinician-reviewed end-to-end case"
([`../paper/submission_readiness.md`](../paper/submission_readiness.md)). Run
**three**, chosen to be the null / real-effect / confounded triad a reviewer
needs:

1. **Q1 (PPI vs H2RA / PEPTIC)** — cleanest active-comparator, largest N,
   well-mapped, near-null. Demonstrates the pipeline does not manufacture an
   effect. Lowest execution risk; run it first.
2. **Q8 (steroids in severe CAP / CAPE COD)** — a genuine protective effect
   with a 7-RCT meta-analysis behind it. Demonstrates the pipeline is not
   merely conservative. Condition-phenotype PPV is the work item.
3. **Q5 (vancomycin + pip-tazo vs vancomycin + cefepime / ACORN)** — the
   headline: the pipeline reproduces a strong observational association
   (OR ~2.5) that a randomized trial shows is ~null, and its E-value /
   negative-control / balance diagnostics present it as fragile. This is the
   demonstration that most directly supports the paper's central claim.

Q2 and Q7 are the natural second wave (both are confounding-by-indication
demonstrations that reuse the Sepsis-3 phenotype work). Q3, Q4, Q6, Q9, Q10 are
documented but lower priority — Q4 and Q9 also carry the largest open ETL /
phenotype risks.

## 7. Open risks that apply to the whole bank

- **Sepsis-3 phenotype (Q2, Q3, Q7).** Not yet built. A work plan scopes it
  (`paper/mimic_omop_coverage_workplan.md`, workstream C.3); it needs only
  `measurement` + `drug_exposure`, so it is not blocked by the `inputevents`
  gap.
- **`measurement` coverage (Q3, Q4, Q5, Q8, Q9).** The 5.85 % unmapped rate is
  concentrated in bare source itemids. Creatinine, lactate, glucose,
  temperature are well covered; CAM-ICU, ICDSC, and urine-output-derived
  measurements must be checked per build before Q4 and Q9 are committed.
- **Condition-code PPV (Q8, Q9, and the secondary endpoints of Q1, Q10).**
  MIMIC's ICD -> SNOMED mapping is complete but diagnosis coding is
  billing-driven. Pair every condition-code phenotype with a corroborating
  drug or procedure requirement, and keep mortality as the robust endpoint.
- **Out-of-hospital follow-up.** MIMIC-IV gives death within one year of the
  last discharge; it gives essentially nothing else post-discharge. Every
  outcome here is mortality or an in-hospital event by design.
- **No dose, no infusion timing (all).** Every "treated" arm is
  "order present in the window". Dose-response, time-varying exposure, and
  infusion-anchored time zero are out of scope until the `inputevents` ETL
  extension lands (`paper/mimic_omop_coverage_workplan.md`, workstream B).

---

## 8. Citations

Trial and meta-analysis citations are inline above with DOIs. Method and
target-trial-emulation references (Hernán & Robins 2016; Matthews et al. 2022;
Wang et al. 2023; Franklin et al. 2021; Doutreligne et al. 2025; the OMOP
reference set; empirical calibration) are in
[`validation_study_candidates.md` §"Cross-cutting method references"](validation_study_candidates.md)
and [`README.md`'s References](../README.md#references). Re-verify every
**[verify …]** tag against the DOI landing page before any citation is used in
the manuscript.
