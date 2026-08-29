# Reference extraction — Doutreligne et al. 2025 (PLOS Digital Health)

Ground-truth extraction for comparing the CausalOMOP pipeline against the
published MIMIC-IV target-trial emulation this study replicates. Compiled
2026-08-29 from the main-text PDF, supplements **S1 Fig, S5 Fig, S6 Fig,
S7 Appendix**, and the authors' analysis code (`soda-inria/causal_ehr_mimic`).
Every claim is tagged so nothing here is mistaken for a verified pipeline
result. Outstanding supplements are listed in §11.

- **Paper:** "Step-by-step causal analysis of EHRs to ground decision-making."
  *PLOS Digit Health* 4(2):e0000721. `doi:10.1371/journal.pdig.0000721`
  (published 2025-02-03).
- **Preprint (same study, more appendix detail):** "Causal thinking for
  decision making on Electronic Health Records: why and how."
  `arXiv:2308.01605` (v4, 2024-12-11). HTML: `arxiv.org/html/2308.01605v4`.
- **Analysis code:** `github.com/soda-inria/causal_ehr_mimic` (package
  `caumim/`). This is the authoritative source for exact parameters — the
  paper text under-specifies several of them.
- **Data:** MIMIC-IV **v2.2** (raw + `mimiciv_derived` MIMIC-Code tables).
  **The paper does not use OMOP.** See §9.

## Tag legend

| Tag | Meaning |
|---|---|
| `[PAPER]` | Stated in the main-text prose, Table 1, or Table 2 |
| `[CODE]` | Read from `soda-inria/causal_ehr_mimic` source |
| `[FIG]` | Transcribed from a forest-plot figure — **low-res, verify against the supplement** |
| `[SUPP]` | Not in what we hold; listed in §11 as needed from the user |
| `[INFERRED]` | Our reconstruction from conventions, not stated anywhere |

---

## 1. Errata and ambiguities in the source (read first)

1. **Cohort size disagreement — resolved.** Prose says "**18,121** patients"
   `[PAPER p.8]`; Table 2 and **S5 Fig** say **18,421**. The S5 Fig flowchart
   is internally consistent at every step (`20,288 − 1,867 = 18,421`;
   `18,421 = 3,559 + 14,862`), so **18,421 is correct and the prose "18,121"
   is a typo.** Use the exact S5 Fig numbers in §2a.
2. **PLOS sidebar metadata is a production error.** The "Citation", "Editor",
   and "Data availability" sidebar on p.1 belongs to a different article
   (a heart-murmur / phonocardiogram paper). The copyright line
   ("© 2025 Doutreligne et al."), the article body, and the reference list
   are the correct ones. Do not cite the sidebar.
3. **"RCT Gold Standard (Caironi et al. 2014)"** in Fig 3b/3c is the **ALBIOS
   trial** (Caironi P et al., *NEJM* 2014;370:1412-21, ref [82]). The
   *pooled* RCT consensus quoted in the text is a **different** source: the
   meta-analysis Xu J-Y et al., *Crit Care* 2014;18:702 (ref [80]) — 28-day
   mortality OR 0.93 (95% CI 0.80–1.08). Keep the two separate when scoring.
4. Minor prose typo p.2 ("does suffice" should read "does not suffice") — no
   analytic impact.

---

## 2. Study design — Step 1 / PICO(T)

| Component | Value | Source |
|---|---|---|
| **Population** | Adults (**age ≥ 18** at ICU admission — paper prose and S5 Fig say "over 18", the code filters `age >= 18`) with **Sepsis-3** during the ICU stay; **first ICU stay** only; **survived and stayed ≥ 24 h** ("sufficient follow-up of at least 24 hours"; code: ICU LOS ≥ 1 day) | `[PAPER p.8]` + `[CODE]` |
| **Intervention** | **Crystalloids + albumin**, both within the **first 24 h** of the ICU stay, with the first albumin dose *after* the first crystalloid dose | `[PAPER]` + `[CODE]` |
| **Control** | **Crystalloids only** in the first 24 h (no albumin in the window) | `[PAPER]` + `[CODE]` |
| **Contrast type** | Add-on (albumin added to crystalloid vs. crystalloid alone); **every** subject receives crystalloids — this is the inclusion trigger, not a covariate | `[CODE]` — matches `protocol.json.query_settings.exposure_contrast = "add_on"` |
| **Outcome** | **28-day all-cause mortality** (binary). `mortality_28days = 1` iff death date − inclusion_start ≤ 28 days | `[PAPER]` + `[CODE]` |
| **Secondary outcome** | 90-day mortality (`mortality_90days`) — defined in code, not a headline result | `[CODE]` |
| **Time** | Follow-up starts at **first crystalloid administration within the first 24 h of ICU stay** (this is "inclusion_start" / time zero). *Not* ICU admission. | `[PAPER Table 1 / Step 1]` + `[CODE]` |
| **Estimand** | Population **ATE on the risk-difference scale** (E[Y(1)] − E[Y(0)]) | `[PAPER Step 2]` |
| **Effect direction key** | Negative ATE = albumin lowers 28-day mortality ("albumin more efficient") | `[PAPER Fig 3]` |

### 2a. Selection flowchart — exact counts (S5 Fig) `[SUPP received]`

| Step | Criterion added | N remaining | N dropped (mean sex/race/age of the dropped) |
|---|---|---:|---|
| 0 | Initial population | **50,920** | — |
| 1 | Age > 18 **and** ICU LOS ≥ 1 day | **40,279** | 10,641 (Female 0.47, White 0.68, age 62.8) |
| 2 | + Sepsis-3 positive | **20,288** | 19,991 (Female 0.45, White 0.67, age 64.4) |
| 3 | + Crystalloids in first 24 h | **18,421** | 1,867 (Female 0.45, White 0.66, age 67.5) |
| 4a | + Albumin in first 24 h → **Treated** | **3,559** | — |
| 4b | Crystalloids only → **Control** | **14,862** | — |

Targets for the pipeline's cohort/feasibility step: eligible **18,421**,
treated **3,559** (19.3 %), control **14,862**. The big drop is step 2
(Sepsis-3): a diagnosis-code sepsis phenotype will not reproduce it — see
§9 and `concept_audit.md` (3,048 with a bare dx code).

### Cohort construction (exact, from `caumim/framing/albumin_for_sepsis.py`) `[CODE]`

1. Start from `mimiciv_derived.sepsis3` stays with `sepsis3 == True`.
2. Keep **first ICU stay** per patient; require **age ≥ 18** and **≥ 1 day
   ICU LOS / ≥ 1 day survival**.
3. `inclusion_start` = earliest crystalloid `inputevents` row in the first
   24 h of ICU admission. Patients with no crystalloid in 24 h are excluded.
4. `intervention_start` = earliest albumin `inputevents` row in the first
   24 h, **kept only if `intervention_start > inclusion_start`**.
5. `intervention_status` = 1 if such an albumin dose exists, else 0 (control).
6. Outcome: `mortality_28days` from `dod` vs `inclusion_start`.

### Drug item IDs `[CODE]` (MIMIC-IV `d_items`, not OMOP)

- **Albumin:** `220862` (albumin 25%), `220864` (albumin 5%).
- **Crystalloids:** `226364, 226375, 225158, 225159, 225160, 225161,
  220964, 220965, 220966, 220967, 220968` — NaCl 0.9% / 0.45% / 3%, dextrose
  combinations, OR/PACU crystalloids. (Verify this list against the current
  repo — item-ID sets have been edited over the repo's history.)

---

## 3. Target-trial / immortal-time design

- The design **deliberately allows a small gap** between time zero (first
  crystalloid) and albumin initiation, up to 24 h. The authors argue 24 h is
  short enough that the induced immortal time is negligible, and they **test
  that claim** in Step 4. `[PAPER Step 1]`
- **Immortal-time-bias vibration** (`immortal_time_bias_albumin_for_sepis.py`,
  note the filename typo): re-run with the exposure-eligibility window set to
  **0.25 d (6 h), 1 d (24 h), 3 d (72 h)** `[CODE]`. Longer window →
  albumin looks more protective (bias). Estimator for this sweep = random
  forest + IPW (`backdoor.propensity_score_weighting`), **30** bootstrap
  reps `[CODE]`.
- Other temporal biases named but not separately quantified: lead-time,
  right-censoring, attrition. `[PAPER]`
- Censoring / analysis strategy: administrative censoring at min(end of data,
  28 d); treatment fixed by the 24 h window, no time-varying update
  (ITT-analog). `[INFERRED]` — consistent with `protocol.json`, not spelled
  out in the paper.

---

## 4. Identification — Step 2: the expert DAG (S6 Fig)

Already transcribed in **`reference_dag.json`** (23 confounder nodes: 20
observed + 3 unobserved). Key facts for scoring:

- Built by the authors with a clinician in **DAGitty**. `[PAPER Step 2]`
- Every white node → arrow to **both** treatment and outcome (dense
  confounder fan). Caption: "only the most important edges" are drawn.
- **Grey / unobserved nodes** ("not controlled for since not available"):
  **Socio-economical factor, Comorbidity, lifestyle**. `[SUPP S6 Fig]` — so
  even the full observed adjustment set leaves an open backdoor path (this is
  asserted by `test_reference_dag_albumin_sepsis.py`).
- Comorbidity → SOFA and Comorbidity → SAPSII are the only non-fan edges;
  transcribed at moderate confidence.
- **`[SUPP]` A higher-resolution S6 Fig (or the DAGitty source) is needed**
  to confirm the AKI-stage ↔ Dialysis(RRT) relationship and rule out any
  other non-fan edges.

### DAG node ↔ analysis feature reconciliation

The DAG has **20 observed nodes**; the estimation code uses **24 features**
because one node fans out and one feature has no clean node:

| DAG node | Analysis feature(s) |
|---|---|
| Age at admission | `admission_age` |
| Gender | `Female` |
| Race | `White` |
| Weight | `Weight` |
| Emergency at admission | `Emergency_admission` |
| Mean Blood Pressure | `mbp` |
| Heart rate | `heart_rate` |
| Lactate | `lactate` |
| Insurance, Medicare | `Insurance_medicare` |
| **Antibiotics** | `Carbapenems`, `Aminoglycosides`, `Beta-lactams`, `Glycopeptide` (4) |
| SOFA | `SOFA` |
| SAPSII | `SAPSII` |
| Temperature | `temperature` |
| Respiratory rate | `resp_rate` |
| Ventilation | `ventilation` |
| Vasopressors | `vasopressors` |
| SpO2 | `spo2` |
| Urine output | `urineoutput` |
| AKI stage | `aki_stage` |
| Dialysis (RRT) | `RRT` |
| Socio-economical factor | *(unobserved — no feature)* |
| Comorbidity | *(unobserved; text says SOFA/SAPS II act as proxies)* |
| lifestyle | *(unobserved — no feature)* |
| *(no explicit node)* | `suspected_infection_blood` (infection/sepsis marker) |

---

## 5. Confounder sets (Step 2 → Step 4b) `[CODE]` — `FEATURE_SETS`

**"All confounders" — 24 features** (matches Fig 3b "24 features"):

```
admission_age, Female, Emergency_admission, Insurance_medicare, White,
Carbapenems, Aminoglycosides, Beta-lactams, Glycopeptide, vasopressors,
Weight, lactate, heart_rate, spo2, mbp, urineoutput, temperature,
aki_stage, SAPSII, resp_rate, SOFA, RRT, ventilation,
suspected_infection_blood
```

| Fig 3b label | Definition | Count | Check |
|---|---|---:|---|
| All confounders | full list above | 24 | ✓ matches paper |
| Without drugs | drop `Carbapenems, Aminoglycosides, Beta-lactams, Glycopeptide, vasopressors` | 19 | ✓ matches paper |
| Without measurements | drop `Weight, lactate, heart_rate, spo2, mbp, urineoutput, temperature, aki_stage, SAPSII, resp_rate, SOFA` (11) | 13 | ✓ matches paper |
| Socio-demographics | `admission_age, Female, Emergency_admission, Insurance_medicare, White` | 5 | ✓ matches paper |
| *(code only, not in Fig 3b)* Without measurements and drugs | demographics + `RRT, ventilation, suspected_infection_blood` | 8 | — |
| Unadjusted risk difference | raw difference in means, no model | 0 | ✓ in Fig 3b |

Charlson comorbidities are defined in the repo but **not** in the 24-feature
set — consistent with "Comorbidity" being a grey DAG node.

---

## 6. Estimation — Step 3 `[PAPER]` + `[CODE]`

- **Packages:** `dowhy` for IPW/PSM; `EconML` for T-learner, AIPW (DRLearner),
  DML.
- **Causal estimators run:** difference-in-means (naive), Propensity Score
  Matching, Inverse Propensity Weighting, Outcome model / G-formula
  (T-learner), Double Machine Learning, Doubly-Robust / AIPW.
- **Nuisance models (2):**
  - `Forests` — RandomForest; randomized search over
    `n_estimators ∈ {10, 100, 200}`, `max_depth ∈ {3, 10, 50}`.
  - `Regularized LR` — ridge (outcome) / logistic (treatment);
    `alpha` resp. `C` over `np.logspace(-3, 2, 10)`.
- **Confidence intervals:** bootstrap, **50** repetitions (main analysis;
  30 for the immortal-time sweep). `[CODE]`
- **`RANDOM_STATE = 42`.** `[CODE]`
- **Confounder aggregation** (time series → one value per patient, all before
  time zero): `first`, `last`, and `first`+`last` as separate features.
  The paper reports the `first`+`last` strategy in the main figures; S9 Fig
  shows the aggregation sweep. `[PAPER Step 3 / Step 4c]`
- **Missing data:** median imputation for numeric features; one-hot encoding
  for categoricals (missing category dropped). `[PAPER Step 3]`
- **No cross-fitting in the dowhy path** is flagged by the authors as a
  limitation (S3 Appendix). `[PAPER]`

---

## 7. Results to compare against — Step 4 (vibration analysis)

> **All numbers in this section tagged `[FIG]` are transcribed from the
> low-resolution Fig 3 forest plot in the main PDF.** They are here so the
> pipeline has a target, but the exact point estimates and CIs must be read
> from **S7 Fig / S8 Fig** (see §11) before being used to score anything.
> Scale = risk difference in 28-day mortality (ATE). Negative = albumin
> protective.

### 7a. Immortal time bias (Fig 3a) `[FIG]`

| Exposure window | ATE (95% bootstrap CI) |
|---|---|
| 6 h | ≈ +0.01 (−0.01 to +0.03) |
| 24 h (main) | ≈ −0.00 (−0.01 to +0.01) |
| 72 h | ≈ −0.02 (−0.03 to −0.01) |

Qualitative claim to reproduce: **longer window → more (spurious) apparent
benefit of albumin.** `[PAPER]`

### 7b. Confounder choice (Fig 3b) `[FIG]`

| Confounder set | ATE (95% bootstrap CI) |
|---|---|
| ALBIOS RCT (Caironi 2014) reference | ≈ −0.00 (−0.05 to +0.05) |
| All confounders (24) | ≈ −0.00 (−0.01 to +0.01) |
| Without drugs (19) | ≈ −0.01 (−0.02 to +0.00) |
| Without measurements (13) | ≈ −0.02 (−0.04 to −0.00) |
| Socio-demographics (5) | ≈ −0.03 (−0.05 to −0.02) |
| Unadjusted | ≈ −0.07 (−0.07 to −0.07) |

Qualitative claims to reproduce: dropping **drugs** barely moves the estimate;
dropping **measurements** or going socio-demographics-only introduces
**sizeable** bias toward apparent benefit; unadjusted is badly confounded.
`[PAPER]`

### 7c. Estimator / model selection (Fig 3c) `[FIG]`

| Estimator | `Regularized LR` | `Forests` |
|---|---|---|
| Difference in means | −0.07 (−0.07 to −0.07) | — |
| Propensity Score Matching | ≈ +0.01 (−0.00 to +0.04) | ≈ +0.67 (+0.07 to +1.48) — **outlier** |
| Inverse Propensity Weighting | ≈ −0.03 (−0.06 to +0.00) | ≈ −0.03 (−0.04 to −0.02) |
| Outcome model (T-learner) | ≈ −0.05 (−0.12 to +0.02) | ≈ −0.01 (−0.21 to +0.18) |
| Double Machine Learning | ≈ −0.06 (−0.07 to −0.04) | ≈ −0.01 (−0.02 to −0.00) |
| Doubly Robust (AIPW) | ≈ −0.08 (−0.15 to −0.02) | ≈ −0.00 (−0.03 to +0.01) |

Qualitative claims to reproduce `[PAPER]`:
- **AIPW + random forests** is the estimator the authors endorse; its CI
  **covers 0**, matching the RCT consensus.
- Forest-based nuisances land closer to "no effect" than linear ones (better
  fit of non-linear relationships).
- Doubly-robust methods give the **narrowest** CIs; outcome-regression the
  widest.
- PSM + forests is unstable (huge positive outlier).

### 7d. RCT anchors `[PAPER]`

- Pooled meta-analysis (Xu 2014, ref [80]): 28-day mortality **OR 0.93
  (0.80–1.08)**; 90-day **OR 0.88 (0.76–1.01)** → "no effect" overall.
- ALBIOS (Caironi 2014, ref [82]): no overall benefit; **post-hoc septic
  shock subgroup benefits.**
- Li 2020 network meta-analysis (ref [81]): consistent with no net benefit.

---

## 8. Treatment heterogeneity — Step 5 / CATE `[PAPER]` + `[CODE]` + `[SUPP S7 Appendix received]`

- **Effect modifiers (4, binary):** age (**< 60 vs ≥ 60**), `septic_shock`
  ("pre-treatment septic shock" / "septic shock at inclusion" — a derived
  clinical state, **not** a diagnosis code), sex (`Female`), race (`White`).
  Fig 2/Fig 3 of S7 Appendix also feed continuous `admission_age` to the
  final model; the ≥ 60 split is for the box-plot display only.
- **`septic_shock` exact definition** `[CODE]` (`caumim.variables.selection.
  get_septic_shock_from_features`): `sepsis3 == True` **AND** any vasopressor
  (dopamine / epinephrine / norepinephrine / phenylephrine / vasopressin /
  dobutamine / milrinone, value > 0) **AND** serum lactate **≥ 2 mmol/L**
  (code uses `>= 2`; the Sepsis-3 text and the code's own docstring say
  `> 2`). MAP < 65 is *simplified to* "on vasopressors". All ascertained
  pre-treatment.
- **CATE method (S7 Appendix):** Double ML residual-on-residual —
  regress `Ỹ = Y − m̂(X)` on `Ã = A − ê(X)` interacted with the
  heterogeneity features `X_CATE`, minimising `Eₙ[(Ỹ − τ(X_CATE)·Ã)²]`.
  `predict_cate(..., alpha=0.05)`.
- **Sample split (S7 Appendix):** main-analysis data split **train 0.8 /
  test 0.2**; causal estimator *and* final CATE model fit on train, CATEs
  **reported on the held-out test set only.** (Same 80/20 split as S1 Fig's
  motivating example.)
- **Final model choice matters (S7 Appendix, its Fig 1–3):**
  - **Ridge = the endorsed final model** (main-text Fig 4). Gives a clean
    monotone CATE-vs-age trend; CATEs straddle 0, consistent with the null
    ATE. Range on the white-male-no-shock stratum ≈ **+0.05 at age 20 →
    −0.05 at age 100** (S7 App Fig 3).
  - **Random forest final model = negative example.** Its CATEs are
    **positive for every subgroup** (box medians ≈ 0 to +0.25, whiskers to
    ~+0.6) — *inconsistent* with the null ATE; the authors read this as the
    RF final stage **underfitting**. Do not treat an all-positive CATE
    picture as a success.
- **Subgroup ordering to reproduce (sign/ordering, not magnitude), ridge
  final model** `[PAPER Fig 4]`: albumin more beneficial (more negative
  CATE) for **age ≥ 60** > **septic shock** > **male**, and **≈ flat for
  race**. In S7 App Fig 2 the 8 crossed strata rank from least to most
  beneficial roughly: *white female w/o shock* (least) → … → *non-white
  male w/ septic shock* (most beneficial).
- **Literature heterogeneity anchors (S7 Appendix "Known heterogeneity"):**
  - Caironi 2014 / ALBIOS: septic shock **RR 0.87 (0.77–0.99)** vs no shock
    **RR 1.13 (0.92–1.39)**.
  - SAFE-TBI 2007: *higher* mortality with albumin in severe traumatic
    brain injury (opposite direction — relevant as a "don't over-generalise"
    note, TBI is out of this cohort).
  - Zhou 2021 (MIMIC-IV retrospective, ref [83]): RMST subgroup gaps —
    sepsis vs septic shock 3.47 vs 2.58; age ≥ 60 3.75 vs 2.44; male 3.4 vs
    2.69; **none statistically significant.**
- **Still `[SUPP]`:** exact ridge-CATE median/IQR per subgroup behind
  main-text Fig 4 (only readable from a high-res Fig 4).

---

## 9. Mapping to the CausalOMOP pipeline + comparison metrics

**Biggest replication gap: the paper is raw-MIMIC, the pipeline is OMOP.**
The paper leans on `mimiciv_derived` MIMIC-Code tables that have no direct
OMOP equivalent and must be reconstructed:

| Paper artifact (`mimiciv_derived`) | OMOP reconstruction needed |
|---|---|
| `sepsis3` (SOFA ≥ 2 + suspected infection) | Sepsis-3 logic over `measurement` + `drug_exposure` + culture `measurement`/`observation`; **cannot** use a sepsis SNOMED code (see `concept_audit.md` — diagnosis-based phenotype gave 3,048 vs ~18.4k) |
| `sofa`, `sapsii` | component-wise from vitals/labs/vasopressors/ventilation/GCS/urine |
| `first_day_*` vitals/labs | `measurement` aggregated in the first-day window |
| `charlson` | condition history (only if a comorbidity feature is added) |
| `aki_stage` (KDIGO) | creatinine trajectory + urine output + RRT |
| albumin / crystalloid `inputevents` itemids | **Not fixable with a concept set alone.** The OHDSI/MIMIC ETL in `external/ohdsi_mimic` does not stage `inputevents` at all (`etl/staging/st_icu.sql` stages only `procedureevents`/`datetimeevents`/`chartevents`/`outputevents`/`d_items`); `drug_exposure` is built from `prescriptions` + `pharmacy` only (`lk_drug.sql`), and every row in the current build is `drug_type_concept_id = 32838 "EHR prescription"`. All ICU infusions — fluids, vasopressors, sedation, IV insulin, blood products — are therefore absent from OMOP. See §9a. |
| ICU stay / `icustays` | `visit_detail` + `care_site` (`icu_time_zero.json`) |
| septic shock | Sepsis-3 shock criteria, not concept `196236` |

### 9a. ETL coverage findings (verified 2026-08-29 against `data/mimiciv_omop.duckdb`)

1. **`drug_exposure` is well mapped: 96.8 % of 15.45 M rows carry a standard
   concept, 3.2 % are `concept_id = 0`.** The local DuckDB export reflects
   the post-2026-08-11 vocab fix — it is **not** stale (an earlier note in
   this doc claiming ~96.7 % *unmapped* was an inverted-predicate error).
   Prescription-drug questions are workable now: aspirin ~177 K rows,
   norepinephrine 19.4 K / 10 K persons, phenylephrine 35 K, epinephrine
   9.7 K, propofol 53.6 K, midazolam 23 K, fentanyl 78 K, hydrocortisone
   23.5 K, sodium chloride 1.87 M.
2. **`inputevents` not in the ETL scope.** `st_icu.sql` never stages it;
   `drug_exposure` is 100 % `drug_type_concept_id = 32838` (prescription
   orders). So even the drugs above are represented as *orders*, not
   titrated administrations — **no dose/rate/duration, and timing is
   order-time, not infusion-start.** Fine for "ever exposed" designs;
   not for dose-response or infusion-precise time-zero (which is exactly
   what this study needs).
3. **Drugs that appear only as infusions are absent entirely** (0 rows via
   `concept_ancestor`): `albumin human`, `lactated ringers` (`%ringer%` → 0),
   `vasopressin`, **`insulin human`** (affects the IV-insulin run). Some
   NaCl resuscitation-volume bags (1000 mL / 500 mL) are also still
   `concept_id = 0` while flushes and small bags map.
4. **`d_items_to_concept.csv` has the infusion itemids but as blank stubs.**
   Rows exist for `inputevents,220862,Albumin 25%`, `225158 NaCl 0.9%`, …
   with every `target_concept_id` column empty — so loading `inputevents`
   also needs those targets curated.
5. **The mapping-coverage gate already exists** (`_check_mapping_coverage` in
   `causal/orchestrator/workflow.py`, floor 5 %, wired into feasibility) and
   fires correctly for *name-matched-but-unmapped* concepts. Blind spot:
   a role that resolves to ~0 total rows (an `inputevents`-only drug
   entirely absent from `drug_exposure`) has `< 50` evidence rows → gate
   stays silent → feasibility returns a tiny cohort with no ETL-scope
   message. Worth a targeted "near-zero rows for a concept that should be
   common" check.
6. **Vocabulary is not the blocker for albumin** — standard RxNorm
   `albumin human … Injection` concepts (`standard_concept='S'`) are present
   in `CONCEPT`. The gap is ETL source→concept mapping + `inputevents` scope.

**Per-stage comparison:**

| Pipeline stage | Compare to | Metric |
|---|---|---|
| DAG construction | `reference_dag.json` | SHD, SID (`causal/agents/dag_metrics.py`) |
| Concept mapping | `reference_concept_mappings.json` | recall@k, MRR (known ETL gaps excluded) |
| Cohort / feasibility | eligible ≈ 18.4k; treated 3,559; control 14,862; Table 2 baseline stats | count ratios, SMD of baseline table vs Table 2 |
| Estimand | risk-difference ATE, CATE by the 4 modifiers | exact-match on estimand spec |
| Point estimate | §7c AIPW+RF row (≈ 0, CI covers 0) | does the pipeline's headline CI cover 0 |
| Vibration pattern | §7a, §7b **gradients** (the direction and rough size of each bias), not single numbers | sign + ordinal agreement |
| CATE | §8 subgroup **sign & ordering** | septic-shock and age≥60 effects more protective; race ~flat |

**The discriminating validation signal is the pattern, not the headline
number.** The headline ATE is a null (RCT consensus is null), so "our CI
covers 0" is weak evidence. The parts of this study that actually test the
pipeline are: (a) the confounder-set bias gradient in §7b, (b) the
immortal-time gradient in §7a, (c) the estimator-instability picture in §7c,
(d) the CATE subgroup ordering in §8. Prioritise reproducing those.

---

## 10. Corrections needed to the existing study scaffold

**Status: all rows below applied 2026-08-29.** The wording/structure corrections
are done; the concept-id work (replacing the `source_value_rule` placeholders
with approved albumin / crystalloid concept ids, adding resuscitation-fluid
exclusions) is *not* here — it depends on the `inputevents` ETL work
(`docs/mimic_omop_coverage_workplan.md` workstream B) and a human concept-set
approval gate.

| File | Current | Should be | Applied |
|---|---|---|---|
| `protocol.json` `target_trial.time_zero` | `"ICU admission"` | first crystalloid administration within the first 24 h of the ICU stay (`inclusion_start`) | ✅ `time_zero` rewritten; `treatment`/`comparator`/`assignment_procedure`/`analysis_strategy`/`*_value` re-anchored off ICU admission for consistency |
| `protocol.json` `target_trial.grace_period_hours` | `24` | 24 h is the **exposure-eligibility window** (albumin must start ≤ 24 h *and* after crystalloid), not a grace period in the usual sense — reword to avoid ambiguity | ✅ numeric key kept at `24` (consumed by the query builder); ambiguity resolved with a `design_constraints` note |
| `protocol.json` `eligibility_criteria` | "Sepsis diagnosis at or within a window of ICU admission" | Sepsis-3 (SOFA ≥ 2 + suspected infection); reconstruct, do not use a sepsis condition code | ✅ full list rewritten (Sepsis-3, first ICU stay per patient, ≥ 24 h survival / ≥ 1 day LOS, crystalloid-triggered inclusion) |
| `protocol.json` `secondary_estimands` | CATE by age / sex / septic shock | add **race**; note age split is ≥ 60 | ✅ `"CATE by race"` added; age ≥ 60 split + "septic shock = derived state, not concept 196236" recorded in `design_constraints` |
| `protocol.json` `published_benchmarks` | `approximately_eligible: 18421`, `albumin_treated: 3559` | correct (S5 Fig confirms exactly); add `control: 14862` and the upstream steps (`sepsis3_and_age_los: 20288`, `initial: 50920`) so the cohort step can be checked stepwise, not just at the end | ✅ full `selection_flowchart` block added (50,920 → 40,279 → 20,288 → 18,421 → 3,559 / 14,862) |
| `icu_time_zero.json` `time_zero` | "first recorded adult ICU visit_detail start" | correct as the **ICU-admission anchor**, but the study's **follow-up clock** starts at first crystalloid — keep both, clearly labelled | ✅ `icu_admission_anchor` + `follow_up_clock_start` split out, limitation added |
| `reference_dag.json` | 23 nodes | fine; add a note that estimation uses 24 *features* (Antibiotics → 4) and one extra feature (`suspected_infection_blood`) with no DAG node | ✅ `analysis_feature_reconciliation` key added (20 observed nodes → 24 features); node count unchanged |

### 10a. Follow-on consistency fixes (applied 2026-08-29, after the table above)

Correcting `time_zero` / `eligibility_criteria` left a few neighbouring fields
inconsistent; those were fixed in the same pass:

| File | Fix |
|---|---|
| `protocol.json` `status` | `"draft_blocked_on_drug_concept_mapping"` → `"draft_blocked_on_inputevents_etl_scope_and_sepsis3_phenotype"` + a `status_detail` string (drug mapping is no longer the blocker per §9a; the blockers are `inputevents` ETL scope, the Sepsis-3 phenotype, and baseline covariate coverage) |
| `protocol.json` `source_question` | "…initiated within 24 hours of ICU admission…" → "…both initiated within the first 24 hours of the ICU stay…" |
| `protocol.json` `concept_sets.sepsis` | `status` → `"placeholder_requires_sepsis3_reconstruction"` + `note`: 132797 stays as a concept-mapping benchmark target but must not define the population (dx-code phenotype gives ~3,048 vs ~18,421) |
| `protocol.json` `concept_sets.septic_shock` | `status` → `"placeholder_derived_state_not_a_condition_code"` + `note`: 196236 is a valid search target, but the effect modifier is a derived Sepsis-3 shock state |
| `reference_concept_mappings.json` note | stale `status='candidate_pending_human_approval'` cross-reference updated to match the placeholder statuses |

### 10c. Wording-precision fixes (applied 2026-08-29, after a consistency re-check)

| File / field | Fix |
|---|---|
| `protocol.json` `design_constraints` grace-period note | "albumin must start within 24 h of **time zero**" → "both the first crystalloid and the first albumin dose must fall within the first 24 h of **the ICU stay**, albumin after the first crystalloid". Time zero is itself inside that window (the first crystalloid), so "24 h of time zero" could nominally admit albumin up to ~48 h post-admission. The `treatment` / `source_question` / `treatment_value` fields were already correct. |
| `protocol.json` `assignment_procedure` | same fix — "window that begins at time zero" → "the first 24 h of the ICU stay (which contains time zero)". |
| `protocol.json` `concept_sets.septic_shock.note` + `design_constraints` | "vasopressor requirement plus lactate **> 2**" → the exact `caumim.variables.selection.get_septic_shock_from_features` definition: Sepsis-3 positive **AND** any of 7 named vasopressors **AND** lactate **`>= 2`** mmol/L (code filters `>= 2`; Sepsis-3 text says `> 2`). Recorded in §8. |
| `reference_paper_extraction.md` §2 / §8 | §2 population age harmonised to `≥ 18` (code-authoritative; paper prose / S5 Fig say "over 18"); §8 gained the verified `septic_shock` formula. |

### 10b. Still open (not yet done)

| File | Correction | Blocked on |
|---|---|---|
| `protocol.json` `concept_sets.albumin` / `crystalloids` | replace the `source_value_rule` regex placeholders with approved standard concept ids; add resuscitation-fluid exclusions (flush, irrigation, drug diluent, nebulised, ophthalmic). Also refresh the `status` strings — `albumin`'s `"present_but_unmapped_to_standard_concept"` is inaccurate per §9a §3 (the infusion rows are *absent entirely*, not present-but-unmapped) | `inputevents` ETL (workplan B) + a human concept-approval gate |
| `protocol.json` `first_pass_covariates` | 9 entries; the reference uses 24 analysis features (§4/§5) — expand once the covariate mapping is scoped | baseline covariate coverage work |
| `reference_concept_mappings.json` `albumin` case | `expected_concept_id: null` is likely wrong for the *search* benchmark: §9a §6 says standard RxNorm "albumin human … Injection" concepts (`standard_concept='S'`) **do** exist in `CONCEPT`. The null belongs on a downstream ETL-coverage metric, not on "can the search find the concept". Needs the actual concept id filled in. | a concept lookup + deciding which metric the gap belongs to |

---

## 11. Supplementary material — status

**Received:** S5 Fig (→ §2a), S7 Appendix + its Fig 1–3 (→ §8), S1 Fig
(motivating example — 80/20 split, HistGradientBoosting; not needed for the
causal comparison, only confirms the split).

| # | Still needed | What it settles | Priority |
|---|---|---|---|
| 1 | **S7 Fig** — complete results, main analysis | Exact ATE point estimates + CIs behind Fig 3c (replaces §7c `[FIG]` guesses) | **High** |
| 2 | **S1 Table** — full confounder description | Baseline stats (treated vs control, missingness) for all 24 features → baseline-table SMD comparison; also the exact `suspected_infection_blood` / comorbidity handling | **High** |
| 3 | **S8 Fig** — complete results, immortal time bias | Exact numbers behind Fig 3a; also causal-forest + other aggregation variants | Medium |
| 4 | **high-res main-text Fig 4** | Exact ridge-CATE median/IQR per subgroup | Medium |
| 5 | **S3 Fig** — graphical timeline | Visual confirmation of the time-zero / grace / follow-up alignment | Medium |
| 6 | **S6 Fig high-res** (or DAGitty source) | Non-fan edges (AKI↔RRT, Comorbidity→scores) for exact SHD/SID | Medium |
| 7 | **S6 Appendix** — smaller confounder sets | Cross-check the 19 / 13 / 5 memberships (largely covered by `[CODE]` in §5) | Low |
| 8 | **S9 Fig** — aggregation vibration | Whether first / last / first+last matter | Low |
| 9 | **S2 / S4 Appendix** — estimator + package details | Cross-fitting, CI formulas, EconML settings (mostly in `[CODE]`) | Low |

Alternative to items 1 and 3: the authors' result CSVs are **not** committed
to `soda-inria/causal_ehr_mimic` (only the report scripts are). Re-running
their `reports/1_*.py` … `reports/4_*.py` against a local MIMIC-IV v2.2 would
regenerate the exact numbers, but that is a separate task from this pipeline
comparison.
