# Design note: target-trial eligibility refinements

Four eligibility/ascertainment/covariate-coverage refinements the validation
question bank ([`validation_question_bank.md`](validation_question_bank.md))
needs and the current cohort builder could not express. Written 2026-09-08;
section 2/3 priority revised the same day after a re-check (see the Priority
section for the reasoning); section 4 added the same day once the first
end-to-end run hit it. Items 1 and 3a are implemented; item 2 is specified
here and deferred; item 3b is future work; item 4 is a documented current
gap, not yet scheduled.

The cohort builder ([`causal/queries/feasibility.py`](../causal/queries/feasibility.py))
today supports: a single population concept set plus ANDed
`additional_population` semi-joins, a grace-period exposure window
`[t0, t0 + grace)`, a fixed calendar follow-up `[t0, t0 + follow_up_days]`,
and a per-row `censored` column with three reviewed strategies
([`causal/roadmap.py`](../causal/roadmap.py) `CENSORING_STRATEGIES`), all of
which assume administrative (non-informative) censoring.

## 1. Washout / new-user exclusion — implemented

**What.** Exclude persons who had a qualifying exposure in the look-back
window `[t0 - washout_days, t0)`, and persons whose observation record does
not actually cover that window (so "no prior exposure" is an observed fact,
not an artefact of a short record).

**Why.** The active-comparator and single-drug new-user designs (Q1, Q5, Q8,
Q10) all require it. Standard, citable: Ray 2003 (*new-user designs*,
Am J Epidemiol 158:915); Lund, Richardson, Stürmer 2015 (*the active
comparator, new user study design*, Curr Epidemiol Rep 2:221); Suissa &
Azoulay 2012 (*time-related biases*, Diabetes Care 35:2665) for the
prevalent-user hazard. The look-back-coverage requirement is the standard
`priorObservation` rule from OHDSI `CohortMethod`.

**MIMIC caveat.** MIMIC-IV patients enter the record at hospital / ED
admission; pre-admission drug data is limited to admission medication
reconciliation. A "no drug in the prior 90 days" rule is therefore close to
vacuous for most patients unless the look-back-coverage check is applied
alongside it. The two diagnostic counts below let a reviewer see which
regime they are in before approving.

**Contract.**

- `protocol["query_settings"]["washout_days"]` (int, default `0` = disabled).
  Reviewer-set via `WorkflowManager.set_washout_window`, mirroring
  `set_exposure_contrast` (writes `query_settings`, never `hypothesis.json`,
  invalidates gates from `analysis_plan`).
- The washout concept set is derived from the exposure concept sets, not
  declared separately:
  - `mutually_exclusive`: treatment class ∪ comparator class (Q1: no PPI
    **and** no H2RA in the look-back).
  - `vs_unexposed`: treatment class only.
  - `add_on`: treatment (the adjunct) only — the comparator is the backbone
    every cohort member is on by design, so a washout on it would empty the
    cohort.
- SQL: a `washout` CTE keyed by `person_id` computes two booleans,
  `washout_new_user` (`NOT EXISTS` prior exposure in
  `[t0 - washout_days, t0)`) and `washout_lookback_covered` (`EXISTS` an
  `observation_period` spanning `t0 - washout_days` through `t0`, matching
  the span convention the `censored` column already uses).
  `analytic_cohort_sql` ANDs both into the inclusion filter;
  `feasibility_sql` reports them as attrition counts and never filters
  (same split as the exposure-combination counts):
  `washout_excluded_prevalent_users`,
  `washout_excluded_insufficient_lookback`, `eligible_after_washout`.
- `washout_days == 0` emits byte-identical SQL to before.

**Not done, follow-up.** `dag_identification` reads the free-text
`eligibility_criteria`; it does not yet get a structured "new user" node
from `washout_days`. The reviewer should still state the washout in the
prose criteria. A UI control (sibling of the exposure-contrast selector) is
not wired.

## 2. Landmark / immortal-time handling — specified, deferred

**Method (this part is settled).** Naively excluding everyone who dies
inside the grace window is **not** a clean immortal-time fix on its own: it
conditions on post-baseline survival, and it leaves the `[t0, L]` span in
follow-up as guaranteed event-free person-time, which is itself immortal
time. The correct construction is a **landmark**: pick `L`, require survival
to `L` and no primary outcome before `L`, classify exposure over `[t0, L]`,
and start the outcome clock at `L` with the horizon measured from `L`.
`L` must be `>=` the end of the exposure-ascertainment window (`L` earlier
than that leaves residual classification-dependent immortal time); the
clean default is `L =` the end of the grace window. Citable: Hernán &
Robins 2016 (Am J Epidemiol 183:758); Hernán 2018 (*how to estimate the
effect of treatment duration*, BMJ 360:k182); Maringe 2020 (*trial
emulation in the presence of immortal-time bias*, IJE 49:1719); Dafni 2011
(landmark analysis, Circ Cardiovasc Qual Outcomes 4:363).

Early **discharge** is not part of the landmark criterion — that is an
eligibility / estimand choice (population becomes sicker-on-average, as in
Q6's "expected to need ≥ 7 more days") and, for in-hospital-mortality
questions, it is item 3's concern (the landmark criterion there extends to
"alive **and** in-hospital at `L`"). The bare landmark criterion is "alive
at `L`".

**Why it is deferred, not next.** A re-check against the bank changed the
priority:

- The bank's own §2 assigns "immortal-time-bias handling (landmark / grace
  period)" to **Q4 and Q6 only** — the procedure-timing questions, which
  §6 lists as "documented but lower priority", not first wave.
- Of the three headline studies (Q1, Q8, Q5), only **Q8** (steroids vs
  none) has material immortal time. Q1 and Q5 are active-comparator
  designs: both arms require having received their drug within the same
  window, so the survival selection is symmetric between arms and largely
  cancels in the contrast. For Q5 a 48 h landmark would additionally drop
  the earliest KDIGO-AKI events, which is the signal the study is built to
  detect — landmark would hurt it.
- For Q8, immortal time (biases toward "steroids protective") runs
  **opposite** to the confounding by indication the study showcases (the
  bank: "confounding by indication runs toward the null or toward harm
  here"). Removing one without the other could make the Q8 estimate more
  conservative and complicate the sensitivity demonstration. The magnitude
  needs to be measured on a real end-to-end run before a fix is worth
  committing.
- The bank already treats early-death handling as a **per-study
  sensitivity analysis**, not a core feature (Q7 risks: "a sensitivity
  analysis excluding early deaths (< 48 h) ... is needed").

**When to build it, and as what.** A per-study *sensitivity-analysis* mode
("require survival and no outcome to day `X`, follow from `X`"), built
correctly (exclusion **and** clock shift, never exclusion alone), once at
least one study runs end to end or when Q4/Q6 enter scope. Contract sketch
if/when it is built: `query_settings.landmark_enabled` (bool, default off)
with `L = ceil(grace_period_hours / 24)` days so there is no `L`-vs-grace
degree of freedom; a `landmark` CTE with `survived_to_landmark` and
`no_outcome_before_landmark` booleans (mirroring the `washout` CTE); the
`analytic` outcome windows rebased to `[t0 + L, t0 + L + follow_up_days]`;
an emitted `landmark_datetime` column and `follow_up_days` documented as
"from the landmark"; feasibility counts `landmark_excluded_early_death` /
`landmark_excluded_early_outcome` / `eligible_after_landmark` per arm; the
estimand gate and report stating the landmark-conditional population and
the from-`L` horizon. `death_date` is DATE-only in the OMOP CDM, so `L` is
in whole days.

**Independent of the schedule:** the paper's methods/limitations should
cite the landmark method and clone-censor-weight, and state that for Q4/Q6
the pipeline does a documented landmark simplification of a time-varying
strategy and has no CCW (Gaber et al. 2024, *the clone-censor-weight method
in pharmacoepidemiology*, Curr Epidemiol Rep).

**Validation when built.** Add an immortal-time DGP to
[`causal/simulation.py`](../causal/simulation.py) (treatment-initiation
time correlated with survival): show assign-at-`t0` is biased and the
landmark recovers the truth, the same way the nonlinear DGP validates
TMLE over g-computation.

## 3. In-hospital mortality

**This was a live correctness gap, not a refinement.** Q1's primary outcome
is "in-hospital all-cause mortality"; Q8's is "in-hospital / 30-day". The
builder could not express "in-hospital": `follow_up_days` is a fixed
calendar window that treats every patient as hospitalised for the same
number of days. Both scaffolded headline studies rested on that fiction.

**Correction to the obvious approach.** A `censor_at_discharge` mode routed
through the existing `censored` column and the IPCW g-formula would be a
methodological error: discharge alive is strongly prognostic (informative
censoring), and every current `CENSORING_STRATEGIES` value and the
TMLE + IPCW gate rejection reason are premised on *administrative*
censoring.

### 3a — in-hospital-death-by-horizon — implemented

**What.** An outcome-definition mode for a death outcome: `Y = 1` iff the
person died on or after `t0`, no later than the follow-up horizon, **and**
no later than the end of the index hospitalization
(`eligible.visit_occurrence_id` -> `visit_occurrence.visit_end_datetime`).
A person discharged alive before the horizon is `Y = 0` even if they later
die out of hospital (the competing-risks subdistribution view: surviving to
discharge is surviving the at-risk period). A person still admitted at the
horizon with no death is `Y = 0` (didn't die within the horizon). It is a
reframe of the outcome definition, matching what PEPTIC (in-hospital),
CAPE COD (28-day) and ADRENAL (90-day) report. **No `censored` column, no
estimator change; `follow_up_days` still measured from `t0`** — the landmark
clock shift is item 2.

**Contract.**

- `protocol["query_settings"]["mortality_ascertainment"]` -- `"any_location"`
  (default, unchanged behavior) or `"in_hospital"`. Reviewer-set via
  `WorkflowManager.set_mortality_ascertainment`, mirroring
  `set_exposure_contrast` (writes `query_settings`, never `hypothesis.json`,
  invalidates gates from `analysis_plan` because `Y` changes).
- SQL: the death branch of `_cohort_ctes` LEFT JOINs `visit_occurrence` on
  the index visit and ANDs
  `d.death_date <= COALESCE(DATE(v.visit_end_datetime), <horizon>)` into
  `outcome_event` (a NULL `visit_end_datetime`, an unclosed source visit, is
  read as "still admitted"). `feasibility_sql` adds one diagnostic column,
  `deaths_after_discharge_within_horizon` -- the deaths the
  `"any_location"` -> `"in_hospital"` choice reclassifies as non-events -- so
  a reviewer sees how far the two estimands diverge before the gate.
  `analytic_cohort_sql` needs no filter change (`Y` picks up the modified
  `outcome_event`). The SQL builders reject `"in_hospital"` with a non-death
  outcome. `visit_occurrence` added to `query_validation.ALLOWED_TABLES`
  (already in OMCP's `OMOP_TABLES`).
- `"any_location"` emits byte-identical SQL to before.

**Not done, follow-up.** Same as item 1: no structured `dag_identification`
awareness (state it in the prose `outcome_definition`), no UI control.
Multi-hospitalization handling is deliberately simple -- only the index
hospitalization counts; a readmit-and-die within the horizon is `Y = 0`.

### 3b — competing-risks survival estimator — future work

Out of scope for the workshop paper. A true survival estimator with
discharge as a competing event (cause-specific / Fine-Gray subdistribution
hazards; Young et al. 2020, *a causal framework for classical statistical
estimands ... with competing events*, Stat Med 39:1199). Name it in the
paper's limitations next to clone-censor-weight. 3a covers the binary
in-hospital-mortality endpoint the bank actually uses; 3b would only be
needed for a time-to-event contrast.

## 4. Covariate coverage: visit_occurrence attributes -- live gap, documented

**What.** `causal.queries.feasibility.CovariateSpec`/`DOMAIN_TABLES` extract a
baseline covariate from exactly one of `condition_occurrence`,
`drug_exposure`, `procedure_occurrence`, `measurement`, `observation` (see
`DOMAIN_TABLES`). Nothing today extracts a covariate from `visit_occurrence`
(admission type, admission source) or from `care_site`/`visit_detail` (the
ICU unit), and nothing derives a covariate directly from `index_datetime`
(e.g. calendar year of the index event) the way the query engine's own
`care_site_names` *cohort filter* already does for eligibility. A
DAG-declared confounder that needs one of these is unmappable, not merely
unmapped: no `--domain` choice in `workflow covariate-review`/
`covariate-map-concepts` can express it.

**Where this bit, live.** Building the PPI-vs-H2RA (Q1) run end to end
(`icu_mv_ppi_vs_h2ra_stress_ulcer_prophylaxis_mortality`), the DAG-review
correction rounds added `Care_site (ICU unit)`, `Calendar year of admission`,
and `Admission type (medical vs surgical vs trauma)` as confounders straight
from the reference DAG comparison
([`studies/ppi_vs_h2ra_sup_mortality/reference_dag.json`](../studies/ppi_vs_h2ra_sup_mortality/reference_dag.json))
and from `first_pass_covariates`. All three stayed `not_ready` once mapping
was attempted, for this reason. `Regional formulary access / local practice
variation` has the same practical effect for a different reason: it names a
facility-policy attribute with no OMOP concept at all, mappable or not.

**Resolution taken for now.** All four were removed from the run's candidate
DAG (`workflow revise-dag`) rather than left `observed: false`: unlike an
honestly-unmeasurable confounder (frailty, goals-of-care limitation -- a
separate correction round on the same run), `care_site` and `calendar_year`
*are* recorded in this OMOP build; the gap is the covariate-extraction
system's coverage, not the data. Marking them `observed: false` would have
been a category error (claiming the data does not exist) and, being on an
open backdoor path, would have re-blocked `identified_by_backdoor_adjustment`
the same way the three truly-unmeasured confounders did. Removing them keeps
the DAG identified but drops real, measurable confounders from the
adjustment set -- for Q1 specifically this is a genuine loss: `care_site` is
PEPTIC's own randomization unit (the closest analogue to cluster assignment
this design has) and calendar year captures the PPI-uptake trend documented
in `docs/validation_question_bank.md`'s Q1 entry. State this explicitly in
any report built from a run missing them (see `paper/main.tex`'s
limitations).

**Fix, when built.** Extend `DOMAIN_TABLES`-style coverage with two more
sources: a `visit_occurrence`/`visit_detail`/`care_site` join (mirroring the
`site_join` the cohort-eligibility filter already builds in `_cohort_ctes`,
but as a covariate column instead of a filter) for admission type and care
site, and an `index_datetime`-derived column (e.g. `EXTRACT(YEAR FROM
e.index_datetime)`) for calendar-time covariates -- no concept lookup needed
for the latter, so it does not need the `ConceptSetSpec` machinery at all,
only a new `CovariateSpec` variant. `Regional formulary access / local
practice variation`-style facility-policy attributes stay out of scope;
there is no OMOP concept for them by construction.

## Priority

Revised 2026-09-08 after re-checking section 2 against the bank.

1. **Washout + look-back coverage** — done.
2. **In-hospital-death-by-horizon (3a)** — done. It was a live correctness
   gap in the primary outcome of both scaffolded headline studies (Q1, Q8),
   cheap as an outcome reframe, zero estimator change.
3. **Whatever unblocks a first end-to-end run** — Sepsis-3 phenotype,
   `measurement` coverage checks, human concept-set approval, clinically
   reviewed DAG. Not a `feasibility.py` change; it is the actual conference
   blocker. Outranks item 2.
4. **Landmark mode (2)** — after a study runs end to end, or when the
   timing questions Q4/Q6 enter scope. Build it as a per-study
   sensitivity-analysis mode (exclusion **and** clock shift), not a
   mandatory primary-analysis mode. The method is settled; only the
   scheduling changed.
5. **Competing-risks survival estimator (3b)** — explicit future work.
6. **`visit_occurrence` covariate coverage (4)** — after a study runs end to
   end without it once (Q1 now does, missing `care_site`/`calendar_year`/
   `admission_type`). Worth prioritizing above the landmark mode once a
   second study also needs it, since it is a pure coverage gap (not a new
   statistical method) and currently costs Q1 two of PEPTIC's most directly
   relevant confounders.

The earlier draft had item 2 as the next build. The re-check found: the
bank assigns immortal-time handling to Q4/Q6 only; of the headline studies
only Q8 has material immortal time, and there it partly offsets the
confounding by indication the study is designed to show; nothing runs end
to end yet, so the bias cannot even be measured. Item 3a, by contrast, is
wrong in the scaffolded studies right now.

## Reusable scaffold

Each knob is a `query_settings` field, reviewer-set through a
`WorkflowManager` method that invalidates gates from `analysis_plan`, with a
feasibility diagnostic count surfaced **before** the gate so the reviewer
sees the attrition, a line in the generated report, and (follow-up)
`dag_identification` awareness.
