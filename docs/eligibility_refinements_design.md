# Design note: target-trial eligibility refinements

Three eligibility/ascertainment refinements the validation question bank
([`validation_question_bank.md`](validation_question_bank.md)) needs and the
current cohort builder cannot express. Written 2026-09-08; section 2/3
priority revised 2026-09-08 after a re-check (see the Priority section for
the reasoning). Item 1 is implemented; items 2 and 3 are specified here and
not yet built. **The recommended next build is item 3a, not item 2** — item
3a fixes a live correctness gap in the two headline studies that are already
scaffolded, whereas the bank assigns item 2 (landmark) to the lower-priority
timing questions and, among the headline studies, it is material only for
Q8, where its bias partly offsets the confounding the study is designed to
show.

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

## 3. In-hospital mortality — specified, recommended next

**This is a live correctness gap, not a refinement.** Q1's primary outcome
is "in-hospital all-cause mortality"; Q8's is "in-hospital / 30-day". The
builder cannot express "in-hospital" today: `follow_up_days` is a fixed
calendar window that treats every patient as hospitalised for the same
number of days. Both scaffolded headline studies currently rest on that
fiction.

**Correction to the obvious approach.** A `censor_at_discharge` mode routed
through the existing `censored` column and the IPCW g-formula would be a
methodological error: discharge alive is strongly prognostic (informative
censoring), and every current `CENSORING_STRATEGIES` value and the
TMLE + IPCW gate rejection reason are premised on *administrative*
censoring.

**What to build (cheap path, 3a).** An outcome-definition mode "in-hospital
death within horizon H": death in hospital within H days = 1; discharged
alive before H = 0 (the competing-risks subdistribution view: discharge
keeps the person in the risk set as a non-event); still inpatient at H
(rare) = administrative censor. For the `death` domain this is a join to
`visit_occurrence` plus a date comparison. **No `censored` column, no
estimator change.** It is a reframe of the outcome definition, and it is
what PEPTIC (in-hospital), CAPE COD (28-day), ADRENAL (90-day) actually
report.

**Expensive path (3b), out of scope for the workshop paper.** A true
survival estimator with discharge as a competing event (cause-specific /
Fine-Gray subdistribution hazards; Young et al. 2020, *a causal framework
for classical statistical estimands ... with competing events*, Stat Med
39:1199). Name it in the paper's limitations next to clone-censor-weight.

## Priority

Revised 2026-09-08 after re-checking section 2 against the bank.

1. **Washout + look-back coverage** — done.
2. **In-hospital-death-by-horizon (3a)** — next. It is a live correctness
   gap in the primary outcome of both scaffolded headline studies (Q1, Q8),
   cheap when scoped as an outcome reframe, zero estimator change.
3. **Whatever unblocks a first end-to-end run** — Sepsis-3 phenotype,
   `measurement` coverage checks, human concept-set approval, clinically
   reviewed DAG. Not a `feasibility.py` change; it is the actual conference
   blocker. Listed here because it outranks item 2.
4. **Landmark mode (2)** — after a study runs end to end, or when the
   timing questions Q4/Q6 enter scope. Build it as a per-study
   sensitivity-analysis mode (exclusion **and** clock shift), not a
   mandatory primary-analysis mode. The method is settled; only the
   scheduling changed.
5. **Competing-risks survival estimator (3b)** — explicit future work.

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
