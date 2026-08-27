# Responsible Demonstration Guide (English)

Present CausalOMOP as a governed study-design prototype that can now produce
one transparent, prespecified estimate (population ATE via g-computation or
TMLE, or a CATE over a prespecified subgroup family via a causal forest --
reviewer's choice), not as a clinical decision system or a source of
clinical-grade causal conclusions. This guide uses the
albumin/sepsis study as its running example because it is the pipeline's
worked literature-comparison case; make clear to any audience that CausalOMOP
itself is general-purpose and disease-agnostic, and that the study is a
validation example, not the tool's scope (see
[README.md's References](../README.md#references)).

## Opening statement

> CausalOMOP turns a clinical question into an auditable target-trial draft,
> evidence-grounded candidate DAG, identification review, OMOP mappings, and
> feasibility diagnostics. LLM outputs remain proposals, and important
> decisions require human review, including the final `results` gate before
> any estimate is reported. The pipeline can now fit a population ATE via
> either g-computation or TMLE, or a CATE over a prespecified subgroup
> family via a causal forest (reviewer's choice per run), with covariate-
> balance diagnostics and E-value quantitative sensitivity analysis; no
> output here is a clinical recommendation.

## Setup

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest tests -q
python -m causal.cli config --json
```

With BigQuery and Application Default Credentials configured:

```powershell
python -m causal.cli preflight --count-persons
```

Show the live output. Do not quote a frozen test count, table count, or person
count from documentation.

## Create a governed run

From the versioned study protocol:

```powershell
$state = python -m causal.cli workflow start studies\albumin_sepsis_28d\protocol.json |
  ConvertFrom-Json
$runId = $state.run_id
python -m causal.cli workflow hypothesis $runId
```

Or from a natural-language question:

```powershell
python -m causal.cli workflow start-question "Among adult ICU patients with sepsis, what is the effect of albumin plus crystalloids versus crystalloids alone on 28-day mortality?"
```

Explain that the first LLM step classifies the question. Descriptive requests
are redirected to FastOMOP; ambiguous requests return a clarification question.
The causal path can retrieve PubMed/OpenAlex evidence, use OptimusKG when
available, and run a bounded DAG critique/revision loop.

In the Streamlit UI the same intake runs as a conversation panel: the
clinical question, the gate verdict, any clarification rounds, the FastOMOP
answer for a descriptive question, and the "study created" event are shown
as a thread and written to `artifacts/runs/<run_id>/conversation.jsonl` once
the run exists. That transcript is an advisory record only -- it carries no
decision authority and never reaches a gate; every approval still goes
through `WorkflowManager` explicitly.

## Show the audit trail

Under `artifacts/runs/<run_id>/`, show:

- `protocol.json` for the target-trial draft;
- `hypothesis.json` for the candidate DAG and identification result;
- `llm_outputs.json` for raw drafting, critique, and revision responses;
- `state.json` for phases, gate decisions, reviewers, and history.

An `executable: false` value is an expected safety state while reviews remain
incomplete.

Export the DAG for independent inspection:

```powershell
python -m causal.cli workflow dag-export <run_id> --highlight-adjustment-set
```

The output can be pasted into [dagitty.net](https://dagitty.net). A graph that
passes a backdoor test is not automatically clinically complete or correct.

## Human review and OMOP mapping

```powershell
python -m causal.cli workflow gate <run_id> protocol approved `
  --reviewer "reviewer-id" `
  --comment "Target trial and DAG accepted for technical exploration"
python -m causal.cli workflow advance <run_id>
python -m causal.cli workflow plan <run_id> --with-mcp
```

OMCP lookup results are candidates only. Review mappings through
`concept-set-map`, the `covariate-*` commands, or the Streamlit interface:

```powershell
streamlit run causal/ui/streamlit_app.py
```

Never approve a mapping or gate merely to make a live demo proceed.

## Registry and aggregate diagnostics

After the required reviews:

```powershell
python -m causal.cli workflow registry-build <run_id>
python -m causal.cli workflow gate <run_id> clinical_mapping provisional `
  --reviewer "reviewer-id" --comment "Aggregate audits only"
python -m causal.cli workflow advance <run_id>
python -m causal.cli workflow feasibility <run_id> --transport mcp
python -m causal.cli workflow execution-readiness <run_id>
python -m causal.cli workflow cohort-plan <run_id>
python -m causal.cli workflow cohort-profile <run_id>
python -m causal.cli workflow followup-audit <run_id>
python -m causal.cli workflow mapping-audit <run_id> <variable_id>
python -m causal.cli workflow demo-report <run_id>
```

Commands correctly fail closed when run state, mappings, or gates are
incomplete. Demonstrating that refusal is part of the governance story.

## Statistical estimand review

Once the analysis plan itself is approved, `estimand_review` is a separate
phase/gate (roadmap stage 8) before any patient-level row is materialized: a
human must record an explicit missing-data/censoring strategy
(`causal.roadmap.StatisticalEstimand`) rather than leaving it implicit in
`EstimationAgent`'s code:

```powershell
python -m causal.cli workflow gate <run_id> analysis_plan approved `
  --reviewer "reviewer-id" --comment "Analysis plan reviewed"
python -m causal.cli workflow advance <run_id>
python -m causal.cli workflow estimand-strategy <run_id> `
  complete_case_analysis ignore_censoring_administrative_waiver `
  --rationale "Administrative censoring reviewed in followup_audit.json and judged non-informative for this cohort" `
  --reviewer "reviewer-id"
python -m causal.cli workflow gate <run_id> statistical_estimand approved `
  --reviewer "reviewer-id" --comment "Adjustment set and censoring strategy reviewed"
python -m causal.cli workflow advance <run_id>
```

Choosing `exclude_censored_before_analysis` instead requires the per-row
`censored` column to already be approved
(`WorkflowManager.approve_censoring_definition`) -- the gate blocks otherwise.

In the Streamlit UI this is a conversation: while the run is in
`estimand_review` the left panel becomes the estimand dialogue
(`causal.companion.companion` via `WorkflowManager.run_companion_turn`). The
study companion walks the reviewer through the three choices one at a time,
grounded in the same pre-gate diagnostic artifacts, and can end a turn by
proposing `set_statistical_estimand_strategy` as a card. The card shows an
editable rationale and applies nothing until the reviewer clicks Confirm; a
strict allow-list (`causal.companion.actions`) means the companion can only
ever propose that one write, with arguments in range, and never the reviewer
identity. Approving the `statistical_estimand` gate itself stays a separate
manual action, and the "Fill the form directly" expander is the same call
without the dialogue. The companion is fed only JSON artifacts, never
`analytic_dataset.parquet`.

## Estimation and the causal report

Only once `clinical_mapping`, `analysis_plan`, and `statistical_estimand` are
all `approved` and the plan's execution flag for the run's mode is set
(`execution-readiness` shows `causal_estimation_allowed: true` -- see
`causal.agents.execution.ExecutionAgent`):

```powershell
python -m causal.cli workflow materialize-dataset <run_id>
python -m causal.cli workflow estimate <run_id> --n-bootstrap 1000
python -m causal.cli workflow balance-diagnostics <run_id>
python -m causal.cli workflow advance <run_id>
python -m causal.cli workflow advance <run_id>
python -m causal.cli workflow gate <run_id> results approved `
  --reviewer "reviewer-id" --comment "Estimate and diagnostics reviewed"
python -m causal.cli workflow advance <run_id>
python -m causal.cli workflow causal-report <run_id>
```

`materialize-dataset` is the one command in this whole walkthrough that pulls
patient-level rows into memory (into a local `analytic_dataset.parquet`,
governed by OMCP's signed `cohort_build` operation) -- it never leaves the run
directory and is never sent to an LLM. `narrative <run_id>` is optional and
calls the configured LLM provider to interpret the already-computed estimate
only; skipping it never blocks `causal-report`.

In the Streamlit UI this interpretation is a conversation too: while the run
is in `results_review` / `report` the left panel becomes the
results-interpretation dialogue (`causal.companion.companion` via
`WorkflowManager.run_results_companion_turn`). The companion talks the
reviewer through direction, magnitude, uncertainty, diagnostics, robustness,
and limitations one topic at a time -- grounded in the same JSON-safe
`NarrativeInput` the one-shot step uses, never `analytic_dataset.parquet` --
then ends a turn with a drafted `{ narrative, claims[] }`. "Save for review"
(`save_results_narrative_draft`) runs the same deterministic
`verify_narrative` and writes `results_narrative.json`; the reviewer still
approves or excludes it under **Estimation & report**
(`review_results_narrative`), and only an approved, clean narrative reaches
`causal_report.md`. The "Draft in one step" button is the same outcome
without the conversation.

## Safe claims

- The system creates versioned, reviewable protocol and DAG drafts.
- Retrieved evidence identifiers are checked against what was actually fetched.
- Human decisions and execution blockers are persisted.
- OMOP mappings and aggregate coverage can be audited.
- Aggregate feasibility results can be reported with their date, query
  definition, and environment.
- The system can generate governed analytic-cohort SQL.
- The system can fit one prespecified, transparent ATE estimate via either
  g-computation/standardization (bootstrap CI) or TMLE (efficient-influence-
  curve CI), reviewer's choice per run, under the assumptions stated in the
  causal report.
- The system can compute an E-value (VanderWeele & Ding 2017) bounding how
  strong an unmeasured confounder would need to be to explain away the
  reported estimate.
- The system can run a negative-control diagnostic: the same reviewed
  estimator, applied to a human-approved exposure or outcome believed to
  have no true causal effect. A null result there is evidence for the
  adjustment set, not proof it is correct.

## Claims to avoid

- A provisional count is a final cohort.
- The generated DAG is clinically authoritative.
- A mortality difference alone (without adjustment) is a treatment effect.
- A large E-value rules out unmeasured confounding -- it does not; it only
  bounds how strong such a confounder would need to be, and does not
  establish that one does not exist.
- An exploratory (non-prespecified) CATE subgroup result is a confirmatory
  finding -- it is not; only the prespecified subgroup family receives
  Holm-stepdown family-wise error control, and exploratory subgroups are
  reported separately and must never be treated as confirmatory.
- The reported ATE estimate has been validated against unmeasured
  confounding, joint positivity, or informative censoring -- it has not;
  see the report's own "Assumptions and limitations" section.
- The albumin study has been reproduced.
- The system is ready for clinical use.

Historical counts in the study's
[concept audit](../studies/albumin_sepsis_28d/concept_audit.md) are dated
feasibility snapshots, not portable current results.

## Live-demo fallback

- Test configuration and external services in advance.
- Keep a pre-generated run for offline inspection.
- Use `--skip-literature` or `--skip-knowledge-graph` only while explicitly
  stating the reduced evidence path.
- Never display secrets, private cloud identifiers, or patient-level rows.
