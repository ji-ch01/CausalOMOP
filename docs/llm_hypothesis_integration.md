# LLM and Evidence Integration

LLMs in CausalOMOP propose and critique structured drafts. They do not approve
scientific decisions, map concepts, authorize database access, or estimate a
causal effect.

## Current flow

```text
User question
  -> causal / descriptive / needs-clarification gate
  -> strict target-trial draft
  -> deterministic PubMed + OpenAlex retrieval
  -> optional OptimusKG relation retrieval
  -> one or more candidate DAG drafts
  -> deterministic citation, role, timing, and graph checks
  -> bounded critique -> revision -> re-critique
  -> optional tournament ranking, SHD deduplication, and edge-union merge
  -> protocol.json + hypothesis.json + llm_outputs.json
  -> human protocol/DAG review
```

The initial question gate is itself LLM-based. It classifies a question as
`causal`, `descriptive`, or `needs_clarification`. The CLI and Streamlit UI stop
before drafting for the latter two categories. `--skip-question-gate` exists
for controlled tests and already-reviewed inputs.

The Streamlit UI runs this intake as a conversation panel
(`causal.ui.conversation` over `causal.companion.intake`): a
`needs_clarification` verdict is answerable in-thread over multiple rounds
(each round re-runs the gate on the accumulated question, capped at
`MAX_CLARIFICATION_ROUNDS`), and the whole exchange is persisted to the run's
`conversation.jsonl` -- a record, not an approval path.

Descriptive follow-ups also carry context. When earlier FastOMOP exchanges
are in the thread, the last `MAX_DESCRIPTIVE_CONTEXT` `(question, answer)`
pairs are folded in (`combine_descriptive_followup`) so a terse follow-up
like "and how many were women?" is self-contained -- the gate classifies it,
and FastOMOP answers it, against that context. The context reaches only the
gate and the FastOMOP call, never the causal drafting step (a target trial
is always drafted from the clean question). This is a bounded in-session
window, not a persisted store.

## Evidence retrieval and citation checks

The first draft supplies short clinical search terms. Deterministic code builds
queries and retrieves records from PubMed and OpenAlex. OptimusKG is optional
and loaded only when available.

The DAG prompt receives the retrieved evidence. A proposed edge is retained
only when its evidence identifier belongs to the identifiers actually retrieved
for that run. This blocks invented citations from entering the candidate DAG,
but it does not prove that a cited paper supports the claimed edge or that the
literature search is complete. Those remain review tasks.

The exact search metadata, candidate DAG, design constraints, and raw model
responses are persisted under `artifacts/runs/<run_id>/`.

## Critique and tournament

The default DAG path uses up to two critique rounds. Each round may request one
revision, after which the revised graph is parsed and checked through the same
deterministic gate. A malformed critique or revision keeps the last valid state
and records the failure; the loop never becomes open ended.

`workflow start-question --max-critique-rounds N` changes the bound.
`--skip-critique` disables the critic.

`--tournament-size N` drafts independent candidates (default `1`, maximum `5`).
For multiple candidates, deterministic code ranks review status, removes
structurally identical candidates using SHD by default, and merges compatible
survivors by edge union. Tournament selection increases diversity; it does not
make a graph clinically valid.

Raw output stages are round-numbered in `llm_outputs.json`, so a reviewer can
reconstruct drafting, critique, and revision history.

## Provider configuration

Development uses the OpenRouter adapter by default:

```text
CAUSALOMOP_ENV=dev
OPENROUTER_API_KEY=...
CAUSALOMOP_LLM_MODEL=openai/gpt-5
```

`CAUSALOMOP_LLM_MODEL` accepts any OpenRouter `vendor/model` slug
(`openai/gpt-5`, `openai/o3`, `anthropic/claude-opus-5`,
`google/gemini-2.5-pro`, ...); it is also overridable per invocation with the
CLI `--model` flag and the Streamlit "Model override" field. Three optional
knobs tune the call, mainly for reasoning models: `CAUSALOMOP_LLM_MAX_TOKENS`
(response ceiling, default 20000 — the DAG stage silently falls back to an
empty graph if the model is truncated mid-response),
`CAUSALOMOP_LLM_REASONING_EFFORT` (`minimal|low|medium|high`, unset by
default), and `CAUSALOMOP_LLM_TIMEOUT` (seconds per attempt, default 180).
The response is requested in JSON mode and any surrounding Markdown code
fence is stripped before parsing.

### Model choice

The default is a frontier reasoning model (`openai/gpt-5`; a newer
same-family snapshot such as `openai/gpt-5.2` is a drop-in replacement and is
what `.env` is normally set to). The demanding step is candidate-DAG drafting
and critique: the model reasons over 15+ retrieved abstracts at once, must
respect the role, timing, and acyclicity constraints checked by
`_parse_dag_response`, and may only propose an edge whose evidence identifier
was actually retrieved for that run. The previous default,
`openai/gpt-oss-120b` (~5B active parameters), was observed truncating
mid-reasoning and silently falling back to an empty `candidate_dag`, and
proposing identifiers absent from the retrieved set; a model with stronger
long-context reasoning and instruction-following reduces both failure modes.
Every stage is parsed with a strict `json.loads`, so the model must also
honour OpenRouter's JSON mode reliably — the OpenAI GPT family and Anthropic
Claude both do, which is a practical constraint on the choice.

The model is swappable by design. Scientific rigor here comes from the
deterministic machinery *around* the model — the citation allow-list, the
structural DAG gate, `verify_narrative`, and the human approval gates — and
from comparing pipeline output against published ground truth (the validation
strategy in the top-level README), not from the model itself. The default is
simply the most capable reasonable option; at this pipeline's call volume
(single digits of requests per run) the cost gap between candidate models is
immaterial next to the review effort each run already requires. This mirrors
the evidence-grounded-co-pilot design of Alaa et al. (2024,
[arXiv:2407.19118](https://arxiv.org/abs/2407.19118)), where verifiable
grounding, not model capability alone, is what makes the output trustworthy.

Production mode requires a local OpenAI-compatible endpoint:

```text
CAUSALOMOP_ENV=prod
LOCAL_LLM_BASE_URL=http://localhost:11434/v1
LOCAL_LLM_MODEL=local-model
LOCAL_LLM_API_KEY=not-required
```

When `CAUSALOMOP_ENV=prod`, the factory refuses to start if a cloud API key is
still configured. The local adapter exists but must be validated against the
actual on-premises model server and institutional confidentiality controls
before clinical deployment.

Install and run the natural-language path with:

```powershell
pip install -r requirements-llm.txt
python -m causal.cli workflow start-question "Among ..., what is the effect of ...?"
```

Useful controls:

- `--skip-question-gate` for an already-reviewed causal question;
- `--skip-literature` to omit literature and DAG drafting;
- `--skip-knowledge-graph` to omit OptimusKG;
- `--skip-critique` to omit critique/revision;
- `--max-critique-rounds` to bound critique rounds;
- `--tournament-size` to draft multiple candidates.

## Estimand dialogue (statistical_estimand gate)

While a run is in the `estimand_review` phase, the Streamlit left panel runs
the study companion (`causal.companion.companion` via
`WorkflowManager.run_companion_turn`). It is the same advisory-LLM posture as
the question gate and the results narrative:

```text
conversation so far + JSON-safe run artifacts (study_specification.json,
  cohort_manifest.json adjustment set, feasibility / follow-up / positivity /
  covariate-missingness counts, whether the per-row censored column is
  approved) + the decision memory (see below) -- never analytic_dataset.parquet
  -> estimand_dialogue prompt
  -> strict-JSON response: { reply, proposed_action | null }
  -> causal.companion.actions.validate_proposed_action
     (allow-list: set_statistical_estimand_strategy only; every arg checked
      against MISSING_DATA_STRATEGIES / CENSORING_STRATEGIES /
      ESTIMATION_PROCEDURES; reviewer / run_id can never appear)
  -> the reviewer clicks Confirm on the card
     -> WorkflowManager.set_statistical_estimand_strategy(reviewer=<sidebar>)
```

`run_companion_turn` is read-only with respect to run state: it takes no
lock, records nothing, writes no file. The one write the dialogue can lead
to is the explicitly confirmed `set_statistical_estimand_strategy`;
approving the `statistical_estimand` gate itself is a separate manual
action. The transcript (`conversation.jsonl`) records every turn, including
which card was confirmed or dismissed. If the companion LLM is unavailable
the panel degrades to a warning and the "Fill the form directly" fallback.

To exercise this panel without a working OMCP/BigQuery stack,
`python scripts/stage_demo_run.py --to estimand_review` stages a fully
synthetic run (auto-approved gates, fabricated cohort) straight into the
phase and prints its run id for the Streamlit sidebar.

### Decision memory

A dialogue turn only replays the natural-language transcript, so on its own
the companion has no memory of what the reviewer already *decided* and will
re-propose a covariate they excluded. `causal.companion.decision_memory`
closes that gap. `distill_decision_memory` is a pure, deterministic digest
(no LLM, same spirit as `verify_narrative`) of:

- `protocol.json`'s `rejected_covariates` -- the audited covariate exclusions
  (this field already existed; nothing read it back before), and
- the decision-shaped entries of `RunState.history` plus the decided
  `ApprovalGate`s with their review comments,

merged into one timeline and rendered into the prompt under `PRIOR
DECISIONS`. The prompt instructs the companion not to re-propose an excluded
covariate or silently re-open a settled choice. `WorkflowManager.
_decision_memory` builds it from the `RunState`/`protocol.json` it already
loads for the turn, so `run_companion_turn` stays read-only.

From the gate objects only a `rejected` gate is replayed (a routine
approval does not change how the companion should behave); gate
*invalidations* come through the history stream. It is inspectable three
ways: live in the Streamlit **Report & audit** tab, via `python -m
causal.cli workflow decision-memory <run_id>` (`--json` for the raw
object), and as a frozen `decision_memory.json` snapshot that
`build_causal_report` writes and lists in the report's Provenance table.

### Reviewer memory (cross-study)

`causal.companion.reviewer_memory` is the outermost, opt-in memory layer:
one file per reviewer at `artifacts/reviewers/<slug>/memory.json`, carrying
their conventions *across* studies -- covariates they have excluded before
(with the study and rationale) and the statistical-estimand strategies they
have settled on. It is populated only when a human asks
(`WorkflowManager.remember_run_for_reviewer` -- the "Add this run's
decisions to my reviewer memory" button, or `workflow remember-run
<run_id> --reviewer <name>`); `merge_run_into_memory` is the pure,
idempotent fold (re-running for the same study replaces that study's
entries). When `run_companion_turn` is given a `reviewer`, that memory is
loaded read-only and rendered into the prompt under `REVIEWER CONVENTIONS`,
with an instruction to use it for consistency but never to treat a past
choice as one already made for this study. Same boundary as the per-run
decision memory: decision-level text only, feeds a prompt, never a gate.
Inspect with `workflow reviewer-memory <name>` or the **Report & audit**
tab.

## Results narrative (post-estimation interpretation)

After `causal.agents.estimation`/`causal.agents.diagnostics` compute the
statistical estimate, an optional, separate LLM step
(`causal.agents.results_narrative`) may translate the already-computed
numbers into plain language:

```text
estimation.json + diagnostics.json
  -> results-narrative prompt (JSON-safe artifacts only, never the
     patient-level analytic_dataset.parquet)
  -> strict-JSON response: { narrative, claims[] }
  -> causal.agents.results_faithfulness.verify_narrative
     (deterministic checks, no second LLM)
  -> WorkflowManager.review_results_narrative (human approve/exclude)
  -> build_causal_report includes the narrative text only if
     review_status == "approved" and verification has no `failed` check
```

### Results dialogue (drafting the narrative conversationally)

While a run is in `results_review` / `report` the Streamlit left panel runs
the results-interpretation dialogue -- the results-review counterpart of the
estimand dialogue, `causal.companion.companion.build_results_dialogue_prompt`
/ `parse_results_companion_turn` via `WorkflowManager.
run_results_companion_turn`. The companion walks the reviewer through the
direction, magnitude, uncertainty, diagnostics, robustness, and limitations
one topic at a time, then ends a turn with a `drafted_narrative` in exactly
the `{ narrative, claims[] }` shape above:

```text
conversation so far + the same NarrativeInput the one-shot step is fed
  -> results_dialogue prompt (JSON-safe artifacts only)
  -> strict-JSON response: { reply, drafted_narrative | null }
     (drafted_narrative validated by
      causal.agents.results_narrative.narrative_result_from_payload)
  -> the reviewer clicks "Save for review" on the card
     -> WorkflowManager.save_results_narrative_draft
        -> verify_narrative + results_narrative.json (source="dialogue",
           review_status="pending")
  -> WorkflowManager.review_results_narrative (unchanged human approve/exclude)
```

`run_results_companion_turn` is read-only with respect to run state (no
lock, no history entry, no file write), matching `run_companion_turn`. The
one-step `run_results_narrative` path stays as the no-conversation fallback;
both write the identical artifact and both go through the same faithfulness
verification and human approval. The `source` field on
`results_narrative.json` records which path produced it.
`python scripts/stage_demo_run.py --to results_review` stages a synthetic
run (fabricated cohort, `estimation.json` / `diagnostics.json` written) for
testing this panel offline.

Every claim in `claims` is typed (`effect_direction`, `effect_magnitude`,
`uncertainty`, `sample_size`, `diagnostic_warning`,
`assumption_or_limitation`) and cites the dotted JSON path(s) it is based
on (e.g. `estimation.estimand.point_estimate`). `verify_narrative` checks,
per claim: every number appears verbatim in the source artifacts; the
stated effect direction matches the point estimate's sign; the stated
uncertainty matches whether the confidence interval crosses the null; each
`source_field` actually resolves inside the artifacts the model was shown
(never a made-up or out-of-scope field); diagnostic warnings and
assumptions/limitations already present in the artifacts are not silently
dropped or contradicted. A `failed` check never blocks estimation, the
`results` gate, or the causal report itself -- it only blocks this
narrative's own automatic approval, and is shown prominently in the
Streamlit "Estimation & causal report" tab (deterministic results ->
generate interpretation -> faithfulness verification -> human review ->
build the final report).

Approving or excluding the narrative (`WorkflowManager.
review_results_narrative`) is entirely separate from `decide_gate(...,
"results", ...)`: the narrative can never approve, invalidate, or otherwise
influence the `results` gate, and the gate's approval never implies the
narrative was reviewed.

## Privacy and governance

Never include patient-level data, direct identifiers, or sensitive free text in
an LLM prompt. Raw outputs can contain the submitted question and must be
protected like other run artifacts. A local provider changes the network
boundary but does not remove the need for institutional review, access control,
logging policy, retention policy, and human scientific oversight.
