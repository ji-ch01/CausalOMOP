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
CAUSALOMOP_LLM_MODEL=openai/gpt-oss-120b
```

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
  approved) -- never analytic_dataset.parquet
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
