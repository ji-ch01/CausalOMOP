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

## Privacy and governance

Never include patient-level data, direct identifiers, or sensitive free text in
an LLM prompt. Raw outputs can contain the submitted question and must be
protected like other run artifacts. A local provider changes the network
boundary but does not remove the need for institutional review, access control,
logging policy, retention policy, and human scientific oversight.
