# LLM hypothesis integration

The current Hypothesis Agent remains deterministic. The provider-neutral
`UserQuestionHypothesisService` is the entry point for natural-language
questions:

```text
User question
  -> versioned prompt (causal/prompts/hypothesis.md)
  -> injected LLM provider
  -> strict JSON parsing
  -> protocol validation
  -> validated short clinical terms -> deterministic PubMed query
  -> second LLM call grounded in retrieved abstracts (causal/prompts/dag.md)
  -> candidate_dag filtered to PMIDs actually retrieved
  -> deterministic Hypothesis Agent
  -> human protocol gate
```

The LLM cannot approve a protocol, create executable OMOP mappings, or authorize
an analysis. Every generated concept set starts with `pending_omop_mapping`, and
the resulting hypothesis therefore has `executable = false`.

## Literature-grounded DAG draft

`UserQuestionHypothesisService._attach_literature_dag` (in
`causal/agents/user_question.py`) runs after the target-trial draft:

1. The first LLM call proposes 1-3 short clinical terms for each PICO role in
   `literature_search`. Deterministic validation rejects field-tag injection,
   control characters, and oversized term lists.
2. Python builds a fielded query from population AND treatment AND a broader
   comparator/outcome context group. Long target-trial descriptions are never
   concatenated into the query.
3. `causal.literature.pubmed.PubMedClient` retrieves real abstracts with real
   PMIDs (NCBI E-utilities; no LLM involved in this step).
4. A second, separate LLM call (`causal/prompts/dag.md`) is given only the
   retrieved abstracts and asked to propose `candidate_dag` edges
   (`confounder_to_treatment` / `confounder_to_outcome`), each citing a PMID.
4. Any edge citing a PMID that was not actually retrieved is rejected — this
   is a deterministic Python check, not something the LLM enforces on itself.

The exact fielded query and returned PMID list are persisted under
`literature_evidence` in `protocol.json` for auditability.

Failure modes degrade gracefully instead of blocking the protocol draft: no
PubMed results, or a malformed/all-rejected second-stage response, both leave
`candidate_dag` empty and add an explanatory entry to `design_constraints` —
they never raise. Genuine transport errors (PubMed or LLM unreachable) still
propagate normally.

`candidate_dag` is always provisional, exactly like the rest of the draft: it
cannot be auto-approved, and `HypothesisAgent` re-validates its structure
independently of how the protocol was produced (LLM or hand-authored).

Pass `--skip-literature` to `workflow start-question` to skip PubMed and the
DAG stage entirely (useful for a fast manual check of just the target-trial
draft, or when NCBI is unreachable).

The first production adapter uses a real OpenAI GPT model through OpenRouter.
It defaults to `openai/gpt-oss-120b`, requests JSON mode, retries transient failures,
and never logs the clinical question or model response. Set the key in `.env`:

```text
OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
CAUSALOMOP_LLM_MODEL=openai/gpt-oss-120b
```

`OPENAI_API_KEY` and `OPENAI_BASE_URL` remain supported as compatibility names.
Install and run the vertical path with:

```powershell
pip install -r requirements-llm.txt
python -m causal.cli workflow start-question "Among ..., what is the effect of ...?"
```

This command creates `protocol.json` and `hypothesis.json`, then stops at the
protocol approval gate. Do not include patient-level data in the question.

For debugging and audit, `workflow start-question` also prints each exact model
response to the terminal on stderr while keeping the final workflow-state JSON
on stdout. The same responses are stored in `llm_outputs.json`, tagged by stage
as `target_trial_draft` and, when literature was retrieved,
`literature_dag_draft`. Raw model text can contain the submitted question and
must be protected like the other run artifacts; patient-level data must never
be placed in a prompt.

## Dev vs prod provider selection

`causal/cli.py` never instantiates an LLM client directly. It calls
`causal.llm.build_hypothesis_llm()`, which reads `CAUSALOMOP_ENV`:

- `CAUSALOMOP_ENV=dev` (default) — uses `OpenRouterGPTClient` with
  `openai/gpt-oss-120b`. The adapter path is unit-tested; the model still
  requires a live structured-output smoke test through OpenRouter.
- `CAUSALOMOP_ENV=prod` — uses `LocalLLMClient` (`causal/llm/local.py`), an
  OpenAI-compatible adapter for an on-premise server (Ollama, vLLM, ...)
  configured through `LOCAL_LLM_BASE_URL` / `LOCAL_LLM_MODEL`. If a cloud key
  (`OPENROUTER_API_KEY` or `OPENAI_API_KEY`) is still set, the factory raises
  instead of silently falling back to the cloud provider.

**Known limitation:** `LocalLLMClient` has not been validated against real
on-premise infrastructure — no local model server was available during
development. Before any SNUH deployment, this must be tested against the
actual local endpoint your encadrants provision, and the confidentiality
assumption (no clinical question ever leaves the hospital network) must be
independently confirmed.
