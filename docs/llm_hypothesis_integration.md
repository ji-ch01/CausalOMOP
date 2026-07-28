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
  -> deterministic structural + clinical pre-check (causal.agents.dag_identification)
  -> third LLM call: independent critique of the draft DAG (causal/prompts/dag_critique.md)
  -> if the critique requests changes: one bounded revision call, same required shape as dag.md
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
   retrieved abstracts and knowledge-graph relations and asked to propose
   `candidate_dag` variables, each with an explicit causal role (confounder,
   mediator, collider, selection, instrument, effect_modifier, ...) and a
   `temporal_classification`, citing a PMID or knowledge-graph id for each.
5. Any variable or relation citing a PMID/id that was not actually retrieved
   is rejected — this is a deterministic Python check
   (`_parse_dag_response`), not something the LLM enforces on itself.
6. If the resulting `candidate_dag` is non-empty, a third LLM call
   independently critiques it (see "Critique and revision" below) before it
   reaches `protocol.json`.

The exact fielded query and returned PMID list are persisted under
`literature_evidence` in `protocol.json` for auditability.

Failure modes degrade gracefully instead of blocking the protocol draft: no
PubMed results, a malformed/all-rejected second-stage response, or a
malformed critique/revision response all leave `candidate_dag` at its last
successfully-produced state (empty, pre-critique, or pre-revision as
applicable) and add an explanatory entry to `design_constraints` — they
never raise. Genuine transport errors (PubMed or LLM unreachable) still
propagate normally.

`candidate_dag` is always provisional, exactly like the rest of the draft: it
cannot be auto-approved, and `HypothesisAgent` re-validates its structure
independently of how the protocol was produced (LLM or hand-authored).

Pass `--skip-literature` to `workflow start-question` to skip PubMed and the
DAG stage entirely (useful for a fast manual check of just the target-trial
draft, or when NCBI is unreachable).

## Critique and revision

After a non-empty `candidate_dag` is drafted, `UserQuestionHypothesisService.
_critique_and_revise_dag` (using the pure helpers in
`causal/agents/dag_critique.py`) runs an independent Generation-Critique
loop, in the spirit of an AI Co-Scientist pattern, before a human ever sees
the draft:

1. The existing deterministic checks (`causal.agents.dag_identification.
   assess_identification` / `assess_clinical_completeness`) run against the
   draft to produce structured findings (is it identified by backdoor
   adjustment? are there disputed-certainty edges?). Note that
   `major_confounders` does not exist yet at this point in the pipeline (it
   is set later, by a human, via the covariate-review CLI flow), so
   `blocking_issues` is typically empty here — the findings are usually
   informative, not "problems found".
2. A third LLM call (`causal/prompts/dag_critique.md`), given the same
   evidence the drafting call saw plus the proposed DAG and the structural
   findings, returns `{"verdict": "approve"|"revise", "concerns": [...]}`.
   The critic never emits DAG-variable objects itself.
3. If `verdict` is `"revise"`, exactly one bounded revision call is made:
   the drafting model gets its own prior proposal plus the critique's
   concerns and produces a complete, revised response in the *same* required
   JSON shape as the original draft — so it is parsed by the same
   `_parse_dag_response` PMID/role/temporal gate as any other draft. This is
   not an open-ended loop: v1 does at most one critique and one revision.

Every round (draft, critique, revision if any) is recorded as its own
`LLMRawOutput` and ends up in `llm_outputs.json`, tagged
`literature_dag_draft`, `literature_dag_critique`, and
`literature_dag_revision` respectively — the human reviewer sees the full
debate trail, not just the final DAG. The critique loop never approves,
rejects, or bypasses anything: whatever it produces still lands in
`protocol["candidate_dag"]` exactly as provisional as an uncritiqued draft
would be, still subject to the human `protocol` approval gate and to
`_block_if_major_confounders_unresolved`/`_block_if_not_identified` in
`causal/orchestrator/workflow.py`.

Pass `--skip-critique` to `workflow start-question` (or `enable_critique=False`
to `UserQuestionHypothesisService.generate`) to skip this step and keep
today's two-call behavior.

`_parse_dag_response` also surfaces the drafting model's own `unresolved`
list (first-pass covariates it explicitly could not classify, with a
`reason`) as `protocol["candidate_dag_unresolved"]` — previously silently
discarded.

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
`literature_dag_draft`, plus (when critique ran and wasn't skipped)
`literature_dag_critique` and, if a revision was requested,
`literature_dag_revision`. Raw model text can contain the submitted question and
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
