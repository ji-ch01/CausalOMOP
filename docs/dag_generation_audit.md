# DAG Generation — Audit (as of 2026-07-24, updated 2026-07-28)

Status snapshot of the causal-DAG generation pipeline, for presentation purposes.
Covers what exists today, how it is governed, what is deliberately out of scope,
and what is still missing. All 220 tests pass (`pytest tests/ -q`, 220 passed).

**Headline caveat:** the machinery below is solid — the governance, the
identification checks, the audit trail all work as designed — but the DAGs it
currently produces are not yet clinically usable as-is. The pipeline is a
rigorous scaffold for producing and checking a candidate DAG, not yet a
source of clinically trustworthy DAGs on its own; see §6 for why.

## 0. Update — 2026-07-28

A correctness pass over evidence sourcing and adjustment-set reporting,
prompted by re-reading this audit's own §6 flags. Three real bugs fixed, two
new metric types added, one external package (`causal-learn`) evaluated for
fit. Full detail folded into the relevant sections below; summary:

- **OptimusKG evidence was structurally unreachable for real confounders.**
  `_fetch_kg_relations` only ever queried OptimusKG pairwise across the four
  trial-defining anchor terms (population/treatment/comparator/outcome) —
  never against an actual candidate covariate label (age, an
  illness-severity score, ...), because OptimusKG has no free-text search
  and those labels do not exist until *after* the DAG-drafting LLM call that
  consumes this evidence. In practice this meant "a variable can be accepted
  on KG evidence alone" (§2.1) was true only for the four core concepts,
  essentially never for a real confounder — the opposite of what that
  evidence source exists for. Fixed by mirroring the literature side's
  per-covariate targeted search: `first_pass_covariates` (already known at
  this point) are now each paired against treatment/outcome directly. See
  §2.1 and §7.
- **`_kg_reinforcement` had a same-endpoint matching bug.** It flagged a KG
  edge as "independently corroborating" a variable whenever the variable's
  label matched *either* endpoint and the anchor term matched *either*
  endpoint — without requiring them to be on *opposite* endpoints. A
  same-side coincidental substring match could be misread as the KG
  confirming a relation it never stated. Fixed to require opposite
  endpoints; more consequential now that the fix above means KG edges
  legitimately mention covariate labels.
- **`workflow dag-export` had a live `AttributeError`.** It read
  `result.warnings`, a field that has never existed on
  `DAGIdentificationResult` (the field is `technical_notes`). Zero test
  coverage on this CLI command meant nobody hit it until the code was read
  directly. Fixed, and `tests/test_cli_dag_export.py` now exercises the real
  `causal.cli.main()` entry point end to end so a future rename here fails a
  test instead of shipping silently.
- **New metrics**: `EvidenceSourceSummary` (literature-only / KG-only /
  corroborated / uncited coverage) and `AdjustmentSetSummary` (core vs.
  alternative vs. unused-confounder structure across
  `minimum_adjustment_sets`) — see §7.
- **`causal-learn` evaluated, not integrated** — see §8 for why and what a
  scoped integration would look like.

## 1. What "DAG generation" means in this codebase

The DAG is never asked for outright. It is assembled in three separable stages,
each with its own trust boundary:

1. **Drafting** — an LLM proposes candidate variables from literature/knowledge-
   graph evidence (`causal/prompts/dag.md`).
2. **Critique** — a second, independent LLM call challenges the draft before it
   is accepted, with at most one bounded revision round
   (`causal/prompts/dag_critique.md`, `causal/agents/dag_critique.py`).
3. **Structural + clinical review** — deterministic, LLM-free checks decide
   whether the resulting graph is identified by backdoor adjustment, and
   whether it is clinically complete enough to approve
   (`causal/agents/dag_identification.py`).

No OMOP concept ever enters this graph — nodes are free-text clinical labels.
Concept mapping happens later, in a separate phase gated by its own approval.

## 2. Pipeline walkthrough

### 2.1 Drafting (`causal/agents/user_question.py`, `causal/prompts/dag.md`)

- Evidence comes from two independent sources: `LiteratureOrchestrator`
  (PubMed/OpenAlex) and `OptimusKGClient` (a biomedical knowledge graph,
  optional dependency, loaded lazily). A variable can be accepted on KG
  evidence alone, with no retrieved abstract discussing it.
- Beyond the generic population/treatment/comparator/outcome search, each
  first-pass covariate gets its own small targeted literature search
  (`_search_covariate_evidence`) — added specifically because classic
  confounders (age, illness-severity scores) were getting crowded out by
  articles that only mention the treatment/outcome terms in passing.
  `_fetch_kg_relations` mirrors this on the OptimusKG side: since OptimusKG
  has no free-text search (a relation is only found by querying two labels
  directly), each first-pass covariate is now also paired against
  treatment/outcome directly, not just the four trial anchor terms
  pairwise — without this, KG evidence could only ever ground a variable
  whose label happened to be a substring of the population/treatment/
  comparator/outcome terms themselves, i.e. essentially never a real
  confounder (fixed 2026-07-28, see §0).
- The prompt forces an explicit 9-question decision tree per variable
  (confounder vs. treatment-cause-only vs. outcome-cause-only vs. mediator vs.
  collider vs. selection vs. instrument vs. effect-modifier, then timing)
  instead of a default "confounder" bucket. Role and `temporal_classification`
  are asserted independently and cross-validated — a variable timed
  `post_treatment` cannot hold a role that implies it precedes treatment.
- `certainty` (`established` / `plausible` / `contested` / `unknown_direction`)
  is tracked separately from citation presence: a cited claim can still be
  contested if the literature disagrees with itself.
- A covariate the LLM cannot confidently classify goes to `unresolved` with a
  reason, rather than being silently dropped.
- Every accepted `pmid` must be an id that was actually shown to the model in
  this call (`known_citation_ids`) — an invented citation is rejected, not
  trusted.

### 2.2 Critique (`causal/agents/dag_critique.py`, `causal/prompts/dag_critique.md`)

- A separate LLM call re-examines the same evidence and the draft, checking
  for: role/temporal inconsistency, a plausible confounder/mediator/collider
  visible in the evidence but omitted, an unsupported citation, a wrong
  `certainty`, or a misplaced `unresolved` entry.
- It also receives the deterministic structural/clinical findings (§2.3) as
  read-only context, not as a checklist it must "find problems" against.
- Verdict is `approve` or `revise`; a `revise` verdict without actionable
  `concerns` is treated as malformed. The critic never emits DAG variables
  itself — only a revision from the *drafting* model, re-validated through the
  exact same acceptance gate as the first draft, can change the protocol.
- Failure mode at every step (empty/malformed LLM output, failed revision) is
  "fall back to the pre-critique draft with an explanatory note" — never a
  hard failure of the whole hypothesis draft.

### 2.3 Structural + clinical review (`causal/agents/dag_identification.py`)

Two deliberately independent assessments, both surfaced before a human
approves:

- **`assess_identification`** — pure graph theory: builds a `roadmap.CausalDAG`
  from the role-classified variables and asks only "does this identify the
  effect by backdoor adjustment?" It knows nothing about clinical
  completeness. An empty `candidate_dag` is trivially "identified" (no
  backdoor path left to block) — `technical_notes` exists specifically so
  that is never misread as "no confounding was found."
- **`assess_clinical_completeness`** — human-judgment concerns: is a known
  major confounder unresolved? Is any edge's direction still contested? This
  can block approval even when the DAG is graph-theoretically identified.
- `major_confounders` is a human-set flag (not inferred from a label like
  "age" or "severity") — the only sanctioned way to force a blocking issue,
  so a reviewer is never ambushed by an automatic guess.

The graph engine underneath (`causal/roadmap.py`) is OMOP- and LLM-independent
by design: nodes carry a role and a timing (`baseline` / `treatment` /
`post_treatment` / `outcome`), edges cannot violate declared temporal order,
and `minimum_adjustment_sets` does a bounded exhaustive search (moralized
ancestral graph, d-separation) over baseline, observed, non-forbidden-role
candidates. Instruments are explicitly excluded from adjustment-set candidates
— conditioning on a pure instrument does not close a backdoor path and can
amplify bias from unmeasured confounding.

## 3. Governance surface

- **Workflow gate** (`causal/orchestrator/workflow.py`): the `protocol` gate
  cannot be approved while (a) any major confounder is unresolved, or (b) the
  declared DAG is not identified by backdoor adjustment. Both checks re-derive
  the hypothesis from `protocol.json` at approval time — they are not trusting
  a stale cached result.
- **Human covariate review** (`WorkflowManager.review_covariate`): accepting a
  first-pass covariate requires an explicit role, temporal classification, and
  a rationale comment; rejecting one requires a reviewer identity. Either way
  `hypothesis.json` is regenerated so `assess_identification` reflects the
  decision immediately.
- **Audit trail**: every LLM call (draft, critique, revision) is captured
  verbatim in `llm_outputs.json`, tagged by stage.
- **CLI/UI**: `workflow dag-export` renders the DAG as `dagitty` text for
  external inspection, plus the evidence/adjustment-set metrics from §7 on
  stderr; the Streamlit app's "Hypothesis & DAG" tab renders the graph
  (Graphviz), the identification/clinical verdicts, the same metrics, and
  the accept/reject covariate review flow.

## 4. Test coverage

| File | Lines | Focus |
|---|---|---|
| `tests/test_dag_identification.py` | 675 | backdoor identification, clinical completeness, role→edge mapping, evidence/adjustment-set metrics |
| `tests/test_dag_critique.py` | 180 | prompt building, critique parsing, revision merging |
| `tests/test_user_question_hypothesis.py` | 940 | drafting prompt, evidence gating, KG reinforcement, covariate-targeted KG lookup, end-to-end generation |
| `tests/test_workflow.py` | 357 | gate blocking behavior, covariate review, run lifecycle |
| `tests/test_cli_dag_export.py` | 90 | `workflow dag-export` end to end through the real `causal.cli.main()` entry point |

## 5. Deliberate scope boundaries (not gaps)

- No OMOP concept mapping happens at this stage — by design, identification
  must be checked on the clinical-language DAG before any concept-mapping
  effort is spent.
- The critique step never redrafts the DAG itself, to keep exactly one code
  path (`_parse_dag_response`) responsible for validating citations/roles/
  timing.
- `assess_identification` intentionally does not judge clinical plausibility,
  and `assess_clinical_completeness` intentionally does not judge graph
  theory — kept apart so a DAG can be "identified but not trustworthy" instead
  of collapsing both concerns into one verdict.

## 6. Open gaps / risks worth flagging in a presentation

- **The generated DAGs are not yet clinically exploitable as-is.** Everything
  in §1–§4 is a governance and consistency framework, not a guarantee of
  clinical correctness: it catches structural contradictions (bad role/timing
  pairs, uncited claims, unidentified graphs) but it cannot tell whether the
  *set of variables* itself is clinically adequate. In practice, drafts seen
  so far are missing well-known confounders that simply weren't surfaced by
  the retrieved literature/KG evidence, occasionally misclassify a variable's
  role in ways the critique step did not catch, and lean on `unresolved`/
  advisory notes rather than a clean, reviewer-ready graph. Every candidate
  DAG must still go through the human covariate-review flow
  (`WorkflowManager.review_covariate`) and clinical judgment before it is
  anything more than a first-pass draft — this is the current bottleneck, not
  a solved part of the pipeline.
- **OptimusKG schema is still unconfirmed against the installed dataset.**
  The client's own docstring flags that Biolink-style column names
  (`name`/`category`/`subject`/`predicate`/`object`) were not fully verified
  against the real schema at the time it was written — worth a spot-check
  before leaning on KG evidence in a live demo. (Distinct from, and not
  fixed by, the 2026-07-28 change: that fix widened *which pairs of terms*
  get queried; it does not touch the column-name assumption `find_relations`
  makes about whatever OptimusKG actually returns.)
- **Single critique pass, single revision round.** There is no loop back to a
  second critique after a revision; a revision that introduces a *new* issue
  would not be caught by this pipeline.
- **`minimum_adjustment_sets` is a bounded exhaustive search** (`max_candidates`,
  default 24 in the identification bridge). Fine for reviewable study DAGs;
  would need a different algorithm if candidate DAGs grow much larger.
- **KG label matching is case-insensitive substring matching** on node names
  (`OptimusKGClient._matching_node_ids`) — adequate for a small anchor/
  covariate-term lookup today, but a source of false positives if reused for
  less curated term lists. `_kg_reinforcement`'s own substring matching had
  a related, now-fixed bug (see §0): it did not require the variable label
  and the anchor term to land on opposite endpoints of the same edge, so a
  coincidental same-side match could be read as corroboration that was
  never actually there.
- **No automated check that the critique step actually improves recall of
  known-missing confounders** — its value (catching an omitted confounder
  visible in the evidence) is currently only exercised by hand-constructed
  unit-test scenarios, not measured against a labeled benchmark. §8's
  proposed `causal-learn` cross-check would give this a data-driven signal
  it does not have today.

## 7. Evidence & adjustment-set metrics (added 2026-07-28)

`causal.agents.dag_identification.review_dag` now bundles two descriptive
metrics alongside the identification/clinical verdicts — both read-only,
neither feeds back into approval (that stays governed exclusively by
`assess_identification`/`assess_clinical_completeness`, see §5).

- **`EvidenceSourceSummary`** (`evidence_summary`) buckets every accepted
  `candidate_dag` variable and `covariate_relations` edge into exactly one
  of: `literature_only`, `kg_only`, `corroborated` (both sources
  independently agree — the strongest tier this pipeline can produce), or
  `uncited` (role asserted on rationale alone). `cited_fraction` and
  `corroborated_fraction` give a one-glance read on how evidence-heavy a
  draft is. This is a *coverage* metric, not a *quality* one — it says
  nothing about whether a cited abstract is itself strong evidence (PubMed's
  evidence-tier grading in `causal.literature.orchestrator` is the
  quality axis, applied earlier, at retrieval time).
- **`AdjustmentSetSummary`** (`adjustment_set_summary`) derives structure
  across `minimum_adjustment_sets` rather than just counting them:
  `core_variable_labels` (in *every* minimum set — non-optional),
  `alternative_variable_labels` (in *some* but not all — interchangeable for
  backdoor-blocking purposes alone, though a reviewer may still prefer one
  clinically), and `unused_confounder_labels` (declared `confounder`-role
  variables that never appear in *any* minimum set). The last one is not
  automatically a bug — a confounder whose only backdoor path is already
  blocked by another selected variable is genuinely redundant for
  identification — but `DAGIdentificationResult.technical_notes` cannot
  flag this class of thing (it only tracks failures in *computing* the
  sets), so it is worth a reviewer's attention: it can also mean a role was
  misassigned.

Both are surfaced in the Streamlit "Hypothesis & DAG" tab and in
`workflow dag-export`'s stderr output. See `tests/test_dag_identification.py`
(`EvidenceSummaryTests`, `AdjustmentSetSummaryTests`,
`ReviewDagIncludesMetricsTests`) for the exact semantics, including the
synthetic-DAG fixture that exercises core/alternative/unused in one graph.

## 8. `causal-learn` (causal discovery) — evaluated, not integrated

[causal-learn](https://causal-learn.readthedocs.io/en/latest/) is the
Python causal-discovery toolkit out of CMU (constraint-based PC/FCI,
score-based GES/exact search, functional-model methods like LiNGAM/ANM, and
the conditional-independence tests — Fisher-Z, chi-square/G-square, KCI —
those algorithms run on). It does **not** provide a backdoor-adjustment-set
API, a notion of expert-declared causal roles, or temporal-tier constraints
on an otherwise-known graph — those are exactly what `causal/roadmap.py`
already implements (moralized-ancestral-graph d-separation, hand-rolled and
covered by 500+ lines of tests). Swapping `roadmap.py` for causal-learn
would be a lateral rewrite onto a general-purpose discovery library, not a
functional improvement, and it would add a non-trivial dependency (numpy,
networkx, graphviz bindings) for a problem this codebase already solves.

Where it *would* add something this pipeline genuinely lacks (see §6's last
bullet): **a data-driven cross-check against the literature/KG-drafted
DAG, once real cohort data exists.** Every DAG produced today is drafted
top-down from evidence (literature + KG) and reviewed by a second LLM
critique (§2.2) — but nothing here ever looks at the actual extracted
patient-level data to ask "does the data support this structure, and does
it suggest anything the evidence search missed?" `causal.data_builder.
analytic.AnalyticCohortBuilder` already tags each analytic column with a
`role` (`treatment` / `outcome` / `baseline_covariate` / ...) and a
`temporal_status` — exactly the shape a discovery run needs:

- **Where it would hook in**: after the analytic cohort is built and
  queried into a DataFrame (a step downstream of the DAG-approval gate
  today — this is genuinely new plumbing, not a small addition), pass the
  `A` (treatment), `Y` (outcome), and `baseline_covariate` columns to
  `causallearn.search.ConstraintBased.FCI.fci` (FCI, not PC — PC assumes
  causal sufficiency, i.e. no unmeasured confounding, which is not a safe
  assumption for EHR-derived cohorts; FCI produces a PAG that represents
  that uncertainty honestly instead of hiding it).
- **Encode what is already known as background knowledge, not discard
  it.** `causallearn.utils.PCUtils.BackgroundKnowledge` (also usable with
  FCI) accepts required/forbidden edges and can encode the same temporal
  tiers `causal.roadmap.NODE_TIMINGS` already enforces (baseline must
  precede treatment must precede post_treatment/outcome) — the discovery
  run should search only what the LLM+KG draft left uncertain, not
  relitigate constraints this pipeline already governs.
- **What to report, not auto-apply**: (a) an edge the draft asserts as
  `established`/`plausible` that the conditional-independence tests reject
  in this cohort's actual data — a signal the literature-derived relation
  may not transfer to this population; (b) a data-driven dependency between
  a baseline covariate and treatment/outcome with no counterpart anywhere in
  `candidate_dag` — a direct, data-driven attack on §6's "missing
  confounders the literature/KG search never surfaced" gap. Both would
  become new `unresolved`-style entries for human review, following the
  same "never silently mutates the DAG" convention `_critique_and_revise_dag`
  already uses for the LLM critique step — a discovery run is a third
  independent check, not a fourth vote that outranks the other two.
- **Not done in this pass**: implementing this needs the DataFrame handoff
  from the analytic-cohort stage wired into the DAG-review stage, which is
  new plumbing, not a bug fix — a bigger, separate scope decision.
  `requirements-causal-discovery.txt` (added 2026-07-28) pins the
  optional dependency so a spike can start without a requirements-file
  detour.
