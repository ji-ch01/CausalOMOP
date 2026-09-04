# DAG Generation — Technical Reference

> Design rationale, evaluation methodology, and known gaps for the causal-DAG
> generation pipeline. Point-in-time status and implementation history belong
> to Git/CI, not this file -- see [LLM and evidence integration](llm_hypothesis_integration.md)
> and the [causal roadmap](causal_roadmap.md) for the maintained overview, and
> `git log` over `causal/agents/dag_identification.py`,
> `causal/agents/dag_critique.py`, `causal/agents/dag_tournament.py`, and
> `causal/agents/dag_metrics.py` for how this pipeline actually evolved.

Covers what the DAG-generation pipeline does, how it is governed, what is
deliberately out of scope, and what is still missing. Run
`python -m pytest tests -q` for the current suite rather than relying on a
frozen count in this document.

**Headline caveat:** the machinery below is solid -- the governance, the
identification checks, the audit trail all work as designed -- but the DAGs it
currently produces are not yet clinically usable as-is. The pipeline is a
rigorous scaffold for producing and checking a candidate DAG, not yet a
source of clinically trustworthy DAGs on its own; see §6 for why.

## 1. What "DAG generation" means in this codebase

The DAG is never asked for outright. It is assembled in three separable stages,
each with its own trust boundary:

1. **Drafting** — an LLM proposes candidate variables from literature/knowledge-
   graph evidence (`causal/prompts/dag.md`).
2. **Critique** — independent LLM calls challenge the draft before it is
   accepted, with a bounded critique/revision loop (two critique rounds by
   default)
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
  directly), each first-pass covariate is also paired against treatment/
  outcome directly, not just the four trial anchor terms pairwise — without
  this, KG evidence could only ever ground a variable whose label happened to
  be a substring of the population/treatment/comparator/outcome terms
  themselves, i.e. essentially never a real confounder.
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
- **Bounded critique → revision → re-critique loop**
  (`UserQuestionHypothesisService._critique_and_revise_dag`): up to
  `max_critique_rounds` (`_MAX_CRITIQUE_ROUNDS = 2` by default, overridable
  per call and via `workflow start-question --max-critique-rounds`) critique
  calls, each followed by at most one revision call. `build_pre_check`
  (the deterministic structural/clinical pre-check, §2.3) is re-run on the
  *current* state each round, so a later round's critique sees what the
  previous round actually produced, not the original draft. This closes the
  gap a single critique-and-revise pass cannot: a revision can introduce a
  new issue the first critique never had a chance to see. `llm_outputs` tags
  every round distinctly (`literature_dag_critique_round1`,
  `literature_dag_revision_round1`, `literature_dag_critique_round2`, ...)
  so the full loop is auditable, not just its final outcome.
- Failure mode at every round (empty/malformed critique or revision output)
  is "stop the loop and keep the last successfully-parsed state, with an
  explanatory note naming the round" — never a hard failure of the whole
  hypothesis draft. Reaching `max_critique_rounds` without a final `approve`
  is a distinct, explicitly-labeled outcome ("reached the maximum number of
  critique rounds without a final approval") — never silently reported the
  same way as an actual approval.

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
| `tests/test_dag_identification.py` | 686 | backdoor identification, clinical completeness, role→edge mapping, evidence/adjustment-set metrics |
| `tests/test_dag_critique.py` | 180 | prompt building, critique parsing, revision merging |
| `tests/test_user_question_hypothesis.py` | 1395 | drafting prompt, evidence gating, KG reinforcement, covariate-targeted KG lookup, evidence-level tagging, bounded critique-loop rounds, tournament-mode end-to-end generation, end-to-end generation |
| `tests/test_dag_tournament.py` | 221 | SHD, lexicographic ranking, SHD-based dedup, and all four merge-conflict types (`role`/`direction`/`cycle`/`structural`) in isolation |
| `tests/test_workflow.py` | 357 | gate blocking behavior, covariate review, run lifecycle |
| `tests/test_cli_dag_export.py` | 90 | `workflow dag-export` end to end through the real `causal.cli.main()` entry point |

Line counts drift with the codebase; re-run `python -m pytest tests -q` rather
than trusting the table above for a current figure.

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
  a solved part of the pipeline. This limitation is surfaced explicitly next
  to every identification/clinical verdict, not only in this document:
  `causal.agents.dag_identification.grounding_caveat(evidence:
  EvidenceSourceSummary) -> str` is a fixed disclaimer stating that no
  published or expert reference DAG was checked against this graph, that
  "identified" / "approved for DAG gate" mean only "no known blocking issue
  was found", and restating §7's citation coverage
  (`cited_fraction`/`corroborated_fraction`) as the closest thing to a
  grounding signal this pipeline has — a *coverage* metric, not a
  *correctness* one. Bundled into `DAGReviewResult.caveat` (`review_dag`), it
  renders as a persistent `st.warning` at the top of the Streamlit
  "Hypothesis & DAG" tab and as a `// caveat: ...` line in `workflow
  dag-export`'s stderr output. See
  `tests/test_dag_identification.py`'s `GroundingCaveatTests`. Automatically
  searching literature/KG evidence for a published reference DAG on every
  question, to run the same SHD/SID comparison the next two bullets describe
  by hand, was considered and set aside: most clinical questions have no
  published DAG to find, so it would add an LLM/search cost for a
  rarely-successful lookup.
- **Reference-DAG comparison artifacts exist, comparison not yet run.**
  Three question-bank studies carry a `reference_dag.json` in the same
  `candidate_dag`/`covariate_relations` shape the rest of the pipeline uses,
  with per-node provenance: `studies/ppi_vs_h2ra_sup_mortality/`,
  `studies/corticosteroids_severe_cap_mortality/`, and
  `studies/vanco_piptazo_vs_cefepime_aki/`. Each is built from the anchor
  trial's covariates plus clinical reasoning (PEPTIC / CAPE COD / ACORN
  publish no DAG figure) and carries explicitly unobserved confounders, so
  conditioning on the full observed set still leaves an open backdoor path.
  `tests/test_reference_dag_question_bank.py` guards the artifacts against
  silently breaking. Comparing one of these references against a
  CausalOMOP-*generated* DAG for the same study — the "how close is automated
  drafting to expert judgment" number — is not done yet: no `candidate_dag`
  has been drafted for these studies through the pipeline itself.
- **`minimum_adjustment_sets` does not scale to a realistically-sized
  reference DAG.** The `corticosteroids_severe_cap_mortality` reference DAG
  has 20 confounders, 17 of them *observed*, and — because 3 are transcribed
  as unobserved (radiographic extent of infiltrates, goals-of-care
  limitation, frailty) — no valid backdoor adjustment set exists among the
  observed ones at *any* size.
  `causal.agents.dag_critique.build_pre_check` (via `assess_identification`'s
  default `max_candidates=24`) calls `CausalDAG.minimum_adjustment_sets`,
  whose exhaustive `combinations` search cannot short-circuit in this
  "no valid set exists" case: it must enumerate every subset of the observed
  confounders before concluding failure, which does not finish in practical
  time.
  `structural_intervention_distance`/`structural_hamming_distance` never
  needed that search in the first place — they only need the graph structure
  — so `tests/test_reference_dag_question_bank.py` builds the `CausalDAG`
  via `build_causal_dag` directly (skipping `assess_identification` entirely)
  instead of `build_pre_check`. This is a real, pre-existing scalability gap
  in `minimum_adjustment_sets` for realistically-sized reference DAGs
  (15–25 variables, well beyond a typical LLM-drafted candidate's handful),
  not yet fixed.
- **OptimusKG schema is still unconfirmed against the installed dataset.**
  The client's own docstring flags that Biolink-style column names
  (`name`/`category`/`subject`/`predicate`/`object`) were not fully verified
  against the real schema at the time it was written — worth a spot-check
  before leaning on KG evidence in a live demo.
- **Critique is bounded, not unbounded-until-approved.** A genuinely
  intractable disagreement between drafting and critique model exhausts the
  round budget and is reported as such (see §2.2's "reached the maximum
  number of critique rounds" outcome) rather than looping indefinitely, and
  there is no automated check that a *third* round would have caught
  anything a second round did not.
- **`minimum_adjustment_sets` is a bounded exhaustive search** (`max_candidates`,
  default 24 in the identification bridge). Fine for reviewable study DAGs;
  would need a different algorithm if candidate DAGs grow much larger (see
  the reference-DAG bullet above for a concrete case where this already
  bites).
- **KG label matching is case-insensitive substring matching** on node names
  (`OptimusKGClient._matching_node_ids`) — adequate for a small anchor/
  covariate-term lookup today, but a source of false positives if reused for
  less curated term lists. `_kg_reinforcement` requires a matched edge's two
  endpoints to be opposite endpoints of the same edge before treating it as
  corroboration, to avoid misreading a coincidental same-side substring match
  as independent confirmation that was never actually there.
- **No automated check that the critique step actually improves recall of
  known-missing confounders** — its value (catching an omitted confounder
  visible in the evidence) is currently only exercised by hand-constructed
  unit-test scenarios, not measured against a labeled benchmark. §8's
  proposed `causal-learn` cross-check would give this a data-driven signal
  it does not have today.

## 7. Evidence & adjustment-set metrics

`causal.agents.dag_identification.review_dag` bundles two descriptive
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

## 7c. Confidence grading — evaluated and removed (2026-07-31)

A GRADE-inspired grading (Guyatt GH, Oxman AD, Vist GE, et al. "GRADE: an
emerging consensus on rating quality of evidence and strength of
recommendations." *BMJ*. 2008;336(7650):924–926.
[doi:10.1136/bmj.39489.470347.AD](https://doi.org/10.1136/bmj.39489.470347.AD))
of `EvidenceSourceSummary` (§7) items -- base grade from evidence tier,
downgraded one level for contested/unknown-direction certainty, upgraded one
level for independent KG corroboration -- was implemented, then removed by
project decision: GRADE rates the certainty of evidence behind *one clinical
claim*, and grading a whole DAG edge/node with it treated a methodology built
for that purpose as if it were a metric for judging graph structure -- not a
fit this project wants to carry forward. The tournament mode's `rank_key`
(§9) dropped the confidence tiebreak accordingly and is a 2-tuple today.
`evidence_level`/`certainty`/`kg_evidence_id` on `DAGVariable`/
`CovariateRelation` are unaffected and still feed `EvidenceSourceSummary`
(§7, a *coverage* metric, never a *quality* grade). Do not reintroduce
GRADE-style grading for DAG structure.

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
- **Not done**: implementing this needs the DataFrame handoff from the
  analytic-cohort stage wired into the DAG-review stage, which is new
  plumbing, not a bug fix — a bigger, separate scope decision.
  `requirements-causal-discovery.txt` pins the optional dependency so a
  spike can start without a requirements-file detour.

## 9. Population + tournament

Today's pipeline (§2.1–§2.2) drafts and critiques exactly **one** candidate
DAG per run by default. That is still one sample from the LLM; §6's "DAGs not
yet clinically exploitable as-is" headline caveat is partly a single-sample
problem. `causal.agents.dag_tournament` (module) drafts `N` independent
candidates, dedups near-duplicates, and combines the survivors into one DAG
before it reaches `protocol["candidate_dag"]` — opt-in via a `tournament_size`
parameter on `UserQuestionHypothesisService.generate`/`_attach_literature_dag`
and a `workflow start-question --tournament-size` CLI flag
(`tournament_size=1` is the default, today's exact single-draft behavior;
`tournament_size>1` is hard-capped at `dag_tournament.MAX_TOURNAMENT_SIZE = 5`).

**Design decision worth flagging explicitly:** an earlier version of this
plan scored candidates with an invented weighted formula, deduplicated them
with plain Jaccard similarity on variable labels, and merged the survivors
with an LLM "Evolution agent" — none of that is literature-grounded. It was
replaced with mechanisms that are each either a named metric from the
causal-discovery literature or a direct consequence of a proven property in
it:

- **Structural Hamming Distance (SHD)** for the dedup pass, in place of
  Jaccard similarity (`causal.agents.dag_metrics`, alongside SID below). SHD
  counts, over the union of both graphs' node labels, every pair whose edge
  type (no edge / `a->b` / `b->a`) differs between two candidates (Acid, S. &
  de Campos, L.M. (2003). Searching for Bayesian network structures in the
  space of restricted acyclic partially directed graphs. *Journal of
  Artificial Intelligence Research*, 18, 445–490; Tsamardinos, I., Brown,
  L.E., & Aliferis, C.F. (2006). The max-min hill-climbing Bayesian network
  structure learning algorithm. *Machine Learning*, 65, 31–78 — both cited
  from, and confirmed via, Peters & Bühlmann 2015 §1.1 below). **Known
  limitation, to document alongside the implementation**: SHD formally
  assumes both graphs share the same variable set; two candidates naming a
  clinically-identical concept with different free-text labels (e.g. "age"
  vs. "patient age") are treated as genuinely different variables, the same
  class of limitation already flagged for OptimusKG's substring matching in
  §6. `dedup_candidates`'s `duplicate_threshold` defaults to `0`: the SHD
  papers cited above define a *distance metric*, not a *near-duplicate
  cutoff*, so only structurally identical candidates are merged away by
  default rather than inventing a non-zero threshold nothing in the reviewed
  literature justifies. Looser dedup is possible by passing a larger
  threshold.
- **A lexicographic ranking**, in place of a weighted score: sort candidates
  by `(identified_by_backdoor_adjustment, approved_for_dag_gate)` descending
  — no new weights or magic numbers (`rank_candidates`, a stable sort, reuses
  `build_pre_check` per candidate rather than recomputing anything).
  Candidates tied on both keys keep their original (drafting) order.
- **A deterministic edge-union merge** (`merge_candidates`), in place of an
  LLM "Evolution agent," justified by Peters, J. & Bühlmann, P. (2015).
  Structural Intervention Distance (SID) for Evaluating Causal Graphs.
  *Neural Computation*, 27(3), 771–799 (arXiv:1306.1043) — specifically its
  **Proposition 7**: `SID(G, H) = 0 ⟺ G ≤ H` (G a subgraph of H). A superset
  graph never predicts a *wrong* intervention distribution relative to any
  DAG it is a superset of, for data actually Markov with respect to the
  weaker graph. Unioning the surviving candidates' edges is therefore sound
  *provided the union stays consistent*: a duplicate label with a
  disagreeing role, a duplicate edge with a disagreeing direction, or a
  union that introduces a cycle (checked by actually building the candidate
  DAG via `causal.agents.dag_identification.build_causal_dag`, which also
  catches the pre-existing "`covariate_relations` cannot reference a
  mediator/collider/selection label" rule for free) are all genuine
  structural conflicts between independently-drafted candidates — recorded
  as a `MergeConflict` (`role_conflict` / `direction_conflict` /
  `cycle_conflict` / `structural_conflict`) in
  `protocol["dag_tournament"]["conflicts"]` for human review, never silently
  resolved by picking one side or guessing which edge to drop. This removes
  an LLM call from the pipeline entirely (cheaper, fully auditable, grounded
  in a proven graph property instead of LLM judgment on which candidates to
  trust) and only reuses the deterministic acyclicity check
  `causal.roadmap.CausalDAG` already has.
- SID is implemented from the paper's own graphical, distribution-free
  characterization (Definition 4 via Proposition 6, which in turn rests on
  Lemma 5's adjustment-set criterion) — not an approximation of it: for every
  ordered label pair `(i, j)`, `i`'s parent set in the estimate is checked
  against Lemma 5's adjustment-validity condition in the true graph, via
  direct enumeration of the (small, clinically-reviewable) graph's simple
  paths and the standard d-separation collider/non-collider blocking rule.
  `tests/test_dag_metrics.py` reproduces the paper's own Example 1 (Figure 2)
  verbatim, including both published numeric results (`SID(G, H1) = 0`,
  `SID(G, H2) = 8`) — the strongest available check that this matches the
  paper rather than an invented reading of it. `compute_pairwise_distances`
  (`causal.agents.dag_tournament`) computes SHD and *both* directions of SID
  (SID is not symmetric) for every pair of ranked, pre-dedup candidates,
  since two candidates can disagree on the same number of edges (equal SHD)
  yet have wildly different SID — i.e. wildly different consequences for the
  causal conclusions drawn from each. `compute_merge_fidelity` computes SID
  from each survivor's own DAG to the final merged DAG: Proposition 7 implies
  this should be exactly 0 (the merge is an edge union, so every survivor's
  edges are a literal subset of the merged DAG's) *unless* a
  role/direction/structural `MergeConflict` excluded some of that survivor's
  own variables or relations from the merge, in which case a positive value
  quantifies how much of that candidate's own causal-inference conclusions
  were not carried into the merge. Both are threaded into
  `protocol["dag_tournament"]["pairwise_distances"]`/`["merge_fidelity"]`
  (`causal.agents.user_question`) and rendered in the Streamlit "Hypothesis &
  DAG" tab under a "Pairwise SHD/SID and merge fidelity" expander
  (`causal.ui.streamlit_app`). See `tests/test_dag_metrics.py` and
  `tests/test_dag_tournament.py` for exact semantics. Both metrics share one
  more limitation worth naming: graphs are compared at the node-*label*
  level, so a label present in only one graph is treated as isolated in the
  other, and SID specifically does not penalize an edge asserted between two
  otherwise-unrelated labels unless it also breaks a valid-adjustment-set
  check elsewhere — a property of SID itself (it only counts errors that
  change a computed intervention distribution, unlike SHD), not an artifact
  of this implementation.
- One further critique round (§2.2, `max_rounds=1`) always runs on the merged
  result, tagged `literature_dag_critique_merged_round1` — a structural union
  cannot judge clinical plausibility on its own, so this is the one LLM call
  the merge step keeps.

Each candidate's own `llm_outputs` stage tags are suffixed
(`literature_dag_draft_candidate2`, `literature_dag_critique_candidate2_round1`,
...) so the audit trail stays unambiguous across the whole population, not
just the final merge.

**`causal.orchestrator.workflow.WorkflowManager` has no tournament-specific
method** — `tournament_size` flows through
`UserQuestionHypothesisService.generate` exactly like `max_critique_rounds`
already does, so no workflow-layer change was needed. A follow-on idea not
pursued here: cross-run reviewer memory plus a `WorkflowManager` auto-run
mode.

**Cost model**: `N × (1 draft + up to max_critique_rounds critique/revision
pairs)` + 1 final critique round on the merge — scales linearly in `N`,
hard-capped in code (`dag_tournament.MAX_TOURNAMENT_SIZE = 5`) since this is
opt-in and explicitly a cost/thoroughness trade-off, not a default-on
behavior.

See `tests/test_dag_tournament.py` (SHD, lexicographic ranking, SHD-based
dedup, and all four merge-conflict types in isolation) and
`test_user_question_hypothesis.py`'s `TournamentModeTests` (end-to-end
through `UserQuestionHypothesisService.generate`) for exact semantics.
