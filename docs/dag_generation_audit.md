# DAG Generation — Audit (as of 2026-07-24, updated 2026-07-28)

Status snapshot of the causal-DAG generation pipeline, for presentation purposes.
Covers what exists today, how it is governed, what is deliberately out of scope,
and what is still missing. All 258 tests pass (`pytest tests/ -q`, 258 passed).

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

## 0b. Update — Phase 1 of the 4-part extension roadmap

Adds a *quality* metric (`confidence_grade`/`ConfidenceSummary`) alongside
the existing *coverage* metric (`EvidenceSourceSummary`) and *structural*
metric (`AdjustmentSetSummary`) from §7 — see §7c. This is the first of four
planned phases (confidence metrics; a bounded critique→revision→re-critique
loop, closing the "single critique pass" gap flagged in §6; a
population/tournament DAG-drafting mode; cross-run reviewer memory plus a
`WorkflowManager` auto-run mode) tracked as an internal roadmap. No new LLM
calls or governance changes in this phase — purely a deterministic,
read-only metric computed from fields the pipeline already records.

## 0c. Update — Phase 2 of the 4-part extension roadmap

Closes the "single critique pass, single revision round" gap §6 flagged
(and §0b's Phase 1 update named as the next phase): `_critique_and_revise_dag`
is now a bounded critique → revision → re-critique loop
(`_MAX_CRITIQUE_ROUNDS = 2` by default, overridable via
`max_critique_rounds=`/`workflow start-question --max-critique-rounds`).
See §2.2 for the mechanics and §6 for the residual bound. Round-numbered
`llm_outputs` stage tags (`literature_dag_critique_round1`,
`literature_dag_revision_round1`, `literature_dag_critique_round2`, ...)
keep every round auditable. No new metric or governance change — the
identification/clinical gates in §2.3/§3 are unaffected; this only changes
how many times the draft is challenged before reaching them.

## 0d. Update — Phase 3 of the 4-part extension roadmap

Implements the population/tournament DAG-drafting mode §9 designed:
`causal.agents.dag_tournament` (new module) plus a `tournament_size`
parameter on `UserQuestionHypothesisService.generate`/`_attach_literature_dag`
and a `workflow start-question --tournament-size` CLI flag. Opt-in —
`tournament_size=1` (the default) is byte-for-byte today's single-draft code
path, untouched; `tournament_size>1` (hard-capped at
`dag_tournament.MAX_TOURNAMENT_SIZE = 5`) drafts that many independent
candidates (each through its own bounded critique loop, §2.2), then:

1. **Ranks** them by the lexicographic key §9 specified
   (`identified_by_backdoor_adjustment`, `approved_for_dag_gate`; see §0e —
   originally a 3-tuple with a GRADE-based confidence tiebreak, since
   removed), reusing `build_pre_check` (§2.2/§2.3's own machinery) per
   candidate rather than recomputing anything.
2. **Dedups** near-duplicates via `structural_hamming_distance`, keeping the
   best-ranked member of each duplicate cluster. One deviation from the
   original design worth flagging: the SHD papers cited in §9 define a
   *distance metric*, not a *near-duplicate cutoff* — there is no citable
   threshold for "how much structural disagreement still counts as a
   duplicate." Rather than invent one (would repeat the exact mistake §9
   itself replaced an invented Jaccard-similarity threshold to avoid),
   `dedup_candidates`'s `duplicate_threshold` defaults to `0`: only
   structurally identical candidates are merged away. Looser dedup is
   possible by passing a larger threshold, but nothing in the reviewed
   literature justifies a specific non-zero default.
3. **Merges** survivors by deterministic edge union (`merge_candidates`),
   exactly as designed: a label with a disagreeing role, a pair with a
   disagreeing edge direction, or a union that introduces a cycle (checked
   by actually building the candidate DAG via
   `causal.agents.dag_identification.build_causal_dag`, which also catches
   the pre-existing "`covariate_relations` cannot reference a
   mediator/collider/selection label" rule for free) are never silently
   resolved — each becomes a `MergeConflict` (`role_conflict` /
   `direction_conflict` / `cycle_conflict` / `structural_conflict`) recorded
   in the new `protocol["dag_tournament"]["conflicts"]` for human review.
4. Runs **one further critique round** (§2.2, `max_rounds=1`) on the merged
   result, tagged `literature_dag_critique_merged_round1` — a structural
   union cannot judge clinical plausibility on its own.

Each candidate's own `llm_outputs` stage tags are suffixed
(`literature_dag_draft_candidate2`, `literature_dag_critique_candidate2_round1`,
...) so the audit trail stays unambiguous across the whole population, not
just the final merge. `protocol["dag_tournament"]` (candidate/survivor/
dropped-duplicate counts, a per-survivor summary, the conflict list, and —
since §0f — pairwise SHD/SID between every ranked candidate plus each
survivor's SID to the final merge) is new, structured, machine-readable
audit output alongside the existing free-text `design_constraints` note;
surfaced in `workflow dag-export`'s stderr and the Streamlit "Hypothesis &
DAG" tab, mirroring how §7's metrics are already surfaced. See
`tests/test_dag_metrics.py` (SHD, SID, and the paper's own Example 1
reproduced as a numeric check) and `tests/test_dag_tournament.py` (ranking,
dedup, all four conflict types, pairwise distances, and merge fidelity, all
in isolation) and `test_user_question_hypothesis.py`'s `TournamentModeTests`
(end-to-end through `UserQuestionHypothesisService.generate`) for exact
semantics.

**Not done in this pass**: `causal.orchestrator.workflow.WorkflowManager`
has no tournament-specific method — `tournament_size` flows through
`UserQuestionHypothesisService.generate` exactly like `max_critique_rounds`
already does, so no workflow-layer change was needed. Phase 4 (cross-run
reviewer memory plus a `WorkflowManager` auto-run mode) remains unstarted.

## 0e. Update — 2026-07-31: GRADE confidence grading removed

`confidence_grade`/`ConfidenceSummary` (§7c, Phase 1 of the roadmap above)
were removed by project decision: GRADE rates the certainty of evidence
backing a specific clinical claim, and grading a whole DAG edge/node with it
treated a methodology built for judging one study's evidence base as if it
were a metric for judging graph structure — not a fit the project wants to
carry forward. `causal.agents.dag_identification.confidence_grade`/
`ConfidenceSummary`/`confidence_summary` and `DAGReviewResult.confidence` are
gone; `evidence_level`/`certainty`/`kg_evidence_id` on `DAGVariable`/
`CovariateRelation` are untouched and still feed `EvidenceSourceSummary`
(§7, a *coverage* metric, never a *quality* grade) and are still shown
per-item (raw, ungraded) in the CLI/Streamlit evidence table. The tournament
mode's `rank_key` (§9, §0d) drops the confidence tiebreak and is now a
2-tuple (`identified_by_backdoor_adjustment`, `approved_for_dag_gate`);
candidates tied on both keep drafting order (stable sort) instead of
breaking the tie on a confidence fraction. §7c below is kept for the
historical record of what existed and why; treat it as removed, not
current.

## 0f. Update — 2026-08-03: SID actually implemented, SHD/SID surfaced as audit metrics

Before this pass, §9's SID citation (Peters & Buhlmann 2015, Proposition 7)
justified *why* the edge-union merge is sound but nothing in the codebase
ever computed an SID value — a real gap between what the design claimed and
what ran. This closes it, and extracts both distance metrics out of the
tournament module into their own:

- **`causal.agents.dag_metrics`** (new module): `structural_hamming_distance`
  (moved, unchanged) and `structural_intervention_distance` (new). SID is
  implemented from the paper's own graphical, distribution-free
  characterization (Definition 4 via Proposition 6, which in turn rests on
  Lemma 5's adjustment-set criterion) — not an approximation of it: for
  every ordered label pair `(i, j)`, `i`'s parent set in the estimate is
  checked against Lemma 5's adjustment-validity condition in the true
  graph, via direct enumeration of the (small, clinically-reviewable)
  graph's simple paths and the standard d-separation collider/non-collider
  blocking rule. `tests/test_dag_metrics.py` reproduces the paper's own
  Example 1 (Figure 2) verbatim, including both published numeric results
  (`SID(G, H1) = 0`, `SID(G, H2) = 8`) — the strongest available check that
  this matches the paper rather than an invented reading of it. Both
  metrics share the same known limitation (graphs compared at the node-
  *label* level, so a label present in only one graph is treated as
  isolated in the other — see §6/§9's OptimusKG/SHD precedent for the same
  class of limitation), and SID has one more of its own worth naming: an
  edge asserted between two labels that are simply unrelated (neither a
  descendant of the other) in the true graph is not penalized unless it
  also breaks a valid-adjustment-set check elsewhere — a property of SID
  itself (Peters & Buhlmann's own point: SID only counts errors that change
  a computed intervention distribution, unlike SHD), not an artifact of
  this implementation.
- **`compute_pairwise_distances`** (`causal.agents.dag_tournament`, new):
  SHD and *both* directions of SID (SID is not symmetric) for every pair of
  ranked, pre-dedup candidates — not just the dedup pass/fail outcome dedup
  itself already computed. Motivation is the paper's own point in the
  passage above: two candidates can disagree on the same number of edges
  (equal SHD) yet have wildly different SID, i.e. wildly different
  consequences for the causal conclusions drawn from each — a distinction
  SHD alone cannot surface, which the previous "N dropped as SHD-duplicates"
  summary hid entirely.
- **`compute_merge_fidelity`** (`causal.agents.dag_tournament`, new): SID
  from each survivor's own DAG to the final merged DAG. Proposition 7
  implies this should be exactly 0 (the merge is an edge union, so every
  survivor's edges are a literal subset of the merged DAG's — by
  construction, not by assumption) *unless* a role/direction/structural
  `MergeConflict` excluded some of that survivor's own variables or
  relations from the merge, in which case a positive value quantifies how
  much of that candidate's own causal-inference conclusions were not
  carried into the merge — a live check on the merge's own soundness,
  surfaced for the human already reviewing the conflict list, rather than a
  bare assertion that Proposition 7 "must" hold.

Both are threaded into `protocol["dag_tournament"]["pairwise_distances"]`/
`["merge_fidelity"]` (`causal.agents.user_question`) and rendered in the
Streamlit "Hypothesis & DAG" tab under a "Pairwise SHD/SID and merge
fidelity" expander (`causal.ui.streamlit_app`). `TournamentCandidate`'s own
`summary_dict` had a `high_or_moderate_fraction` key from the now-removed
GRADE tiebreak (§0e); that key is gone from it, unrelated to this pass.

**Reference-DAG comparison, started same day**: `studies/albumin_sepsis_28d/
reference_dag.json` transcribes the published study's own expert DAG (S6
Fig, built by the paper's authors with a clinician via DAGitty) into the
same `candidate_dag`/`covariate_relations` shape the rest of the pipeline
uses, with per-node provenance and explicit transcription-confidence notes
(`source.transcription_notes`) for the handful of edges that could not be
read from the figure with full confidence. `tests/test_reference_dag_
albumin_sepsis.py` guards the artifact against silently breaking. Comparing
this reference against a CausalOMOP-*generated* DAG for the same study (the
"how close is automated drafting to expert judgment" number) is not done
yet — no `candidate_dag` has been drafted for this study through the
pipeline itself.

**Scalability limitation surfaced by this transcription, not fixed**: the
reference DAG has 20 *observed* baseline confounders, and — because 3 more
confounders are transcribed as unobserved (matching the paper's own gray
"not available in the data" nodes) — no valid backdoor adjustment set
exists among those 20 at *any* size. `causal.agents.dag_critique.
build_pre_check` (via `assess_identification`'s default `max_candidates=24`)
calls `CausalDAG.minimum_adjustment_sets`, whose exhaustive `combinations`
search cannot short-circuit in this "no valid set exists" case: it must
enumerate all `2**20` (~1e6) subsets before concluding failure, which does
not finish in practical time. `structural_intervention_distance`/
`structural_hamming_distance` never needed that search in the first place —
they only need the graph structure — so `tests/test_reference_dag_
albumin_sepsis.py` builds the `CausalDAG` via `build_causal_dag` directly
(skipping `assess_identification` entirely) instead of `build_pre_check`.
This is a real, pre-existing scalability gap in `minimum_adjustment_sets`
for realistically-sized *published* DAGs (15–25 variables, well beyond a
typical LLM-drafted candidate's handful) — noted here, not fixed, since
nothing asked for it in this pass and it is unrelated to the SID/SHD work
itself.

## 0g. Update — 2026-08-03: a grounding caveat, surfaced everywhere a DAG verdict is

Prompted by the question this audit's own headline caveat (§6, first bullet)
begs: for an arbitrary user question — not the one study
(`albumin_sepsis_28d`) that happens to have a hand-transcribed published
reference DAG (§0f) — there is no way to *guarantee* a generated DAG is
literature-grounded or clinically correct. Two options were considered:
(a) automatically search literature/KG evidence for a published reference
DAG on every question and run the same SHD/SID comparison §0f did by hand,
or (b) make the *absence* of that guarantee explicit wherever a DAG's
verdict is shown, so `identified_by_backdoor_adjustment` /
`approved_for_dag_gate` are never mistaken for "clinically validated". (a)
was set aside — most clinical questions have no published DAG to find, so
it would add an LLM/search cost for a rarely-successful lookup — in favor
of (b), by project decision.

`causal.agents.dag_identification.grounding_caveat(evidence:
EvidenceSourceSummary) -> str` is a fixed disclaimer, not a new computed
verdict: it states plainly that no published or expert reference DAG was
checked against this graph, that "identified" / "approved for DAG gate"
mean only "no known blocking issue was found", and restates §7's citation
coverage (`cited_fraction`/`corroborated_fraction`) as the closest thing to
a grounding signal this pipeline has — explicitly a *coverage* metric, not
a *correctness* one. Bundled into `DAGReviewResult.caveat` (`review_dag`),
so every existing caller of `review_dag` gets it automatically: rendered as
a persistent `st.warning` at the top of the Streamlit "Hypothesis & DAG"
tab's DAG section, and as a `// caveat: ...` line in `workflow dag-export`'s
stderr output, alongside the identification/evidence/adjustment lines §7
already prints there. No governance change and no new LLM call — purely
additive, read-only text derived from a metric the pipeline already
computes. See `tests/test_dag_identification.py`'s `GroundingCaveatTests`.

**Not a fix for §6's first bullet** — the DAGs this pipeline drafts are
exactly as clinically complete as before this change; only the *legibility*
of that limitation changed, from "documented in this audit" to "shown next
to every verdict a reviewer sees."

## 0h. Update — 2026-08-03: the same benchmarking gap, closed for concept mapping (adjacent to this audit, not DAG generation itself)

§0f closed "SID is cited but never computed" for DAG structure. The
clinical-term → OMOP concept-lookup step (`causal.agents.data_science.
DataScienceAgent._lookup_candidates`, backed by OMCP's `Lookup_Condition`/
`Lookup_Drug`/`Lookup_Concept`) had the exact same shape of gap: no
precision/recall benchmark existed anywhere in the repo for whether that
search actually surfaces the concept a reviewer would approve. This is not
a DAG-generation change — noted here only because it follows the same
"measure with an established, citable metric before/after changing the
underlying method" discipline as §0f, and because OMCP's search itself was
improved in the same pass (see below).

- **`causal.mapping.metrics`** (new module): `recall_at_k` and
  `reciprocal_rank`/mean reciprocal rank (MRR) — standard information-
  retrieval metrics (Manning, Raghavan & Schuetze 2008; Voorhees 1999),
  not an invented heuristic. `evaluate_lookup_accuracy` aggregates both
  over a list of `LookupCase`s, excluding cases with no achievable standard
  concept today (a documented upstream ETL/vocabulary gap) from the score
  rather than letting them silently look like search failures.
- **`studies/albumin_sepsis_28d/reference_concept_mappings.json`** (new):
  seeded from the manual concept review already written down in
  `concept_audit.md` — sepsis/septic_shock as real recall/MRR targets,
  albumin/crystalloids as documented known-gap cases (`expected_concept_id:
  null`). Same caveat as `reference_dag.json` (§0f): this encodes a
  documented clinical/data judgment, not a `protocol.json` workflow
  approval — see the file's own `source.description`.
- **OMCP search itself** (`omcp/src/omcp/concept_lookup.py`): every
  name-search query now also computes `name_edit_distance` (Levenshtein
  distance from the search term to a candidate's own name) and orders by
  it instead of plain shortest-name-first, and automatically retries with a
  length-scaled fuzzy (edit-distance) match — following the same
  length-scaled tolerance Elasticsearch/Lucene document for "AUTO"
  fuzziness (Schulz & Mihov 2002) — when the first page of an exact/
  substring search returns nothing. `causal.agents.data_science.
  ConceptCandidate` and the Streamlit review UI both surface the resulting
  `match_score` to the reviewer.
- **`causal.agents.data_science._attach_covariate_concept_candidates`**: a
  covariate whose DAG entry has no OMOP `domain` yet was already,
  deliberately, left unmapped rather than guessed ("clinical judgment is
  explicit and checked") — but that used to leave it silently stuck at
  `definition_pending` with nothing telling a reviewer why. It now also
  appears in `unresolved_issues` as `covariate:{label}:missing_domain`, the
  same pattern `covariate:{label}:unclassified_by_dag` already used.

**Not done**: running `evaluate_lookup_accuracy` against a live OMCP/
BigQuery connection is a manual step (no credentials in CI) —
`tests/test_concept_mapping_metrics.py` only checks the metric functions
themselves (hand-built cases) and the reference file's shape, the same
split §0f's `tests/test_dag_metrics.py`/`tests/test_reference_dag_
albumin_sepsis.py` use. No actual before/after recall@k/MRR numbers exist
yet for this pipeline's own OMOP dataset.

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
  a solved part of the pipeline. Since §0g, this limitation is at least
  surfaced explicitly next to every identification/clinical verdict
  (`DAGReviewResult.caveat`) rather than living only in this document — the
  limitation itself is unchanged, only its visibility to a reviewer is.
- **OptimusKG schema is still unconfirmed against the installed dataset.**
  The client's own docstring flags that Biolink-style column names
  (`name`/`category`/`subject`/`predicate`/`object`) were not fully verified
  against the real schema at the time it was written — worth a spot-check
  before leaning on KG evidence in a live demo. (Distinct from, and not
  fixed by, the 2026-07-28 change: that fix widened *which pairs of terms*
  get queried; it does not touch the column-name assumption `find_relations`
  makes about whatever OptimusKG actually returns.)
- ~~Single critique pass, single revision round~~ — **fixed**: the critique
  step is now a bounded critique → revision → re-critique loop
  (`_MAX_CRITIQUE_ROUNDS = 2` by default; see §2.2). The remaining bound to
  flag: it is still bounded, not unbounded-until-approved — a genuinely
  intractable disagreement between drafting and critique model exhausts the
  round budget and is reported as such (see §2.2's "reached the maximum
  number of critique rounds" outcome) rather than looping indefinitely, and
  there is no automated check that a *third* round would have caught
  anything a second round did not.
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

## 7c. Confidence grading (added — Phase 1 of the 4-part extension roadmap; **removed 2026-07-31, see §0e**)

**Removed.** This section originally documented
`causal.agents.dag_identification.confidence_grade`/`ConfidenceSummary`
(`confidence_summary`, bundled into `review_dag` as `DAGReviewResult.
confidence`): a GRADE-inspired grading (Guyatt GH, Oxman AD, Vist GE, et al.
"GRADE: an emerging consensus on rating quality of evidence and strength of
recommendations." *BMJ*. 2008;336(7650):924–926.
[doi:10.1136/bmj.39489.470347.AD](https://doi.org/10.1136/bmj.39489.470347.AD))
of `EvidenceSourceSummary` (§7) items — base grade from evidence tier,
downgraded one level for contested/unknown-direction certainty, upgraded
one level for independent KG corroboration, clamped to a four-level
`very_low`/`low`/`moderate`/`high` scale. It was removed by project
decision that GRADE — a framework for rating the certainty of evidence
behind one clinical claim — is not an appropriate metric for grading DAG
structure (nodes/edges), even in this simplified single-function form. See
§0e for what replaced it (nothing — the tournament's `rank_key`, §9, simply
dropped the confidence tiebreak) and what still uses the underlying
`evidence_level`/`certainty`/`kg_evidence_id` fields (§7's
`EvidenceSourceSummary`, unaffected).

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

## 9. Population + tournament — implemented, Phase 3 of the 4-part extension roadmap (see §0d)

Today's pipeline (§2.1–§2.2) drafts and critiques exactly **one** candidate
DAG per run. That is still one sample from the LLM; §6's "DAGs not yet
clinically exploitable as-is" headline caveat is partly a single-sample
problem. Phase 3 drafts `N` independent candidates, dedups near-duplicates,
and combines the survivors into one DAG before it reaches
`protocol["candidate_dag"]` — opt-in (`tournament_size=1` stays the default,
today's exact single-draft behavior).

**Design decision worth flagging explicitly:** an earlier version of this
plan scored candidates with an invented weighted formula, deduplicated them
with plain Jaccard similarity on variable labels, and merged the survivors
with an LLM "Evolution agent" — none of that is literature-grounded. It was
replaced after reviewing two papers (Peters & Bühlmann's SID paper, and a
GRADE/Bradford-Hill paper that independently confirmed the §7c GRADE
citation via its own reference list) with mechanisms that are each either a
named metric from the causal-discovery literature or a direct consequence
of a proven property in it:

- **Structural Hamming Distance (SHD)** for the dedup pass, in place of
  Jaccard similarity (now in `causal.agents.dag_metrics`, alongside SID —
  see §0f). SHD counts, over the union of both graphs' node
  labels, every pair whose edge type (no edge / `a->b` / `b->a`) differs
  between two candidates (Acid, S. & de Campos, L.M. (2003). Searching for
  Bayesian network structures in the space of restricted acyclic partially
  directed graphs. *Journal of Artificial Intelligence Research*, 18,
  445–490; Tsamardinos, I., Brown, L.E., & Aliferis, C.F. (2006). The
  max-min hill-climbing Bayesian network structure learning algorithm.
  *Machine Learning*, 65, 31–78 — both cited from, and confirmed via, Peters
  & Bühlmann 2015 §1.1 below). **Known limitation, to document alongside the
  implementation**: SHD formally assumes both graphs share the same
  variable set; two candidates naming a clinically-identical concept with
  different free-text labels (e.g. "age" vs. "patient age") are treated as
  genuinely different variables, the same class of limitation already
  flagged for OptimusKG's substring matching in §6.
- **A lexicographic ranking**, in place of a weighted score: sort candidates
  by `(identified_by_backdoor_adjustment, approved_for_dag_gate)`
  descending — no new weights or magic numbers. Originally a 3-tuple with a
  §7c GRADE-based confidence fraction as a tiebreak; removed along with §7c
  (§0e). Candidates tied on both remaining keys keep their original
  (drafting) order (`rank_candidates` is a stable sort) rather than being
  broken by a third key.
- **A deterministic edge-union merge**, in place of an LLM "Evolution
  agent," justified by Peters, J. & Bühlmann, P. (2015). Structural
  Intervention Distance (SID) for Evaluating Causal Graphs. *Neural
  Computation*, 27(3), 771–799 (arXiv:1306.1043) — specifically its
  **Proposition 7**: `SID(G, H) = 0 ⟺ G ≤ H` (G a subgraph of H). A superset
  graph never predicts a *wrong* intervention distribution relative to any
  DAG it is a superset of, for data actually Markov with respect to the
  weaker graph. Unioning the surviving candidates' edges is therefore sound
  *provided the union stays consistent*: a duplicate label with a
  disagreeing role, a duplicate edge with a disagreeing direction, or a
  union that introduces a cycle are all genuine structural conflicts
  between independently-drafted candidates — recorded for human review,
  never silently resolved by picking one side or guessing which edge to
  drop. This removes an LLM call from the pipeline entirely (cheaper, fully
  auditable, grounded in a proven graph property instead of LLM judgment on
  which candidates to trust) and only reuses the deterministic acyclicity
  check `causal.roadmap.CausalDAG` already has. At the time this bullet was
  originally written, SID itself was cited but never computed — closed in
  §0f, which also uses SID to check Proposition 7 against the *actual*
  merge output (`compute_merge_fidelity`) rather than only asserting it
  algebraically here.
- One more critique round (§2.2) still runs on the merged result — a
  structural union cannot judge clinical plausibility on its own, so this
  is the one LLM call the merge step keeps.

**Cost model**: `N × (1 draft + up to max_critique_rounds critique/revision
pairs)` + 1 final critique round on the merge — scales linearly in `N`,
hard-capped in code (`dag_tournament.MAX_TOURNAMENT_SIZE = 5`) since this is
opt-in and explicitly a cost/thoroughness trade-off, not a default-on
behavior.

**Implemented as designed, see §0d** for the concrete module
(`causal/agents/dag_tournament.py`), the `tournament_size` parameter, the
`workflow start-question --tournament-size` CLI flag, and the one
deliberate deviation (the SHD near-duplicate threshold, which the cited
literature does not itself define).
