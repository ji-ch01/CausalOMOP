"""Population/tournament DAG-drafting mode (Phase 3 of the 4-part extension
roadmap tracked in docs/dag_generation_audit.md; see that doc's SS9 for the
full design rationale, the literature grounding each mechanism below is
justified by, and the known limitations).

Today's pipeline (`causal.agents.user_question`) drafts and critiques exactly
one candidate DAG per run -- one sample from the drafting LLM. This module
lets a caller draft several *independent* candidates from the same evidence
and combine them into one DAG before it reaches `protocol["candidate_dag"]`,
using only deterministic, literature-grounded mechanisms:

- `structural_hamming_distance` / `dedup_candidates` -- SHD-based
  near-duplicate detection, in place of an invented similarity heuristic.
- `rank_candidates` -- a lexicographic ranking on already-computed
  identification/clinical/confidence verdicts, in place of a weighted score.
- `merge_candidates` -- a deterministic edge-union merge (justified by
  Peters & Buhlmann 2015's Proposition 7: a superset DAG never predicts a
  *wrong* intervention distribution relative to any DAG it is a superset
  of), in place of an LLM "evolution" step. Genuine disagreements between
  candidates (a label's role, an edge's direction, or a cycle introduced by
  the union) are recorded as `MergeConflict` entries for human review, never
  silently resolved by picking one side.
- `compute_pairwise_distances` / `compute_merge_fidelity` -- both metrics
  from `causal.agents.dag_metrics` surfaced as audit output, not just used
  internally: SHD alone only counts edge disagreements between candidates,
  blind to whether any of them would actually change a computed causal
  effect; pairing it with SID answers that second question directly, and
  `compute_merge_fidelity` uses Proposition 7 itself as a live check that
  the merge actually preserved each survivor's own identification
  conclusions (never assumed).

No new LLM calls happen inside this module: `UserQuestionHypothesisService`
owns every `llm(prompt)` call (each candidate's own draft/critique loop, and
the one additional critique round the merged result gets) and simply hands
the finished candidates here to be deduped, ranked, and merged.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from causal.agents.dag_critique import build_pre_check
from causal.agents.dag_identification import DAGReviewResult, build_causal_dag
from causal.agents.dag_metrics import structural_hamming_distance, structural_intervention_distance
from causal.agents.hypothesis import HypothesisAgent
from causal.roadmap import CausalDAG


#: Opt-in, hard-capped: N independent drafts cost N x (1 draft + up to
#: max_critique_rounds critique/revision pairs) + 1 final critique round on
#: the merge -- a cost/thoroughness trade-off that must stay bounded, not a
#: default-on behavior (see SS9's cost model).
MAX_TOURNAMENT_SIZE = 5

#: SHD (see `structural_hamming_distance`) is a distance metric, not a
#: "near-duplicate" cutoff -- the literature it comes from does not define
#: one. Rather than invent a threshold, `dedup_candidates` defaults to the
#: only value with an unambiguous meaning: exact structural agreement.
DEFAULT_DUPLICATE_THRESHOLD = 0


@dataclass(frozen=True)
class TournamentCandidate:
    """One independently-drafted-and-critiqued DAG in the tournament population.

    `variables`/`relations`/`unresolved` are the same dict shape
    `UserQuestionHypothesisService._parse_dag_response` already produces --
    this module never re-parses or re-validates their content, only dedups,
    ranks, and merges across candidates already gated by that trust
    boundary. `review` is `causal.agents.dag_critique.build_pre_check`'s
    result for this candidate (identification/clinical/confidence verdicts
    computed the same way the rest of the pipeline already computes them).
    """

    index: int
    variables: tuple[dict[str, Any], ...]
    relations: tuple[dict[str, Any], ...]
    unresolved: tuple[dict[str, Any], ...]
    review: DAGReviewResult
    note: str

    def summary_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "variable_count": len(self.variables),
            "relation_count": len(self.relations),
            "identified_by_backdoor_adjustment": self.review.identification.identified_by_backdoor_adjustment,
            "approved_for_dag_gate": self.review.clinical.approved_for_dag_gate,
            "note": self.note,
        }


@dataclass(frozen=True)
class MergeConflict:
    """A genuine structural disagreement between surviving candidates that a
    deterministic union cannot resolve on its own.

    Recorded for human review, never silently resolved by picking one side
    or guessing which edge to drop (see SS9). `kind` is one of
    `role_conflict` (the same label drafted with disagreeing roles),
    `direction_conflict` (the same pair of labels drafted with disagreeing
    edge directions), or `cycle_conflict`/`structural_conflict` (the union
    of otherwise-compatible edges violates `causal.roadmap.CausalDAG`'s
    acyclicity or role-based structural rules -- only visible once the
    union is actually built into a graph).
    """

    kind: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "description": self.description}


@dataclass(frozen=True)
class PairwiseDistance:
    """SHD and (both-direction) SID between two ranked, pre-dedup tournament
    candidates -- computed for every pair, not only the ones dedup happens
    to compare, so a reviewer can see the full population's spread rather
    than only the dedup pass/fail outcomes.

    Indexed by each candidate's own `TournamentCandidate.index` (drafting
    order), not position in `ranked`, so labels stay meaningful regardless
    of rank-sort or dedup order. SHD (Acid & de Campos 2003 / Tsamardinos et
    al. 2006) is symmetric, so one value suffices; SID (Peters & Buhlmann
    2015) is not (Definition 4; see `causal.agents.dag_metrics`'s module
    docstring), so both directions are reported. The paper's own Example 1
    (reproduced in `tests/test_dag_metrics.py`) is the reason both metrics
    are worth surfacing together: two candidates can share the same SHD
    (same number of disagreeing edges) yet have wildly different SID
    (whether those disagreements would change a computed causal effect).
    """

    candidate_a: int
    candidate_b: int
    shd: int
    sid_a_given_b: int
    sid_b_given_a: int
    #: Size of the union of both candidates' node labels -- the raw counts
    #: above are only meaningful relative to how many labels were even in
    #: play. Kept alongside them (not just used internally) so a normalized
    #: agreement score (`causal.agents.dag_metrics.normalized_agreement`) can
    #: be recomputed from this stored record alone, e.g. for display.
    variable_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "candidate_a": self.candidate_a,
            "candidate_b": self.candidate_b,
            "shd": self.shd,
            "sid_a_given_b": self.sid_a_given_b,
            "sid_b_given_a": self.sid_b_given_a,
            "variable_count": self.variable_count,
        }


@dataclass(frozen=True)
class MergeFidelity:
    """SID from one surviving candidate's own DAG to the final merged DAG.

    Peters & Buhlmann (2015) Proposition 7: `SID(G, H) = 0 <=> G <= H` (G a
    subgraph of H). Since `merge_candidates` builds the merged DAG as an
    edge union of the survivors (SS9), each survivor's own edges are
    expected to be a literal subset of the merged DAG's edges -- SID should
    be exactly 0 -- *unless* a role/direction/structural `MergeConflict`
    excluded some of that survivor's own variables or relations from the
    merge. A non-zero value here is therefore a live, literature-grounded
    check on the merge's own soundness, not a re-assertion of Proposition 7:
    it quantifies how much of that survivor's causal-inference conclusions
    the merge did *not* preserve, for the human already reviewing the
    conflict list to weigh.
    """

    candidate_index: int
    sid_to_merged: int
    #: Size of the union of the survivor's and the merged DAG's node labels.
    #: See `PairwiseDistance.variable_count`.
    variable_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "candidate_index": self.candidate_index,
            "sid_to_merged": self.sid_to_merged,
            "variable_count": self.variable_count,
        }


def compute_pairwise_distances(ranked: tuple[TournamentCandidate, ...]) -> tuple[PairwiseDistance, ...]:
    """SHD and both-direction SID for every pair among `ranked` (see
    `PairwiseDistance`)."""
    distances = []
    for a, b in combinations(ranked, 2):
        dag_a, dag_b = a.review.identification.dag, b.review.identification.dag
        variable_count = len({node.label for node in dag_a.nodes} | {node.label for node in dag_b.nodes})
        distances.append(PairwiseDistance(
            candidate_a=a.index,
            candidate_b=b.index,
            shd=structural_hamming_distance(dag_a, dag_b),
            sid_a_given_b=structural_intervention_distance(dag_a, dag_b),
            sid_b_given_a=structural_intervention_distance(dag_b, dag_a),
            variable_count=variable_count,
        ))
    return tuple(distances)


def compute_merge_fidelity(
    survivors: tuple[TournamentCandidate, ...], merged_dag: CausalDAG | None
) -> tuple[MergeFidelity, ...]:
    """SID from each survivor's own DAG to `merged_dag` (see `MergeFidelity`).

    `merged_dag` is `None` when `build_pre_check` could not build a
    `CausalDAG` from the merged variables/relations at all (e.g. an empty
    merge) -- nothing to compare against, so this returns no entries rather
    than raising, matching this module's "never blocks on a single bad
    candidate" convention.
    """
    if merged_dag is None:
        return ()
    merged_labels = {node.label for node in merged_dag.nodes}
    return tuple(
        MergeFidelity(
            candidate_index=survivor.index,
            sid_to_merged=structural_intervention_distance(survivor.review.identification.dag, merged_dag),
            variable_count=len({node.label for node in survivor.review.identification.dag.nodes} | merged_labels),
        )
        for survivor in survivors
    )


@dataclass(frozen=True)
class TournamentResult:
    survivors: tuple[TournamentCandidate, ...]
    dropped_duplicate_count: int
    merged_variables: tuple[dict[str, Any], ...]
    merged_relations: tuple[dict[str, Any], ...]
    merged_unresolved: tuple[dict[str, Any], ...]
    conflicts: tuple[MergeConflict, ...]
    pairwise_distances: tuple[PairwiseDistance, ...]
    merge_fidelity: tuple[MergeFidelity, ...]


def rank_key(candidate: TournamentCandidate) -> tuple[bool, bool]:
    """Lexicographic ranking (SS9): identification, then clinical
    completeness -- no weighted score, no magic numbers. Candidates tied on
    both keep their original (drafting) order (`rank_candidates` is a
    stable sort).
    """
    return (
        candidate.review.identification.identified_by_backdoor_adjustment,
        candidate.review.clinical.approved_for_dag_gate,
    )


def rank_candidates(candidates: tuple[TournamentCandidate, ...]) -> tuple[TournamentCandidate, ...]:
    """Sort candidates best-first by `rank_key`. Stable: candidates tied on
    every key keep their original (drafting) order.
    """
    return tuple(sorted(candidates, key=rank_key, reverse=True))


def dedup_candidates(
    ranked: tuple[TournamentCandidate, ...], duplicate_threshold: int = DEFAULT_DUPLICATE_THRESHOLD
) -> tuple[tuple[TournamentCandidate, ...], int]:
    """Drop near-duplicate candidates, keeping the higher-ranked survivor of
    each duplicate cluster.

    `ranked` must already be sorted best-first (see `rank_candidates`): a
    candidate is dropped as soon as its SHD to any already-kept (i.e.
    higher- or equal-ranked) survivor is at or below `duplicate_threshold`,
    so the best-ranked member of every duplicate cluster is always the one
    kept. See `DEFAULT_DUPLICATE_THRESHOLD` for why the default is 0.
    """
    survivors: list[TournamentCandidate] = []
    dropped = 0
    for candidate in ranked:
        is_duplicate = any(
            structural_hamming_distance(
                candidate.review.identification.dag, kept.review.identification.dag
            )
            <= duplicate_threshold
            for kept in survivors
        )
        if is_duplicate:
            dropped += 1
            continue
        survivors.append(candidate)
    return tuple(survivors), dropped


def _merge_variables(
    survivors: tuple[TournamentCandidate, ...],
) -> tuple[list[dict[str, Any]], list[MergeConflict], set[str]]:
    """Union surviving candidates' `candidate_dag` entries by label.

    `survivors` must already be rank-sorted best-first: when the same label
    appears in multiple candidates with the *same* role, the first
    (best-ranked) occurrence is kept as the merged entry -- deterministic,
    not a field-by-field re-merge of the individual entries. A label
    asserted with a *disagreeing* role across candidates is a genuine
    structural conflict (SS9): excluded from the merge and recorded, never
    resolved by guessing which candidate is right.
    """
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in survivors:
        for variable in candidate.variables:
            by_label[variable["label"]].append(variable)

    merged: list[dict[str, Any]] = []
    conflicts: list[MergeConflict] = []
    excluded: set[str] = set()
    for label, occurrences in by_label.items():
        roles = {entry["role"] for entry in occurrences}
        if len(roles) > 1:
            conflicts.append(MergeConflict(
                "role_conflict",
                f"variable '{label}' was assigned disagreeing roles across tournament "
                f"candidates ({', '.join(sorted(roles))}); excluded from the merged DAG "
                "for human review",
            ))
            excluded.add(label)
            continue
        merged.append(occurrences[0])
    return merged, conflicts, excluded


def _merge_relations(
    survivors: tuple[TournamentCandidate, ...],
    merged_variables: list[dict[str, Any]],
    excluded_labels: set[str],
) -> tuple[list[dict[str, Any]], list[MergeConflict]]:
    """Union surviving candidates' `covariate_relations` by endpoint pair.

    A relation touching a label `_merge_variables` excluded (a role
    conflict), or a label whose merged role is mediator/collider/selection
    (forbidden as a `covariate_relations` endpoint by
    `causal.agents.dag_identification.build_causal_dag`), is dropped as a
    direct consequence rather than a separate conflict -- the endpoint
    itself is already flagged, or the rule is a pre-existing structural one
    this merge does not relitigate. A pair asserted with *disagreeing
    directions* across candidates is its own conflict type: recorded and
    excluded rather than guessing which direction is right.
    """
    forbidden_endpoints = excluded_labels | {
        variable["label"] for variable in merged_variables
        if variable["role"] in {"mediator", "collider", "selection"}
    }
    by_pair: dict[frozenset[str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in survivors:
        for relation in candidate.relations:
            source, target = relation["source"], relation["target"]
            if source in forbidden_endpoints or target in forbidden_endpoints:
                continue
            by_pair[frozenset((source, target))].append(relation)

    merged: list[dict[str, Any]] = []
    conflicts: list[MergeConflict] = []
    for pair, occurrences in by_pair.items():
        directions = {(entry["source"], entry["target"]) for entry in occurrences}
        if len(directions) > 1:
            source_label, target_label = sorted(pair)
            conflicts.append(MergeConflict(
                "direction_conflict",
                f"a relation between '{source_label}' and '{target_label}' was drafted "
                "with disagreeing directions across tournament candidates; excluded from "
                "the merged DAG for human review",
            ))
            continue
        merged.append(occurrences[0])
    return merged, conflicts


def _resolve_structural_conflicts(
    protocol: dict[str, Any], variables: list[dict[str, Any]], relations: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[MergeConflict]]:
    """Drop the minimum trailing relations needed to make the merged DAG
    buildable, in deterministic (insertion) order.

    Role- and direction-conflict resolution (`_merge_variables`/
    `_merge_relations`) only rule out disagreement between candidates about
    a single label or a single edge; a longer cycle spanning three or more
    independently-drafted, individually-non-conflicting relations (e.g.
    A->B from one candidate, B->C from another, C->A from a third) is only
    visible once the union is actually built into a graph
    (`causal.roadmap.CausalDAG`'s acyclicity check, reached via
    `causal.agents.dag_identification.build_causal_dag`). Per SS9, such a
    conflict is never silently broken by guessing which edge to drop --
    only that a deterministic, order-based choice is made, recorded as its
    own conflict for human review.
    """
    remaining = list(relations)
    conflicts: list[MergeConflict] = []
    while True:
        draft = {**protocol, "candidate_dag": variables, "covariate_relations": remaining}
        try:
            build_causal_dag(HypothesisAgent().run(draft))
            return remaining, conflicts
        except ValueError as exc:
            if not remaining:
                # Nothing left to drop; the failure does not originate from
                # a mergeable relation -- surface the merged DAG as-is and
                # let the caller's own downstream checks report it.
                return remaining, conflicts
            dropped = remaining.pop()
            kind = "cycle_conflict" if "acyclic" in str(exc).lower() else "structural_conflict"
            conflicts.append(MergeConflict(
                kind,
                f"relation '{dropped['source']}' -> '{dropped['target']}' was dropped from the "
                f"merged DAG because the union of surviving candidates' relations was "
                f"structurally invalid ({exc}); recorded for human review, not a clinical "
                "judgment this merge makes",
            ))


def merge_candidates(
    protocol: dict[str, Any], survivors: tuple[TournamentCandidate, ...]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[MergeConflict]]:
    """Deterministic edge-union merge of the surviving candidates (SS9).

    Justified by Peters, J. & Buhlmann, P. (2015). Structural Intervention
    Distance (SID) for Evaluating Causal Graphs. Neural Computation, 27(3),
    771-799 (arXiv:1306.1043), specifically Proposition 7:
    `SID(G, H) = 0 <=> G <= H` (G a subgraph of H) -- a superset graph never
    predicts a *wrong* intervention distribution relative to any DAG it is
    a superset of, for data actually Markov with respect to the weaker
    graph. Unioning the survivors' edges is therefore sound *provided the
    union stays consistent*, which is exactly what the conflict handling
    below exists to guarantee.

    Returns `(variables, relations, unresolved, conflicts)`.
    """
    merged_variables, role_conflicts, excluded_labels = _merge_variables(survivors)
    merged_relations, direction_conflicts = _merge_relations(survivors, merged_variables, excluded_labels)
    merged_relations, structural_conflicts = _resolve_structural_conflicts(
        protocol, merged_variables, merged_relations
    )

    merged_unresolved: dict[str, dict[str, Any]] = {}
    for candidate in survivors:
        for entry in candidate.unresolved:
            merged_unresolved.setdefault(entry["label"], entry)

    return (
        merged_variables,
        merged_relations,
        list(merged_unresolved.values()),
        role_conflicts + direction_conflicts + structural_conflicts,
    )


def run_tournament(
    protocol: dict[str, Any],
    candidates_raw: tuple[tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str], ...],
) -> TournamentResult:
    """Dedup, rank, and merge a population of independently-drafted DAG
    candidates (see module docstring and SS9).

    `candidates_raw` is `((variables, relations, unresolved, note), ...)` --
    the same shape `UserQuestionHypothesisService`'s per-candidate
    draft -> critique -> revision loop already produces, one tuple per
    independent draft. This function owns none of the LLM calls: the
    caller runs each candidate's own loop and passes the finished results
    here only once every candidate is final. `protocol` is the shared base
    protocol (target trial, study id, ...) all candidates were drafted
    against -- used only to build a `HypothesisArtifact` per candidate for
    ranking/dedup, exactly as `causal.agents.dag_critique.build_pre_check`
    already does for the single-draft pipeline.
    """
    candidates: list[TournamentCandidate] = []
    for index, (variables, relations, unresolved, note) in enumerate(candidates_raw):
        review = build_pre_check(protocol, variables, relations)
        if review is None:
            # Should not happen in practice: a structurally valid candidate
            # already passed this same check inside its own critique loop.
            # Kept defensive, matching this pipeline's "never blocks on a
            # single bad candidate" convention, rather than asserted.
            continue
        candidates.append(TournamentCandidate(
            index=index,
            variables=tuple(variables),
            relations=tuple(relations),
            unresolved=tuple(unresolved),
            review=review,
            note=note,
        ))

    ranked = rank_candidates(tuple(candidates))
    pairwise_distances = compute_pairwise_distances(ranked)
    survivors, dropped = dedup_candidates(ranked)
    merged_variables, merged_relations, merged_unresolved, conflicts = merge_candidates(protocol, survivors)

    merged_review = build_pre_check(protocol, merged_variables, merged_relations)
    merge_fidelity = compute_merge_fidelity(
        survivors, merged_review.identification.dag if merged_review is not None else None
    )

    return TournamentResult(
        survivors=survivors,
        dropped_duplicate_count=dropped,
        merged_variables=tuple(merged_variables),
        merged_relations=tuple(merged_relations),
        merged_unresolved=tuple(merged_unresolved),
        conflicts=tuple(conflicts),
        pairwise_distances=pairwise_distances,
        merge_fidelity=merge_fidelity,
    )
