"""Literature-grounded distance metrics between two candidate DAGs.

Both metrics compare graphs at the node-*label* level (not `node_id`), since
independently-drafted candidates never share a `node_id` namespace -- only
their free-text labels can possibly line up. Both therefore inherit the same
known limitation: a label present in only one graph is treated as an
isolated node (no edges) in that graph, so two candidates naming a
clinically-identical concept with different free-text labels (e.g. "age" vs.
"patient age") count as genuinely unrelated variables. See
docs/dag_generation_audit.md SS9 for the full rationale and
`causal.knowledge_graph.optimuskg_client`'s substring matching for the same
class of limitation elsewhere in this pipeline.

`align_labels` / `compare_generated_to_reference` (bottom of this module) are
the deterministic, LLM-free mitigation for that limitation, for the specific
case of scoring a pipeline-generated candidate DAG against a hand-built
`studies/*/reference_dag.json`: they reconcile the two label sets first and
report every label left unmatched, so the score is computed over a
correspondence a reviewer can check rather than over accidental string
equality.

- `structural_hamming_distance` (SHD): Acid, S. & de Campos, L.M. (2003).
  Searching for Bayesian network structures in the space of restricted
  acyclic partially directed graphs. Journal of Artificial Intelligence
  Research, 18, 445-490; Tsamardinos, I., Brown, L.E., & Aliferis, C.F.
  (2006). The max-min hill-climbing Bayesian network structure learning
  algorithm. Machine Learning, 65, 31-78 (both cited from, and confirmed
  via, Peters & Buhlmann 2015 SS1.1 below). Counts edge-*type* disagreements
  -- a purely structural distance, blind to whether a disagreement would
  ever change a computed causal effect.
- `structural_intervention_distance` (SID): Peters, J. & Buhlmann, P.
  (2015). Structural Intervention Distance (SID) for Evaluating Causal
  Graphs. Neural Computation, 27(3), 771-799 (arXiv:1306.1043). Counts,
  over every ordered pair of labels (i, j), whether using `b`'s parent set
  of i as the adjustment set for i's effect on j would recover the correct
  interventional distribution *in `a`* (Definition 3/4). Unlike SHD, SID is
  not symmetric and does not directly bound or get bounded by SHD (the
  paper's own Proposition 8 and Example 1 -- reproduced almost exactly in
  `tests/test_dag_metrics.py` -- show two graphs can share the same SHD to a
  reference graph yet have wildly different SID, precisely because SHD
  cannot see which edges matter for causal inference and which do not).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations

from causal.roadmap import CausalDAG, CausalEdge, CausalNode


def _edge_types(dag: CausalDAG) -> dict[frozenset[str], str]:
    label_of = {node.node_id: node.label for node in dag.nodes}
    types: dict[frozenset[str], str] = {}
    for edge in dag.edges:
        source_label, target_label = label_of[edge.source], label_of[edge.target]
        types[frozenset((source_label, target_label))] = f"{source_label}->{target_label}"
    return types


def max_structural_hamming_distance(variable_count: int) -> int:
    """Theoretical worst-case `structural_hamming_distance` for two graphs
    sharing `variable_count` node labels: every unordered pair of labels
    disagreeing on edge type. Used to normalize a raw SHD count into a
    0.0-1.0 agreement score (see `normalized_agreement`) for display -- the
    normalization itself, not a judgment about what counts as "close"."""
    return variable_count * (variable_count - 1) // 2


def max_structural_intervention_distance(variable_count: int) -> int:
    """Theoretical worst-case `structural_intervention_distance` for two
    graphs sharing `variable_count` node labels: every ordered pair of
    distinct labels falsely estimated. See `max_structural_hamming_distance`."""
    return variable_count * (variable_count - 1)


def normalized_agreement(distance: int, max_distance: int) -> float:
    """`distance` (an SHD or SID count) rescaled to a 0.0-1.0 agreement
    score against its theoretical worst case `max_distance`
    (`max_structural_hamming_distance`/`max_structural_intervention_distance`):
    1.0 means identical, 0.0 means maximally disagreeing. `max_distance <= 0`
    (0 or 1 shared label -- nothing for the two graphs to disagree about) is
    defined as full agreement.
    """
    if max_distance <= 0:
        return 1.0
    return max(0.0, 1.0 - (distance / max_distance))


def structural_hamming_distance(a: CausalDAG, b: CausalDAG) -> int:
    """Structural Hamming Distance between two candidate DAGs, compared at
    the node-*label* level (see module docstring).

    Counts, over the union of both graphs' node labels, every unordered
    pair of labels whose edge type (no edge / `a->b` / `b->a`) differs
    between the two graphs.
    """
    labels = sorted({node.label for node in a.nodes} | {node.label for node in b.nodes})
    edges_a, edges_b = _edge_types(a), _edge_types(b)
    distance = 0
    for source, target in combinations(labels, 2):
        pair = frozenset((source, target))
        if edges_a.get(pair, "none") != edges_b.get(pair, "none"):
            distance += 1
    return distance


def _children_by_label(dag: CausalDAG) -> dict[str, set[str]]:
    label_of = {node.node_id: node.label for node in dag.nodes}
    children: dict[str, set[str]] = defaultdict(set)
    for edge in dag.edges:
        children[label_of[edge.source]].add(label_of[edge.target])
    return children


def _parents_by_label(children: Mapping[str, set[str]]) -> dict[str, set[str]]:
    parents: dict[str, set[str]] = defaultdict(set)
    for source, targets in children.items():
        for target in targets:
            parents[target].add(source)
    return parents


def _reachable(adjacency: Mapping[str, set[str]], start: str) -> set[str]:
    """Nodes reachable from `start` by >=1 hop, following `adjacency` (either
    a children map, giving descendants, or a parents map, giving
    ancestors)."""
    found: set[str] = set()
    pending = list(adjacency.get(start, ()))
    while pending:
        current = pending.pop()
        if current not in found:
            found.add(current)
            pending.extend(adjacency.get(current, ()))
    return found


def _forbidden_set(children: Mapping[str, set[str]], parents: Mapping[str, set[str]], x: str, y: str) -> set[str]:
    """`forb(x, y, G)` (Lemma 5's condition (*), first clause): every label
    lying on some directed path from `x` to `y` (mediators and `y` itself,
    excluding `x`), plus every descendant of those labels.

    A label lies on some directed path from `x` to `y` iff it is reachable
    from `x` *and* can itself reach `y` (`y` always qualifies, trivially, by
    the second half of that conjunction).
    """
    descendants_of_x = _reachable(children, x)
    if y not in descendants_of_x:
        return set()
    ancestors_of_y = _reachable(parents, y) | {y}
    on_path = descendants_of_x & ancestors_of_y
    forbidden = set(on_path)
    for label in on_path:
        forbidden |= _reachable(children, label)
    return forbidden


def _all_simple_paths(neighbors: Mapping[str, set[str]], start: str, end: str) -> list[list[str]]:
    """Every simple (no repeated node) path between `start` and `end` in the
    undirected skeleton `neighbors`. Graphs handled by this module are small,
    reviewable clinical DAGs (see docs/dag_generation_audit.md SS9's cost
    model and `causal.roadmap.CausalDAG.minimum_adjustment_sets`'s own
    bounded-search precedent) -- exhaustive enumeration is deliberately not
    optimized for dense, high-node-count graphs.
    """
    paths: list[list[str]] = []

    def _walk(current: str, visited: set[str], path: list[str]) -> None:
        if current == end:
            paths.append(list(path))
            return
        for neighbor in neighbors.get(current, ()):
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                _walk(neighbor, visited, path)
                path.pop()
                visited.discard(neighbor)

    _walk(start, {start}, [start])
    return paths


def _is_directed_path(children: Mapping[str, set[str]], path: list[str]) -> bool:
    return all(path[index + 1] in children.get(path[index], ()) for index in range(len(path) - 1))


def _path_is_blocked(children: Mapping[str, set[str]], path: list[str], conditioned: set[str]) -> bool:
    """Standard d-separation path-blocking rule: a non-collider blocks the
    path iff it is conditioned on; a collider blocks the path unless it (or
    one of its descendants) is conditioned on.
    """
    for index in range(1, len(path) - 1):
        previous_node, node, next_node = path[index - 1], path[index], path[index + 1]
        into_node_from_previous = node in children.get(previous_node, ())
        into_node_from_next = node in children.get(next_node, ())
        is_collider = into_node_from_previous and into_node_from_next
        if is_collider:
            opened = node in conditioned or bool(_reachable(children, node) & conditioned)
            if not opened:
                return True
        elif node in conditioned:
            return True
    return False


def _blocks_all_non_directed_paths(
    children: Mapping[str, set[str]], x: str, y: str, conditioned: set[str]
) -> bool:
    """Lemma 5's condition (*), second clause: `conditioned` blocks every
    path between `x` and `y` in `G` other than a directed path from `x` to
    `y` (directed `x`-to-`y` paths are exempt -- they are the genuine causal
    effect being queried, not a confound to block).
    """
    neighbors: dict[str, set[str]] = defaultdict(set)
    for source, targets in children.items():
        for target in targets:
            neighbors[source].add(target)
            neighbors[target].add(source)
    for path in _all_simple_paths(neighbors, x, y):
        if _is_directed_path(children, path):
            continue
        if not _path_is_blocked(children, path, conditioned):
            return False
    return True


def _is_valid_adjustment_set(
    children: Mapping[str, set[str]], parents: Mapping[str, set[str]], x: str, y: str, adjustment: set[str]
) -> bool:
    """Lemma 5 (Characterization of valid Adjustment Sets), applied to `G`:
    whether `adjustment` is a valid adjustment set for `y | do(x)` in `G`.
    """
    if not adjustment.isdisjoint(_forbidden_set(children, parents, x, y)):
        return False
    return _blocks_all_non_directed_paths(children, x, y, adjustment)


def structural_intervention_distance(a: CausalDAG, b: CausalDAG) -> int:
    """Structural Intervention Distance from `a` (treated as the true DAG)
    to `b` (treated as the estimate), compared at the node-*label* level
    (see module docstring).

    Implements Definition 4 via Proposition 6's graphical, distribution-free
    equivalent (Peters & Buhlmann 2015): for every ordered pair of labels
    `(i, j)`, `i != j`, the intervention distribution from `i` to `j` is
    counted as falsely estimated by `b` (relative to `a`) if

    - `j` is one of `i`'s parents in `b`: exactly when `j` actually is a
      descendant of `i` in `a` (adjusting for the very variable whose
      effect is being queried, when a real effect exists, is always wrong);
      otherwise
    - `pa_b(i)` is not a valid adjustment set for `j | do(i)` in `a`
      (Lemma 5).

    Not symmetric (`structural_intervention_distance(a, b) !=
    structural_intervention_distance(b, a)` in general) and not bounded by,
    or a bound on, `structural_hamming_distance` (Proposition 8(2)): unlike
    SHD, an edge disagreement only counts against SID if it would actually
    change a computed intervention distribution. `tests/test_dag_metrics.py`
    reproduces the paper's own Example 1 (Figure 2) as a numeric check.

    Known limitation shared with `structural_hamming_distance`: a directed
    edge asserted by `b` between two labels that are simply unrelated in
    `a` (neither a descendant of the other) is not penalized by this metric
    unless it also breaks a valid-adjustment-set check elsewhere -- this is
    a property of SID itself (it only counts errors that change an
    intervention distribution), not an artifact of comparing candidates
    with different label sets.
    """
    labels = {node.label for node in a.nodes} | {node.label for node in b.nodes}
    children_a = _children_by_label(a)
    parents_a = _parents_by_label(children_a)
    parents_b = _parents_by_label(_children_by_label(b))

    distance = 0
    for i in labels:
        descendants_of_i_in_a = _reachable(children_a, i)
        parents_of_i_in_b = parents_b.get(i, set())
        for j in labels:
            if i == j:
                continue
            if j in parents_of_i_in_b:
                falsely_estimated = j in descendants_of_i_in_a
            else:
                falsely_estimated = not _is_valid_adjustment_set(
                    children_a, parents_a, i, j, parents_of_i_in_b
                )
            if falsely_estimated:
                distance += 1
    return distance


# ---------------------------------------------------------------------------
# Label alignment
# ---------------------------------------------------------------------------
# Both metrics above compare at the node-*label* level, so a pipeline-drafted
# candidate DAG and an independently authored reference DAG that name the same
# clinical concept differently ("age" vs "Age at ICU admission") are scored as
# if they were unrelated variables -- inflating every distance (see the module
# docstring and docs/dag_generation_audit.md SS9). The functions below are the
# deterministic, LLM-free mitigation: `align_labels` proposes a one-to-one
# correspondence between the two label sets and, crucially, *reports every
# unmatched label on each side* so a reviewer can audit the correspondence
# before trusting a score computed over it. This is the same class of
# heuristic -- and carries the same "a human must check it" caveat -- as
# `causal.knowledge_graph.optimuskg_client`'s case-insensitive substring
# matching (docs/dag_generation_audit.md SS6).

#: Pure function words dropped before token-overlap scoring. Clinically loaded
#: words that merely look like stopwords ("baseline", "prior", "chronic",
#: "history", "acute") are deliberately kept -- dropping them would merge
#: genuinely different variables.
_LABEL_STOPWORDS = frozenset(
    {"a", "an", "and", "at", "by", "for", "in", "of", "on", "or", "the", "to", "vs", "versus", "with"}
)

#: Default Sorensen-Dice token-overlap a `(reference, candidate)` label pair
#: must clear to be matched by `align_labels`'s third pass. 0.5 accepts "age"
#: vs "Age at ICU admission" (one shared token out of {age} and {age, icu,
#: admission}: 2*1/(1+3)) while rejecting pairs that share only a generic
#: qualifier. Raise it for a stricter alignment; lower it only with manual
#: review of the extra matches it lets through.
DEFAULT_MIN_LABEL_SIMILARITY = 0.5


def _normalize_label(label: str) -> str:
    """Lowercase, replace every run of non-alphanumeric characters with a
    single space, and strip -- so "Age at ICU admission." and
    "age  at ICU  admission" normalize to the same string."""
    return re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()


def _label_tokens(label: str) -> frozenset[str]:
    """Content tokens of a label: `_normalize_label` then whitespace-split,
    with `_LABEL_STOPWORDS` removed."""
    return frozenset(_normalize_label(label).split()) - _LABEL_STOPWORDS


def _dice_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    """Sorensen-Dice set-overlap coefficient, ``2|A n B| / (|A| + |B|)`` (Dice,
    L.R. 1945. Measures of the amount of ecologic association between species.
    Ecology 26(3):297-302) -- a standard symmetric token-set similarity, the
    same family used for record linkage and ontology alignment. 1.0 when both
    token sets are empty (two labels that were entirely stopwords), 0.0 when
    exactly one is empty."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return 2 * len(left & right) / (len(left) + len(right))


@dataclass(frozen=True)
class LabelMatch:
    """One entry of a `LabelAlignment`: a reference label, the candidate label
    it was matched to, how (`manual` / `exact` / `token_overlap`), and the
    `_dice_similarity` that produced it (1.0 for `manual` and `exact`)."""

    reference_label: str
    candidate_label: str
    method: str
    similarity: float

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_label": self.reference_label,
            "candidate_label": self.candidate_label,
            "method": self.method,
            "similarity": round(self.similarity, 4),
        }


@dataclass(frozen=True)
class LabelAlignment:
    """The full, auditable output of `align_labels`: the matched pairs plus
    every label left unmatched on each side (never silently dropped)."""

    matches: tuple[LabelMatch, ...]
    unmatched_reference: tuple[str, ...]
    unmatched_candidate: tuple[str, ...]

    @property
    def candidate_to_reference(self) -> dict[str, str]:
        """`{candidate_label: reference_label}` remap for `relabel_dag`, moving
        a candidate DAG into the reference DAG's label namespace."""
        return {match.candidate_label: match.reference_label for match in self.matches}

    def to_dict(self) -> dict[str, object]:
        return {
            "matched_count": len(self.matches),
            "matches": [match.to_dict() for match in self.matches],
            "unmatched_reference": list(self.unmatched_reference),
            "unmatched_candidate": list(self.unmatched_candidate),
        }


def align_labels(
    reference_labels: Iterable[str],
    candidate_labels: Iterable[str],
    *,
    manual_aliases: Mapping[str, str] | None = None,
    min_similarity: float = DEFAULT_MIN_LABEL_SIMILARITY,
) -> LabelAlignment:
    """Propose a one-to-one correspondence between two independently authored
    sets of free-text DAG node labels (a reference set and a candidate set).

    Deterministic and LLM-free. Matching runs in three passes, each removing
    the labels it consumes from both pools:

    1. ``manual_aliases`` -- an explicit ``{candidate_label: reference_label}``
       map from a reviewer. Both labels must still be unmatched and present
       (a stale alias raises rather than being silently ignored).
    2. exact match after `_normalize_label` (case / punctuation / whitespace).
    3. greedy token overlap: the still-unmatched ``(reference, candidate)``
       pair with the highest ``_dice_similarity >= min_similarity`` is
       matched, then the next highest, and so on. Ties break lexicographically
       so the result does not depend on input order.

    Never maps one label to two: `structural_hamming_distance`,
    `structural_intervention_distance` and `relabel_dag` all assume an
    injective correspondence. Whatever is left over is returned in
    ``unmatched_reference`` / ``unmatched_candidate`` -- the metrics then treat
    those labels exactly as they did before any alignment step existed, as
    variables present in only one graph.

    Matching stays one-to-one. When a reference DAG folds a covariate family
    into one node while the candidate lists each member separately (e.g.
    ``vanco_piptazo_vs_cefepime_aki``'s single "Nephrotoxin co-exposure
    cluster" node vs ~7 separate drug-exposure covariates), the members land
    in ``unmatched_candidate`` -- many-vs-one is not a 1:1 correspondence and
    is not guessed. Collapse them first with `merge_dag_nodes` (or
    `compare_generated_to_reference`'s ``candidate_label_groups``), which
    makes that grouping an explicit, reviewed decision.
    """
    reference_pool = list(dict.fromkeys(str(label) for label in reference_labels))
    candidate_pool = list(dict.fromkeys(str(label) for label in candidate_labels))
    reference_remaining = set(reference_pool)
    candidate_remaining = set(candidate_pool)
    matches: list[LabelMatch] = []

    for candidate_label, reference_label in (manual_aliases or {}).items():
        if candidate_label not in candidate_remaining:
            raise ValueError(
                "manual_aliases key is not a candidate-DAG label still available for matching: "
                f"{candidate_label!r}"
            )
        if reference_label not in reference_remaining:
            raise ValueError(
                f"manual_aliases maps {candidate_label!r} to a reference-DAG label that is not "
                f"available for matching: {reference_label!r}"
            )
        matches.append(LabelMatch(reference_label, candidate_label, "manual", 1.0))
        reference_remaining.discard(reference_label)
        candidate_remaining.discard(candidate_label)

    by_normalized: dict[str, str] = {}
    for label in sorted(reference_remaining):
        by_normalized.setdefault(_normalize_label(label), label)
    for candidate_label in sorted(candidate_remaining):
        reference_label = by_normalized.get(_normalize_label(candidate_label))
        if reference_label is not None and reference_label in reference_remaining:
            matches.append(LabelMatch(reference_label, candidate_label, "exact", 1.0))
            reference_remaining.discard(reference_label)
            candidate_remaining.discard(candidate_label)

    reference_tokens = {label: _label_tokens(label) for label in reference_remaining}
    candidate_tokens = {label: _label_tokens(label) for label in candidate_remaining}
    scored: list[tuple[float, str, str]] = []
    for reference_label in reference_remaining:
        for candidate_label in candidate_remaining:
            similarity = _dice_similarity(
                reference_tokens[reference_label], candidate_tokens[candidate_label]
            )
            if similarity >= min_similarity:
                scored.append((similarity, reference_label, candidate_label))
    # Highest similarity first; lexicographic tie-break keeps this deterministic
    # regardless of set iteration order.
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    for similarity, reference_label, candidate_label in scored:
        if reference_label in reference_remaining and candidate_label in candidate_remaining:
            matches.append(LabelMatch(reference_label, candidate_label, "token_overlap", similarity))
            reference_remaining.discard(reference_label)
            candidate_remaining.discard(candidate_label)

    matches.sort(key=lambda match: (match.reference_label, match.candidate_label))
    return LabelAlignment(
        matches=tuple(matches),
        unmatched_reference=tuple(sorted(reference_remaining)),
        unmatched_candidate=tuple(sorted(candidate_remaining)),
    )


def relabel_dag(dag: CausalDAG, label_map: Mapping[str, str]) -> CausalDAG:
    """Return a copy of `dag` with each node label ``l`` replaced by
    ``label_map.get(l, l)``, keeping every ``node_id`` (and therefore every
    edge) unchanged. Used to move a candidate DAG into a reference DAG's label
    namespace before `structural_hamming_distance` /
    `structural_intervention_distance`.

    Raises if the remap would collapse two distinct nodes onto one label --
    which would make the metrics silently treat them as the same variable --
    so a `LabelAlignment` whose `candidate_to_reference` is injective (it
    always is) is safe to pass straight in.
    """
    relabeled = [
        CausalNode(
            node_id=node.node_id,
            label=label_map.get(node.label, node.label),
            role=node.role,
            timing=node.timing,
            observed=node.observed,
        )
        for node in dag.nodes
    ]
    new_labels = [node.label for node in relabeled]
    if len(set(new_labels)) != len(new_labels):
        raise ValueError("relabel_dag produced duplicate node labels -- the label map is not injective")
    return CausalDAG(
        nodes=tuple(relabeled),
        edges=dag.edges,
        exposure_id=dag.exposure_id,
        outcome_id=dag.outcome_id,
        comparator=dag.comparator,
    )


def merge_dag_nodes(dag: CausalDAG, groups: Mapping[str, Iterable[str]]) -> CausalDAG:
    """Collapse each named group of node labels into one node, labelled by the
    group key, carrying the union of the group's in/out edges.

    The use case is scoring a fine-grained candidate DAG against a reference
    DAG that represents a whole covariate family as a single node -- e.g.
    ``vanco_piptazo_vs_cefepime_aki``'s one "Nephrotoxin co-exposure cluster"
    node standing in for six or seven separate drug-exposure covariates. Left
    as-is, `align_labels` correctly leaves all of them unmatched (one concept
    vs many is not a 1:1 correspondence); merging first turns it into one.

    Each group member must be an existing node label, none may be the
    exposure or outcome, and a group's members must share a role and a timing
    -- a heterogeneous group is a modelling decision (do these really act the
    same way?), not a label-granularity one, and is rejected rather than
    silently resolved. The merged node is `observed` only if every member is.
    Intra-group edges collapse away; a parallel edge that several members had
    to the same third node is kept once (metrics are edge-existence based, so
    the retained edge's rationale/evidence is the earliest member's and does
    not affect any score).
    """
    label_to_node = {node.label: node for node in dag.nodes}
    if len(label_to_node) != len(dag.nodes):
        raise ValueError("merge_dag_nodes requires unique node labels")

    remap: dict[str, str] = {}
    merged_nodes: list[CausalNode] = []
    consumed: set[str] = set()
    for index, (group_label, members) in enumerate(groups.items()):
        member_labels = [str(member) for member in members]
        unknown = [label for label in member_labels if label not in label_to_node]
        if unknown:
            raise ValueError(f"merge group {group_label!r} names unknown node label(s): {unknown}")
        member_nodes = [label_to_node[label] for label in member_labels]
        member_ids = {node.node_id for node in member_nodes}
        if {dag.exposure_id, dag.outcome_id} & member_ids:
            raise ValueError(f"merge group {group_label!r} cannot include the exposure or outcome node")
        roles = {node.role for node in member_nodes}
        timings = {node.timing for node in member_nodes}
        if len(roles) != 1 or len(timings) != 1:
            raise ValueError(
                f"merge group {group_label!r} mixes roles/timings ({sorted(roles)} / "
                f"{sorted(timings)}) -- a heterogeneous group is a modelling decision, not a "
                "label-granularity one"
            )
        merged_id = f"merged__{index}"
        merged_nodes.append(
            CausalNode(
                node_id=merged_id,
                label=group_label,
                role=member_nodes[0].role,
                timing=member_nodes[0].timing,
                observed=all(node.observed for node in member_nodes),
            )
        )
        for member_id in member_ids:
            remap[member_id] = merged_id
        consumed |= member_ids

    new_nodes = [node for node in dag.nodes if node.node_id not in consumed] + merged_nodes
    seen: set[tuple[str, str]] = set()
    new_edges: list[CausalEdge] = []
    for edge in dag.edges:
        source = remap.get(edge.source, edge.source)
        target = remap.get(edge.target, edge.target)
        if source == target or (source, target) in seen:
            continue
        seen.add((source, target))
        new_edges.append(
            CausalEdge(
                source=source,
                target=target,
                rationale=edge.rationale,
                evidence_ids=edge.evidence_ids,
                status=edge.status,
                certainty=edge.certainty,
            )
        )
    return CausalDAG(
        nodes=tuple(new_nodes),
        edges=tuple(new_edges),
        exposure_id=dag.exposure_id,
        outcome_id=dag.outcome_id,
        comparator=dag.comparator,
    )


@dataclass(frozen=True)
class ReferenceDagComparison:
    """How close a pipeline-*generated* candidate DAG is to a hand-built
    reference DAG (`studies/*/reference_dag.json`) for the same question,
    scored after `align_labels` reconciles the two label sets.

    `sid_generated_vs_reference` is the headline evaluation number: treating
    the expert reference as the true graph, how many ordered intervention
    distributions does the generated DAG get wrong
    (`structural_intervention_distance`, Peters & Buhlmann 2015).
    `sid_reference_vs_generated` is the other, non-symmetric direction. `shd`
    is the coarser structural count. Both distances are also reported as
    0.0-1.0 agreement scores (`normalized_agreement`) over the union of the
    two aligned label sets.

    `alignment` is the audit trail -- always inspect `alignment.
    unmatched_reference` / `unmatched_candidate` before quoting a score: a
    generated DAG that simply phrased half its variables differently would
    otherwise look far worse than it is.
    """

    alignment: LabelAlignment
    shared_label_count: int
    union_label_count: int
    shd: int
    sid_generated_vs_reference: int
    sid_reference_vs_generated: int
    structural_agreement: float
    causal_agreement: float

    def to_dict(self) -> dict[str, object]:
        return {
            "alignment": self.alignment.to_dict(),
            "shared_label_count": self.shared_label_count,
            "union_label_count": self.union_label_count,
            "shd": self.shd,
            "sid_generated_vs_reference": self.sid_generated_vs_reference,
            "sid_reference_vs_generated": self.sid_reference_vs_generated,
            "structural_agreement": round(self.structural_agreement, 4),
            "causal_agreement": round(self.causal_agreement, 4),
        }


def compare_generated_to_reference(
    reference: CausalDAG,
    generated: CausalDAG,
    *,
    manual_aliases: Mapping[str, str] | None = None,
    candidate_label_groups: Mapping[str, Iterable[str]] | None = None,
    min_similarity: float = DEFAULT_MIN_LABEL_SIMILARITY,
) -> ReferenceDagComparison:
    """Align `generated`'s node labels onto `reference`'s (`align_labels`),
    then score the two graphs with `structural_hamming_distance` /
    `structural_intervention_distance` over the aligned labels.

    `reference` is treated as the true graph: `sid_generated_vs_reference` is
    ``structural_intervention_distance(reference, aligned_generated)``.
    Unmatched labels are not dropped -- they remain in their own graph as
    variables the other graph lacks, exactly as the metrics handled every
    label before this alignment step existed.

    `candidate_label_groups` (``{merged_label: [member_label, ...]}``) runs
    `merge_dag_nodes` on `generated` *before* aligning -- for a reference DAG
    that folds a covariate family into one node while the generated draft
    lists each member separately (see `merge_dag_nodes`). The merge happens
    on the generated side only; the reference graph is scored as authored.
    """
    if candidate_label_groups:
        generated = merge_dag_nodes(generated, candidate_label_groups)
    alignment = align_labels(
        (node.label for node in reference.nodes),
        (node.label for node in generated.nodes),
        manual_aliases=manual_aliases,
        min_similarity=min_similarity,
    )
    aligned_generated = relabel_dag(generated, alignment.candidate_to_reference)
    reference_labels = {node.label for node in reference.nodes}
    generated_labels = {node.label for node in aligned_generated.nodes}
    union_label_count = len(reference_labels | generated_labels)

    shd = structural_hamming_distance(reference, aligned_generated)
    sid_generated_vs_reference = structural_intervention_distance(reference, aligned_generated)
    sid_reference_vs_generated = structural_intervention_distance(aligned_generated, reference)
    return ReferenceDagComparison(
        alignment=alignment,
        shared_label_count=len(reference_labels & generated_labels),
        union_label_count=union_label_count,
        shd=shd,
        sid_generated_vs_reference=sid_generated_vs_reference,
        sid_reference_vs_generated=sid_reference_vs_generated,
        structural_agreement=normalized_agreement(
            shd, max_structural_hamming_distance(union_label_count)
        ),
        causal_agreement=normalized_agreement(
            sid_generated_vs_reference, max_structural_intervention_distance(union_label_count)
        ),
    )
