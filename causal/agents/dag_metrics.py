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

from collections import defaultdict
from itertools import combinations
from typing import Mapping

from causal.roadmap import CausalDAG


def _edge_types(dag: CausalDAG) -> dict[frozenset[str], str]:
    label_of = {node.node_id: node.label for node in dag.nodes}
    types: dict[frozenset[str], str] = {}
    for edge in dag.edges:
        source_label, target_label = label_of[edge.source], label_of[edge.target]
        types[frozenset((source_label, target_label))] = f"{source_label}->{target_label}"
    return types


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
