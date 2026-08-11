import unittest

from causal.agents.dag_critique import build_pre_check
from causal.agents.dag_metrics import (
    max_structural_hamming_distance,
    max_structural_intervention_distance,
    normalized_agreement,
    structural_hamming_distance,
    structural_intervention_distance,
)
from causal.roadmap import CausalDAG, CausalEdge, CausalNode


def _protocol() -> dict:
    return {
        "study_id": "test_study",
        "source_question": "Does treatment affect outcome?",
        "target_trial": {
            "population": "population", "treatment": "treatment",
            "comparator": "comparator", "time_zero": "index",
            "grace_period_hours": 24, "outcome": "outcome",
            "follow_up_days": 28, "primary_estimand": "ATE",
        },
        "concept_sets": {"treatment": {"status": "pending"}},
    }


def _variable(
    label: str, role: str = "confounder", temporal: str = "pre_treatment", pmid: str = "111",
    observed: bool = True,
) -> dict:
    return {
        "label": label,
        "role": role,
        "rationale": f"{label} affects both treatment choice and outcome.",
        "temporal_classification": temporal,
        "pmid": pmid,
        "observed": observed,
    }


def _dag_from_candidate(variables: list[dict], relations: list[dict] | None = None) -> CausalDAG:
    review = build_pre_check(_protocol(), variables, relations or [])
    assert review is not None
    return review.identification.dag


def _node(node_id: str, label: str, role: str = "other", timing: str = "baseline") -> CausalNode:
    return CausalNode(node_id=node_id, label=label, role=role, timing=timing, observed=True)


def _edge(source: str, target: str) -> CausalEdge:
    return CausalEdge(source=source, target=target, rationale=f"{source} affects {target}.")


class StructuralHammingDistanceTests(unittest.TestCase):
    def test_identical_graphs_have_zero_distance(self):
        a = _dag_from_candidate([_variable("age")])
        b = _dag_from_candidate([_variable("age")])
        self.assertEqual(structural_hamming_distance(a, b), 0)

    def test_disagreeing_role_changes_distance(self):
        a = _dag_from_candidate([_variable("age", role="confounder")])
        b = _dag_from_candidate([_variable("age", role="effect_modifier")])
        # confounder wires age->exposure and age->outcome; effect_modifier
        # wires only age->outcome -- exactly one pair (age, exposure)
        # differs (edge vs. none).
        self.assertEqual(structural_hamming_distance(a, b), 1)

    def test_extra_variable_in_one_graph_counts_its_pairs_as_mismatches(self):
        a = _dag_from_candidate([_variable("age")])
        b = _dag_from_candidate([_variable("age"), _variable("severity")])
        # "severity" contributes two edges (severity->exposure,
        # severity->outcome) absent from graph a, and the (age, severity)
        # pair is "none" in both -- so exactly 2 pairs differ.
        self.assertEqual(structural_hamming_distance(a, b), 2)


class StructuralInterventionDistanceTests(unittest.TestCase):
    """Peters & Buhlmann (2015), Neural Computation 27(3), 771-799
    (arXiv:1306.1043). `test_paper_example_1_*` reproduces Figure 2 / Example
    1 verbatim, including its two published numeric results (SID(G,H1)=0,
    SID(G,H2)=8) -- the strongest available check that this implementation
    matches the paper rather than an approximation of it.
    """

    def _example_1_true_graph(self) -> CausalDAG:
        # G: X1 -> X2, and both X1 and X2 point into each of Y1, Y2, Y3.
        nodes = (
            _node("x1", "X1", role="exposure", timing="treatment"),
            _node("x2", "X2", timing="treatment"),
            _node("y1", "Y1", role="outcome", timing="outcome"),
            _node("y2", "Y2", timing="outcome"),
            _node("y3", "Y3", timing="outcome"),
        )
        edges = (
            _edge("x1", "x2"),
            _edge("x1", "y1"), _edge("x1", "y2"), _edge("x1", "y3"),
            _edge("x2", "y1"), _edge("x2", "y2"), _edge("x2", "y3"),
        )
        return CausalDAG(nodes=nodes, edges=edges, exposure_id="x1", outcome_id="y1")

    def test_paper_example_1_h1_adds_a_mediator_edge_and_stays_sid_zero(self):
        g = self._example_1_true_graph()
        h1_nodes = g.nodes
        h1_edges = g.edges + (_edge("y1", "y2"),)
        h1 = CausalDAG(nodes=h1_nodes, edges=h1_edges, exposure_id="x1", outcome_id="y1")
        self.assertEqual(structural_intervention_distance(g, h1), 0)

    def test_paper_example_1_h2_reverses_x1_x2_and_sid_is_eight(self):
        g = self._example_1_true_graph()
        h2_nodes = g.nodes
        h2_edges = tuple(
            _edge("x2", "x1") if (edge.source, edge.target) == ("x1", "x2") else edge
            for edge in g.edges
        )
        h2 = CausalDAG(nodes=h2_nodes, edges=h2_edges, exposure_id="x1", outcome_id="y1")
        self.assertEqual(structural_intervention_distance(g, h2), 8)
        # Both "mistakes" (H1's extra edge, H2's reversed edge) have the same
        # SHD to G -- this is exactly the paper's point (SS1): SHD cannot
        # tell them apart, SID can.
        h1 = CausalDAG(
            nodes=g.nodes, edges=g.edges + (_edge("y1", "y2"),), exposure_id="x1", outcome_id="y1"
        )
        self.assertEqual(structural_hamming_distance(g, h1), structural_hamming_distance(g, h2))

    def test_identical_graphs_have_zero_sid(self):
        g = self._example_1_true_graph()
        self.assertEqual(structural_intervention_distance(g, g), 0)

    def test_proposition_7_superset_graph_has_zero_sid_from_the_subgraph(self):
        # SID(G, H) = 0 <=> G <= H (Proposition 7): a candidate whose edges
        # are a literal subset of a merged/superset DAG's edges must always
        # score SID=0 against that superset -- exactly the property
        # `causal.agents.dag_tournament.merge_candidates`'s edge-union merge
        # relies on.
        subset = _dag_from_candidate([_variable("age")])
        superset = _dag_from_candidate([_variable("age"), _variable("severity")])
        self.assertEqual(structural_intervention_distance(subset, superset), 0)

    def test_omitting_a_confounder_entirely_is_a_nonzero_sid(self):
        # H omits "severity" outright: the adjustment set H would use for
        # exposure's effect on outcome (empty) fails to block the backdoor
        # path exposure<-severity->outcome that exists in the true graph G.
        g = _dag_from_candidate([_variable("severity")])
        h = _dag_from_candidate([])
        self.assertGreater(structural_intervention_distance(g, h), 0)

    def test_sid_is_not_symmetric(self):
        # A mediator chain (X causes Y only through M) versus H asserting a
        # direct X->Y edge with M unrelated: which graph is "true" changes
        # which ordered pairs come out falsely estimated, so the two
        # directions need not (and here do not) agree.
        x = _node("x", "X", role="exposure", timing="treatment")
        m = _node("m", "M", role="mediator", timing="post_treatment")
        y = _node("y", "Y", role="outcome", timing="outcome")
        chain = CausalDAG(
            nodes=(x, m, y), edges=(_edge("x", "m"), _edge("m", "y")),
            exposure_id="x", outcome_id="y",
        )
        direct = CausalDAG(nodes=(x, m, y), edges=(_edge("x", "y"),), exposure_id="x", outcome_id="y")
        self.assertNotEqual(
            structural_intervention_distance(chain, direct),
            structural_intervention_distance(direct, chain),
        )


class NormalizedAgreementTests(unittest.TestCase):
    """`normalized_agreement` / `max_structural_*_distance`: rescale a raw
    SHD or SID count into a 0.0-1.0 score for display (see
    `causal.ui.streamlit_app`'s DAG-comparison section), without inventing a
    similarity judgment -- just the arithmetic normalization against each
    metric's own theoretical worst case for a given number of shared labels.
    """

    def test_max_shd_is_the_unordered_pair_count(self):
        self.assertEqual(max_structural_hamming_distance(4), 6)  # C(4, 2)

    def test_max_sid_is_the_ordered_pair_count(self):
        self.assertEqual(max_structural_intervention_distance(4), 12)  # 4 * 3

    def test_zero_distance_is_full_agreement(self):
        self.assertEqual(normalized_agreement(0, max_structural_hamming_distance(4)), 1.0)

    def test_worst_case_distance_is_zero_agreement(self):
        max_shd = max_structural_hamming_distance(4)
        self.assertEqual(normalized_agreement(max_shd, max_shd), 0.0)

    def test_halfway_distance_is_half_agreement(self):
        max_sid = max_structural_intervention_distance(4)
        self.assertEqual(normalized_agreement(max_sid // 2, max_sid), 0.5)

    def test_zero_or_one_shared_label_is_defined_as_full_agreement(self):
        # 0 or 1 shared label leaves no pair to disagree about -- max
        # distance is 0, which would otherwise divide by zero.
        self.assertEqual(normalized_agreement(0, max_structural_hamming_distance(1)), 1.0)
        self.assertEqual(normalized_agreement(0, max_structural_hamming_distance(0)), 1.0)


if __name__ == "__main__":
    unittest.main()
