import unittest

from causal.agents.dag_critique import build_pre_check
from causal.agents.dag_metrics import (
    LabelAlignment,
    ReferenceDagComparison,
    align_labels,
    compare_generated_to_reference,
    max_structural_hamming_distance,
    max_structural_intervention_distance,
    merge_dag_nodes,
    normalized_agreement,
    relabel_dag,
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


class LabelAlignmentTests(unittest.TestCase):
    """`align_labels`: the deterministic, LLM-free label reconciliation that
    lets a pipeline-generated DAG be scored against a hand-built
    `reference_dag.json` without accidental string equality driving the
    result (module docstring, docs/dag_generation_audit.md SS9)."""

    def test_exact_match_after_normalization(self):
        alignment = align_labels(["Chronic Liver Disease."], ["chronic liver disease"])
        self.assertEqual(len(alignment.matches), 1)
        match = alignment.matches[0]
        self.assertEqual(match.method, "exact")
        self.assertEqual(match.similarity, 1.0)
        self.assertEqual(match.reference_label, "Chronic Liver Disease.")
        self.assertEqual(match.candidate_label, "chronic liver disease")
        self.assertEqual(alignment.unmatched_reference, ())
        self.assertEqual(alignment.unmatched_candidate, ())

    def test_token_overlap_match_above_threshold(self):
        alignment = align_labels(["Age at ICU admission"], ["age"])
        self.assertEqual(len(alignment.matches), 1)
        self.assertEqual(alignment.matches[0].method, "token_overlap")
        # {age} vs {age, icu, admission}: 2 * 1 / (1 + 3)
        self.assertAlmostEqual(alignment.matches[0].similarity, 0.5)

    def test_below_threshold_pair_is_left_unmatched(self):
        alignment = align_labels(["Serum creatinine"], ["serum sodium"], min_similarity=0.75)
        self.assertEqual(alignment.matches, ())
        self.assertEqual(alignment.unmatched_reference, ("Serum creatinine",))
        self.assertEqual(alignment.unmatched_candidate, ("serum sodium",))

    def test_manual_alias_takes_precedence_and_is_recorded(self):
        alignment = align_labels(
            ["Illness severity (SOFA)"],
            ["baseline organ-dysfunction burden"],
            manual_aliases={"baseline organ-dysfunction burden": "Illness severity (SOFA)"},
        )
        self.assertEqual(len(alignment.matches), 1)
        self.assertEqual(alignment.matches[0].method, "manual")
        self.assertEqual(alignment.matches[0].similarity, 1.0)

    def test_stale_manual_alias_raises(self):
        with self.assertRaises(ValueError):
            align_labels(["Age"], ["age"], manual_aliases={"not a candidate label": "Age"})
        with self.assertRaises(ValueError):
            align_labels(["Age"], ["age"], manual_aliases={"age": "not a reference label"})

    def test_matching_is_one_to_one_and_greedy_by_similarity(self):
        # Both reference labels share the token "disease" with the single
        # candidate, but "chronic kidney disease" overlaps more -- it wins the
        # candidate, and "chronic liver disease" is left unmatched rather than
        # both mapping to the same candidate.
        alignment = align_labels(
            ["Chronic kidney disease", "Chronic liver disease"],
            ["chronic kidney disease / renal replacement therapy"],
        )
        self.assertEqual(len(alignment.matches), 1)
        self.assertEqual(alignment.matches[0].reference_label, "Chronic kidney disease")
        self.assertEqual(alignment.unmatched_reference, ("Chronic liver disease",))

    def test_result_is_deterministic_regardless_of_input_order(self):
        reference = ["Age at admission", "Baseline SOFA score", "Prior GI bleeding"]
        candidate = ["prior gi bleeding", "age at admission", "baseline sofa score"]
        first = align_labels(reference, candidate).to_dict()
        second = align_labels(list(reversed(reference)), list(reversed(candidate))).to_dict()
        self.assertEqual(first["matches"], second["matches"])

    def test_to_dict_is_json_safe_and_complete(self):
        alignment = align_labels(["Age at ICU admission", "Frailty"], ["age", "sepsis"])
        payload = alignment.to_dict()
        self.assertEqual(payload["matched_count"], 1)
        self.assertEqual(payload["unmatched_reference"], ["Frailty"])
        self.assertEqual(payload["unmatched_candidate"], ["sepsis"])
        self.assertEqual(
            set(payload["matches"][0]), {"reference_label", "candidate_label", "method", "similarity"}
        )


class RelabelDagTests(unittest.TestCase):
    def test_relabel_changes_labels_but_preserves_edges(self):
        original = _dag_from_candidate([_variable("age"), _variable("severity")])
        relabeled = relabel_dag(original, {"age": "Age at ICU admission"})
        labels = {node.label for node in relabeled.nodes}
        self.assertIn("Age at ICU admission", labels)
        self.assertNotIn("age", labels)
        # Same structure -- only a label changed -- so SHD/SID to the original
        # (which still says "age") is driven purely by that one renamed node.
        self.assertEqual(
            structural_hamming_distance(original, relabel_dag(original, {})), 0
        )

    def test_relabel_rejects_a_non_injective_map(self):
        original = _dag_from_candidate([_variable("age"), _variable("severity")])
        with self.assertRaises(ValueError):
            relabel_dag(original, {"age": "shared", "severity": "shared"})


class ReferenceDagComparisonTests(unittest.TestCase):
    """`compare_generated_to_reference`: align first, then score with
    SHD/SID, treating the reference as the true graph."""

    def test_same_structure_different_labels_scores_zero_after_alignment(self):
        reference = _dag_from_candidate([_variable("Age at ICU admission")])
        generated = _dag_from_candidate([_variable("age")])
        comparison = compare_generated_to_reference(reference, generated)
        self.assertIsInstance(comparison, ReferenceDagComparison)
        self.assertEqual(comparison.shd, 0)
        self.assertEqual(comparison.sid_generated_vs_reference, 0)
        self.assertEqual(comparison.structural_agreement, 1.0)
        self.assertEqual(comparison.alignment.unmatched_reference, ())

    def test_alignment_collapses_a_distance_that_bare_labels_inflate(self):
        reference = _dag_from_candidate([_variable("Age at ICU admission"), _variable("Baseline SOFA score")])
        generated = _dag_from_candidate([_variable("age"), _variable("baseline SOFA score")])
        aligned = compare_generated_to_reference(reference, generated).shd
        bare = structural_hamming_distance(reference, generated)
        self.assertEqual(aligned, 0)
        self.assertGreater(bare, aligned)

    def test_missing_confounder_is_a_nonzero_sid_and_is_reported_unmatched(self):
        reference = _dag_from_candidate([_variable("Age at ICU admission"), _variable("Septic shock")])
        generated = _dag_from_candidate([_variable("age")])
        comparison = compare_generated_to_reference(reference, generated)
        self.assertGreater(comparison.sid_generated_vs_reference, 0)
        self.assertEqual(comparison.alignment.unmatched_reference, ("Septic shock",))
        self.assertEqual(comparison.shared_label_count, 3)  # exposure, outcome, age
        self.assertEqual(comparison.union_label_count, 4)

    def test_manual_aliases_flow_through_to_the_comparison(self):
        reference = _dag_from_candidate([_variable("Illness severity (SOFA)")])
        generated = _dag_from_candidate([_variable("organ dysfunction burden")])
        without = compare_generated_to_reference(reference, generated)
        self.assertEqual(without.alignment.unmatched_reference, ("Illness severity (SOFA)",))
        with_alias = compare_generated_to_reference(
            reference, generated,
            manual_aliases={"organ dysfunction burden": "Illness severity (SOFA)"},
        )
        self.assertEqual(with_alias.shd, 0)
        self.assertEqual(with_alias.alignment.matches[0].method, "manual")

    def test_to_dict_is_json_safe(self):
        reference = _dag_from_candidate([_variable("Age at ICU admission")])
        generated = _dag_from_candidate([_variable("age")])
        payload = compare_generated_to_reference(reference, generated).to_dict()
        self.assertEqual(
            set(payload),
            {
                "alignment", "shared_label_count", "union_label_count", "shd",
                "sid_generated_vs_reference", "sid_reference_vs_generated",
                "structural_agreement", "causal_agreement",
            },
        )
        self.assertIsInstance(payload["alignment"]["matches"], list)


class MergeDagNodesTests(unittest.TestCase):
    """`merge_dag_nodes`: collapse a covariate family the generated DAG lists
    separately into the one node a reference DAG represents it as."""

    def test_merge_collapses_members_into_one_node_with_unioned_edges(self):
        dag = _dag_from_candidate(
            [_variable("nsaid"), _variable("contrast"), _variable("aminoglycoside"), _variable("age")]
        )
        merged = merge_dag_nodes(
            dag, {"Nephrotoxin cluster": ["nsaid", "contrast", "aminoglycoside"]}
        )
        labels = {node.label for node in merged.nodes}
        self.assertIn("Nephrotoxin cluster", labels)
        self.assertEqual(labels & {"nsaid", "contrast", "aminoglycoside"}, set())
        self.assertIn("age", labels)
        # The three confounders each wired to exposure and outcome; the merged
        # node keeps exactly one edge to each (deduped), so it scores like a
        # single confounder.
        single = _dag_from_candidate([_variable("Nephrotoxin cluster"), _variable("age")])
        self.assertEqual(structural_hamming_distance(merged, single), 0)

    def test_merge_is_observed_only_if_every_member_is(self):
        dag = _dag_from_candidate(
            [_variable("measured", observed=True), _variable("latent", observed=False)]
        )
        merged = merge_dag_nodes(dag, {"family": ["measured", "latent"]})
        family = next(node for node in merged.nodes if node.label == "family")
        self.assertFalse(family.observed)

    def test_unknown_member_label_raises(self):
        dag = _dag_from_candidate([_variable("age")])
        with self.assertRaises(ValueError):
            merge_dag_nodes(dag, {"family": ["age", "not a node"]})

    def test_group_cannot_include_exposure_or_outcome(self):
        dag = _dag_from_candidate([_variable("age")])
        exposure_label = dag.node_map[dag.exposure_id].label
        with self.assertRaises(ValueError):
            merge_dag_nodes(dag, {"family": ["age", exposure_label]})

    def test_heterogeneous_group_is_rejected(self):
        dag = _dag_from_candidate(
            [_variable("conf", role="confounder"), _variable("mod", role="effect_modifier")]
        )
        with self.assertRaises(ValueError):
            merge_dag_nodes(dag, {"family": ["conf", "mod"]})

    def test_candidate_label_groups_flows_through_the_comparison(self):
        reference = _dag_from_candidate([_variable("Nephrotoxin co-exposure"), _variable("Age")])
        generated = _dag_from_candidate(
            [_variable("NSAID use"), _variable("contrast use"), _variable("age")]
        )
        without = compare_generated_to_reference(reference, generated)
        merged = compare_generated_to_reference(
            reference, generated,
            candidate_label_groups={"Nephrotoxin co-exposure": ["NSAID use", "contrast use"]},
            manual_aliases={"age": "Age"},
        )
        self.assertLess(merged.shd, without.shd)
        self.assertEqual(merged.alignment.unmatched_candidate, ())


class LabelAlignmentTypeTests(unittest.TestCase):
    def test_align_labels_returns_a_label_alignment(self):
        self.assertIsInstance(align_labels(["a"], ["b"]), LabelAlignment)


if __name__ == "__main__":
    unittest.main()
