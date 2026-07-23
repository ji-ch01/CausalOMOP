import unittest

from causal.agents.dag_identification import build_causal_dag
from causal.agents.hypothesis import HypothesisAgent
from causal.roadmap import CausalDAG, CausalEdge, CausalNode, to_dagitty


def protocol(candidate_dag=()):
    return {
        "study_id": "albumin_sepsis",
        "source_question": "Does albumin affect mortality?",
        "target_trial": {
            "population": "Adults with sepsis",
            "treatment": "Albumin plus crystalloids",
            "comparator": "Crystalloids alone",
            "time_zero": "ICU admission",
            "grace_period_hours": 24,
            "outcome": "28-day mortality",
            "follow_up_days": 28,
            "primary_estimand": "ATE",
            "secondary_estimands": [],
        },
        "first_pass_covariates": [],
        "concept_sets": {
            "sepsis": {"status": "approved", "concept_ids": [132797]},
        },
        "candidate_dag": candidate_dag,
    }


class DagittyExportTests(unittest.TestCase):
    def test_exports_exposure_outcome_and_confounder_markers(self):
        hypothesis = HypothesisAgent().run(protocol([
            {
                "label": "SOFA score",
                "role": "confounder",
                "pmid": "111",
                "temporal_classification": "pre_treatment",
                "rationale": "SOFA influences treatment choice and predicts mortality.",
            },
        ]))
        dag = build_causal_dag(hypothesis)
        text = to_dagitty(dag)
        self.assertTrue(text.startswith("dag {"))
        self.assertTrue(text.endswith("}"))
        self.assertIn('"Albumin plus crystalloids" [exposure', text)
        self.assertIn('"28-day mortality" [outcome', text)
        self.assertIn('"SOFA score" [pos=', text)
        self.assertIn('"SOFA score" -> "Albumin plus crystalloids"', text)
        self.assertIn('"SOFA score" -> "28-day mortality"', text)
        self.assertIn("PMID: 111", text)
        self.assertIn('active comparator: "Albumin plus crystalloids" vs "Crystalloids alone"', text)

    def test_highlights_adjustment_set_when_provided(self):
        hypothesis = HypothesisAgent().run(protocol([
            {
                "label": "SOFA score",
                "role": "confounder",
                "pmid": "111",
                "temporal_classification": "pre_treatment",
                "rationale": "x",
            },
        ]))
        dag = build_causal_dag(hypothesis)
        confounder_id = next(n.node_id for n in dag.nodes if n.role == "confounder")
        without = to_dagitty(dag)
        with_highlight = to_dagitty(dag, adjustment_set=(confounder_id,))
        self.assertNotIn("adjusted", without)
        self.assertIn('"SOFA score" [adjusted,pos=', with_highlight)

    def test_rejects_duplicate_node_labels(self):
        dag = CausalDAG(
            nodes=(
                CausalNode("exposure", "Same label", "exposure", "treatment", True),
                CausalNode("outcome", "Different outcome", "outcome", "outcome", True),
                CausalNode("confounder_x", "Same label", "confounder", "baseline", True),
            ),
            edges=(
                CausalEdge("confounder_x", "exposure", "rationale a"),
                CausalEdge("confounder_x", "outcome", "rationale b"),
            ),
            exposure_id="exposure",
            outcome_id="outcome",
        )
        with self.assertRaises(ValueError):
            to_dagitty(dag)

    def test_marks_contested_edges_in_the_comment_tag(self):
        hypothesis = HypothesisAgent().run(protocol([
            {
                "label": "SOFA score",
                "role": "confounder",
                "pmid": "111",
                "temporal_classification": "pre_treatment",
                "rationale": "Different cited studies disagree on this association's direction.",
                "certainty": "contested",
            },
        ]))
        dag = build_causal_dag(hypothesis)
        text = to_dagitty(dag)
        self.assertIn("[evidence|contested]", text)

    def test_marks_unobserved_nodes_as_latent(self):
        dag = CausalDAG(
            nodes=(
                CausalNode("exposure", "Treatment", "exposure", "treatment", True),
                CausalNode("outcome", "Outcome", "outcome", "outcome", True),
                CausalNode("u", "Unmeasured factor", "confounder", "baseline", False),
            ),
            edges=(
                CausalEdge("u", "exposure", "assumed"),
                CausalEdge("u", "outcome", "assumed"),
            ),
            exposure_id="exposure",
            outcome_id="outcome",
        )
        text = to_dagitty(dag)
        self.assertIn('"Unmeasured factor" [latent,pos=', text)


if __name__ == "__main__":
    unittest.main()
