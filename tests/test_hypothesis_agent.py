import unittest

from causal.agents import HypothesisAgent


def protocol():
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
            "secondary_estimands": ["CATE by age", "CATE by sex"],
        },
        "first_pass_covariates": ["SOFA", "lactate"],
        "concept_sets": {
            "sepsis": {"status": "approved", "concept_ids": [132797]},
            "albumin": {"status": "present_but_unmapped", "concept_ids": []},
        },
    }


class HypothesisAgentTests(unittest.TestCase):
    def test_normalizes_protocol_and_surfaces_mapping_issue(self):
        result = HypothesisAgent().run(protocol())
        self.assertEqual(result.primary_estimand, "ATE")
        self.assertEqual(result.candidate_effect_modifiers, ("age", "sex"))
        self.assertFalse(result.executable)
        self.assertIn("concept_set:albumin:present_but_unmapped", result.unresolved_issues)
        self.assertEqual(result.metadata["agent_name"], "hypothesis_agent")
        self.assertEqual(len(result.metadata["input_sha256"]), 64)

    def test_rejects_incomplete_target_trial(self):
        value = protocol()
        del value["target_trial"]["time_zero"]
        with self.assertRaises(ValueError):
            HypothesisAgent().run(value)

    def test_defaults_to_empty_dag_when_absent(self):
        result = HypothesisAgent().run(protocol())
        self.assertEqual(result.candidate_dag, ())

    def test_accepts_and_normalizes_valid_dag_edges(self):
        value = protocol()
        value["candidate_dag"] = [
            {
                "confounder": "SOFA score",
                "relation": "confounder_to_treatment",
                "pmid": "12345678",
                "rationale": "Higher SOFA scores are associated with albumin use in some ICUs.",
            }
        ]
        result = HypothesisAgent().run(value)
        self.assertEqual(len(result.candidate_dag), 1)
        self.assertEqual(result.candidate_dag[0].confounder, "SOFA score")
        self.assertEqual(result.candidate_dag[0].pmid, "12345678")

    def test_rejects_invalid_dag_relation(self):
        value = protocol()
        value["candidate_dag"] = [
            {"confounder": "SOFA", "relation": "causes_everything", "pmid": "1", "rationale": "x"}
        ]
        with self.assertRaises(ValueError):
            HypothesisAgent().run(value)

    def test_rejects_dag_edge_missing_pmid(self):
        value = protocol()
        value["candidate_dag"] = [
            {"confounder": "SOFA", "relation": "confounder_to_outcome", "pmid": "", "rationale": "x"}
        ]
        with self.assertRaises(ValueError):
            HypothesisAgent().run(value)


if __name__ == "__main__":
    unittest.main()
