import unittest

from causal.agents import DataScienceAgent


class DataScienceAgentTests(unittest.TestCase):
    def test_builds_provisional_plan_without_enabling_final_execution(self):
        hypothesis = {
            "schema_version": "1.0", "study_id": "study", "time_zero": "ICU admission",
            "grace_period_hours": 24, "follow_up_days": 28, "primary_estimand": "ATE",
            "secondary_estimands": ["CATE by age"], "candidate_confounders": ["age", "SOFA"],
            "unresolved_issues": ["concept_set:albumin:unmapped"],
        }
        protocol = {
            "data_source": "project.dataset",
            "concept_sets": {
                "sepsis": {"status": "candidate", "concept_ids": [132797]},
                "septic_shock": {"status": "candidate", "concept_ids": [196236]},
                "albumin": {"status": "unmapped", "concept_ids": []},
                "crystalloids": {"status": "partial", "concept_ids": []},
            },
        }
        result = DataScienceAgent().run(hypothesis, protocol)
        albumin = next(item for item in result.phenotypes if item.name == "albumin")
        self.assertEqual(albumin.mode, "source_value_provisional")
        self.assertFalse(result.exploratory_execution_allowed)
        self.assertFalse(result.final_execution_allowed)
        self.assertIn("death", result.required_tables)
        self.assertEqual(result.metadata["agent_name"], "data_science_agent")

    def test_mcp_candidates_are_review_only_and_domain_filtered(self):
        class Response:
            text = (
                "concept_id,concept_name,concept_code,vocabulary_id,domain_id\n"
                "1,Albumin,x,RxNorm,Drug\n2,Wrong domain,x,SNOMED,Condition\n"
            )

        class Lookup:
            def lookup_drug(self, term):
                return Response()

            def lookup_condition(self, term):
                response = Response()
                response.text = response.text.replace("Albumin", "Sepsis").replace(
                    "RxNorm,Drug", "SNOMED,Condition"
                )
                return response

        hypothesis = {
            "schema_version": "1.0", "study_id": "study", "time_zero": "ICU admission",
            "grace_period_hours": 24, "follow_up_days": 28, "primary_estimand": "ATE",
        }
        protocol = {
            "data_source": "project.dataset",
            "concept_sets": {name: {"status": "pending", "concept_ids": []} for name in (
                "sepsis", "septic_shock", "albumin", "crystalloids"
            )},
        }

        result = DataScienceAgent().run(hypothesis, protocol, concept_lookup=Lookup())
        albumin = next(item for item in result.phenotypes if item.name == "albumin")

        self.assertEqual([item.concept_id for item in albumin.concept_candidates], [1])
        self.assertEqual(albumin.concept_ids, ())
        self.assertEqual(albumin.status, "pending")
        self.assertFalse(result.final_execution_allowed)


if __name__ == "__main__":
    unittest.main()
