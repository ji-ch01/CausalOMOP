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
                "sepsis": {
                    "role": "population", "domain": "condition",
                    "status": "candidate", "concept_ids": [132797],
                },
                "septic_shock": {
                    "role": "effect_modifier", "domain": "condition",
                    "status": "candidate", "concept_ids": [196236],
                },
                "albumin": {
                    "role": "treatment", "domain": "drug",
                    "status": "unmapped", "concept_ids": [],
                    "source_value_rule": r"(?i)^Albumin(?:, Human)?\b",
                },
                "crystalloids": {
                    "role": "treatment_and_comparator", "domain": "drug",
                    "status": "partial", "concept_ids": [],
                    "source_value_rule": r"(?i)(Lactated Ringers|Sodium Chloride).*(Bag|Bottle|Injection)$",
                    "lookup_term": "sodium chloride",
                },
            },
        }
        result = DataScienceAgent().run(hypothesis, protocol)
        albumin = next(item for item in result.phenotypes if item.name == "albumin")
        self.assertEqual(albumin.mode, "source_value_provisional")
        self.assertFalse(result.exploratory_execution_allowed)
        self.assertFalse(result.final_execution_allowed)
        self.assertIn("death", result.required_tables)
        self.assertEqual(result.metadata["agent_name"], "data_science_agent")
        # The two OMOP-standard scaffolds are still added because this
        # protocol declares no concept_sets entry with role="outcome"/"time_zero".
        self.assertIn("all_cause_death", [item.name for item in result.phenotypes])
        self.assertIn("index_event", [item.name for item in result.phenotypes])
        self.assertIn("albumin exposure within grace period", result.feasibility_checks)
        self.assertIn("crystalloids exposure within grace period", result.feasibility_checks)
        self.assertEqual(result.design, "add_on_target_trial_emulation")

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
            "concept_sets": {
                "sepsis": {"role": "population", "domain": "condition", "status": "pending", "concept_ids": []},
                "septic_shock": {
                    "role": "effect_modifier", "domain": "condition", "status": "pending", "concept_ids": [],
                },
                "albumin": {"role": "treatment", "domain": "drug", "status": "pending", "concept_ids": []},
                "crystalloids": {
                    "role": "treatment_and_comparator", "domain": "drug", "status": "pending", "concept_ids": [],
                },
            },
        }

        result = DataScienceAgent().run(hypothesis, protocol, concept_lookup=Lookup())
        albumin = next(item for item in result.phenotypes if item.name == "albumin")

        self.assertEqual([item.concept_id for item in albumin.concept_candidates], [1])
        self.assertEqual(albumin.concept_ids, ())
        self.assertEqual(albumin.status, "pending")
        self.assertFalse(result.final_execution_allowed)

    def test_generalizes_to_an_unrelated_study_without_touching_python_code(self):
        """The whole point of deriving phenotypes from concept_sets: a study
        about a completely different clinical domain, with a mutually
        exclusive active-comparator design and its own outcome, works
        without any hardcoded "sepsis"/"albumin" name in the agent."""
        hypothesis = {
            "schema_version": "1.0", "study_id": "study", "time_zero": "diagnosis date",
            "grace_period_hours": 720, "follow_up_days": 365, "primary_estimand": "ATE",
            "candidate_confounders": ["age"],
        }
        protocol = {
            "data_source": "project.dataset",
            "query_settings": {"exposure_contrast": "mutually_exclusive"},
            "concept_sets": {
                "type_2_diabetes": {
                    "role": "population", "domain": "condition",
                    "status": "approved", "concept_ids": [201826],
                },
                "metformin": {
                    "role": "treatment", "domain": "drug",
                    "status": "approved", "concept_ids": [1503297],
                },
                "sulfonylurea": {
                    "role": "comparator", "domain": "drug",
                    "status": "approved", "concept_ids": [1560171],
                },
                "myocardial_infarction": {
                    "role": "outcome", "domain": "condition",
                    "status": "approved", "concept_ids": [4329847],
                },
            },
        }
        result = DataScienceAgent().run(hypothesis, protocol)
        names = {item.name for item in result.phenotypes}
        self.assertIn("metformin", names)
        self.assertIn("sulfonylurea", names)
        self.assertIn("myocardial_infarction", names)
        # A protocol-declared outcome overrides the all_cause_death default,
        # and role="time_zero" is not declared here so the default still applies.
        self.assertNotIn("all_cause_death", names)
        self.assertIn("index_event", names)
        self.assertEqual(result.design, "active_comparator_target_trial_emulation")
        self.assertIn("metformin exposure within grace period", result.feasibility_checks)
        self.assertIn("sulfonylurea exposure within grace period", result.feasibility_checks)
        self.assertIn("mutually exclusive treatment assignment", result.feasibility_checks)

    def test_concept_set_without_role_is_rejected(self):
        hypothesis = {
            "schema_version": "1.0", "study_id": "study", "time_zero": "index",
            "grace_period_hours": 24, "follow_up_days": 28, "primary_estimand": "ATE",
        }
        protocol = {
            "data_source": "project.dataset",
            "concept_sets": {"sepsis": {"domain": "condition", "status": "pending", "concept_ids": []}},
        }
        with self.assertRaises(ValueError):
            DataScienceAgent().run(hypothesis, protocol)

    def test_concept_set_with_unknown_domain_is_rejected(self):
        hypothesis = {
            "schema_version": "1.0", "study_id": "study", "time_zero": "index",
            "grace_period_hours": 24, "follow_up_days": 28, "primary_estimand": "ATE",
        }
        protocol = {
            "data_source": "project.dataset",
            "concept_sets": {
                "sepsis": {"role": "population", "domain": "not_a_domain", "status": "pending", "concept_ids": []},
            },
        }
        with self.assertRaises(ValueError):
            DataScienceAgent().run(hypothesis, protocol)


if __name__ == "__main__":
    unittest.main()
