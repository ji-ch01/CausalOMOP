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
        self.assertEqual(result.covariate_relations, ())
        self.assertEqual(result.rejected_covariates, ())

    def test_accepts_and_normalizes_valid_dag_variable(self):
        value = protocol()
        value["candidate_dag"] = [
            {
                "label": "SOFA score",
                "role": "confounder",
                "pmid": "12345678",
                "temporal_classification": "pre_treatment",
                "rationale": "Higher SOFA scores are associated with albumin use and with mortality.",
            }
        ]
        result = HypothesisAgent().run(value)
        self.assertEqual(len(result.candidate_dag), 1)
        self.assertEqual(result.candidate_dag[0].label, "SOFA score")
        self.assertEqual(result.candidate_dag[0].role, "confounder")
        self.assertEqual(result.candidate_dag[0].pmid, "12345678")
        self.assertTrue(result.candidate_dag[0].observed)

    def test_domain_defaults_to_empty_string_when_omitted(self):
        value = protocol()
        value["candidate_dag"] = [
            {
                "label": "SOFA score", "role": "confounder", "pmid": "",
                "temporal_classification": "pre_treatment", "rationale": "x",
            }
        ]
        result = HypothesisAgent().run(value)
        self.assertEqual(result.candidate_dag[0].domain, "")

    def test_accepts_a_valid_domain(self):
        value = protocol()
        value["candidate_dag"] = [
            {
                "label": "SOFA score", "role": "confounder", "pmid": "",
                "temporal_classification": "pre_treatment", "rationale": "x",
                "domain": "measurement",
            }
        ]
        result = HypothesisAgent().run(value)
        self.assertEqual(result.candidate_dag[0].domain, "measurement")

    def test_rejects_an_unknown_domain(self):
        value = protocol()
        value["candidate_dag"] = [
            {
                "label": "SOFA score", "role": "confounder", "pmid": "",
                "temporal_classification": "pre_treatment", "rationale": "x",
                "domain": "not_a_domain",
            }
        ]
        with self.assertRaises(ValueError):
            HypothesisAgent().run(value)

    def test_concept_mapping_defaults_to_unmapped_with_no_ids(self):
        value = protocol()
        value["candidate_dag"] = [
            {
                "label": "SOFA score", "role": "confounder", "pmid": "",
                "temporal_classification": "pre_treatment", "rationale": "x",
            }
        ]
        result = HypothesisAgent().run(value)
        self.assertEqual(result.candidate_dag[0].mapping_status, "unmapped")
        self.assertEqual(result.candidate_dag[0].concept_ids, ())

    def test_accepts_an_approved_concept_mapping(self):
        value = protocol()
        value["candidate_dag"] = [
            {
                "label": "SOFA score", "role": "confounder", "pmid": "",
                "temporal_classification": "pre_treatment", "rationale": "x",
                "domain": "measurement", "concept_ids": [4301868], "mapping_status": "approved",
            }
        ]
        result = HypothesisAgent().run(value)
        self.assertEqual(result.candidate_dag[0].concept_ids, (4301868,))
        self.assertEqual(result.candidate_dag[0].mapping_status, "approved")

    def test_rejects_approved_mapping_with_no_concept_ids(self):
        value = protocol()
        value["candidate_dag"] = [
            {
                "label": "SOFA score", "role": "confounder", "pmid": "",
                "temporal_classification": "pre_treatment", "rationale": "x",
                "mapping_status": "approved",
            }
        ]
        with self.assertRaises(ValueError):
            HypothesisAgent().run(value)

    def test_aggregation_defaults_to_definition_pending(self):
        value = protocol()
        value["candidate_dag"] = [
            {
                "label": "SOFA score", "role": "confounder", "pmid": "",
                "temporal_classification": "pre_treatment", "rationale": "x",
            }
        ]
        result = HypothesisAgent().run(value)
        self.assertEqual(result.candidate_dag[0].aggregation, "definition_pending")

    def test_accepts_an_aggregation_compatible_with_its_domain(self):
        value = protocol()
        value["candidate_dag"] = [
            {
                "label": "SOFA score", "role": "confounder", "pmid": "",
                "temporal_classification": "pre_treatment", "rationale": "x",
                "domain": "measurement", "concept_ids": [4301868], "mapping_status": "approved",
                "aggregation": "last_value_before_index",
            }
        ]
        result = HypothesisAgent().run(value)
        self.assertEqual(result.candidate_dag[0].aggregation, "last_value_before_index")

    def test_rejects_an_unknown_aggregation(self):
        value = protocol()
        value["candidate_dag"] = [
            {
                "label": "SOFA score", "role": "confounder", "pmid": "",
                "temporal_classification": "pre_treatment", "rationale": "x",
                "domain": "measurement", "aggregation": "not_a_real_aggregation",
            }
        ]
        with self.assertRaises(ValueError):
            HypothesisAgent().run(value)

    def test_rejects_an_aggregation_incompatible_with_its_domain(self):
        """A condition covariate has no numeric value -- a value aggregation
        like max_in_window cannot be reviewed onto it."""
        value = protocol()
        value["candidate_dag"] = [
            {
                "label": "diabetes", "role": "confounder", "pmid": "",
                "temporal_classification": "pre_treatment", "rationale": "x",
                "domain": "condition", "aggregation": "max_in_window",
            }
        ]
        with self.assertRaises(ValueError):
            HypothesisAgent().run(value)

    def test_window_defaults_to_none(self):
        value = protocol()
        value["candidate_dag"] = [
            {
                "label": "SOFA score", "role": "confounder", "pmid": "",
                "temporal_classification": "pre_treatment", "rationale": "x",
            }
        ]
        result = HypothesisAgent().run(value)
        self.assertIsNone(result.candidate_dag[0].window_start_hours)
        self.assertIsNone(result.candidate_dag[0].window_end_hours)

    def test_accepts_a_valid_window_override(self):
        value = protocol()
        value["candidate_dag"] = [
            {
                "label": "SOFA score", "role": "confounder", "pmid": "",
                "temporal_classification": "pre_treatment", "rationale": "x",
                "window_start_hours": -6, "window_end_hours": 0,
            }
        ]
        result = HypothesisAgent().run(value)
        self.assertEqual(result.candidate_dag[0].window_start_hours, -6)
        self.assertEqual(result.candidate_dag[0].window_end_hours, 0)

    def test_rejects_a_window_with_only_one_bound_set(self):
        value = protocol()
        value["candidate_dag"] = [
            {
                "label": "SOFA score", "role": "confounder", "pmid": "",
                "temporal_classification": "pre_treatment", "rationale": "x",
                "window_start_hours": -6,
            }
        ]
        with self.assertRaises(ValueError):
            HypothesisAgent().run(value)

    def test_rejects_a_window_extending_past_the_index_event(self):
        value = protocol()
        value["candidate_dag"] = [
            {
                "label": "SOFA score", "role": "confounder", "pmid": "",
                "temporal_classification": "pre_treatment", "rationale": "x",
                "window_start_hours": -6, "window_end_hours": 6,
            }
        ]
        with self.assertRaises(ValueError):
            HypothesisAgent().run(value)

    def test_rejects_a_window_with_start_after_end(self):
        value = protocol()
        value["candidate_dag"] = [
            {
                "label": "SOFA score", "role": "confounder", "pmid": "",
                "temporal_classification": "pre_treatment", "rationale": "x",
                "window_start_hours": 0, "window_end_hours": -6,
            }
        ]
        with self.assertRaises(ValueError):
            HypothesisAgent().run(value)

    def test_rejects_non_integer_concept_ids(self):
        value = protocol()
        value["candidate_dag"] = [
            {
                "label": "SOFA score", "role": "confounder", "pmid": "",
                "temporal_classification": "pre_treatment", "rationale": "x",
                "concept_ids": ["not_an_int"],
            }
        ]
        with self.assertRaises(ValueError):
            HypothesisAgent().run(value)

    def test_accepts_unobserved_flag(self):
        value = protocol()
        value["candidate_dag"] = [
            {
                "label": "unmeasured frailty",
                "role": "confounder",
                "pmid": "",
                "temporal_classification": "pre_treatment",
                "rationale": "Frailty plausibly affects both treatment choice and mortality.",
                "observed": False,
            }
        ]
        result = HypothesisAgent().run(value)
        self.assertFalse(result.candidate_dag[0].observed)

    def test_pmid_is_optional_for_expert_asserted_roles(self):
        value = protocol()
        value["candidate_dag"] = [
            {
                "label": "age", "role": "confounder", "pmid": "",
                "temporal_classification": "pre_treatment",
                "rationale": "Well-established baseline confounder.",
            }
        ]
        result = HypothesisAgent().run(value)
        self.assertEqual(result.candidate_dag[0].pmid, "")

    def test_accepts_unknown_direction_certainty(self):
        value = protocol()
        value["candidate_dag"] = [
            {
                "label": "SOFA score",
                "role": "confounder",
                "pmid": "",
                "temporal_classification": "pre_treatment",
                "rationale": "Associated with both treatment and outcome; direction unclear from abstracts.",
                "certainty": "unknown_direction",
            }
        ]
        result = HypothesisAgent().run(value)
        self.assertEqual(result.candidate_dag[0].certainty, "unknown_direction")

    def test_defaults_certainty_to_established(self):
        value = protocol()
        value["candidate_dag"] = [
            {
                "label": "SOFA score", "role": "confounder", "pmid": "",
                "temporal_classification": "pre_treatment", "rationale": "x",
            }
        ]
        result = HypothesisAgent().run(value)
        self.assertEqual(result.candidate_dag[0].certainty, "established")

    def test_rejects_invalid_certainty(self):
        value = protocol()
        value["candidate_dag"] = [
            {
                "label": "SOFA score",
                "role": "confounder",
                "pmid": "",
                "rationale": "x",
                "certainty": "very confident",
            }
        ]
        with self.assertRaises(ValueError):
            HypothesisAgent().run(value)

    def test_rejects_invalid_dag_role(self):
        value = protocol()
        value["candidate_dag"] = [
            {"label": "SOFA", "role": "causes_everything", "pmid": "1", "rationale": "x"}
        ]
        with self.assertRaises(ValueError):
            HypothesisAgent().run(value)

    def test_rejects_dag_variable_missing_rationale(self):
        value = protocol()
        value["candidate_dag"] = [
            {"label": "SOFA", "role": "confounder", "pmid": "1", "rationale": ""}
        ]
        with self.assertRaises(ValueError):
            HypothesisAgent().run(value)

    def test_rejects_duplicate_dag_labels(self):
        value = protocol()
        value["candidate_dag"] = [
            {"label": "SOFA", "role": "confounder", "pmid": "1", "rationale": "x"},
            {"label": "SOFA", "role": "outcome_cause_only", "pmid": "2", "rationale": "y"},
        ]
        with self.assertRaises(ValueError):
            HypothesisAgent().run(value)

    def test_accepts_covariate_relation_between_declared_variables(self):
        value = protocol()
        value["candidate_dag"] = [
            {
                "label": "renal function", "role": "confounder", "pmid": "",
                "temporal_classification": "pre_treatment", "rationale": "x",
            },
            {
                "label": "SOFA score", "role": "confounder", "pmid": "",
                "temporal_classification": "pre_treatment", "rationale": "y",
            },
        ]
        value["covariate_relations"] = [
            {
                "source": "renal function",
                "target": "SOFA score",
                "pmid": "",
                "rationale": "Renal function is a component of the SOFA score.",
            }
        ]
        result = HypothesisAgent().run(value)
        self.assertEqual(len(result.covariate_relations), 1)
        self.assertEqual(result.covariate_relations[0].source, "renal function")
        self.assertEqual(result.covariate_relations[0].target, "SOFA score")

    def test_rejects_covariate_relation_referencing_undeclared_label(self):
        value = protocol()
        value["candidate_dag"] = [
            {"label": "renal function", "role": "confounder", "pmid": "", "rationale": "x"},
        ]
        value["covariate_relations"] = [
            {"source": "renal function", "target": "unknown variable", "pmid": "", "rationale": "y"}
        ]
        with self.assertRaises(ValueError):
            HypothesisAgent().run(value)

    def test_rejects_self_referencing_covariate_relation(self):
        value = protocol()
        value["candidate_dag"] = [
            {"label": "renal function", "role": "confounder", "pmid": "", "rationale": "x"},
        ]
        value["covariate_relations"] = [
            {"source": "renal function", "target": "renal function", "pmid": "", "rationale": "y"}
        ]
        with self.assertRaises(ValueError):
            HypothesisAgent().run(value)

    def test_accepts_rejected_covariate(self):
        value = protocol()
        value["rejected_covariates"] = [
            {
                "label": "age",
                "reviewer": "Dr. Chen",
                "comment": "Not plausibly linked to treatment choice in this cohort.",
                "decided_at": "2026-07-23T00:00:00+00:00",
            }
        ]
        result = HypothesisAgent().run(value)
        self.assertEqual(len(result.rejected_covariates), 1)
        self.assertEqual(result.rejected_covariates[0].label, "age")
        self.assertEqual(result.rejected_covariates[0].reviewer, "Dr. Chen")

    def test_rejected_covariate_comment_and_decided_at_are_optional(self):
        value = protocol()
        value["rejected_covariates"] = [{"label": "age", "reviewer": "Dr. Chen"}]
        result = HypothesisAgent().run(value)
        self.assertEqual(result.rejected_covariates[0].comment, "")
        self.assertEqual(result.rejected_covariates[0].decided_at, "")

    def test_rejects_rejected_covariate_missing_reviewer(self):
        value = protocol()
        value["rejected_covariates"] = [{"label": "age", "reviewer": ""}]
        with self.assertRaises(ValueError):
            HypothesisAgent().run(value)

    def test_rejects_duplicate_rejected_covariate_labels(self):
        value = protocol()
        value["rejected_covariates"] = [
            {"label": "age", "reviewer": "Dr. Chen"},
            {"label": "age", "reviewer": "Dr. Lee"},
        ]
        with self.assertRaises(ValueError):
            HypothesisAgent().run(value)

    def test_rejects_dag_variable_missing_temporal_classification(self):
        value = protocol()
        value["candidate_dag"] = [
            {"label": "SOFA score", "role": "confounder", "pmid": "", "rationale": "x"}
        ]
        with self.assertRaises(ValueError):
            HypothesisAgent().run(value)

    def test_rejects_role_temporal_classification_mismatch(self):
        value = protocol()
        value["candidate_dag"] = [
            {
                "label": "new organism on culture",
                "role": "confounder",
                "pmid": "",
                "temporal_classification": "post_treatment",
                "rationale": "Cultured after treatment began.",
            }
        ]
        with self.assertRaises(ValueError):
            HypothesisAgent().run(value)

    def test_mediator_requires_post_treatment_temporal_classification(self):
        value = protocol()
        value["candidate_dag"] = [
            {
                "label": "downstream complication",
                "role": "mediator",
                "pmid": "",
                "temporal_classification": "pre_treatment",
                "rationale": "On the causal pathway from treatment to outcome.",
            }
        ]
        with self.assertRaises(ValueError):
            HypothesisAgent().run(value)

    def test_major_confounders_must_reference_a_first_pass_covariate(self):
        value = protocol()
        value["major_confounders"] = ["age not in first pass list"]
        with self.assertRaises(ValueError):
            HypothesisAgent().run(value)

    def test_unresolved_major_confounder_marks_the_artifact_not_executable(self):
        value = protocol()
        value["concept_sets"] = {"sepsis": {"status": "approved", "concept_ids": [132797]}}
        value["major_confounders"] = ["SOFA"]
        result = HypothesisAgent().run(value)
        self.assertFalse(result.unresolved_issues)
        self.assertIn("SOFA", result.unresolved_covariates)
        self.assertEqual(result.unresolved_major_confounders, ("SOFA",))
        self.assertFalse(result.executable)

    def test_classified_major_confounder_is_not_unresolved(self):
        value = protocol()
        value["major_confounders"] = ["SOFA"]
        value["candidate_dag"] = [
            {
                "label": "SOFA", "role": "confounder", "pmid": "",
                "temporal_classification": "pre_treatment",
                "rationale": "Baseline severity affects both treatment choice and outcome.",
            }
        ]
        result = HypothesisAgent().run(value)
        self.assertEqual(result.unresolved_major_confounders, ())


if __name__ == "__main__":
    unittest.main()
