import unittest

from causal.agents import ExecutionAgent


def plan(exploratory_allowed=False, unresolved=()):
    return {
        "schema_version": "1.0",
        "study_id": "study",
        "exploratory_execution_allowed": exploratory_allowed,
        "final_execution_allowed": False,
        "unresolved_issues": list(unresolved),
        "phenotypes": [
            {"name": "population", "status": "approved", "approval_required": True}
        ],
        "covariates": [{"name": "age", "status": "ready"}],
    }


class ExecutionAgentTests(unittest.TestCase):
    def test_provisional_mapping_only_allows_aggregate_audits(self):
        result = ExecutionAgent().run(
            plan(unresolved=("albumin mapping",)),
            mode="exploratory",
            clinical_mapping_status="provisional",
            analysis_plan_status="pending",
        )
        self.assertTrue(result.aggregate_feasibility_allowed)
        self.assertFalse(result.analytic_dataset_build_allowed)
        self.assertFalse(result.causal_estimation_allowed)
        self.assertIn("estimate_or_publish_causal_effect", result.prohibited_actions)

    def test_all_gates_and_definitions_are_required_for_estimation(self):
        result = ExecutionAgent().run(
            plan(exploratory_allowed=True),
            mode="exploratory",
            clinical_mapping_status="approved",
            analysis_plan_status="approved",
        )
        self.assertTrue(result.analytic_dataset_build_allowed)
        self.assertTrue(result.causal_estimation_allowed)
        self.assertEqual(result.status, "ready_for_causal_execution")

    def test_a_covariate_not_yet_ready_blocks_estimation_even_with_gates_approved(self):
        """A non-demographic covariate stuck at `status="definition_pending"`
        (mapping approved but aggregation not yet reviewed, or vice versa --
        see `causal.agents.data_science.DataScienceAgent._covariate`) must
        still block `causal_estimation_allowed`, even once the top-level
        `clinical_mapping` gate is approved -- the top-level gate alone is
        not proof every individual covariate is actually ready."""
        study_plan = plan(exploratory_allowed=True)
        study_plan["covariates"] = [{"name": "sofa_score", "status": "definition_pending"}]
        result = ExecutionAgent().run(
            study_plan,
            mode="exploratory",
            clinical_mapping_status="approved",
            analysis_plan_status="approved",
        )
        self.assertFalse(result.causal_estimation_allowed)
        self.assertIn("covariate_not_ready:sofa_score", result.blockers)


if __name__ == "__main__":
    unittest.main()
