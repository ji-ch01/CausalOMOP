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


if __name__ == "__main__":
    unittest.main()
