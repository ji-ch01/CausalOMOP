import unittest

from causal.agents import ExecutionAgent, analysis_plan_blockers


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

    def test_final_mode_allows_aggregate_feasibility_once_mapping_is_approved(self):
        """`omcp.governance.authorize_causal_operation` has a dedicated
        `mode == "final"` branch for `aggregate_feasibility` that requires
        `clinical_mapping_status == "approved"` -- that branch is only
        reachable at all if this agent can actually grant
        `aggregate_feasibility_allowed=True` for `mode="final"` in the first
        place. It previously never did (bug: `aggregate_allowed` was
        unconditionally `mode == "exploratory"`), permanently blocking every
        read-only diagnostic for final-mode runs regardless of approval."""
        result = ExecutionAgent().run(
            plan(),
            mode="final",
            clinical_mapping_status="approved",
            analysis_plan_status="pending",
        )
        self.assertTrue(result.aggregate_feasibility_allowed)

    def test_final_mode_blocks_aggregate_feasibility_when_mapping_only_provisional(self):
        result = ExecutionAgent().run(
            plan(),
            mode="final",
            clinical_mapping_status="provisional",
            analysis_plan_status="pending",
        )
        self.assertFalse(result.aggregate_feasibility_allowed)

    def test_malformed_phenotype_entry_blocks_like_a_malformed_covariate_would(self):
        """`incomplete_covariates` always treated a non-dict list entry as
        incomplete; `incomplete_phenotypes` used to only check `isinstance
        (item, dict) AND ...`, so a non-dict entry (e.g. a stray `None` from
        a hand-edited or buggily-generated analysis_plan.json) fell out of
        the filter entirely instead of blocking -- the two lists must treat
        malformed entries the same way."""
        unresolved, incomplete_covariates, incomplete_phenotypes = analysis_plan_blockers(
            {"phenotypes": [None, {"name": "population", "status": "approved"}], "covariates": []}
        )
        self.assertEqual(incomplete_phenotypes, ("unnamed",))


if __name__ == "__main__":
    unittest.main()
