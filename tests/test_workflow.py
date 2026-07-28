import json
import tempfile
import unittest
from pathlib import Path

from causal.orchestrator import WorkflowManager


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.protocol = root / "protocol.json"
        self.protocol.write_text(json.dumps({"study_id": "test_study"}), encoding="utf-8")
        self.manager = WorkflowManager(root / "runs")

    def tearDown(self):
        self.temp.cleanup()

    def test_start_snapshots_protocol(self):
        state = self.manager.start(self.protocol)
        self.assertEqual(state.phase, "protocol_review")
        self.assertTrue((Path(self.temp.name) / "runs" / state.run_id / "protocol.json").is_file())

    def test_gate_required_to_advance(self):
        state = self.manager.start(self.protocol)
        with self.assertRaises(RuntimeError):
            self.manager.advance(state.run_id)
        self.manager.decide_gate(state.run_id, "protocol", "approved", "tester")
        state = self.manager.advance(state.run_id)
        self.assertEqual(state.phase, "concept_mapping")

    def test_provisional_mapping_only_allows_exploration(self):
        exploratory = self.manager.start(self.protocol, "exploratory")
        self.manager.decide_gate(exploratory.run_id, "protocol", "approved", "tester")
        self.manager.advance(exploratory.run_id)
        self.manager.decide_gate(exploratory.run_id, "clinical_mapping", "provisional", "tester")
        self.assertEqual(self.manager.advance(exploratory.run_id).phase, "feasibility")

        final = self.manager.start(self.protocol, "final")
        self.manager.decide_gate(final.run_id, "protocol", "approved", "tester")
        self.manager.advance(final.run_id)
        self.manager.decide_gate(final.run_id, "clinical_mapping", "provisional", "tester")
        with self.assertRaises(RuntimeError):
            self.manager.advance(final.run_id)

    def test_hypothesis_agent_writes_artifact(self):
        full_protocol = {
            "study_id": "test_study",
            "source_question": "Does treatment affect outcome?",
            "target_trial": {
                "population": "population", "treatment": "treatment",
                "comparator": "comparator", "time_zero": "index",
                "grace_period_hours": 24, "outcome": "outcome",
                "follow_up_days": 28, "primary_estimand": "ATE"
            },
            "concept_sets": {"treatment": {"status": "pending"}}
        }
        self.protocol.write_text(json.dumps(full_protocol), encoding="utf-8")
        state = self.manager.start(self.protocol)
        state = self.manager.run_hypothesis_agent(state.run_id)
        self.assertIn("hypothesis.json", state.artifacts)
        self.assertTrue((Path(self.temp.name) / "runs" / state.run_id / "hypothesis.json").is_file())

    def test_demo_report_is_explicitly_non_causal(self):
        state = self.manager.start(self.protocol)
        report = self.manager.build_demo_report(state.run_id)
        text = report.read_text(encoding="utf-8")
        self.assertIn("No causal effect has been estimated", text)
        self.assertNotIn("Aucun effet causal", text)

    def test_execution_readiness_is_written_and_reported(self):
        run_dir = Path(self.temp.name) / "runs"
        state = self.manager.start(self.protocol)
        state.phase = "analysis_plan"
        self.manager.save(state)
        (run_dir / state.run_id / "analysis_plan.json").write_text(json.dumps({
            "schema_version": "1.0",
            "study_id": "test_study",
            "data_source": "project.dataset",
            "design": "target_trial_emulation",
            "exploratory_execution_allowed": False,
            "final_execution_allowed": False,
            "unresolved_issues": ["mapping pending"],
            "phenotypes": [],
            "covariates": [],
            "feasibility_checks": [],
        }), encoding="utf-8")
        state = self.manager.run_execution_readiness(state.run_id)
        self.assertIn("execution_readiness.json", state.artifacts)
        context = json.loads(
            (run_dir / state.run_id / "mcp_context.json").read_text(encoding="utf-8")
        )
        self.assertEqual(context["run_id"], state.run_id)
        self.assertFalse(context["causal_estimation_allowed"])
        report = self.manager.build_demo_report(state.run_id).read_text(encoding="utf-8")
        self.assertIn("Execution Agent decision", report)
        self.assertIn("Causal estimation allowed: `False`", report)

    def test_demo_report_includes_aggregate_cohort_profile_without_causal_claim(self):
        run_dir = Path(self.temp.name) / "runs"
        state = self.manager.start(self.protocol)
        artifact_dir = run_dir / state.run_id
        (artifact_dir / "cohort_manifest.json").write_text(json.dumps({
            "status": "draft_not_executable",
            "contains_person_level_data": True,
            "execution_allowed": False,
            "delivery_to_llm_allowed": False,
            "unresolved_issues": ["censoring_definition_pending"],
        }), encoding="utf-8")
        (artifact_dir / "cohort_profile.json").write_text(json.dumps({
            "causal_effect_estimated": False,
            "patient_level_rows_returned": False,
            "profile": {
                "cohort_persons": 100,
                "treated_persons": 20,
                "comparator_persons": 80,
                "outcome_events": 30,
                "missing_race": 5,
                "unresolved_censoring": 100,
            },
            "warnings": ["Mechanics only."],
        }), encoding="utf-8")

        report = self.manager.build_demo_report(state.run_id).read_text(encoding="utf-8")

        self.assertIn("Aggregate cohort profile", report)
        self.assertIn("Cohort persons: **100**", report)
        self.assertIn("Unresolved censoring: **100**", report)
        self.assertIn("does not estimate a causal effect", report)
        self.assertNotIn("causal effect is", report.lower())

    def test_demo_report_surfaces_grace_period_outcome_overlap_diagnostic(self):
        run_dir = Path(self.temp.name) / "runs"
        state = self.manager.start(self.protocol)
        (run_dir / state.run_id / "feasibility.json").write_text(json.dumps({
            "status": "provisional_cohort_feasibility",
            "causal_effect_estimated": False,
            "counts": {
                "eligible_persons": 100,
                "outcome_events": 20,
                "outcome_events_during_grace_period": 4,
            },
            "warnings": ["Counts are for pipeline feasibility and must not be interpreted causally."],
        }), encoding="utf-8")

        report = self.manager.build_demo_report(state.run_id).read_text(encoding="utf-8")

        self.assertIn("Outcome events during grace period: **4**", report)


class WorkflowCovariateReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.protocol = root / "protocol.json"
        self.protocol.write_text(json.dumps({
            "study_id": "test_study",
            "source_question": "Does treatment affect outcome?",
            "target_trial": {
                "population": "population", "treatment": "treatment",
                "comparator": "comparator", "time_zero": "index",
                "grace_period_hours": 24, "outcome": "outcome",
                "follow_up_days": 28, "primary_estimand": "ATE"
            },
            "first_pass_covariates": ["age", "SOFA score"],
            "concept_sets": {"treatment": {"status": "approved"}},
        }), encoding="utf-8")
        self.manager = WorkflowManager(root / "runs")
        state = self.manager.start(self.protocol)
        self.state = self.manager.run_hypothesis_agent(state.run_id)
        self.run_dir = root / "runs" / self.state.run_id

    def tearDown(self):
        self.temp.cleanup()

    def _hypothesis(self) -> dict:
        return json.loads((self.run_dir / "hypothesis.json").read_text(encoding="utf-8"))

    def test_accept_adds_a_role_classified_candidate_dag_entry(self):
        self.manager.review_covariate(
            self.state.run_id, "age", "accept", "Dr. Chen",
            "Institutional practice affects who receives treatment.",
            role="treatment_cause_only", temporal_classification="pre_treatment",
        )
        hypothesis = self._hypothesis()
        entry = next(item for item in hypothesis["candidate_dag"] if item["label"] == "age")
        self.assertEqual(entry["role"], "treatment_cause_only")
        self.assertEqual(entry["temporal_classification"], "pre_treatment")

    def test_accept_without_temporal_classification_raises(self):
        with self.assertRaises(ValueError):
            self.manager.review_covariate(
                self.state.run_id, "age", "accept", "Dr. Chen",
                "Institutional practice affects who receives treatment.",
                role="treatment_cause_only",
            )

    def test_reject_records_an_audited_exclusion(self):
        self.manager.review_covariate(
            self.state.run_id, "age", "reject", "Dr. Chen", "Not plausible in this cohort.",
        )
        hypothesis = self._hypothesis()
        self.assertEqual(hypothesis["candidate_dag"], [])
        entry = next(item for item in hypothesis["rejected_covariates"] if item["label"] == "age")
        self.assertEqual(entry["reviewer"], "Dr. Chen")
        self.assertEqual(entry["comment"], "Not plausible in this cohort.")

    def test_reviewing_a_label_again_replaces_rather_than_duplicates(self):
        self.manager.review_covariate(
            self.state.run_id, "age", "reject", "Dr. Chen", "First pass.",
        )
        self.manager.review_covariate(
            self.state.run_id, "age", "accept", "Dr. Lee", "Reconsidered as a treatment cause.",
            role="treatment_cause_only", temporal_classification="pre_treatment",
        )
        hypothesis = self._hypothesis()
        self.assertEqual(hypothesis["rejected_covariates"], [])
        self.assertEqual(len([item for item in hypothesis["candidate_dag"] if item["label"] == "age"]), 1)

    def test_accept_without_role_raises(self):
        with self.assertRaises(ValueError):
            self.manager.review_covariate(
                self.state.run_id, "age", "accept", "Dr. Chen", "Some rationale."
            )

    def test_reject_records_history_event(self):
        state = self.manager.review_covariate(
            self.state.run_id, "age", "reject", "Dr. Chen", "Not plausible.",
        )
        self.assertTrue(any(event["event"] == "covariate_reviewed" for event in state.history))

    def test_requires_hypothesis_agent_to_have_run_first(self):
        state = self.manager.start(self.protocol)
        with self.assertRaises(RuntimeError):
            self.manager.review_covariate(state.run_id, "age", "reject", "Dr. Chen", "x")


class WorkflowMajorConfounderGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.protocol = root / "protocol.json"
        self.protocol.write_text(json.dumps({
            "study_id": "test_study",
            "source_question": "Does treatment affect outcome?",
            "target_trial": {
                "population": "population", "treatment": "treatment",
                "comparator": "comparator", "time_zero": "index",
                "grace_period_hours": 24, "outcome": "outcome",
                "follow_up_days": 28, "primary_estimand": "ATE"
            },
            "first_pass_covariates": ["age", "SOFA score"],
            "major_confounders": ["age"],
            "concept_sets": {"treatment": {"status": "approved"}},
        }), encoding="utf-8")
        self.manager = WorkflowManager(root / "runs")
        state = self.manager.start(self.protocol)
        self.state = self.manager.run_hypothesis_agent(state.run_id)

    def tearDown(self):
        self.temp.cleanup()

    def test_protocol_gate_cannot_be_approved_while_major_confounder_unresolved(self):
        with self.assertRaises(ValueError):
            self.manager.decide_gate(self.state.run_id, "protocol", "approved", "tester")

    def test_protocol_gate_can_still_be_rejected_or_made_provisional(self):
        self.manager.decide_gate(self.state.run_id, "protocol", "provisional", "tester")
        self.manager.decide_gate(self.state.run_id, "protocol", "rejected", "tester")

    def test_classifying_the_major_confounder_unblocks_approval(self):
        self.manager.review_covariate(
            self.state.run_id, "age", "accept", "Dr. Chen",
            "Baseline age affects both treatment choice and outcome.",
            role="confounder", temporal_classification="pre_treatment",
        )
        state = self.manager.decide_gate(self.state.run_id, "protocol", "approved", "tester")
        self.assertEqual(state.gates["protocol"].status, "approved")

    def test_rejecting_the_major_confounder_unblocks_approval(self):
        self.manager.review_covariate(
            self.state.run_id, "age", "reject", "Dr. Chen",
            "Cohort is age-restricted at enrollment.",
        )
        state = self.manager.decide_gate(self.state.run_id, "protocol", "approved", "tester")
        self.assertEqual(state.gates["protocol"].status, "approved")


class WorkflowIdentificationGateTests(unittest.TestCase):
    """The protocol/DAG gate must also refuse approval when the declared DAG
    is not identified by backdoor adjustment -- e.g. a confounder was
    classified but marked latent/unobserved, so no baseline set can block
    the backdoor path it opens. See `_block_if_not_identified`."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.protocol = root / "protocol.json"
        self.protocol.write_text(json.dumps({
            "study_id": "test_study",
            "source_question": "Does treatment affect outcome?",
            "target_trial": {
                "population": "population", "treatment": "treatment",
                "comparator": "comparator", "time_zero": "index",
                "grace_period_hours": 24, "outcome": "outcome",
                "follow_up_days": 28, "primary_estimand": "ATE"
            },
            "first_pass_covariates": ["age"],
            "concept_sets": {"treatment": {"status": "approved"}},
        }), encoding="utf-8")
        self.manager = WorkflowManager(root / "runs")
        state = self.manager.start(self.protocol)
        self.state = self.manager.run_hypothesis_agent(state.run_id)

    def tearDown(self):
        self.temp.cleanup()

    def test_protocol_gate_cannot_be_approved_when_the_only_confounder_is_unobserved(self):
        self.manager.review_covariate(
            self.state.run_id, "age", "accept", "Dr. Chen",
            "Baseline age affects both treatment choice and outcome.",
            role="confounder", temporal_classification="pre_treatment",
            observed=False,
        )
        with self.assertRaises(ValueError):
            self.manager.decide_gate(self.state.run_id, "protocol", "approved", "tester")

    def test_protocol_gate_approves_when_the_confounder_is_observed(self):
        self.manager.review_covariate(
            self.state.run_id, "age", "accept", "Dr. Chen",
            "Baseline age affects both treatment choice and outcome.",
            role="confounder", temporal_classification="pre_treatment",
        )
        state = self.manager.decide_gate(self.state.run_id, "protocol", "approved", "tester")
        self.assertEqual(state.gates["protocol"].status, "approved")

    def test_protocol_gate_can_still_be_rejected_while_unidentified(self):
        self.manager.review_covariate(
            self.state.run_id, "age", "accept", "Dr. Chen",
            "Baseline age affects both treatment choice and outcome.",
            role="confounder", temporal_classification="pre_treatment",
            observed=False,
        )
        self.manager.decide_gate(self.state.run_id, "protocol", "provisional", "tester")
        self.manager.decide_gate(self.state.run_id, "protocol", "rejected", "tester")

    def test_empty_dag_is_trivially_identified_and_not_blocked_here(self):
        """An empty candidate_dag is trivially backdoor-identified (no path
        to block); this gate must not be the thing that catches that --
        `_block_if_major_confounders_unresolved` is."""
        state = self.manager.decide_gate(self.state.run_id, "protocol", "approved", "tester")
        self.assertEqual(state.gates["protocol"].status, "approved")


if __name__ == "__main__":
    unittest.main()
