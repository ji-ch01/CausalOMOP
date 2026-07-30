import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from causal.orchestrator import WorkflowManager
from causal.orchestrator.workflow import (
    _LOCK_STALE_SECONDS,
    _RunLock,
    _load_concept_specs,
    _mismatched_concept_domains,
)
from causal.queries import ConceptSetSpec


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

    def test_start_rejects_a_study_id_with_characters_omcp_would_later_reject(self):
        """`study_id` is embedded verbatim in `run_id` (checked by
        `WorkflowManager._state_path`) and in `mcp_context.json` (checked by
        `omcp.governance.CausalExecutionContext`, the same character class).
        A charset mistake must fail loudly here, before any run directory or
        file is written -- not mid-`start()` after `protocol.json` was
        already copied to disk, and not deep inside an OMCP subprocess call."""
        self.protocol.write_text(json.dumps({"study_id": "sepsis mortality"}), encoding="utf-8")
        runs_dir = Path(self.temp.name) / "runs"
        with self.assertRaises(ValueError):
            self.manager.start(self.protocol)
        self.assertEqual(list(runs_dir.iterdir()) if runs_dir.is_dir() else [], [])

    def test_start_rejects_a_study_id_shorter_than_omcps_minimum(self):
        self.protocol.write_text(json.dumps({"study_id": "ab"}), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.manager.start(self.protocol)

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

    def test_accept_with_a_domain_reaches_the_candidate_dag_entry(self):
        self.manager.review_covariate(
            self.state.run_id, "age", "accept", "Dr. Chen",
            "Institutional practice affects who receives treatment.",
            role="treatment_cause_only", temporal_classification="pre_treatment",
            domain="measurement",
        )
        hypothesis = self._hypothesis()
        entry = next(item for item in hypothesis["candidate_dag"] if item["label"] == "age")
        self.assertEqual(entry["domain"], "measurement")

    def test_accept_without_a_domain_still_works(self):
        """`domain` stays optional -- an existing caller that never supplies
        it (the overwhelming majority today) must keep working exactly as
        before, just with no OMOP mapping attempted for that covariate."""
        self.manager.review_covariate(
            self.state.run_id, "age", "accept", "Dr. Chen",
            "Institutional practice affects who receives treatment.",
            role="treatment_cause_only", temporal_classification="pre_treatment",
        )
        hypothesis = self._hypothesis()
        entry = next(item for item in hypothesis["candidate_dag"] if item["label"] == "age")
        self.assertEqual(entry["domain"], "")

    def test_accept_with_an_invalid_domain_raises(self):
        with self.assertRaises(ValueError):
            self.manager.review_covariate(
                self.state.run_id, "age", "accept", "Dr. Chen", "Some rationale.",
                role="treatment_cause_only", temporal_classification="pre_treatment",
                domain="not_a_domain",
            )

    def test_re_accepting_preserves_already_reviewed_concept_mapping_and_aggregation(self):
        """A reviewer reconsidering role/temporal_classification/rationale for
        an already-mapped covariate must not silently wipe the separate
        concept-mapping/aggregation/window review work another reviewer
        already did (`map_covariate_concepts`/`review_covariate_aggregation`/
        `review_covariate_window` each document that same "leave every other
        field untouched" contract -- `review_covariate` must honor it too)."""
        self.manager.review_covariate(
            self.state.run_id, "SOFA score", "accept", "Dr. Chen",
            "Higher SOFA is associated with treatment choice and outcome.",
            role="confounder", temporal_classification="pre_treatment",
            domain="measurement",
        )
        self.manager.map_covariate_concepts(
            self.state.run_id, "SOFA score", [4301868], "Dr. Lee",
        )
        self.manager.review_covariate_aggregation(
            self.state.run_id, "SOFA score", "last_value_before_index", "Dr. Lee",
        )
        self.manager.review_covariate_window(
            self.state.run_id, "SOFA score", -6, 0, "Dr. Lee",
        )

        # Reconsidered later: same domain, just a corrected rationale/role.
        self.manager.review_covariate(
            self.state.run_id, "SOFA score", "accept", "Dr. Patel",
            "Corrected rationale wording.",
            role="confounder", temporal_classification="pre_treatment",
        )
        entry = next(item for item in self._hypothesis()["candidate_dag"] if item["label"] == "SOFA score")
        self.assertEqual(entry["domain"], "measurement")
        self.assertEqual(entry["concept_ids"], [4301868])
        self.assertEqual(entry["mapping_status"], "approved")
        self.assertEqual(entry["aggregation"], "last_value_before_index")
        self.assertEqual(entry["window_start_hours"], -6)
        self.assertEqual(entry["window_end_hours"], 0)
        self.assertEqual(entry["rationale"], "Corrected rationale wording.")

    def test_re_accepting_with_a_changed_domain_resets_the_now_invalid_mapping(self):
        """Unlike role/rationale, `concept_ids`/`mapping_status`/`aggregation`
        are domain-specific -- carrying them over across an actual domain
        change would leave a `measurement` aggregation choice attached to a
        covariate now declared `condition`, which `COVARIATE_AGGREGATIONS_BY_
        DOMAIN` would never have allowed directly."""
        self.manager.review_covariate(
            self.state.run_id, "SOFA score", "accept", "Dr. Chen",
            "Higher SOFA is associated with treatment choice and outcome.",
            role="confounder", temporal_classification="pre_treatment",
            domain="measurement",
        )
        self.manager.map_covariate_concepts(
            self.state.run_id, "SOFA score", [4301868], "Dr. Lee",
        )
        self.manager.review_covariate_aggregation(
            self.state.run_id, "SOFA score", "last_value_before_index", "Dr. Lee",
        )

        self.manager.review_covariate(
            self.state.run_id, "SOFA score", "accept", "Dr. Patel",
            "Actually this should be tracked as a coded condition, not a lab value.",
            role="confounder", temporal_classification="pre_treatment",
            domain="condition",
        )
        entry = next(item for item in self._hypothesis()["candidate_dag"] if item["label"] == "SOFA score")
        self.assertEqual(entry["domain"], "condition")
        self.assertEqual(entry["concept_ids"], [])
        self.assertEqual(entry["mapping_status"], "unmapped")
        self.assertEqual(entry["aggregation"], "definition_pending")

    def test_reject_records_history_event(self):
        state = self.manager.review_covariate(
            self.state.run_id, "age", "reject", "Dr. Chen", "Not plausible.",
        )
        event = next(event for event in state.history if event["event"] == "covariate_reviewed")
        self.assertIsNone(event["changes"]["before"])
        self.assertEqual(event["changes"]["after"]["reviewer"], "Dr. Chen")

    def test_re_reviewing_records_the_prior_entry_as_before(self):
        """Regression for gap #8: reconsidering a covariate must let
        `state.json` show what it changed from, not just what it changed to."""
        self.manager.review_covariate(
            self.state.run_id, "age", "reject", "Dr. Chen", "First pass.",
        )
        state = self.manager.review_covariate(
            self.state.run_id, "age", "accept", "Dr. Lee", "Reconsidered as a treatment cause.",
            role="treatment_cause_only", temporal_classification="pre_treatment",
        )
        event = next(
            event for event in reversed(state.history) if event["event"] == "covariate_reviewed"
        )
        self.assertIsNone(event["changes"]["before"])  # rejection lives in rejected_covariates, not candidate_dag
        self.assertEqual(event["changes"]["after"]["role"], "treatment_cause_only")

    def test_requires_hypothesis_agent_to_have_run_first(self):
        state = self.manager.start(self.protocol)
        with self.assertRaises(RuntimeError):
            self.manager.review_covariate(state.run_id, "age", "reject", "Dr. Chen", "x")


class WorkflowMapCovariateConceptsTests(unittest.TestCase):
    """`map_covariate_concepts` is the governed way to record which OMOP
    concept(s) a reviewer picked for an already-classified covariate --
    distinct from `review_covariate` (causal role) on purpose."""

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
            "first_pass_covariates": ["SOFA score"],
            "concept_sets": {"treatment": {"status": "approved"}},
        }), encoding="utf-8")
        self.manager = WorkflowManager(root / "runs")
        state = self.manager.start(self.protocol)
        self.state = self.manager.run_hypothesis_agent(state.run_id)
        self.run_dir = root / "runs" / self.state.run_id
        self.manager.review_covariate(
            self.state.run_id, "SOFA score", "accept", "Dr. Chen",
            "Higher SOFA is associated with treatment choice and outcome.",
            role="confounder", temporal_classification="pre_treatment",
            domain="measurement",
        )

    def tearDown(self):
        self.temp.cleanup()

    def _hypothesis(self) -> dict:
        return json.loads((self.run_dir / "hypothesis.json").read_text(encoding="utf-8"))

    def test_maps_concepts_without_touching_the_causal_role(self):
        self.manager.map_covariate_concepts(
            self.state.run_id, "SOFA score", [4301868], "Dr. Lee",
        )
        entry = next(item for item in self._hypothesis()["candidate_dag"] if item["label"] == "SOFA score")
        self.assertEqual(entry["concept_ids"], [4301868])
        self.assertEqual(entry["mapping_status"], "approved")
        self.assertEqual(entry["role"], "confounder")
        self.assertEqual(entry["domain"], "measurement")

    def test_raises_when_the_covariate_was_never_accepted(self):
        with self.assertRaises(RuntimeError):
            self.manager.map_covariate_concepts(
                self.state.run_id, "lactate", [123], "Dr. Lee",
            )

    def test_raises_when_approved_with_no_concept_ids(self):
        with self.assertRaises(ValueError):
            self.manager.map_covariate_concepts(
                self.state.run_id, "SOFA score", [], "Dr. Lee",
            )

    def test_allows_a_non_approved_status_with_no_concept_ids(self):
        self.manager.map_covariate_concepts(
            self.state.run_id, "SOFA score", [], "Dr. Lee",
            mapping_status="no_standard_concept",
        )
        entry = next(item for item in self._hypothesis()["candidate_dag"] if item["label"] == "SOFA score")
        self.assertEqual(entry["mapping_status"], "no_standard_concept")

    def test_records_history_event(self):
        state = self.manager.map_covariate_concepts(
            self.state.run_id, "SOFA score", [4301868], "Dr. Lee",
        )
        event = next(event for event in state.history if event["event"] == "covariate_concepts_mapped")
        self.assertEqual(event["changes"]["before"]["concept_ids"], [])
        self.assertEqual(event["changes"]["before"]["mapping_status"], "unmapped")
        self.assertEqual(event["changes"]["after"]["concept_ids"], [4301868])
        self.assertEqual(event["changes"]["after"]["mapping_status"], "approved")

    def test_a_second_mapping_records_the_first_as_before(self):
        """Regression for gap #8: without this, `state.json` alone cannot
        reconstruct what a re-mapping actually changed."""
        self.manager.map_covariate_concepts(
            self.state.run_id, "SOFA score", [111], "Dr. Lee",
        )
        state = self.manager.map_covariate_concepts(
            self.state.run_id, "SOFA score", [222], "Dr. Lee",
        )
        event = next(
            event for event in reversed(state.history) if event["event"] == "covariate_concepts_mapped"
        )
        self.assertEqual(event["changes"]["before"]["concept_ids"], [111])
        self.assertEqual(event["changes"]["after"]["concept_ids"], [222])


class MismatchedConceptDomainsTests(unittest.TestCase):
    """Pure-parsing unit tests for the `Get_Concepts_By_Id` CSV response
    handling behind `map_covariate_concepts`'s domain-coherence check."""

    CSV = (
        "concept_id,concept_name,concept_code,vocabulary_id,domain_id\n"
        "4301868,SOFA score,x,LOINC,Measurement\n"
        "1,Albumin,x,RxNorm,Drug\n"
    )

    def test_no_mismatch_when_every_domain_matches(self):
        mismatches = _mismatched_concept_domains(self.CSV, [4301868], "measurement")
        self.assertEqual(mismatches, [])

    def test_flags_a_wrong_domain(self):
        mismatches = _mismatched_concept_domains(self.CSV, [1], "measurement")
        self.assertEqual(len(mismatches), 1)
        self.assertIn("1", mismatches[0])
        self.assertIn("Drug", mismatches[0])

    def test_flags_a_concept_id_not_found(self):
        mismatches = _mismatched_concept_domains(self.CSV, [999999], "measurement")
        self.assertEqual(len(mismatches), 1)
        self.assertIn("not found", mismatches[0])

    def test_checks_every_concept_id_independently(self):
        mismatches = _mismatched_concept_domains(self.CSV, [4301868, 1], "measurement")
        self.assertEqual(len(mismatches), 1)
        self.assertIn("1", mismatches[0])


class _FakeConceptDomainLookup:
    """Minimal `OMCPClient`-shaped fixture exposing only `get_concepts_by_id`,
    returning a fixed CSV response regardless of the ids asked for."""

    def __init__(self, csv_text: str):
        self._csv_text = csv_text
        self.calls: list[list[int]] = []

    class _Response:
        def __init__(self, text: str):
            self.text = text

    def get_concepts_by_id(self, concept_ids: list[int]):
        self.calls.append(list(concept_ids))
        return self._Response(self._csv_text)


class WorkflowMapCovariateConceptsDomainCheckTests(unittest.TestCase):
    """Gap #4: nothing previously stopped a reviewer from approving a
    concept_id whose real OMOP domain doesn't match the covariate's
    declared `domain` -- `map_covariate_concepts(..., concept_lookup=...)`
    now checks this when a live OMCP client is supplied."""

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
            "first_pass_covariates": ["SOFA score"],
            "concept_sets": {"treatment": {"status": "approved"}},
        }), encoding="utf-8")
        self.manager = WorkflowManager(root / "runs")
        state = self.manager.start(self.protocol)
        self.state = self.manager.run_hypothesis_agent(state.run_id)
        self.run_dir = root / "runs" / self.state.run_id
        self.manager.review_covariate(
            self.state.run_id, "SOFA score", "accept", "Dr. Chen",
            "Higher SOFA is associated with treatment choice and outcome.",
            role="confounder", temporal_classification="pre_treatment",
            domain="measurement",
        )

    def tearDown(self):
        self.temp.cleanup()

    def _hypothesis(self) -> dict:
        return json.loads((self.run_dir / "hypothesis.json").read_text(encoding="utf-8"))

    def test_accepts_a_concept_id_with_a_matching_domain(self):
        lookup = _FakeConceptDomainLookup(
            "concept_id,concept_name,concept_code,vocabulary_id,domain_id\n"
            "4301868,SOFA score,x,LOINC,Measurement\n"
        )
        self.manager.map_covariate_concepts(
            self.state.run_id, "SOFA score", [4301868], "Dr. Lee", concept_lookup=lookup,
        )
        entry = next(item for item in self._hypothesis()["candidate_dag"] if item["label"] == "SOFA score")
        self.assertEqual(entry["concept_ids"], [4301868])
        self.assertEqual(lookup.calls, [[4301868]])

    def test_rejects_a_concept_id_from_the_wrong_domain(self):
        lookup = _FakeConceptDomainLookup(
            "concept_id,concept_name,concept_code,vocabulary_id,domain_id\n"
            "1,Albumin,x,RxNorm,Drug\n"
        )
        with self.assertRaises(ValueError):
            self.manager.map_covariate_concepts(
                self.state.run_id, "SOFA score", [1], "Dr. Lee", concept_lookup=lookup,
            )
        # Rejected before ever writing -- the prior (empty) mapping must be untouched.
        entry = next(item for item in self._hypothesis()["candidate_dag"] if item["label"] == "SOFA score")
        self.assertEqual(entry.get("concept_ids", []), [])

    def test_skipped_entirely_without_a_concept_lookup(self):
        """Regression: this method must stay filesystem-only/offline by
        default -- no OMCPClient means no domain check at all, exactly like
        before this feature existed."""
        self.manager.map_covariate_concepts(
            self.state.run_id, "SOFA score", [1], "Dr. Lee",
        )
        entry = next(item for item in self._hypothesis()["candidate_dag"] if item["label"] == "SOFA score")
        self.assertEqual(entry["concept_ids"], [1])

    def test_skipped_for_a_non_approved_mapping_status(self):
        lookup = _FakeConceptDomainLookup("concept_id,concept_name,concept_code,vocabulary_id,domain_id\n")
        self.manager.map_covariate_concepts(
            self.state.run_id, "SOFA score", [], "Dr. Lee",
            mapping_status="no_standard_concept", concept_lookup=lookup,
        )
        self.assertEqual(lookup.calls, [])


class WorkflowMapConceptSetTests(unittest.TestCase):
    """`map_concept_set` is the governed way to record a `concept_sets[name]`
    phenotype's OMOP domain + concept(s) -- the step `DataScienceAgent`
    requires before it will build an analysis plan at all (see
    `_phenotype_from_concept_set`'s domain validation), and also the only
    concept-mapping step: `_load_concept_specs` (feasibility/cohort SQL
    generation) resolves its `ConceptSetSpec`s from this same `concept_sets`
    store, by role -- there is no longer a separate `query_concept_sets`
    mapping step for the same role."""

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
            "concept_sets": {
                "population": {
                    "role": "population", "status": "pending_omop_mapping",
                    "concept_ids": [], "domain": "",
                },
                "outcome": {
                    "role": "outcome", "status": "pending_omop_mapping",
                    "concept_ids": [], "domain": "",
                },
            },
        }), encoding="utf-8")
        self.manager = WorkflowManager(root / "runs")
        state = self.manager.start(self.protocol)
        self.state = self.manager.run_hypothesis_agent(state.run_id)
        self.run_dir = root / "runs" / self.state.run_id

    def tearDown(self):
        self.temp.cleanup()

    def _protocol(self) -> dict:
        return json.loads((self.run_dir / "protocol.json").read_text(encoding="utf-8"))

    def _hypothesis(self) -> dict:
        return json.loads((self.run_dir / "hypothesis.json").read_text(encoding="utf-8"))

    def test_maps_domain_and_concepts_without_touching_role(self):
        self.manager.map_concept_set(
            self.state.run_id, "population", "condition", [132797], "Dr. Chen",
        )
        entry = self._protocol()["concept_sets"]["population"]
        self.assertEqual(entry["domain"], "condition")
        self.assertEqual(entry["concept_ids"], [132797])
        self.assertEqual(entry["status"], "approved")
        self.assertEqual(entry["role"], "population")  # untouched

    def test_regenerates_hypothesis_so_unresolved_issues_clear(self):
        before = self._hypothesis()["unresolved_issues"]
        self.assertIn("concept_set:population:pending_omop_mapping", before)
        self.manager.map_concept_set(
            self.state.run_id, "population", "condition", [132797], "Dr. Chen",
        )
        self.assertNotIn("concept_set:population:pending_omop_mapping", self._hypothesis()["unresolved_issues"])

    def test_rejects_an_unknown_domain(self):
        with self.assertRaises(ValueError):
            self.manager.map_concept_set(
                self.state.run_id, "population", "not_a_domain", [132797], "Dr. Chen",
            )

    def test_approved_status_requires_concept_ids(self):
        with self.assertRaises(ValueError):
            self.manager.map_concept_set(
                self.state.run_id, "population", "condition", [], "Dr. Chen",
            )

    def test_a_non_approved_status_allows_no_concept_ids(self):
        self.manager.map_concept_set(
            self.state.run_id, "population", "condition", [], "Dr. Chen",
            status="no_standard_concept",
        )
        entry = self._protocol()["concept_sets"]["population"]
        self.assertEqual(entry["status"], "no_standard_concept")

    def test_raises_for_an_unknown_concept_set_name(self):
        with self.assertRaises(RuntimeError):
            self.manager.map_concept_set(
                self.state.run_id, "not_declared", "condition", [1], "Dr. Chen",
            )

    def test_rejects_a_concept_id_from_the_wrong_domain_when_checked(self):
        lookup = _FakeConceptDomainLookup(
            "concept_id,concept_name,concept_code,vocabulary_id,domain_id\n"
            "1,Albumin,x,RxNorm,Drug\n"
        )
        with self.assertRaises(ValueError):
            self.manager.map_concept_set(
                self.state.run_id, "population", "condition", [1], "Dr. Chen",
                concept_lookup=lookup,
            )

    def test_records_history_event(self):
        state = self.manager.map_concept_set(
            self.state.run_id, "population", "condition", [132797], "Dr. Chen",
        )
        event = next(event for event in state.history if event["event"] == "concept_set_mapped")
        self.assertEqual(event["changes"]["before"]["status"], "pending_omop_mapping")
        self.assertEqual(event["changes"]["after"]["status"], "approved")

    def test_outcome_role_accepts_death_with_no_concept_ids(self):
        self.manager.map_concept_set(
            self.state.run_id, "outcome", "death", [], "Dr. Chen",
        )
        entry = self._protocol()["concept_sets"]["outcome"]
        self.assertEqual(entry["domain"], "death")
        self.assertEqual(entry["concept_ids"], [])
        self.assertEqual(entry["status"], "approved")

    def test_death_domain_is_rejected_for_a_non_outcome_role(self):
        with self.assertRaises(ValueError):
            self.manager.map_concept_set(
                self.state.run_id, "population", "death", [], "Dr. Chen",
            )


class LoadConceptSpecsTests(unittest.TestCase):
    """`_load_concept_specs` resolves the `ConceptSetSpec`s every SQL
    generator (`feasibility_sql`, `AnalyticCohortBuilder.build`,
    `profile_sql`, `followup_audit_sql`) reads, directly from `concept_sets`
    by role -- unified from the formerly separate, independently-mapped
    `query_concept_sets` store (see `map_concept_set`'s docstring)."""

    @staticmethod
    def _approved(role: str, domain: str, concept_ids: list[int]) -> dict:
        return {
            "role": role, "status": "approved", "domain": domain, "concept_ids": concept_ids,
        }

    def test_resolves_each_role_directly(self):
        specs = _load_concept_specs({"concept_sets": {
            "sepsis": self._approved("population", "condition", [1]),
            "albumin": self._approved("treatment", "drug", [2]),
            "crystalloids_alone": self._approved("comparator", "drug", [3]),
            "death_outcome": self._approved("outcome", "death", []),
        }})
        self.assertEqual(specs["population"], ConceptSetSpec(domain="condition", concept_ids=(1,)))
        self.assertEqual(specs["treatment"], ConceptSetSpec(domain="drug", concept_ids=(2,)))
        self.assertEqual(specs["comparator"], ConceptSetSpec(domain="drug", concept_ids=(3,)))
        self.assertEqual(specs["outcome"], ConceptSetSpec(domain="death", concept_ids=()))

    def test_treatment_and_comparator_shared_backbone_resolves_both_roles(self):
        """Mirrors `studies/albumin_sepsis_28d/protocol.json`'s "crystalloids"
        entry: one phenotype (role='treatment_and_comparator') serves as
        both arms of an `add_on` design."""
        specs = _load_concept_specs({"concept_sets": {
            "sepsis": self._approved("population", "condition", [1]),
            "crystalloids": self._approved("treatment_and_comparator", "drug", [2]),
        }})
        self.assertEqual(specs["treatment"], ConceptSetSpec(domain="drug", concept_ids=(2,)))
        self.assertEqual(specs["comparator"], ConceptSetSpec(domain="drug", concept_ids=(2,)))

    def test_an_exact_role_entry_takes_priority_over_the_fallback(self):
        specs = _load_concept_specs({"concept_sets": {
            "sepsis": self._approved("population", "condition", [1]),
            "albumin": self._approved("treatment", "drug", [2]),
            "crystalloids": self._approved("treatment_and_comparator", "drug", [3]),
        }})
        self.assertEqual(specs["treatment"], ConceptSetSpec(domain="drug", concept_ids=(2,)))
        self.assertEqual(specs["comparator"], ConceptSetSpec(domain="drug", concept_ids=(3,)))

    def test_missing_outcome_role_defaults_to_death(self):
        """Matches `DataScienceAgent.run`'s own "all_cause_death" fallback
        when no phenotype declares role='outcome' -- both readers of
        `concept_sets` must agree on what an omitted outcome role means."""
        specs = _load_concept_specs({"concept_sets": {
            "sepsis": self._approved("population", "condition", [1]),
            "albumin": self._approved("treatment", "drug", [2]),
            "crystalloids": self._approved("comparator", "drug", [3]),
        }})
        self.assertEqual(specs["outcome"], ConceptSetSpec(domain="death", concept_ids=()))

    def test_missing_population_role_raises(self):
        with self.assertRaises(RuntimeError):
            _load_concept_specs({"concept_sets": {
                "albumin": self._approved("treatment", "drug", [2]),
                "crystalloids": self._approved("comparator", "drug", [3]),
            }})

    def test_an_unapproved_entry_raises_rather_than_falling_back_to_death(self):
        """An outcome entry that exists but isn't approved yet must block --
        silently defaulting to death here would bypass the approval gate."""
        with self.assertRaises(RuntimeError):
            _load_concept_specs({"concept_sets": {
                "sepsis": self._approved("population", "condition", [1]),
                "albumin": self._approved("treatment", "drug", [2]),
                "crystalloids": self._approved("comparator", "drug", [3]),
                "mortality": {
                    "role": "outcome", "status": "pending_omop_mapping", "domain": "", "concept_ids": [],
                },
            }})

    def test_two_entries_declaring_the_same_role_raises(self):
        with self.assertRaises(RuntimeError):
            _load_concept_specs({"concept_sets": {
                "sepsis_v1": self._approved("population", "condition", [1]),
                "sepsis_v2": self._approved("population", "condition", [2]),
                "albumin": self._approved("treatment", "drug", [3]),
                "crystalloids": self._approved("comparator", "drug", [4]),
            }})

    def test_missing_concept_sets_raises(self):
        with self.assertRaises(RuntimeError):
            _load_concept_specs({})


class WorkflowReviewCovariateDomainTests(unittest.TestCase):
    """`review_covariate_domain` lets a reviewer confirm or correct a
    covariate's OMOP domain -- either one the DAG-drafting LLM proposed, or
    one a human already set via `review_covariate` -- without re-supplying
    role/temporal_classification/rationale."""

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
            "first_pass_covariates": ["SOFA score", "weight"],
            "concept_sets": {"treatment": {"status": "approved"}},
        }), encoding="utf-8")
        self.manager = WorkflowManager(root / "runs")
        state = self.manager.start(self.protocol)
        self.state = self.manager.run_hypothesis_agent(state.run_id)
        self.run_dir = root / "runs" / self.state.run_id
        self.manager.review_covariate(
            self.state.run_id, "SOFA score", "accept", "Dr. Chen",
            "Higher SOFA is associated with treatment choice and outcome.",
            role="confounder", temporal_classification="pre_treatment",
            domain="measurement",
        )
        self.manager.review_covariate(
            self.state.run_id, "weight", "accept", "Dr. Chen",
            "Weight is associated with treatment choice and outcome.",
            role="confounder", temporal_classification="pre_treatment",
        )

    def tearDown(self):
        self.temp.cleanup()

    def _hypothesis(self) -> dict:
        return json.loads((self.run_dir / "hypothesis.json").read_text(encoding="utf-8"))

    def test_sets_domain_for_a_covariate_with_none_yet(self):
        self.manager.review_covariate_domain(
            self.state.run_id, "weight", "measurement", "Dr. Lee",
        )
        entry = next(item for item in self._hypothesis()["candidate_dag"] if item["label"] == "weight")
        self.assertEqual(entry["domain"], "measurement")
        self.assertEqual(entry["role"], "confounder")

    def test_confirming_the_same_domain_does_not_reset_mapping(self):
        self.manager.map_covariate_concepts(
            self.state.run_id, "SOFA score", [4301868], "Dr. Lee",
        )
        self.manager.review_covariate_domain(
            self.state.run_id, "SOFA score", "measurement", "Dr. Lee",
        )
        entry = next(item for item in self._hypothesis()["candidate_dag"] if item["label"] == "SOFA score")
        self.assertEqual(entry["concept_ids"], [4301868])
        self.assertEqual(entry["mapping_status"], "approved")

    def test_changing_domain_resets_concept_mapping_and_aggregation(self):
        self.manager.map_covariate_concepts(
            self.state.run_id, "SOFA score", [4301868], "Dr. Lee",
        )
        self.manager.review_covariate_aggregation(
            self.state.run_id, "SOFA score", "last_value_before_index", "Dr. Lee",
        )
        self.manager.review_covariate_domain(
            self.state.run_id, "SOFA score", "observation", "Dr. Lee",
        )
        entry = next(item for item in self._hypothesis()["candidate_dag"] if item["label"] == "SOFA score")
        self.assertEqual(entry["domain"], "observation")
        self.assertEqual(entry["concept_ids"], [])
        self.assertEqual(entry["mapping_status"], "unmapped")
        self.assertEqual(entry["aggregation"], "definition_pending")

    def test_does_not_touch_window(self):
        self.manager.review_covariate_window(
            self.state.run_id, "SOFA score", -6, 0, "Dr. Lee",
        )
        self.manager.review_covariate_domain(
            self.state.run_id, "SOFA score", "observation", "Dr. Lee",
        )
        entry = next(item for item in self._hypothesis()["candidate_dag"] if item["label"] == "SOFA score")
        self.assertEqual(entry["window_start_hours"], -6)
        self.assertEqual(entry["window_end_hours"], 0)

    def test_raises_when_the_covariate_was_never_accepted(self):
        with self.assertRaises(RuntimeError):
            self.manager.review_covariate_domain(
                self.state.run_id, "unknown_covariate", "measurement", "Dr. Lee",
            )

    def test_rejects_an_unknown_domain(self):
        with self.assertRaises(ValueError):
            self.manager.review_covariate_domain(
                self.state.run_id, "weight", "not_a_real_domain", "Dr. Lee",
            )

    def test_records_history_event(self):
        state = self.manager.review_covariate_domain(
            self.state.run_id, "weight", "measurement", "Dr. Lee",
        )
        event = next(event for event in state.history if event["event"] == "covariate_domain_reviewed")
        self.assertEqual(event["changes"]["before"]["domain"], "")
        self.assertEqual(event["changes"]["after"]["domain"], "measurement")


class WorkflowReviewCovariateAggregationTests(unittest.TestCase):
    """`review_covariate_aggregation` is the governed way to record how to
    combine repeated occurrences of an already-mapped covariate's concept(s)
    -- distinct from `map_covariate_concepts` (which concepts) on purpose."""

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
            "first_pass_covariates": ["SOFA score", "weight"],
            "concept_sets": {"treatment": {"status": "approved"}},
        }), encoding="utf-8")
        self.manager = WorkflowManager(root / "runs")
        state = self.manager.start(self.protocol)
        self.state = self.manager.run_hypothesis_agent(state.run_id)
        self.run_dir = root / "runs" / self.state.run_id
        self.manager.review_covariate(
            self.state.run_id, "SOFA score", "accept", "Dr. Chen",
            "Higher SOFA is associated with treatment choice and outcome.",
            role="confounder", temporal_classification="pre_treatment",
            domain="measurement",
        )
        self.manager.review_covariate(
            self.state.run_id, "weight", "accept", "Dr. Chen",
            "Weight is associated with treatment choice and outcome.",
            role="confounder", temporal_classification="pre_treatment",
        )

    def tearDown(self):
        self.temp.cleanup()

    def _hypothesis(self) -> dict:
        return json.loads((self.run_dir / "hypothesis.json").read_text(encoding="utf-8"))

    def test_reviews_aggregation_without_touching_concept_mapping(self):
        self.manager.map_covariate_concepts(
            self.state.run_id, "SOFA score", [4301868], "Dr. Lee",
        )
        self.manager.review_covariate_aggregation(
            self.state.run_id, "SOFA score", "last_value_before_index", "Dr. Lee",
        )
        entry = next(item for item in self._hypothesis()["candidate_dag"] if item["label"] == "SOFA score")
        self.assertEqual(entry["aggregation"], "last_value_before_index")
        self.assertEqual(entry["concept_ids"], [4301868])
        self.assertEqual(entry["role"], "confounder")

    def test_raises_when_the_covariate_was_never_accepted(self):
        with self.assertRaises(RuntimeError):
            self.manager.review_covariate_aggregation(
                self.state.run_id, "lactate", "last_value_before_index", "Dr. Lee",
            )

    def test_raises_when_domain_was_never_set(self):
        with self.assertRaises(RuntimeError):
            self.manager.review_covariate_aggregation(
                self.state.run_id, "weight", "last_value_before_index", "Dr. Lee",
            )

    def test_raises_when_aggregation_incompatible_with_domain(self):
        with self.assertRaises(ValueError):
            self.manager.review_covariate_aggregation(
                self.state.run_id, "SOFA score", "presence_in_window", "Dr. Lee",
            )

    def test_records_history_event(self):
        state = self.manager.review_covariate_aggregation(
            self.state.run_id, "SOFA score", "last_value_before_index", "Dr. Lee",
        )
        event = next(event for event in state.history if event["event"] == "covariate_aggregation_reviewed")
        self.assertEqual(event["changes"]["before"]["aggregation"], "definition_pending")
        self.assertEqual(event["changes"]["after"]["aggregation"], "last_value_before_index")


class WorkflowReviewCovariateWindowTests(unittest.TestCase):
    """`review_covariate_window` lets a reviewer override the auto-derived
    ascertainment window -- there was previously no mechanism to correct it
    for a covariate that genuinely needs a window other than the two
    temporal_classification-derived defaults."""

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
            "first_pass_covariates": ["lactate"],
            "concept_sets": {"treatment": {"status": "approved"}},
        }), encoding="utf-8")
        self.manager = WorkflowManager(root / "runs")
        state = self.manager.start(self.protocol)
        self.state = self.manager.run_hypothesis_agent(state.run_id)
        self.run_dir = root / "runs" / self.state.run_id
        self.manager.review_covariate(
            self.state.run_id, "lactate", "accept", "Dr. Chen",
            "Lactate is associated with treatment choice and outcome.",
            role="confounder", temporal_classification="pre_treatment",
        )

    def tearDown(self):
        self.temp.cleanup()

    def _hypothesis(self) -> dict:
        return json.loads((self.run_dir / "hypothesis.json").read_text(encoding="utf-8"))

    def test_overrides_the_window_without_touching_other_fields(self):
        self.manager.review_covariate_window(
            self.state.run_id, "lactate", -6, 0, "Dr. Lee",
        )
        entry = next(item for item in self._hypothesis()["candidate_dag"] if item["label"] == "lactate")
        self.assertEqual(entry["window_start_hours"], -6)
        self.assertEqual(entry["window_end_hours"], 0)
        self.assertEqual(entry["role"], "confounder")

    def test_raises_when_the_covariate_was_never_accepted(self):
        with self.assertRaises(RuntimeError):
            self.manager.review_covariate_window(
                self.state.run_id, "unknown_covariate", -6, 0, "Dr. Lee",
            )

    def test_raises_when_window_end_is_after_index(self):
        with self.assertRaises(ValueError):
            self.manager.review_covariate_window(
                self.state.run_id, "lactate", -6, 6, "Dr. Lee",
            )

    def test_raises_when_start_is_after_end(self):
        with self.assertRaises(ValueError):
            self.manager.review_covariate_window(
                self.state.run_id, "lactate", 0, -6, "Dr. Lee",
            )

    def test_records_structured_history_event(self):
        state = self.manager.review_covariate_window(
            self.state.run_id, "lactate", -6, 0, "Dr. Lee",
        )
        event = next(event for event in state.history if event["event"] == "covariate_window_reviewed")
        self.assertIsNone(event["changes"]["before"]["window_start_hours"])
        self.assertEqual(event["changes"]["after"]["window_start_hours"], -6)
        self.assertEqual(event["changes"]["after"]["window_end_hours"], 0)


class RunLockTests(unittest.TestCase):
    """`_RunLock` is what turns two reviewers editing the same run's
    `protocol.json` at the same time from a silent last-write-wins race into
    a bounded wait / clear failure (see `WorkflowMapCovariateConceptsTests`
    and friends for the read-modify-write methods it protects)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_acquire_and_release_removes_the_lock_file(self):
        lock_path = self.run_dir / ".review.lock"
        with _RunLock(self.run_dir):
            self.assertTrue(lock_path.is_file())
        self.assertFalse(lock_path.exists())

    def test_second_acquire_times_out_while_held(self):
        lock_path = self.run_dir / ".review.lock"
        lock_path.touch()
        with self.assertRaises(TimeoutError):
            with _RunLock(self.run_dir, timeout_seconds=0.2, poll_seconds=0.02):
                pass

    def test_a_stale_lock_is_stolen_instead_of_blocking_forever(self):
        """A lock file left behind by a crashed process must not permanently
        block every future review on this run."""
        lock_path = self.run_dir / ".review.lock"
        lock_path.touch()
        stale_time = time.time() - _LOCK_STALE_SECONDS - 1
        os.utime(lock_path, (stale_time, stale_time))
        with _RunLock(self.run_dir, timeout_seconds=1.0, poll_seconds=0.02):
            pass  # Must not raise TimeoutError.

    def test_lock_is_released_even_if_the_critical_section_raises(self):
        lock_path = self.run_dir / ".review.lock"
        with self.assertRaises(ValueError):
            with _RunLock(self.run_dir):
                raise ValueError("boom")
        self.assertFalse(lock_path.exists())


class WorkflowConcurrentCovariateReviewTests(unittest.TestCase):
    """Two reviewers racing to `map_covariate_concepts` on the same
    covariate must not corrupt `protocol.json` -- exactly one write must
    survive intact, never an interleaved/truncated write from both."""

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
            "first_pass_covariates": ["SOFA score"],
            "concept_sets": {"treatment": {"status": "approved"}},
        }), encoding="utf-8")
        self.manager = WorkflowManager(root / "runs")
        state = self.manager.start(self.protocol)
        self.state = self.manager.run_hypothesis_agent(state.run_id)
        self.run_dir = root / "runs" / self.state.run_id
        self.manager.review_covariate(
            self.state.run_id, "SOFA score", "accept", "Dr. Chen",
            "Higher SOFA is associated with treatment choice and outcome.",
            role="confounder", temporal_classification="pre_treatment",
            domain="measurement",
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_concurrent_mappings_do_not_corrupt_protocol_json(self):
        errors: list[Exception] = []

        def _map(concept_id: int) -> None:
            try:
                self.manager.map_covariate_concepts(
                    self.state.run_id, "SOFA score", [concept_id], f"Dr. {concept_id}",
                )
            except Exception as exc:  # noqa: BLE001 -- surfaced via `errors`, not raised in-thread
                errors.append(exc)

        threads = [threading.Thread(target=_map, args=(cid,)) for cid in (111, 222)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        protocol = json.loads((self.run_dir / "protocol.json").read_text(encoding="utf-8"))
        entry = next(item for item in protocol["candidate_dag"] if item["label"] == "SOFA score")
        self.assertIn(entry["concept_ids"], ([111], [222]))
        self.assertFalse((self.run_dir / ".review.lock").exists())


class _FakeConceptLookup:
    """Minimal `ConceptLookup` fixture -- same shape as
    `test_data_science_agent.DataScienceAgentTests._FakeLookup`, duplicated
    here rather than imported since that one is a TestCase-nested class."""

    class _Response:
        text = (
            "concept_id,concept_name,concept_code,vocabulary_id,domain_id\n"
            "4301868,Found concept,x,VOCAB,Measurement\n"
        )

    def lookup_drug(self, term):
        return self._Response()

    def lookup_condition(self, term):
        return self._Response()

    def lookup_concept(self, term, domain):
        return self._Response()


class WorkflowConceptCandidatePersistenceTests(unittest.TestCase):
    """Gap #6: `concept_candidates` used to be regenerated from scratch on
    every `run_data_science_agent` call and silently dropped whenever
    `--with-mcp` (a live `concept_lookup`) was not passed -- a reviewer
    could lose the candidate list they were working from just by re-running
    `workflow plan` without the flag."""

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
            "data_source": "project.dataset",
            "first_pass_covariates": ["SOFA score"],
            "concept_sets": {},
        }), encoding="utf-8")
        self.manager = WorkflowManager(root / "runs")
        state = self.manager.start(self.protocol)
        self.state = self.manager.run_hypothesis_agent(state.run_id)
        self.run_dir = root / "runs" / self.state.run_id
        self.manager.review_covariate(
            self.state.run_id, "SOFA score", "accept", "Dr. Chen",
            "Higher SOFA is associated with treatment choice and outcome.",
            role="confounder", temporal_classification="pre_treatment",
            domain="measurement",
        )
        self.manager.decide_gate(self.state.run_id, "protocol", "approved", "tester")
        self.state = self.manager.advance(self.state.run_id)

    def tearDown(self):
        self.temp.cleanup()

    def _plan(self) -> dict:
        return json.loads((self.run_dir / "analysis_plan.json").read_text(encoding="utf-8"))

    def test_first_run_without_mcp_has_no_candidates(self):
        state = self.manager.run_data_science_agent(self.state.run_id)
        covariate = next(c for c in self._plan()["covariates"] if c["name"] == "SOFA score")
        self.assertEqual(covariate["concept_candidates"], [])
        self.assertIn("mapping_source=protocol_only", state.history[-1]["detail"])

    def test_a_later_run_without_mcp_carries_forward_prior_candidates(self):
        self.manager.run_data_science_agent(self.state.run_id, concept_lookup=_FakeConceptLookup())
        covariate = next(c for c in self._plan()["covariates"] if c["name"] == "SOFA score")
        self.assertEqual(len(covariate["concept_candidates"]), 1)

        state = self.manager.run_data_science_agent(self.state.run_id)
        covariate = next(c for c in self._plan()["covariates"] if c["name"] == "SOFA score")
        self.assertEqual(len(covariate["concept_candidates"]), 1)
        self.assertEqual(covariate["concept_candidates"][0]["concept_id"], 4301868)
        self.assertIn("mapping_source=carried_forward_candidates", state.history[-1]["detail"])

    def test_a_fresh_mcp_lookup_is_never_overridden_by_carry_forward(self):
        """Carry-forward must only fill in when nothing was looked up this
        run -- it must never mask a genuine empty result from a live
        lookup."""
        self.manager.run_data_science_agent(self.state.run_id, concept_lookup=_FakeConceptLookup())

        class _EmptyLookup(_FakeConceptLookup):
            class _Response:
                text = "concept_id,concept_name,concept_code,vocabulary_id,domain_id\n"

        self.manager.run_data_science_agent(self.state.run_id, concept_lookup=_EmptyLookup())
        covariate = next(c for c in self._plan()["covariates"] if c["name"] == "SOFA score")
        self.assertEqual(covariate["concept_candidates"], [])


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


class WorkflowDataScienceAgentTests(unittest.TestCase):
    """`run_data_science_agent` rebuilds the DAG's `AdjustmentSetSummary` from
    `protocol` fresh (see `review_dag` call in
    `WorkflowManager.run_data_science_agent`) and passes it into
    `DataScienceAgent.run`, so `analysis_plan.json`'s covariates carry real
    core/alternative/unused adjustment-role provenance -- exercised end to
    end here since this method previously had no direct test coverage."""

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
            "data_source": "project.dataset",
            "first_pass_covariates": ["age"],
            "concept_sets": {},
        }), encoding="utf-8")
        self.manager = WorkflowManager(root / "runs")
        state = self.manager.start(self.protocol)
        self.state = self.manager.run_hypothesis_agent(state.run_id)
        self.run_dir = root / "runs" / self.state.run_id

    def tearDown(self):
        self.temp.cleanup()

    def test_analysis_plan_covariate_carries_the_real_adjustment_role(self):
        self.manager.review_covariate(
            self.state.run_id, "age", "accept", "Dr. Chen",
            "Baseline age affects both treatment choice and outcome.",
            role="confounder", temporal_classification="pre_treatment",
        )
        self.manager.decide_gate(self.state.run_id, "protocol", "approved", "tester")
        state = self.manager.advance(self.state.run_id)
        self.assertEqual(state.phase, "concept_mapping")

        state = self.manager.run_data_science_agent(state.run_id)
        self.assertIn("analysis_plan.json", state.artifacts)
        plan = json.loads((self.run_dir / "analysis_plan.json").read_text(encoding="utf-8"))
        covariate = next(item for item in plan["covariates"] if item["name"] == "age")
        self.assertEqual(covariate["causal_role"], "confounder")
        # "age" is the DAG's only confounder, so it is in every minimum
        # backdoor adjustment set -- "core", not "alternative"/"unused".
        self.assertEqual(covariate["adjustment_role"], "core")


class WorkflowBuildCohortPlanCovariateTests(unittest.TestCase):
    """`build_cohort_plan` is the one place a reviewed, mapped, and
    aggregation-reviewed covariate actually becomes part of the generated
    SQL -- everything upstream of it (DAG review, concept mapping,
    aggregation review) is still just a review trail until this runs. Builds
    a full real pipeline (not a hand-written analysis_plan.json) to exercise
    the whole DAGVariable -> CovariatePlan -> CovariateSpec -> SQL chain."""

    def setUp(self):
        from causal.config import BigQueryDatasets

        self.datasets = BigQueryDatasets()
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
            "data_source": "project.dataset",
            "first_pass_covariates": ["SOFA score"],
            "concept_sets": {
                "population": {
                    "role": "population", "status": "approved",
                    "domain": "condition", "concept_ids": [132797],
                },
                "treatment": {
                    "role": "treatment", "status": "approved",
                    "domain": "drug", "concept_ids": [1],
                },
                "comparator": {
                    "role": "comparator", "status": "approved",
                    "domain": "drug", "concept_ids": [2],
                },
                "outcome": {
                    "role": "outcome", "status": "approved",
                    "domain": "condition", "concept_ids": [316866],
                },
            },
        }), encoding="utf-8")
        self.manager = WorkflowManager(root / "runs")
        state = self.manager.start(self.protocol)
        state = self.manager.run_hypothesis_agent(state.run_id)
        self.run_id = state.run_id
        self.run_dir = root / "runs" / self.run_id

        self.manager.review_covariate(
            self.run_id, "SOFA score", "accept", "Dr. Chen",
            "Higher SOFA is associated with treatment choice and outcome.",
            role="confounder", temporal_classification="pre_treatment",
            domain="measurement",
        )
        self.manager.map_covariate_concepts(
            self.run_id, "SOFA score", [4301868], "Dr. Lee",
        )
        self.manager.review_covariate_aggregation(
            self.run_id, "SOFA score", "last_value_before_index", "Dr. Lee",
        )
        self.manager.decide_gate(self.run_id, "protocol", "approved", "tester")
        state = self.manager.advance(self.run_id)
        state = self.manager.run_data_science_agent(self.run_id)
        state.phase = "analysis_plan"
        self.manager.save(state)

    def tearDown(self):
        self.temp.cleanup()

    def test_ready_covariate_becomes_a_real_sql_column(self):
        plan = json.loads((self.run_dir / "analysis_plan.json").read_text(encoding="utf-8"))
        covariate = next(item for item in plan["covariates"] if item["name"] == "SOFA score")
        self.assertEqual(covariate["status"], "ready")

        self.manager.build_cohort_plan(self.run_id, self.datasets)
        sql = (self.run_dir / "cohort.sql").read_text(encoding="utf-8")
        self.assertIn("cov_sofa_score", sql)
        self.assertIn("value_as_number", sql)
        manifest = json.loads((self.run_dir / "cohort_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(any(c["name"] == "cov_sofa_score" for c in manifest["columns"]))
        self.assertNotIn(
            "non-demographic baseline covariates are not implemented", manifest["unresolved_issues"],
        )

    def test_a_not_ready_covariate_is_skipped(self):
        plan_path = self.run_dir / "analysis_plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        for covariate in plan["covariates"]:
            if covariate["name"] == "SOFA score":
                covariate["status"] = "definition_pending"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")

        self.manager.build_cohort_plan(self.run_id, self.datasets)
        sql = (self.run_dir / "cohort.sql").read_text(encoding="utf-8")
        self.assertNotIn("cov_sofa_score", sql)


class WorkflowVariableRegistryTests(unittest.TestCase):
    """`build_variable_registry` only needs `protocol.json` -- it must not
    depend on the Data Science Agent or on any OMOP data being reachable
    (see `docs/causal_roadmap.md`)."""

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
            "data_source": "project.dataset",
            "first_pass_covariates": ["SOFA score"],
            "concept_sets": {
                "population": {
                    "role": "population", "status": "approved",
                    "domain": "condition", "concept_ids": [132797],
                },
                "treatment": {
                    "role": "treatment", "status": "approved",
                    "domain": "drug", "concept_ids": [1],
                },
                "comparator": {
                    "role": "comparator", "status": "approved",
                    "domain": "drug", "concept_ids": [2],
                },
                "outcome": {
                    "role": "outcome", "status": "approved",
                    "domain": "condition", "concept_ids": [316866],
                },
            },
        }), encoding="utf-8")
        self.manager = WorkflowManager(root / "runs")
        state = self.manager.start(self.protocol)
        self.run_id = state.run_id
        self.run_dir = root / "runs" / self.run_id

    def tearDown(self):
        self.temp.cleanup()

    def test_builds_registry_from_protocol_alone_without_the_data_science_agent(self):
        state = self.manager.build_variable_registry(self.run_id)
        registry_path = self.run_dir / "registry.json"
        self.assertTrue(registry_path.is_file())
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        self.assertEqual(registry["study_id"], "test_study")
        self.assertEqual(len(registry["variables"]), 4)
        self.assertIn("registry.json", state.artifacts)
        self.assertTrue(any(entry["event"] == "registry_built" for entry in state.history))

    def test_registry_reflects_an_approved_covariate_after_dag_review(self):
        self.manager.run_hypothesis_agent(self.run_id)
        self.manager.review_covariate(
            self.run_id, "SOFA score", "accept", "Dr. Chen",
            "Higher SOFA is associated with treatment choice and outcome.",
            role="confounder", temporal_classification="pre_treatment",
            domain="measurement",
        )
        self.manager.map_covariate_concepts(self.run_id, "SOFA score", [4301868], "Dr. Lee")
        self.manager.review_covariate_aggregation(
            self.run_id, "SOFA score", "last_value_before_index", "Dr. Lee",
        )
        self.manager.build_variable_registry(self.run_id)
        registry = json.loads((self.run_dir / "registry.json").read_text(encoding="utf-8"))
        covariate = next(v for v in registry["variables"] if v["dag_node_label"] == "SOFA score")
        self.assertEqual(covariate["mapping_status"], "approved")
        self.assertEqual(covariate["root_concept_ids"], [4301868])

    def test_registry_carries_approved_root_concepts_with_no_stray_candidates(self):
        """Every declared `concept_sets` entry here is already
        `status="approved"`, so the registry must resolve each straight to
        `root_concept_ids` with nothing left in `candidate_concept_ids` --
        the pending-vs-approved distinction itself (a `candidate_pending_
        human_approval` entry never becoming a root concept) is covered in
        `tests/test_omop_variable_registry.py::ConceptSetConversionTests`.
        """
        self.manager.build_variable_registry(self.run_id)
        registry = json.loads((self.run_dir / "registry.json").read_text(encoding="utf-8"))
        for variable in registry["variables"]:
            self.assertEqual(variable["mapping_status"], "approved")
            self.assertEqual(variable["candidate_concept_ids"], [])


if __name__ == "__main__":
    unittest.main()
