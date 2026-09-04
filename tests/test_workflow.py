import io
import json
import os
import random
import tempfile
import threading
import time
import unittest
from pathlib import Path

import pandas as pd

from causal.config import BigQueryDatasets
from causal.mcp_client import MCPToolResponse
from causal.companion.companion import CompanionMessage
from causal.orchestrator import WorkflowManager
from causal.provenance import artifact_metadata
from causal.orchestrator.workflow import (
    _LOCK_STALE_SECONDS,
    _RunLock,
    _atomic_write_json,
    _atomic_write_text,
    _check_mapping_coverage,
    _dispatch_estimator,
    _gather_assumptions,
    _load_concept_specs,
    _mapping_coverage_roles,
    _mismatched_concept_domains,
)
from causal.queries import ConceptSetSpec


class AtomicWriteTests(unittest.TestCase):
    """Every persisted run artifact goes through `_atomic_write_json` /
    `_atomic_write_text` -- the Streamlit UI reads run files holding no
    `_RunLock`, so a partial write must never be observable."""

    def test_text_write_is_atomic_and_leaves_no_sidecar(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cohort.sql"
            _atomic_write_text(path, "SELECT 1")
            self.assertEqual(path.read_text(encoding="utf-8"), "SELECT 1")
            self.assertEqual(list(Path(temp).glob("*.tmp")), [])

    def test_json_write_round_trips_and_leaves_no_sidecar(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "estimation.json"
            _atomic_write_json(path, {"b": 2, "a": [1, 2, 3]})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"b": 2, "a": [1, 2, 3]})
            self.assertEqual(list(Path(temp).glob("*.tmp")), [])

    def test_sidecar_name_keeps_the_original_suffix(self):
        # `foo.json` and `foo.sql` written to the same dir must not collide
        # on one `foo.tmp` sidecar.
        with tempfile.TemporaryDirectory() as temp:
            _atomic_write_text(Path(temp) / "foo.sql", "SELECT 1")
            _atomic_write_json(Path(temp) / "foo.json", {"ok": True})
            self.assertEqual((Path(temp) / "foo.sql").read_text(encoding="utf-8"), "SELECT 1")
            self.assertEqual(json.loads((Path(temp) / "foo.json").read_text(encoding="utf-8")), {"ok": True})


class GatherAssumptionsTests(unittest.TestCase):
    def _call(self, cate=None):
        return _gather_assumptions(
            study_spec={}, readiness={}, cohort_manifest={}, dataset_manifest={},
            diagnostics={}, estimand_block={}, cate=cate,
        )

    def test_no_cate_run_states_heterogeneity_was_not_examined_not_out_of_scope(self):
        assumptions = self._call(cate=None)
        joined = " ".join(assumptions)
        self.assertIn("Treatment-effect heterogeneity was not examined", joined)
        self.assertNotIn("Out of scope", joined)
        self.assertNotIn("milestone #4", joined)

    def test_cate_run_reports_the_subgroup_family_caveat(self):
        assumptions = self._call(cate={"estimand": {"n_prespecified_subgroups_tested": 3}})
        joined = " ".join(assumptions)
        self.assertIn("prespecified subgroup family only", joined)
        self.assertNotIn("Treatment-effect heterogeneity was not examined", joined)


def _approve_pre_mapping_gates(manager, run_id, reviewer="tester"):
    """Approve the three gates that replaced the old combined "protocol"
    gate (see docs/causal_roadmap.md's governance split: question/target
    trial, causal DAG, identification/estimand) -- the boilerplate every
    test that just needs to reach the concept_mapping phase now needs,
    instead of one `decide_gate(..., "protocol", "approved", ...)` call."""
    manager.decide_gate(run_id, "target_trial", "approved", reviewer)
    manager.decide_gate(run_id, "dag_review", "approved", reviewer)
    manager.decide_gate(run_id, "identification_estimand", "approved", reviewer)


_FULL_TARGET_TRIAL_FIELDS = {
    "eligibility_criteria": ["adult"],
    "assignment_procedure": "Emulated baseline assignment",
    "censoring_strategy": "Administrative censoring at end of follow-up",
    "analysis_strategy": "Intention-to-treat analogue",
    "effect_measure": "risk_difference",
    "treatment_value": "treatment_value",
    "comparator_value": "comparator_value",
}


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
        self.assertEqual(state.phase, "target_trial_review")
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

    def _full_protocol_path(self):
        path = Path(self.temp.name) / "full_protocol.json"
        path.write_text(json.dumps({
            "study_id": "test_study",
            "source_question": "Does treatment affect outcome?",
            "target_trial": {
                "population": "population", "treatment": "treatment",
                "comparator": "comparator", "time_zero": "index",
                "grace_period_hours": 24, "outcome": "outcome",
                "follow_up_days": 28, "primary_estimand": "ATE",
                **_FULL_TARGET_TRIAL_FIELDS,
            },
        }), encoding="utf-8")
        return path

    def test_gate_required_to_advance(self):
        state = self.manager.start(self._full_protocol_path())
        with self.assertRaises(RuntimeError):
            self.manager.advance(state.run_id)
        _approve_pre_mapping_gates(self.manager, state.run_id)
        for _ in range(3):
            state = self.manager.advance(state.run_id)
        self.assertEqual(state.phase, "concept_mapping")

    def test_set_target_trial_estimand_corrects_primary_estimand(self):
        state = self.manager.start(self._full_protocol_path())
        self.manager.decide_gate(state.run_id, "target_trial", "rejected", "tester", "Should be CATE, not ATE")

        updated = self.manager.set_target_trial_estimand(
            state.run_id,
            primary_estimand="CATE",
            secondary_estimands=("ATE", ""),
            reviewer="tester",
            comment="Study prespecifies age-group subgroups",
        )

        protocol = json.loads(
            (Path(self.temp.name) / "runs" / state.run_id / "protocol.json").read_text(encoding="utf-8")
        )
        self.assertEqual(protocol["target_trial"]["primary_estimand"], "CATE")
        self.assertEqual(protocol["target_trial"]["secondary_estimands"], ["ATE"])
        # `_invalidate_gate` is a no-op on an already-`rejected` gate: the reviewer still has to
        # explicitly re-decide the gate after fixing the estimand, same as any other correction.
        self.assertEqual(updated.gates["target_trial"].status, "rejected")

    def test_set_target_trial_estimand_rejects_invalid_value(self):
        state = self.manager.start(self._full_protocol_path())
        with self.assertRaises(ValueError):
            self.manager.set_target_trial_estimand(
                state.run_id, primary_estimand="RISK_RATIO", reviewer="tester", comment="typo"
            )

    def test_set_target_trial_estimand_requires_comment(self):
        state = self.manager.start(self._full_protocol_path())
        with self.assertRaises(ValueError):
            self.manager.set_target_trial_estimand(
                state.run_id, primary_estimand="ATE", reviewer="tester", comment=""
            )

    def test_set_target_trial_estimand_blocked_once_approved(self):
        state = self.manager.start(self._full_protocol_path())
        self.manager.decide_gate(state.run_id, "target_trial", "approved", "tester")
        with self.assertRaises(ValueError):
            self.manager.set_target_trial_estimand(
                state.run_id, primary_estimand="CATE", reviewer="tester", comment="change of mind"
            )

    def test_set_target_trial_timing_corrects_only_the_given_fields(self):
        state = self.manager.start(self._full_protocol_path())

        updated = self.manager.set_target_trial_timing(
            state.run_id,
            follow_up_days=90,
            time_zero="first qualifying antibiotic order",
            reviewer="tester",
            comment="CAPE COD primary endpoint is 90-day mortality",
        )

        protocol = json.loads(
            (Path(self.temp.name) / "runs" / state.run_id / "protocol.json").read_text(encoding="utf-8")
        )
        self.assertEqual(protocol["target_trial"]["follow_up_days"], 90)
        self.assertEqual(protocol["target_trial"]["time_zero"], "first qualifying antibiotic order")
        # untouched field keeps its drafted value
        self.assertEqual(protocol["target_trial"]["grace_period_hours"], 24)
        entry = next(e for e in updated.history if e["event"] == "target_trial_timing_corrected")
        self.assertEqual(entry["changes"]["before"]["follow_up_days"], 28)
        self.assertEqual(entry["changes"]["after"]["follow_up_days"], 90)

    def test_set_target_trial_timing_invalidates_downstream_gates(self):
        state = self.manager.start(self._full_protocol_path())
        # Simulate a run that already got these gates decided in a later phase,
        # without tripping their approval blockers (analysis_plan.json etc.).
        state = self.manager.load(state.run_id)
        state.gates["analysis_plan"].status = "provisional"
        state.gates["statistical_estimand"].status = "provisional"
        self.manager.save(state)

        updated = self.manager.set_target_trial_timing(
            state.run_id, grace_period_hours=48, reviewer="tester", comment="wider grace window"
        )

        self.assertEqual(updated.gates["analysis_plan"].status, "pending")
        self.assertEqual(updated.gates["statistical_estimand"].status, "pending")
        self.assertTrue(
            any(e["event"] == "gate_invalidated" for e in updated.history)
        )

    def test_set_target_trial_timing_blocked_once_target_trial_approved(self):
        state = self.manager.start(self._full_protocol_path())
        self.manager.decide_gate(state.run_id, "target_trial", "approved", "tester")
        with self.assertRaises(ValueError):
            self.manager.set_target_trial_timing(
                state.run_id, follow_up_days=90, reviewer="tester", comment="too late"
            )

    def test_set_target_trial_timing_rejects_bad_values(self):
        state = self.manager.start(self._full_protocol_path())
        for kwargs in (
            {"follow_up_days": 0},
            {"grace_period_hours": -1},
            {"time_zero": "   "},
            {},  # nothing to change
        ):
            with self.assertRaises(ValueError):
                self.manager.set_target_trial_timing(
                    state.run_id, reviewer="tester", comment="x", **kwargs
                )

    def test_set_target_trial_timing_refreshes_hypothesis_when_present(self):
        state = self.manager.start(self._full_protocol_path())
        self.manager.run_hypothesis_agent(state.run_id)

        self.manager.set_target_trial_timing(
            state.run_id, follow_up_days=90, reviewer="tester", comment="anchor trial horizon"
        )

        hypothesis = json.loads(
            (Path(self.temp.name) / "runs" / state.run_id / "hypothesis.json").read_text(encoding="utf-8")
        )
        self.assertEqual(hypothesis["follow_up_days"], 90)

    def test_set_target_trial_timing_and_estimand_write_a_correction_log(self):
        state = self.manager.start(self._full_protocol_path())
        self.manager.decide_gate(state.run_id, "target_trial", "rejected", "tester", "wrong")
        self.manager.set_target_trial_timing(
            state.run_id, follow_up_days=90, reviewer="tester", comment="D90 endpoint"
        )
        self.manager.set_target_trial_estimand(
            state.run_id, primary_estimand="CATE", reviewer="tester", comment="prespecified subgroups"
        )
        protocol = json.loads(
            (Path(self.temp.name) / "runs" / state.run_id / "protocol.json").read_text(encoding="utf-8")
        )
        kinds = [entry["kind"] for entry in protocol["correction_log"]]
        self.assertEqual(kinds, ["target_trial_timing", "target_trial_estimand"])
        self.assertEqual(protocol["correction_log"][0]["comment"], "D90 endpoint")

    def _dag_protocol_path(self):
        path = self._full_protocol_path()
        protocol = json.loads(path.read_text(encoding="utf-8"))
        protocol["first_pass_covariates"] = ["age"]
        protocol["candidate_dag"] = [
            {
                "label": "age", "role": "confounder", "temporal_classification": "pre_treatment",
                "rationale": "prognostic and affects treatment choice", "pmid": "",
                "observed": True, "certainty": "established", "evidence_level": "uncited",
            },
            {
                "label": "vasopressor use", "role": "confounder",
                "temporal_classification": "pre_treatment", "rationale": "severity marker",
                "pmid": "", "observed": True, "certainty": "established", "evidence_level": "uncited",
            },
        ]
        path.write_text(json.dumps(protocol), encoding="utf-8")
        return path

    def test_revise_candidate_dag_from_points_applies_operations(self):
        state = self.manager.start(self._dag_protocol_path())
        self.manager.run_hypothesis_agent(state.run_id)

        response = json.dumps({
            "operations": [
                {
                    "op": "add", "label": "baseline creatinine", "role": "confounder",
                    "temporal_classification": "pre_treatment", "observed": True,
                    "certainty": "established", "rationale": "reviewer: confounds nephrotoxin and AKI",
                },
                {
                    "op": "reclassify", "label": "vasopressor use", "role": "mediator",
                    "temporal_classification": "post_treatment",
                    "rationale": "reviewer: on the causal pathway",
                },
            ],
            "acknowledged_errors": ["missed baseline renal function"],
            "unaddressed_points": [],
        })
        updated = self.manager.revise_candidate_dag_from_points(
            state.run_id,
            ["add baseline creatinine as a confounder", "vasopressors are a mediator"],
            _CapturingLLM(response),
            "tester",
            comment="clinical pass 1",
        )

        protocol = json.loads(
            (Path(self.temp.name) / "runs" / state.run_id / "protocol.json").read_text(encoding="utf-8")
        )
        labels = {v["label"]: v["role"] for v in protocol["candidate_dag"]}
        self.assertEqual(labels["baseline creatinine"], "confounder")
        self.assertEqual(labels["vasopressor use"], "mediator")
        self.assertEqual(protocol["correction_log"][-1]["kind"], "dag_points")
        for gate in ("dag_review", "clinical_mapping", "analysis_plan"):
            self.assertEqual(updated.gates[gate].status, "pending")

    def test_revise_candidate_dag_from_points_rejects_an_invalid_graph(self):
        state = self.manager.start(self._dag_protocol_path())
        self.manager.run_hypothesis_agent(state.run_id)
        # A confounder cannot be post_treatment -- HypothesisAgent rejects it,
        # and protocol.json must be left untouched.
        bad = json.dumps({
            "operations": [{
                "op": "reclassify", "label": "age", "role": "confounder",
                "temporal_classification": "post_treatment", "rationale": "reviewer: nonsense",
            }],
            "acknowledged_errors": [], "unaddressed_points": [],
        })
        with self.assertRaises(ValueError):
            self.manager.revise_candidate_dag_from_points(
                state.run_id, ["make age a post-treatment confounder"], _CapturingLLM(bad), "tester"
            )
        protocol = json.loads(
            (Path(self.temp.name) / "runs" / state.run_id / "protocol.json").read_text(encoding="utf-8")
        )
        self.assertNotIn("correction_log", protocol)
        self.assertEqual(
            next(v["temporal_classification"] for v in protocol["candidate_dag"] if v["label"] == "age"),
            "pre_treatment",
        )

    def test_revise_candidate_dag_from_points_requires_points(self):
        state = self.manager.start(self._dag_protocol_path())
        self.manager.run_hypothesis_agent(state.run_id)
        with self.assertRaises(ValueError):
            self.manager.revise_candidate_dag_from_points(
                state.run_id, ["   "], _CapturingLLM("{}"), "tester"
            )

    def test_run_design_companion_turn_returns_a_validated_card(self):
        state = self.manager.start(self._dag_protocol_path())
        self.manager.run_hypothesis_agent(state.run_id)
        response = json.dumps({
            "reply": "The anchor RCT used 90-day mortality; I'll propose that.",
            "proposed_action": {
                "action": "set_target_trial_timing",
                "args": {"follow_up_days": 90, "comment": "anchor RCT primary endpoint"},
            },
        })
        llm = _CapturingLLM(response)
        turn = self.manager.run_design_companion_turn(
            state.run_id, llm, [CompanionMessage("user", "follow-up should be 90 days")]
        )
        self.assertEqual(turn.proposed_action.action, "set_target_trial_timing")
        self.assertIn("DRAFTED CANDIDATE DAG", llm.prompts[0])
        self.assertIn("Reviewer: follow-up should be 90 days", llm.prompts[0])
        # read-only: no history entry, no protocol write
        before = self.manager.load(state.run_id)
        self.manager.run_design_companion_turn(state.run_id, _CapturingLLM(response), [])
        self.assertEqual(len(self.manager.load(state.run_id).history), len(before.history))

    def test_run_design_companion_turn_rejects_wrong_phase(self):
        state = self.manager.start(self._dag_protocol_path())
        _approve_pre_mapping_gates(self.manager, state.run_id)
        for _ in range(3):
            state = self.manager.advance(state.run_id)
        with self.assertRaises(RuntimeError):
            self.manager.run_design_companion_turn(state.run_id, _CapturingLLM("{}"), [])

    def test_run_design_companion_turn_drops_timing_card_once_target_trial_approved(self):
        state = self.manager.start(self._dag_protocol_path())
        self.manager.run_hypothesis_agent(state.run_id)
        self.manager.decide_gate(state.run_id, "target_trial", "approved", "tester")
        state = self.manager.advance(state.run_id)
        self.assertEqual(state.phase, "dag_review")
        response = json.dumps({
            "reply": "You want 90-day follow-up.",
            "proposed_action": {
                "action": "set_target_trial_timing",
                "args": {"follow_up_days": 90, "comment": "anchor RCT"},
            },
        })
        turn = self.manager.run_design_companion_turn(
            state.run_id, _CapturingLLM(response), []
        )
        # `set_target_trial_timing` would only raise on confirm once the
        # gate is approved, so the card is dropped and the reply says why.
        self.assertIsNone(turn.proposed_action)
        self.assertIn("target_trial gate rejected first", turn.reply)

    def _protocol_path_with_approved_concept_sets(self):
        """`_full_protocol_path` plus an approved population/treatment/
        comparator, needed as a base for `add_population_concept_set`
        (requires `hypothesis.json`, which in turn needs `concept_sets`
        present to run cleanly)."""
        path = self._full_protocol_path()
        protocol = json.loads(path.read_text(encoding="utf-8"))
        protocol["concept_sets"] = {
            "t2dm": {"role": "population", "status": "approved", "domain": "condition", "concept_ids": [1]},
            "metformin": {"role": "treatment", "status": "approved", "domain": "drug", "concept_ids": [2]},
            "no_metformin": {"role": "comparator", "status": "approved", "domain": "drug", "concept_ids": [3]},
        }
        path.write_text(json.dumps(protocol), encoding="utf-8")
        return path

    def test_add_population_concept_set_creates_pending_entry(self):
        state = self.manager.start(self._protocol_path_with_approved_concept_sets())
        self.manager.run_hypothesis_agent(state.run_id)

        self.manager.add_population_concept_set(
            state.run_id, name="copd", label="Chronic obstructive pulmonary disease",
            lookup_term="COPD", reviewer="tester", comment="second required diagnosis",
        )

        run_dir = Path(self.temp.name) / "runs" / state.run_id
        protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(
            protocol["concept_sets"]["copd"],
            {
                "status": "pending_omop_mapping", "concept_ids": [], "label": "Chronic obstructive pulmonary disease",
                "lookup_term": "COPD", "role": "population_secondary", "domain": "",
            },
        )
        self.assertEqual(protocol["additional_population_concept_sets"], ["copd"])
        hypothesis = json.loads((run_dir / "hypothesis.json").read_text(encoding="utf-8"))
        self.assertIn("concept_set:copd:pending_omop_mapping", hypothesis["unresolved_issues"])
        self.assertFalse(hypothesis["executable"])

    def test_add_population_concept_set_requires_hypothesis_json(self):
        state = self.manager.start(self._protocol_path_with_approved_concept_sets())
        with self.assertRaises(RuntimeError):
            self.manager.add_population_concept_set(
                state.run_id, name="copd", label="COPD", reviewer="tester",
            )

    def test_add_population_concept_set_rejects_a_name_already_in_use(self):
        state = self.manager.start(self._protocol_path_with_approved_concept_sets())
        self.manager.run_hypothesis_agent(state.run_id)
        with self.assertRaises(ValueError):
            self.manager.add_population_concept_set(
                state.run_id, name="t2dm", label="duplicate", reviewer="tester",
            )

    def test_add_population_concept_set_requires_name_and_label_and_reviewer(self):
        state = self.manager.start(self._protocol_path_with_approved_concept_sets())
        self.manager.run_hypothesis_agent(state.run_id)
        with self.assertRaises(ValueError):
            self.manager.add_population_concept_set(state.run_id, name="", label="x", reviewer="tester")
        with self.assertRaises(ValueError):
            self.manager.add_population_concept_set(state.run_id, name="copd", label="", reviewer="tester")
        with self.assertRaises(ValueError):
            self.manager.add_population_concept_set(state.run_id, name="copd", label="x", reviewer="")

    def test_remove_population_concept_set_deletes_the_entry_and_unblocks_executable(self):
        state = self.manager.start(self._protocol_path_with_approved_concept_sets())
        self.manager.run_hypothesis_agent(state.run_id)
        self.manager.add_population_concept_set(
            state.run_id, name="copd", label="COPD", reviewer="tester",
        )

        self.manager.remove_population_concept_set(state.run_id, "copd", "tester", "added by mistake")

        run_dir = Path(self.temp.name) / "runs" / state.run_id
        protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
        self.assertNotIn("copd", protocol["concept_sets"])
        self.assertEqual(protocol["additional_population_concept_sets"], [])
        hypothesis = json.loads((run_dir / "hypothesis.json").read_text(encoding="utf-8"))
        self.assertNotIn("concept_set:copd:pending_omop_mapping", hypothesis["unresolved_issues"])

    def test_remove_population_concept_set_requires_a_declared_name(self):
        state = self.manager.start(self._protocol_path_with_approved_concept_sets())
        self.manager.run_hypothesis_agent(state.run_id)
        with self.assertRaises(ValueError):
            self.manager.remove_population_concept_set(state.run_id, "never_added", "tester")

    def test_provisional_mapping_only_allows_exploration(self):
        exploratory = self.manager.start(self._full_protocol_path(), "exploratory")
        _approve_pre_mapping_gates(self.manager, exploratory.run_id)
        for _ in range(3):
            self.manager.advance(exploratory.run_id)
        self.manager.decide_gate(exploratory.run_id, "clinical_mapping", "provisional", "tester")
        self.assertEqual(self.manager.advance(exploratory.run_id).phase, "feasibility")

        final = self.manager.start(self._full_protocol_path(), "final")
        _approve_pre_mapping_gates(self.manager, final.run_id)
        for _ in range(3):
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

    def test_persists_the_reviewer_onto_the_candidate_dag_entry(self):
        """`causal.mapping.registry.variable_from_candidate_dag_entry` reads
        `entry["mapping_reviewer"]` back out into `OMOPVariableDefinition.
        reviewer` -- this was previously never written here, so the
        registry's reviewer field (and its "an automated mapping needs a
        named reviewer" check) stayed blank for every real approval."""
        self.manager.map_covariate_concepts(
            self.state.run_id, "SOFA score", [4301868], "Dr. Lee",
        )
        protocol = json.loads((self.run_dir / "protocol.json").read_text(encoding="utf-8"))
        entry = next(item for item in protocol["candidate_dag"] if item["label"] == "SOFA score")
        self.assertEqual(entry["mapping_reviewer"], "Dr. Lee")

        from causal.mapping.registry import variable_from_candidate_dag_entry
        variable = variable_from_candidate_dag_entry(entry)
        self.assertEqual(variable.reviewer, "Dr. Lee")

    def test_defaults_provenance_to_protocol_declared_without_a_concept_lookup(self):
        """Without a live `concept_lookup` cross-check, this mapping is
        purely human-supplied -- `provenance` must stay outside
        `causal.mapping.registry._UNSUPERVISED_PROVENANCE` (see
        `test_persists_omcp_candidate_provenance_when_concept_lookup_is_used`
        in `WorkflowMapCovariateConceptsDomainCheckTests` for the contrasting
        case)."""
        self.manager.map_covariate_concepts(
            self.state.run_id, "SOFA score", [4301868], "Dr. Lee",
        )
        protocol = json.loads((self.run_dir / "protocol.json").read_text(encoding="utf-8"))
        entry = next(item for item in protocol["candidate_dag"] if item["label"] == "SOFA score")
        self.assertEqual(entry["provenance"], "protocol_declared")


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

    def test_persists_omcp_candidate_provenance_when_concept_lookup_is_used(self):
        """A live `concept_lookup` cross-check is this codebase's one signal
        that the approved concept_id was OMCP-vetted, not purely
        human-supplied -- `causal.mapping.registry`'s
        `_UNSUPERVISED_PROVENANCE` check treats `"omcp_candidate"` as
        needing a named reviewer (already required here regardless), rather
        than the always-trusted `"protocol_declared"` default."""
        lookup = _FakeConceptDomainLookup(
            "concept_id,concept_name,concept_code,vocabulary_id,domain_id\n"
            "4301868,SOFA score,x,LOINC,Measurement\n"
        )
        self.manager.map_covariate_concepts(
            self.state.run_id, "SOFA score", [4301868], "Dr. Lee", concept_lookup=lookup,
        )
        protocol = json.loads((self.run_dir / "protocol.json").read_text(encoding="utf-8"))
        entry = next(item for item in protocol["candidate_dag"] if item["label"] == "SOFA score")
        self.assertEqual(entry["provenance"], "omcp_candidate")

        from causal.mapping.registry import _UNSUPERVISED_PROVENANCE
        self.assertIn(entry["provenance"], _UNSUPERVISED_PROVENANCE)


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

    def test_persists_the_reviewer_and_default_provenance_onto_the_entry(self):
        """`causal.mapping.registry.variable_from_concept_set` reads
        `entry["reviewer"]`/`entry["provenance"]` back out into
        `OMOPVariableDefinition.reviewer`/`.provenance` -- neither was ever
        written here, so both stayed permanently blank/defaulted for every
        real phenotype approval, the same dead-safety-net gap
        `map_covariate_concepts` had for `candidate_dag` (see
        `WorkflowMapCovariateConceptsTests.
        test_persists_the_reviewer_onto_the_candidate_dag_entry`)."""
        self.manager.map_concept_set(
            self.state.run_id, "population", "condition", [132797], "Dr. Chen",
        )
        entry = self._protocol()["concept_sets"]["population"]
        self.assertEqual(entry["reviewer"], "Dr. Chen")
        self.assertEqual(entry["provenance"], "protocol_declared")

        from causal.mapping.registry import variable_from_concept_set
        variable = variable_from_concept_set("population", entry)
        self.assertEqual(variable.reviewer, "Dr. Chen")

    def test_persists_omcp_candidate_provenance_when_concept_lookup_is_used(self):
        lookup = _FakeConceptDomainLookup(
            "concept_id,concept_name,concept_code,vocabulary_id,domain_id\n"
            "132797,Sepsis,x,SNOMED,Condition\n"
        )
        self.manager.map_concept_set(
            self.state.run_id, "population", "condition", [132797], "Dr. Chen",
            concept_lookup=lookup,
        )
        entry = self._protocol()["concept_sets"]["population"]
        self.assertEqual(entry["provenance"], "omcp_candidate")

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


class WorkflowSetConceptSetLookupTermsTests(unittest.TestCase):
    """`set_concept_set_lookup_terms` only seeds the UI concept-search box; it
    writes `protocol.json` but touches no mapping decision and no gate."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        protocol = root / "protocol.json"
        protocol.write_text(json.dumps({
            "study_id": "test_study",
            "source_question": "Does treatment affect outcome?",
            "target_trial": {
                "population": "population", "treatment": "treatment",
                "comparator": "comparator", "time_zero": "index",
                "grace_period_hours": 24, "outcome": "outcome",
                "follow_up_days": 28, "primary_estimand": "ATE",
            },
            "concept_sets": {
                "population": {"role": "population", "status": "pending_omop_mapping", "concept_ids": [], "domain": ""},
                "outcome": {"role": "outcome", "status": "pending_omop_mapping", "concept_ids": [], "domain": ""},
            },
        }), encoding="utf-8")
        self.manager = WorkflowManager(root / "runs")
        self.state = self.manager.run_hypothesis_agent(self.manager.start(protocol).run_id)
        self.run_dir = root / "runs" / self.state.run_id

    def _protocol(self) -> dict:
        return json.loads((self.run_dir / "protocol.json").read_text(encoding="utf-8"))

    def test_updates_lookup_term_without_touching_other_fields(self):
        self.manager.map_concept_set(self.state.run_id, "population", "condition", [132797], "Dr. Chen")
        self.manager.set_concept_set_lookup_terms(
            self.state.run_id, {"population": "  sepsis ", "outcome": "in-hospital death"}
        )
        concept_sets = self._protocol()["concept_sets"]
        self.assertEqual(concept_sets["population"]["lookup_term"], "sepsis")
        self.assertEqual(concept_sets["outcome"]["lookup_term"], "in-hospital death")
        # The mapping approved above is untouched.
        self.assertEqual(concept_sets["population"]["concept_ids"], [132797])
        self.assertEqual(concept_sets["population"]["status"], "approved")

    def test_records_an_event_and_no_gate_is_invalidated(self):
        self.manager.decide_gate(self.state.run_id, "clinical_mapping", "approved", "Dr. Lee")
        state = self.manager.set_concept_set_lookup_terms(self.state.run_id, {"population": "sepsis"})
        self.assertEqual(state.gates["clinical_mapping"].status, "approved")
        self.assertTrue(any(e["event"] == "concept_set_lookup_terms_set" for e in state.history))

    def test_unknown_names_are_ignored_when_at_least_one_matches(self):
        self.manager.set_concept_set_lookup_terms(
            self.state.run_id, {"population": "sepsis", "not_a_set": "x"}
        )
        self.assertEqual(self._protocol()["concept_sets"]["population"]["lookup_term"], "sepsis")

    def test_raises_when_no_name_matches(self):
        with self.assertRaises(RuntimeError):
            self.manager.set_concept_set_lookup_terms(self.state.run_id, {"not_a_set": "x"})

    def test_rejects_empty_or_non_string_input(self):
        for bad in ({}, {"population": "   "}, {"population": 5}, "not a dict"):
            with self.assertRaises(ValueError):
                self.manager.set_concept_set_lookup_terms(self.state.run_id, bad)


class WorkflowGateInvalidationTests(unittest.TestCase):
    """An `approved`/`provisional` gate must not silently keep authorizing
    execution once the content it approved changes underneath it --
    `ExecutionAgent`/`omcp.governance` only ever see the gate's bare status
    string, with nothing tying it to the artifact version a human actually
    reviewed. Every review action that mutates `concept_sets`/`candidate_dag`
    must reset an already-decided `clinical_mapping`/`analysis_plan` gate
    back to `pending` (see `WorkflowManager._invalidate_gate`)."""

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
            },
        }), encoding="utf-8")
        self.manager = WorkflowManager(root / "runs")
        state = self.manager.start(self.protocol)
        self.state = self.manager.run_hypothesis_agent(state.run_id)
        self.run_dir = root / "runs" / self.state.run_id

    def tearDown(self):
        self.temp.cleanup()

    def test_map_concept_set_invalidates_an_approved_clinical_mapping_gate(self):
        self.manager.decide_gate(self.state.run_id, "clinical_mapping", "approved", "tester")
        state = self.manager.map_concept_set(
            self.state.run_id, "population", "condition", [132797], "Dr. Chen",
        )
        self.assertEqual(state.gates["clinical_mapping"].status, "pending")
        self.assertIsNone(state.gates["clinical_mapping"].reviewer)
        self.assertTrue(any(event["event"] == "gate_invalidated" for event in state.history))

    def test_map_concept_set_invalidates_a_provisional_analysis_plan_gate(self):
        self.manager.decide_gate(self.state.run_id, "analysis_plan", "provisional", "tester")
        state = self.manager.map_concept_set(
            self.state.run_id, "population", "condition", [132797], "Dr. Chen",
        )
        self.assertEqual(state.gates["analysis_plan"].status, "pending")

    def test_map_concept_set_leaves_a_pending_gate_alone(self):
        """The ordinary review flow (map, then approve) must not spam the
        audit trail with spurious invalidation events for a gate nobody had
        decided yet."""
        state = self.manager.map_concept_set(
            self.state.run_id, "population", "condition", [132797], "Dr. Chen",
        )
        self.assertEqual(state.gates["clinical_mapping"].status, "pending")
        self.assertFalse(any(event["event"] == "gate_invalidated" for event in state.history))


class WorkflowAnalysisPlanApprovalGateTests(unittest.TestCase):
    """`decide_gate(..., "analysis_plan", "approved", ...)` must not rubber-
    stamp an incomplete plan -- `causal.agents.execution.ExecutionAgent`
    would refuse to authorize execution against it anyway (see
    `analysis_plan_blockers`), so approval should fail loudly with the same
    reasons instead of silently succeeding and only surfacing the problem
    later at execution time."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.protocol = root / "protocol.json"
        self.protocol.write_text(json.dumps({"study_id": "test_study"}), encoding="utf-8")
        self.manager = WorkflowManager(root / "runs")
        self.state = self.manager.start(self.protocol)
        self.run_dir = root / "runs" / self.state.run_id

    def tearDown(self):
        self.temp.cleanup()

    def _write_plan(self, **overrides):
        plan = {
            "schema_version": "1.0",
            "study_id": "test_study",
            "data_source": "project.dataset",
            "design": "target_trial_emulation",
            "exploratory_execution_allowed": False,
            "final_execution_allowed": False,
            "unresolved_issues": [],
            "phenotypes": [{"name": "population", "status": "approved", "approval_required": True}],
            "covariates": [{"name": "age", "status": "ready"}],
            "feasibility_checks": [],
        }
        plan.update(overrides)
        (self.run_dir / "analysis_plan.json").write_text(json.dumps(plan), encoding="utf-8")

    def test_blocks_approval_before_a_plan_exists(self):
        with self.assertRaises(ValueError):
            self.manager.decide_gate(self.state.run_id, "analysis_plan", "approved", "tester")

    def test_blocks_approval_while_unresolved_issues_remain(self):
        self._write_plan(unresolved_issues=["albumin mapping pending"])
        with self.assertRaises(ValueError):
            self.manager.decide_gate(self.state.run_id, "analysis_plan", "approved", "tester")
        state = self.manager.load(self.state.run_id)
        self.assertEqual(state.gates["analysis_plan"].status, "pending")

    def test_blocks_approval_while_a_covariate_is_not_ready(self):
        self._write_plan(covariates=[{"name": "sofa_score", "status": "definition_pending"}])
        with self.assertRaises(ValueError):
            self.manager.decide_gate(self.state.run_id, "analysis_plan", "approved", "tester")

    def test_allows_approval_once_the_plan_is_complete(self):
        self._write_plan()
        state = self.manager.decide_gate(self.state.run_id, "analysis_plan", "approved", "tester")
        self.assertEqual(state.gates["analysis_plan"].status, "approved")

    def test_rejecting_an_incomplete_plan_is_still_allowed(self):
        """The completeness guard only protects `approved`, mirroring
        `_block_if_major_confounders_unresolved`/`_block_if_not_identified`
        for the `protocol` gate: a reviewer must always be able to reject an
        incomplete plan outright."""
        self._write_plan(unresolved_issues=["albumin mapping pending"])
        state = self.manager.decide_gate(self.state.run_id, "analysis_plan", "rejected", "tester")
        self.assertEqual(state.gates["analysis_plan"].status, "rejected")


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
        """One phenotype (role='treatment_and_comparator') serves as the
        shared backbone for both arms of an `add_on` design."""
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


class LoadConceptSpecsAdditionalPopulationTests(unittest.TestCase):
    """`additional_population_concept_sets` (see `WorkflowManager.
    add_population_concept_set`) ANDs one or more further, independently-
    required population concept sets onto `specs["population"]` -- resolved
    directly by name (`dag_node_label`), not via `resolve_role`, since two
    entries sharing one `causal_role` would otherwise raise (see
    `test_two_entries_declaring_the_same_role_raises` above)."""

    @staticmethod
    def _approved(role: str, domain: str, concept_ids: list[int]) -> dict:
        return {"role": role, "status": "approved", "domain": domain, "concept_ids": concept_ids}

    def _base_concept_sets(self) -> dict:
        return {
            "t2dm": self._approved("population", "condition", [1]),
            "metformin": self._approved("treatment", "drug", [2]),
            "no_metformin": self._approved("comparator", "drug", [3]),
        }

    def test_absent_by_default(self):
        specs = _load_concept_specs({"concept_sets": self._base_concept_sets()})
        self.assertEqual(specs["population_additional"], ())
        self.assertEqual(specs["population_additional_names"], ())

    def test_resolves_one_additional_entry(self):
        concept_sets = self._base_concept_sets()
        concept_sets["copd"] = self._approved("population_secondary", "condition", [255573])
        specs = _load_concept_specs({
            "concept_sets": concept_sets,
            "additional_population_concept_sets": ["copd"],
        })
        self.assertEqual(specs["population_additional"], (ConceptSetSpec(domain="condition", concept_ids=(255573,)),))
        self.assertEqual(specs["population_additional_names"], ("copd",))

    def test_resolves_multiple_entries_in_declared_order(self):
        concept_sets = self._base_concept_sets()
        concept_sets["copd"] = self._approved("population_secondary", "condition", [255573])
        concept_sets["ckd"] = self._approved("population_secondary", "condition", [46271022])
        specs = _load_concept_specs({
            "concept_sets": concept_sets,
            "additional_population_concept_sets": ["ckd", "copd"],
        })
        self.assertEqual(
            specs["population_additional"],
            (
                ConceptSetSpec(domain="condition", concept_ids=(46271022,)),
                ConceptSetSpec(domain="condition", concept_ids=(255573,)),
            ),
        )
        self.assertEqual(specs["population_additional_names"], ("ckd", "copd"))

    def test_undeclared_name_raises(self):
        with self.assertRaises(RuntimeError):
            _load_concept_specs({
                "concept_sets": self._base_concept_sets(),
                "additional_population_concept_sets": ["copd"],
            })

    def test_unapproved_entry_raises(self):
        concept_sets = self._base_concept_sets()
        concept_sets["copd"] = {
            "role": "population_secondary", "status": "pending_omop_mapping", "domain": "", "concept_ids": [],
        }
        with self.assertRaises(RuntimeError):
            _load_concept_specs({
                "concept_sets": concept_sets,
                "additional_population_concept_sets": ["copd"],
            })

    def test_does_not_affect_the_primary_population_role(self):
        concept_sets = self._base_concept_sets()
        concept_sets["copd"] = self._approved("population_secondary", "condition", [255573])
        specs = _load_concept_specs({
            "concept_sets": concept_sets,
            "additional_population_concept_sets": ["copd"],
        })
        self.assertEqual(specs["population"], ConceptSetSpec(domain="condition", concept_ids=(1,)))


class LoadConceptSpecsVsUnexposedComparatorTests(unittest.TestCase):
    """`exposure_contrast="vs_unexposed"` (e.g. "aspirin vs no aspirin") is
    the one case where an unapproved, domain-less, or entirely absent
    comparator is not a blocker: there is no OMOP concept for "absence of a
    drug", so nothing can ever be approved for it. Every other
    exposure_contrast keeps requiring a real, approved comparator concept
    (see `LoadConceptSpecsTests` above)."""

    @staticmethod
    def _approved(role: str, domain: str, concept_ids: list[int]) -> dict:
        return {
            "role": role, "status": "approved", "domain": domain, "concept_ids": concept_ids,
        }

    def _base_concept_sets(self) -> dict:
        return {
            "sepsis": self._approved("population", "condition", [1]),
            "albumin": self._approved("treatment", "drug", [2]),
        }

    def test_no_standard_concept_comparator_resolves_without_raising(self):
        concept_sets = self._base_concept_sets()
        concept_sets["no_albumin"] = {
            "role": "comparator", "status": "no_standard_concept", "domain": "", "concept_ids": [],
        }
        specs = _load_concept_specs({
            "query_settings": {"exposure_contrast": "vs_unexposed"},
            "concept_sets": concept_sets,
        })
        self.assertEqual(specs["comparator"], ConceptSetSpec(domain="", concept_ids=()))

    def test_no_comparator_entry_at_all_resolves_without_raising(self):
        specs = _load_concept_specs({
            "query_settings": {"exposure_contrast": "vs_unexposed"},
            "concept_sets": self._base_concept_sets(),
        })
        self.assertEqual(specs["comparator"], ConceptSetSpec(domain="", concept_ids=()))

    def test_a_real_comparator_concept_still_resolves_normally(self):
        """Choosing "vs_unexposed" does not force the comparator empty --
        a reviewer who happens to have mapped one is not blocked either."""
        concept_sets = self._base_concept_sets()
        concept_sets["clopidogrel"] = self._approved("comparator", "drug", [3])
        specs = _load_concept_specs({
            "query_settings": {"exposure_contrast": "vs_unexposed"},
            "concept_sets": concept_sets,
        })
        self.assertEqual(specs["comparator"], ConceptSetSpec(domain="drug", concept_ids=(3,)))

    def test_same_no_standard_concept_comparator_still_raises_under_add_on(self):
        concept_sets = self._base_concept_sets()
        concept_sets["no_albumin"] = {
            "role": "comparator", "status": "no_standard_concept", "domain": "", "concept_ids": [],
        }
        with self.assertRaises(RuntimeError):
            _load_concept_specs({
                "query_settings": {"exposure_contrast": "add_on"},
                "concept_sets": concept_sets,
            })

    def test_missing_comparator_still_raises_under_the_default_exposure_contrast(self):
        with self.assertRaises(RuntimeError):
            _load_concept_specs({"concept_sets": self._base_concept_sets()})


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

    def test_window_change_reopens_the_statistical_estimand_gate(self):
        # A window change does not rename the materialized column, so
        # `run_estimation`'s adjustment-set equality check never notices --
        # the downstream `statistical_estimand`/`results` gates must reopen.
        state = self.manager.load(self.state.run_id)
        for gate in ("clinical_mapping", "analysis_plan", "statistical_estimand", "results"):
            state.gates[gate].status = "approved"
        self.manager.save(state)
        state = self.manager.review_covariate_window(
            self.state.run_id, "lactate", -6, 0, "Dr. Lee",
        )
        for gate in ("clinical_mapping", "analysis_plan", "statistical_estimand", "results"):
            self.assertEqual(state.gates[gate].status, "pending", gate)


class WorkflowSetExposureContrastTests(unittest.TestCase):
    """`set_exposure_contrast` records a reviewer's choice of
    treatment/comparator inclusion semantics (see
    `causal.queries.feasibility.EXPOSURE_CONTRASTS`) -- previously only
    settable by hand-editing protocol.json, with every run silently
    defaulting to "add_on" otherwise."""

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
            "concept_sets": {"treatment": {"status": "approved"}},
        }), encoding="utf-8")
        self.manager = WorkflowManager(root / "runs")
        self.state = self.manager.start(self.protocol)
        self.run_dir = root / "runs" / self.state.run_id

    def tearDown(self):
        self.temp.cleanup()

    def _protocol(self) -> dict:
        return json.loads((self.run_dir / "protocol.json").read_text(encoding="utf-8"))

    def test_saves_the_choice_to_protocol_json(self):
        self.manager.set_exposure_contrast(self.state.run_id, "vs_unexposed", "Dr. Lee")
        self.assertEqual(self._protocol()["query_settings"]["exposure_contrast"], "vs_unexposed")

    def test_rejects_an_unknown_value(self):
        with self.assertRaises(ValueError):
            self.manager.set_exposure_contrast(self.state.run_id, "not_a_real_mode", "Dr. Lee")

    def test_requires_a_reviewer(self):
        with self.assertRaises(ValueError):
            self.manager.set_exposure_contrast(self.state.run_id, "vs_unexposed", "  ")

    def test_records_structured_history_event(self):
        state = self.manager.set_exposure_contrast(self.state.run_id, "mutually_exclusive", "Dr. Lee")
        event = next(event for event in state.history if event["event"] == "exposure_contrast_set")
        self.assertEqual(event["changes"]["before"], "add_on")
        self.assertEqual(event["changes"]["after"], "mutually_exclusive")

    def test_invalidates_the_analysis_plan_gate(self):
        self.manager.decide_gate(self.state.run_id, "analysis_plan", "provisional", "Dr. Lee")
        state = self.manager.set_exposure_contrast(self.state.run_id, "vs_unexposed", "Dr. Lee")
        self.assertEqual(state.gates["analysis_plan"].status, "pending")


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


class WorkflowConcurrentGateDecisionTests(unittest.TestCase):
    """`decide_gate`/`advance`/`run_data_science_agent`/
    `run_execution_readiness`/`build_cohort_plan` all load-mutate-save
    `state.json` (`RunState`) the same way `map_covariate_concepts` etc.
    load-mutate-save `protocol.json` -- without `_RunLock` around that
    cycle too, two reviewers deciding *different* gates on the same run at
    the same time could race: both `load()` the same on-disk state, and
    whichever `save()`s second silently discards the other's gate
    decision, even though neither reviewer touched the same gate."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        protocol = root / "protocol.json"
        protocol.write_text(json.dumps({"study_id": "test_study"}), encoding="utf-8")
        self.manager = WorkflowManager(root / "runs")
        state = self.manager.start(protocol)
        self.run_id = state.run_id
        # `_block_if_results_incomplete` requires estimation.json to exist
        # before the `results` gate can be approved; this test exercises
        # `_RunLock` concurrency, not results-completeness, so stub it in.
        run_dir = root / "runs" / self.run_id
        (run_dir / "estimation.json").write_text("{}", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_concurrent_decisions_on_different_gates_both_survive(self):
        errors: list[Exception] = []

        def _decide(gate: str, reviewer: str) -> None:
            try:
                self.manager.decide_gate(self.run_id, gate, "approved", reviewer)
            except Exception as exc:  # noqa: BLE001 -- surfaced via `errors`, not raised in-thread
                errors.append(exc)

        threads = [
            threading.Thread(target=_decide, args=("clinical_mapping", "Dr. A")),
            threading.Thread(target=_decide, args=("results", "Dr. B")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        state = self.manager.load(self.run_id)
        self.assertEqual(state.gates["clinical_mapping"].status, "approved")
        self.assertEqual(state.gates["clinical_mapping"].reviewer, "Dr. A")
        self.assertEqual(state.gates["results"].status, "approved")
        self.assertEqual(state.gates["results"].reviewer, "Dr. B")
        self.assertFalse((Path(self.temp.name) / "runs" / self.run_id / ".review.lock").exists())


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
                "follow_up_days": 28, "primary_estimand": "ATE",
                **_FULL_TARGET_TRIAL_FIELDS,
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
        _approve_pre_mapping_gates(self.manager, self.state.run_id)
        for _ in range(3):
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

    def test_dag_review_gate_cannot_be_approved_while_major_confounder_unresolved(self):
        with self.assertRaises(ValueError):
            self.manager.decide_gate(self.state.run_id, "dag_review", "approved", "tester")

    def test_dag_review_gate_can_still_be_rejected_or_made_provisional(self):
        self.manager.decide_gate(self.state.run_id, "dag_review", "provisional", "tester")
        self.manager.decide_gate(self.state.run_id, "dag_review", "rejected", "tester")

    def test_classifying_the_major_confounder_unblocks_approval(self):
        self.manager.review_covariate(
            self.state.run_id, "age", "accept", "Dr. Chen",
            "Baseline age affects both treatment choice and outcome.",
            role="confounder", temporal_classification="pre_treatment",
        )
        state = self.manager.decide_gate(self.state.run_id, "dag_review", "approved", "tester")
        self.assertEqual(state.gates["dag_review"].status, "approved")

    def test_rejecting_the_major_confounder_unblocks_approval(self):
        self.manager.review_covariate(
            self.state.run_id, "age", "reject", "Dr. Chen",
            "Cohort is age-restricted at enrollment.",
        )
        state = self.manager.decide_gate(self.state.run_id, "dag_review", "approved", "tester")
        self.assertEqual(state.gates["dag_review"].status, "approved")


class WorkflowIdentificationGateTests(unittest.TestCase):
    """The identification_estimand gate must refuse approval when the
    declared DAG is not identified by backdoor adjustment -- e.g. a
    confounder was classified but marked latent/unobserved, so no baseline
    set can block the backdoor path it opens. See `_block_if_not_identified`.
    The fixture's `target_trial` carries the full estimand-contract fields
    (see `causal.agents.dag_identification.build_study_specification`) so
    this gate's *other* new check -- building the full
    `CausalStudySpecification` -- passes whenever identification does,
    keeping these tests isolated to identification behavior specifically."""

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
                "follow_up_days": 28, "primary_estimand": "ATE",
                "eligibility_criteria": ["adult"],
                "assignment_procedure": "Emulated baseline assignment",
                "censoring_strategy": "Administrative censoring at end of follow-up",
                "analysis_strategy": "Intention-to-treat analogue",
                "effect_measure": "risk_difference",
                "treatment_value": "treatment_value",
                "comparator_value": "comparator_value",
            },
            "first_pass_covariates": ["age"],
            "concept_sets": {"treatment": {"status": "approved"}},
        }), encoding="utf-8")
        self.manager = WorkflowManager(root / "runs")
        state = self.manager.start(self.protocol)
        self.state = self.manager.run_hypothesis_agent(state.run_id)

    def tearDown(self):
        self.temp.cleanup()

    def test_identification_estimand_gate_cannot_be_approved_when_the_only_confounder_is_unobserved(self):
        self.manager.review_covariate(
            self.state.run_id, "age", "accept", "Dr. Chen",
            "Baseline age affects both treatment choice and outcome.",
            role="confounder", temporal_classification="pre_treatment",
            observed=False,
        )
        with self.assertRaises(ValueError):
            self.manager.decide_gate(self.state.run_id, "identification_estimand", "approved", "tester")

    def test_identification_estimand_gate_approves_when_the_confounder_is_observed(self):
        self.manager.review_covariate(
            self.state.run_id, "age", "accept", "Dr. Chen",
            "Baseline age affects both treatment choice and outcome.",
            role="confounder", temporal_classification="pre_treatment",
        )
        state = self.manager.decide_gate(self.state.run_id, "identification_estimand", "approved", "tester")
        self.assertEqual(state.gates["identification_estimand"].status, "approved")
        spec_path = self.manager.runs_dir / self.state.run_id / "study_specification.json"
        self.assertTrue(spec_path.is_file())
        self.assertIn("study_specification.json", state.artifacts)

    def test_identification_estimand_gate_can_still_be_rejected_while_unidentified(self):
        self.manager.review_covariate(
            self.state.run_id, "age", "accept", "Dr. Chen",
            "Baseline age affects both treatment choice and outcome.",
            role="confounder", temporal_classification="pre_treatment",
            observed=False,
        )
        self.manager.decide_gate(self.state.run_id, "identification_estimand", "provisional", "tester")
        self.manager.decide_gate(self.state.run_id, "identification_estimand", "rejected", "tester")

    def test_empty_dag_is_trivially_identified_and_not_blocked_here(self):
        """An empty candidate_dag is trivially backdoor-identified (no path
        to block); this gate must not be the thing that catches that --
        `_block_if_major_confounders_unresolved` is."""
        state = self.manager.decide_gate(self.state.run_id, "identification_estimand", "approved", "tester")
        self.assertEqual(state.gates["identification_estimand"].status, "approved")


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
                "follow_up_days": 28, "primary_estimand": "ATE",
                **_FULL_TARGET_TRIAL_FIELDS,
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
        _approve_pre_mapping_gates(self.manager, self.state.run_id)
        for _ in range(3):
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
                "follow_up_days": 28, "primary_estimand": "ATE",
                **_FULL_TARGET_TRIAL_FIELDS,
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
        _approve_pre_mapping_gates(self.manager, self.run_id)
        for _ in range(3):
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


class WorkflowCovariateRejectAfterMappingFailureTests(unittest.TestCase):
    """Regression coverage for the exact support scenario that motivated
    `_render_covariate_card`'s `Reject` tab (`causal.ui.streamlit_app`): an
    already role-classified covariate (accepted, with a domain) that then
    fails OMOP concept mapping (`no_standard_concept` -- a facility-level
    statistic with no representable OMOP concept, not a genuine vocabulary
    gap) must be rejectable from that stage, and rejecting it must actually
    unblock the `analysis_plan` gate rather than leave a stale
    `analysis_plan.json` snapshot still listing it as not ready. Also checks
    that mapping a *different*, genuinely mappable covariate afterward still
    works normally -- the reject path must not interfere with it."""

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
                "follow_up_days": 28, "primary_estimand": "ATE",
                **_FULL_TARGET_TRIAL_FIELDS,
            },
            "data_source": "project.dataset",
            "first_pass_covariates": ["hospital_quality_process_measures", "SOFA score"],
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
            self.run_id, "hospital_quality_process_measures", "accept", "Dr. Chen",
            "AMI process-measure performance correlates with mortality and aspirin use.",
            role="confounder", temporal_classification="pre_treatment",
            domain="observation",
        )
        self.manager.map_covariate_concepts(
            self.run_id, "hospital_quality_process_measures", [], "Dr. Chen",
            mapping_status="no_standard_concept",
        )
        self.manager.review_covariate(
            self.run_id, "SOFA score", "accept", "Dr. Chen",
            "Higher SOFA is associated with treatment choice and outcome.",
            role="confounder", temporal_classification="pre_treatment",
            domain="measurement",
        )
        _approve_pre_mapping_gates(self.manager, self.run_id)
        for _ in range(3):
            state = self.manager.advance(self.run_id)
        state = self.manager.run_data_science_agent(self.run_id)
        state.phase = "analysis_plan"
        self.manager.save(state)

    def tearDown(self):
        self.temp.cleanup()

    def _plan(self) -> dict:
        return json.loads((self.run_dir / "analysis_plan.json").read_text(encoding="utf-8"))

    def test_unmappable_covariate_blocks_the_gate_before_rejection(self):
        plan = self._plan()
        entry = next(c for c in plan["covariates"] if c["name"] == "hospital_quality_process_measures")
        self.assertEqual(entry["mapping_status"], "no_standard_concept")
        self.assertNotEqual(entry["status"], "ready")
        with self.assertRaises(ValueError):
            self.manager.decide_gate(self.run_id, "analysis_plan", "approved", "tester")

    def test_rejecting_then_regenerating_removes_it_from_the_plan(self):
        """Exactly the two calls `_render_covariate_card`'s Reject tab makes."""
        self.manager.review_covariate(
            self.run_id, "hospital_quality_process_measures", "reject", "tester",
            "Facility-level statistic, not representable as an OMOP concept_id.",
        )
        self.manager.run_data_science_agent(self.run_id)
        plan = self._plan()
        self.assertFalse(
            any(c["name"] == "hospital_quality_process_measures" for c in plan["covariates"])
        )
        rejected = json.loads((self.run_dir / "protocol.json").read_text(encoding="utf-8"))["rejected_covariates"]
        self.assertTrue(any(item["label"] == "hospital_quality_process_measures" for item in rejected))

    def test_rejecting_without_regenerating_leaves_the_plan_stale(self):
        """Regression guard for the bug this whole class exists to prevent:
        without the `run_data_science_agent` call, `review_covariate`'s
        reject is fully recorded in protocol.json/hypothesis.json, but
        analysis_plan.json -- what the gate actually reads -- is untouched,
        so the covariate keeps blocking approval even though it was, from
        the reviewer's point of view, already rejected."""
        self.manager.review_covariate(
            self.run_id, "hospital_quality_process_measures", "reject", "tester",
            "Facility-level statistic, not representable as an OMOP concept_id.",
        )
        plan = self._plan()
        self.assertTrue(
            any(c["name"] == "hospital_quality_process_measures" for c in plan["covariates"])
        )
        with self.assertRaises(ValueError):
            self.manager.decide_gate(self.run_id, "analysis_plan", "approved", "tester")

    def test_gate_approves_once_the_unmappable_covariate_is_rejected(self):
        """Both covariates stop blocking once the unmappable one is rejected
        and the mappable one is fully reviewed, and the gate approves
        outright -- this fixture's protocol never declares a `time_zero`
        concept set, but the auto-scaffolded `index_event` fallback
        (`DataScienceAgent.run`) resolves via a standard OMOP table, exactly
        like its `all_cause_death` sibling, so it does not block either."""
        self.manager.review_covariate(
            self.run_id, "hospital_quality_process_measures", "reject", "tester",
            "Facility-level statistic, not representable as an OMOP concept_id.",
        )
        self.manager.run_data_science_agent(self.run_id)
        self.manager.map_covariate_concepts(
            self.run_id, "SOFA score", [4301868], "tester",
        )
        self.manager.review_covariate_aggregation(
            self.run_id, "SOFA score", "last_value_before_index", "tester",
        )
        self.manager.run_data_science_agent(self.run_id)
        plan = self._plan()
        self.assertFalse(plan["unresolved_issues"])
        sofa = next(c for c in plan["covariates"] if c["name"] == "SOFA score")
        self.assertEqual(sofa["status"], "ready")
        self.assertTrue(all(c["status"] == "ready" for c in plan["covariates"]))
        state = self.manager.decide_gate(self.run_id, "analysis_plan", "approved", "tester")
        self.assertEqual(state.gates["analysis_plan"].status, "approved")

    def test_mapping_a_different_covariate_after_a_rejection_still_works(self):
        """The reject path must not interfere with concept mapping for an
        unrelated covariate reviewed afterward."""
        self.manager.review_covariate(
            self.run_id, "hospital_quality_process_measures", "reject", "tester",
            "Facility-level statistic, not representable as an OMOP concept_id.",
        )
        self.manager.run_data_science_agent(self.run_id)
        self.manager.map_covariate_concepts(
            self.run_id, "SOFA score", [4301868], "tester",
        )
        self.manager.run_data_science_agent(self.run_id)
        plan = self._plan()
        sofa = next(c for c in plan["covariates"] if c["name"] == "SOFA score")
        self.assertEqual(sofa["mapping_status"], "approved")
        self.assertEqual(sofa["concept_ids"], [4301868])


class _FakeCohortRowsClient:
    """Duck-typed OMCP stand-in returning a fixed multi-row cohort CSV for
    every `causal_select` call, regardless of the query -- enough to test
    materialization's CSV-parsing/writing path without a real OMCP checkout.
    """

    def __init__(self, csv_text: str):
        self._csv_text = csv_text
        self.calls: list[tuple[str, str, dict]] = []

    def causal_select(self, query: str, operation: str, causal_context: dict) -> MCPToolResponse:
        self.calls.append((query, operation, causal_context))
        return MCPToolResponse(text=self._csv_text, metadata={"rows": 2})


_COHORT_ROWS_CSV = (
    "person_id,A,Y,index_datetime,follow_up_days,censored,age,gender_concept_id,"
    "race_concept_id,cov_sofa_score\n"
    "101,1,1,2024-01-01T00:00:00,28,,60,8507,8527,4.5\n"
    "102,0,0,2024-01-02T00:00:00,28,,45,8532,8527,\n"
)


class WorkflowMaterializeAnalyticDatasetTests(unittest.TestCase):
    """`materialize_analytic_dataset` is the one place patient-level rows are
    ever pulled into memory -- same fixture shape as
    `WorkflowBuildCohortPlanCovariateTests`, carried all the way to the
    `execution` phase with `cohort.sql`/`cohort_manifest.json` already built.

    `ESTIMATION_PROCEDURE` is a class attribute (not a `setUp` parameter) so
    `WorkflowRunEstimationTMLETests` below can subclass and override it to
    reuse this entire fixture unchanged except for which reviewed estimator
    the `statistical_estimand` gate records.
    """

    ESTIMATION_PROCEDURE = "plug_in_standardization"

    def setUp(self):
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
                "follow_up_days": 28, "primary_estimand": "ATE",
                **_FULL_TARGET_TRIAL_FIELDS,
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
        self.manager.map_covariate_concepts(self.run_id, "SOFA score", [4301868], "Dr. Lee")
        self.manager.review_covariate_aggregation(
            self.run_id, "SOFA score", "last_value_before_index", "Dr. Lee",
        )
        _approve_pre_mapping_gates(self.manager, self.run_id)
        for _ in range(3):
            state = self.manager.advance(self.run_id)
        self.manager.decide_gate(self.run_id, "clinical_mapping", "approved", "Dr. Lee")
        self.manager.advance(self.run_id)
        self.manager.run_data_science_agent(self.run_id)
        self.manager.advance(self.run_id)
        plan_path = self.run_dir / "analysis_plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan["exploratory_execution_allowed"] = True
        plan["unresolved_issues"] = []
        for phenotype in plan["phenotypes"]:
            phenotype["status"] = "approved"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        self.manager.decide_gate(self.run_id, "analysis_plan", "approved", "Dr. Lee")
        state = self.manager.advance(self.run_id)
        self.assertEqual(state.phase, "estimand_review")
        self.manager.build_cohort_plan(self.run_id, self.datasets)
        self.manager.set_statistical_estimand_strategy(
            self.run_id,
            missing_data_strategy="complete_case_analysis",
            censoring_strategy="ignore_censoring_administrative_waiver",
            censoring_rationale="No informative censoring expected in this synthetic fixture.",
            reviewer="Dr. Lee",
            estimation_procedure=self.ESTIMATION_PROCEDURE,
        )
        self.manager.decide_gate(self.run_id, "statistical_estimand", "approved", "Dr. Lee")
        state = self.manager.advance(self.run_id)
        self.assertEqual(state.phase, "execution")

    def tearDown(self):
        self.temp.cleanup()

    def test_materializes_the_cohort_and_writes_a_safe_manifest(self):
        client = _FakeCohortRowsClient(_COHORT_ROWS_CSV)
        state = self.manager.materialize_analytic_dataset(self.run_id, client)
        self.assertEqual(len(client.calls), 1)
        _, operation, _ = client.calls[0]
        self.assertEqual(operation, "cohort_build")
        self.assertTrue((self.run_dir / "analytic_dataset.parquet").is_file())
        manifest = json.loads((self.run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["row_count"], 2)
        self.assertIn("analytic_dataset.parquet", state.artifacts)
        self.assertIn("dataset_manifest.json", state.artifacts)
        self.assertTrue(
            any(entry["event"] == "analytic_dataset_materialized" for entry in state.history)
        )

    def test_audit_history_never_carries_a_raw_covariate_value(self):
        """The materialization event is the audit trail for the one place
        patient-level rows are ever pulled into memory -- it must record a
        row count, never a value from the data itself. Checks only the
        materialization event's own `event`/`detail` fields, not its `at`
        timestamp -- an ISO timestamp's fractional-seconds digits are
        effectively random and will eventually contain any given short
        digit/decimal needle by pure coincidence (already seen for both
        "101" and "4.5" in earlier runs of this suite), which would be a
        false positive, not a real leak: `at` can never structurally carry
        a patient value in the first place."""
        client = _FakeCohortRowsClient(_COHORT_ROWS_CSV)
        state = self.manager.materialize_analytic_dataset(self.run_id, client)
        materialize_events = [
            entry for entry in state.history if entry["event"] == "analytic_dataset_materialized"
        ]
        self.assertGreaterEqual(len(materialize_events), 1)
        latest = materialize_events[-1]
        event_text = json.dumps({"event": latest["event"], "detail": latest["detail"]})
        for needle in ("101", "102", "4.5", "8507", "8532"):
            self.assertNotIn(needle, event_text)
        self.assertIn("2 row(s)", event_text)

    def test_requires_execution_phase(self):
        state = self.manager.load(self.run_id)
        state.phase = "analysis_plan"
        self.manager.save(state)
        with self.assertRaises(RuntimeError):
            self.manager.materialize_analytic_dataset(self.run_id, _FakeCohortRowsClient(_COHORT_ROWS_CSV))

    def test_requires_cohort_plan_to_already_exist(self):
        other = self.manager.start(self.protocol)
        with self.assertRaises(RuntimeError):
            self.manager.materialize_analytic_dataset(
                other.run_id, _FakeCohortRowsClient(_COHORT_ROWS_CSV)
            )

    def test_rejects_when_analytic_dataset_build_is_not_allowed(self):
        plan_path = self.run_dir / "analysis_plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan["exploratory_execution_allowed"] = False
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.manager.materialize_analytic_dataset(
                self.run_id, _FakeCohortRowsClient(_COHORT_ROWS_CSV)
            )

    def test_rejects_unsupported_estimation_procedure(self):
        with self.assertRaisesRegex(ValueError, "estimation_procedure"):
            self.manager.set_statistical_estimand_strategy(
                self.run_id,
                missing_data_strategy="complete_case_analysis",
                censoring_strategy="ignore_censoring_administrative_waiver",
                censoring_rationale="test",
                reviewer="Dr. Lee",
                estimation_procedure="ordinary_least_squares",
            )

    def test_old_run_missing_estimation_procedure_defaults_to_plug_in(self):
        """A run whose `statistical_estimand_settings` was persisted before
        `estimation_procedure` existed (simulated here by deleting the key
        after `setUp` already wrote it) must still approve the gate and
        default to `plug_in_standardization` -- `_load_statistical_estimand_
        settings`' backward-compatibility path, not a forced re-review."""
        protocol_path = self.run_dir / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        del protocol["statistical_estimand_settings"]["estimation_procedure"]
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
        state = self.manager.load(self.run_id)
        self.manager._invalidate_gate(state, "statistical_estimand", "test setup")
        self.manager.save(state)
        self.manager.decide_gate(self.run_id, "statistical_estimand", "approved", "Dr. Lee")
        statistical_estimand = json.loads(
            (self.run_dir / "statistical_estimand.json").read_text(encoding="utf-8")
        )
        self.assertEqual(statistical_estimand["estimation_procedure"], "plug_in_standardization")
        self.assertEqual(statistical_estimand["outcome_model"], "logistic_regression_unpenalized")


def _fake_estimation_procedure_llm(prompt: str) -> str:
    return json.dumps({
        "estimation_procedure": "targeted_minimum_loss_based_estimation",
        "rationale": "Doubly robust given the diagnostics passed in this prompt.",
    })


def _never_call_estimation_procedure_llm(prompt: str) -> str:
    raise AssertionError("llm() must not be called when population_summary == 'CATE'")


def _bring_run_to_estimand_review(root: Path, datasets) -> tuple[WorkflowManager, str, Path]:
    """Advance a fresh single-covariate (SOFA) run through every gate to the
    `estimand_review` phase, with `study_specification.json` and
    `cohort_manifest.json` on disk. Stops right after `build_cohort_plan` --
    before the human strategy/gate/advance calls -- since both the
    estimation-procedure suggestion and the companion dialogue are meant to
    run *before* the reviewer has decided anything.
    """
    protocol = root / "protocol.json"
    protocol.write_text(json.dumps({
        "study_id": "test_study",
        "source_question": "Does treatment affect outcome?",
        "target_trial": {
            "population": "population", "treatment": "treatment",
            "comparator": "comparator", "time_zero": "index",
            "grace_period_hours": 24, "outcome": "outcome",
            "follow_up_days": 28, "primary_estimand": "ATE",
            **_FULL_TARGET_TRIAL_FIELDS,
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
    manager = WorkflowManager(root / "runs")
    state = manager.start(protocol)
    state = manager.run_hypothesis_agent(state.run_id)
    run_id = state.run_id
    run_dir = root / "runs" / run_id

    manager.review_covariate(
        run_id, "SOFA score", "accept", "Dr. Chen",
        "Higher SOFA is associated with treatment choice and outcome.",
        role="confounder", temporal_classification="pre_treatment",
        domain="measurement",
    )
    manager.map_covariate_concepts(run_id, "SOFA score", [4301868], "Dr. Lee")
    manager.review_covariate_aggregation(
        run_id, "SOFA score", "last_value_before_index", "Dr. Lee",
    )
    _approve_pre_mapping_gates(manager, run_id)
    for _ in range(3):
        manager.advance(run_id)
    manager.decide_gate(run_id, "clinical_mapping", "approved", "Dr. Lee")
    manager.advance(run_id)
    manager.run_data_science_agent(run_id)
    manager.advance(run_id)
    plan_path = run_dir / "analysis_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["exploratory_execution_allowed"] = True
    plan["unresolved_issues"] = []
    for phenotype in plan["phenotypes"]:
        phenotype["status"] = "approved"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    manager.decide_gate(run_id, "analysis_plan", "approved", "Dr. Lee")
    state = manager.advance(run_id)
    assert state.phase == "estimand_review", state.phase
    manager.build_cohort_plan(run_id, datasets)
    return manager, run_id, run_dir


class WorkflowProposeEstimationProcedureTests(unittest.TestCase):
    """`propose_estimation_procedure` sits at the `estimand_review` gate; see
    `_bring_run_to_estimand_review` for the shared fixture."""

    def setUp(self):
        self.datasets = BigQueryDatasets()
        self.temp = tempfile.TemporaryDirectory()
        self.manager, self.run_id, self.run_dir = _bring_run_to_estimand_review(
            Path(self.temp.name), self.datasets
        )

    def tearDown(self):
        self.temp.cleanup()

    def _protocol(self) -> dict:
        return json.loads((self.run_dir / "protocol.json").read_text(encoding="utf-8"))

    def test_proposes_a_draft_under_the_new_key_without_touching_settings(self):
        state = self.manager.propose_estimation_procedure(self.run_id, _fake_estimation_procedure_llm)
        protocol = self._protocol()
        suggestion = protocol["statistical_estimand_suggestion"]
        self.assertEqual(suggestion["estimation_procedure"], "targeted_minimum_loss_based_estimation")
        self.assertEqual(suggestion["provenance"], "llm_proposed")
        self.assertNotIn("statistical_estimand_settings", protocol)
        self.assertTrue(
            any(entry["event"] == "estimation_procedure_proposed" for entry in state.history)
        )

    def test_cate_short_circuit_never_calls_the_llm(self):
        study_spec_path = self.run_dir / "study_specification.json"
        study_spec = json.loads(study_spec_path.read_text(encoding="utf-8"))
        study_spec["estimand"]["population_summary"] = "CATE"
        study_spec_path.write_text(json.dumps(study_spec), encoding="utf-8")

        self.manager.propose_estimation_procedure(self.run_id, _never_call_estimation_procedure_llm)
        suggestion = self._protocol()["statistical_estimand_suggestion"]
        self.assertEqual(suggestion["estimation_procedure"], "causal_forest_dml")
        self.assertEqual(suggestion["provenance"], "deterministic_cate_rule")

    def test_re_proposing_never_clobbers_an_already_approved_run(self):
        self.manager.set_statistical_estimand_strategy(
            self.run_id,
            missing_data_strategy="complete_case_analysis",
            censoring_strategy="ignore_censoring_administrative_waiver",
            censoring_rationale="No informative censoring expected in this synthetic fixture.",
            reviewer="Dr. Lee",
            estimation_procedure="plug_in_standardization",
        )
        self.manager.decide_gate(self.run_id, "statistical_estimand", "approved", "Dr. Lee")
        before = self._protocol()["statistical_estimand_settings"]

        self.manager.propose_estimation_procedure(self.run_id, _fake_estimation_procedure_llm)
        protocol = self._protocol()
        self.assertEqual(protocol["statistical_estimand_settings"], before)
        self.assertEqual(
            protocol["statistical_estimand_suggestion"]["estimation_procedure"],
            "targeted_minimum_loss_based_estimation",
        )

    def test_requires_study_specification_to_exist(self):
        (self.run_dir / "study_specification.json").unlink()
        with self.assertRaises(RuntimeError):
            self.manager.propose_estimation_procedure(self.run_id, _fake_estimation_procedure_llm)

    def test_rejects_llm_suggesting_causal_forest_dml_for_non_cate_study(self):
        def _fake_llm(prompt: str) -> str:
            return json.dumps({"estimation_procedure": "causal_forest_dml", "rationale": "x"})

        with self.assertRaises(ValueError):
            self.manager.propose_estimation_procedure(self.run_id, _fake_llm)


class _CapturingLLM:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


_COMPANION_REPLY_ONLY = json.dumps({"reply": "Let's start with the censoring strategy."})
_COMPANION_WITH_ACTION = json.dumps({
    "reply": "Recording IPCW plus TMLE, as we discussed.",
    "proposed_action": {
        "action": "set_statistical_estimand_strategy",
        "args": {
            "missing_data_strategy": "complete_case_analysis",
            "censoring_strategy": "ipcw_administrative_censoring",
            "estimation_procedure": "targeted_minimum_loss_based_estimation",
            "censoring_rationale": "Follow-up audit shows non-trivial administrative censoring; reweight rather than drop.",
        },
    },
})


class WorkflowRunCompanionTurnTests(unittest.TestCase):
    """`run_companion_turn` is the LLM-owning half of the estimand dialogue;
    the pure prompt-build/parse is covered by `test_companion_agent.py`."""

    def setUp(self):
        self.datasets = BigQueryDatasets()
        self.temp = tempfile.TemporaryDirectory()
        self.manager, self.run_id, self.run_dir = _bring_run_to_estimand_review(
            Path(self.temp.name), self.datasets
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_returns_a_parsed_reply_only_turn(self):
        turn = self.manager.run_companion_turn(self.run_id, _CapturingLLM(_COMPANION_REPLY_ONLY), [])
        self.assertEqual(turn.reply, "Let's start with the censoring strategy.")
        self.assertIsNone(turn.proposed_action)

    def test_returns_a_validated_proposed_action(self):
        turn = self.manager.run_companion_turn(self.run_id, _CapturingLLM(_COMPANION_WITH_ACTION), [])
        self.assertIsNotNone(turn.proposed_action)
        self.assertEqual(turn.proposed_action.action, "set_statistical_estimand_strategy")
        self.assertEqual(
            turn.proposed_action.args["censoring_strategy"], "ipcw_administrative_censoring"
        )

    def test_prompt_is_grounded_in_the_run_artifacts_and_history(self):
        manifest = json.loads((self.run_dir / "cohort_manifest.json").read_text(encoding="utf-8"))
        covariate = next(
            column["name"] for column in manifest["columns"]
            if column["role"] == "baseline_covariate"
        )
        llm = _CapturingLLM(_COMPANION_REPLY_ONLY)
        self.manager.run_companion_turn(
            self.run_id, llm, [CompanionMessage("user", "what does IPCW change?")]
        )
        prompt = llm.prompts[0]
        self.assertIn(covariate, prompt)
        self.assertIn("Reviewer: what does IPCW change?", prompt)

    def test_does_not_mutate_run_state(self):
        before = self.manager.load(self.run_id)
        self.manager.run_companion_turn(self.run_id, _CapturingLLM(_COMPANION_WITH_ACTION), [])
        after = self.manager.load(self.run_id)
        self.assertEqual(len(after.history), len(before.history))
        self.assertEqual(after.updated_at, before.updated_at)
        self.assertNotIn(
            "statistical_estimand_settings",
            json.loads((self.run_dir / "protocol.json").read_text(encoding="utf-8")),
        )

    def test_prompt_carries_the_decision_memory(self):
        protocol_path = self.run_dir / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["rejected_covariates"] = [
            {
                "label": "baseline_lactate",
                "reviewer": "Dr. Chen",
                "comment": "measured after the exposure; post-treatment",
                "decided_at": "2026-08-20T09:00:00+00:00",
            }
        ]
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

        llm = _CapturingLLM(_COMPANION_REPLY_ONLY)
        self.manager.run_companion_turn(self.run_id, llm, [])
        prompt = llm.prompts[0]
        self.assertIn("PRIOR DECISIONS", prompt)
        self.assertIn("baseline_lactate", prompt)
        # a covariate review the fixture recorded on the way to estimand_review
        self.assertIn("SOFA score", prompt)

    def test_decision_memory_read_helper_is_pure(self):
        protocol_path = self.run_dir / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["rejected_covariates"] = [
            {"label": "baseline_lactate", "reviewer": "Dr. Chen", "comment": "post-treatment",
             "decided_at": "2026-08-20T09:00:00+00:00"}
        ]
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

        before = self.manager.load(self.run_id)
        memory = self.manager.decision_memory(self.run_id)
        after = self.manager.load(self.run_id)

        self.assertEqual([item.label for item in memory.rejected_covariates], ["baseline_lactate"])
        self.assertTrue(memory.decisions)  # covariate reviews from the fixture
        self.assertEqual(len(after.history), len(before.history))
        self.assertFalse((self.run_dir / "decision_memory.json").exists())

    def test_reviewer_memory_reaches_the_prompt_only_when_a_reviewer_is_given(self):
        protocol_path = self.run_dir / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["rejected_covariates"] = [
            {"label": "old_covariate", "reviewer": "Dr. Chen", "comment": "post-treatment",
             "decided_at": "2026-07-01T00:00:00+00:00"}
        ]
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
        self.manager.remember_run_for_reviewer(self.run_id, "Dr. Chen")

        without = _CapturingLLM(_COMPANION_REPLY_ONLY)
        self.manager.run_companion_turn(self.run_id, without, [])
        self.assertIn("No cross-study conventions recorded", without.prompts[0])

        with_reviewer = _CapturingLLM(_COMPANION_REPLY_ONLY)
        self.manager.run_companion_turn(self.run_id, with_reviewer, [], reviewer="Dr. Chen")
        prompt = with_reviewer.prompts[0]
        self.assertIn("REVIEWER CONVENTIONS", prompt)
        self.assertIn("excluded in earlier studies", prompt)
        self.assertIn("old_covariate", prompt)

    def test_rejects_the_wrong_phase(self):
        state = self.manager.load(self.run_id)
        state.phase = "execution"
        self.manager.save(state)
        with self.assertRaises(RuntimeError):
            self.manager.run_companion_turn(self.run_id, _CapturingLLM(_COMPANION_REPLY_ONLY), [])

    def test_requires_study_specification(self):
        (self.run_dir / "study_specification.json").unlink()
        with self.assertRaises(RuntimeError):
            self.manager.run_companion_turn(self.run_id, _CapturingLLM(_COMPANION_REPLY_ONLY), [])

    def test_malformed_llm_response_raises(self):
        with self.assertRaises(ValueError):
            self.manager.run_companion_turn(self.run_id, _CapturingLLM("not json"), [])

    def test_smuggled_reviewer_in_action_fails_the_turn(self):
        smuggled = json.dumps({
            "reply": "recording",
            "proposed_action": {
                "action": "set_statistical_estimand_strategy",
                "args": {
                    "missing_data_strategy": "complete_case_analysis",
                    "censoring_strategy": "ignore_censoring_administrative_waiver",
                    "estimation_procedure": "plug_in_standardization",
                    "censoring_rationale": "waiver",
                    "reviewer": "not_a_real_person",
                },
            },
        })
        with self.assertRaises(ValueError):
            self.manager.run_companion_turn(self.run_id, _CapturingLLM(smuggled), [])


class WorkflowReviewerMemoryTests(unittest.TestCase):
    """`remember_run_for_reviewer` / `reviewer_memory` -- the opt-in
    cross-study memory (`causal.companion.reviewer_memory`)."""

    def setUp(self):
        self.datasets = BigQueryDatasets()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.manager, self.run_id, self.run_dir = _bring_run_to_estimand_review(
            Path(self.temp.name), self.datasets
        )
        protocol_path = self.run_dir / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["rejected_covariates"] = [
            {"label": "baseline_lactate", "reviewer": "Dr. Chen",
             "comment": "post-treatment", "decided_at": "2026-08-01T00:00:00+00:00"}
        ]
        protocol["statistical_estimand_settings"] = {
            "missing_data_strategy": "complete_case_analysis",
            "censoring_strategy": "ignore_censoring_administrative_waiver",
            "estimation_procedure": "plug_in_standardization",
        }
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    def test_unknown_reviewer_has_empty_memory(self):
        self.assertTrue(self.manager.reviewer_memory("Nobody New").is_empty())

    def test_remember_folds_the_run_and_persists_under_a_slug(self):
        merged = self.manager.remember_run_for_reviewer(self.run_id, "Dr. Chen")
        self.assertEqual([e.label for e in merged.exclusions], ["baseline_lactate"])
        self.assertEqual(len(merged.estimand_choices), 1)

        path = Path(self.temp.name) / "reviewers" / "dr_chen" / "memory.json"
        self.assertTrue(path.is_file())
        self.assertEqual(self.manager.reviewer_memory("Dr. Chen"), merged)

    def test_remember_is_idempotent_for_the_same_study(self):
        self.manager.remember_run_for_reviewer(self.run_id, "Dr. Chen")
        second = self.manager.remember_run_for_reviewer(self.run_id, "Dr. Chen")
        self.assertEqual([e.label for e in second.exclusions], ["baseline_lactate"])
        self.assertEqual(len(second.estimand_choices), 1)

    def test_remember_does_not_touch_the_run(self):
        before = self.manager.load(self.run_id)
        self.manager.remember_run_for_reviewer(self.run_id, "Dr. Chen")
        after = self.manager.load(self.run_id)
        self.assertEqual(len(after.history), len(before.history))
        self.assertFalse((self.run_dir / "decision_memory.json").exists())

    def test_empty_reviewer_name_raises(self):
        with self.assertRaises(ValueError):
            self.manager.remember_run_for_reviewer(self.run_id, "   ")


def _synthetic_cohort_csv(n: int = 300, seed: int = 11) -> str:
    """A larger, statistically usable cohort CSV (unlike `_COHORT_ROWS_CSV`,
    which only has 2 rows for the plumbing-only materialization tests) --
    enough rows and outcome variance in both arms for `EstimationAgent`'s
    complete-case + bootstrap requirements to succeed."""
    rng = random.Random(seed)
    lines = [
        "person_id,A,Y,index_datetime,follow_up_days,censored,age,"
        "gender_concept_id,race_concept_id,cov_sofa_score"
    ]
    for i in range(n):
        treatment = rng.randint(0, 1)
        age = rng.randint(18, 90)
        risk = min(max(0.2 + 0.15 * treatment + 0.002 * age, 0.01), 0.99)
        outcome = 1 if rng.random() < risk else 0
        gender = rng.choice((8507, 8532))
        race = rng.choice((8527, 8516))
        sofa = round(rng.uniform(0, 20), 1)
        lines.append(
            f"{100 + i},{treatment},{outcome},2024-01-01T00:00:00,28,,"
            f"{age},{gender},{race},{sofa}"
        )
    return "\n".join(lines) + "\n"


class WorkflowRunEstimationTests(WorkflowMaterializeAnalyticDatasetTests):
    """`run_estimation` is the wiring that makes `EstimationAgent` actually
    runnable inside a workflow run -- same fixture as
    `WorkflowMaterializeAnalyticDatasetTests`, carried one step further
    through a real materialization with a statistically usable cohort."""

    def setUp(self):
        super().setUp()
        self.manager.materialize_analytic_dataset(
            self.run_id, _FakeCohortRowsClient(_synthetic_cohort_csv())
        )

    def test_writes_an_estimation_artifact(self):
        state = self.manager.run_estimation(self.run_id, n_bootstrap=30, random_seed=1)
        artifact_path = self.run_dir / "estimation.json"
        self.assertTrue(artifact_path.is_file())
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.assertEqual(artifact["status"], "estimated")
        self.assertEqual(artifact["estimand"]["method"], "g_computation_standardization")
        self.assertIn("cov_sofa_score", artifact["estimand"]["covariates_adjusted"])
        self.assertIn("estimation.json", state.artifacts)
        self.assertTrue(any(entry["event"] == "estimation_completed" for entry in state.history))

    def test_requires_materialized_dataset(self):
        (self.run_dir / "analytic_dataset.parquet").unlink()
        with self.assertRaises(RuntimeError):
            self.manager.run_estimation(self.run_id)

    def test_requires_execution_or_diagnostics_phase(self):
        state = self.manager.load(self.run_id)
        state.phase = "analysis_plan"
        self.manager.save(state)
        with self.assertRaises(RuntimeError):
            self.manager.run_estimation(self.run_id)

    def test_rejects_when_causal_estimation_is_not_allowed(self):
        self.manager.decide_gate(self.run_id, "clinical_mapping", "rejected", "Dr. Lee")
        with self.assertRaises(RuntimeError):
            self.manager.run_estimation(self.run_id)

    def test_rejects_when_the_statistical_estimand_gate_was_invalidated(self):
        """A reviewer who changes the censoring/procedure strategy after the
        gate was approved re-opens it (`_invalidate_gate` -> pending) but
        `statistical_estimand.json` deliberately stays on disk. `run_
        estimation` must refuse rather than silently estimate against the
        retracted strategy. The happy path (re-approve, then estimate) is
        the standard fixture every other test in this class already runs."""
        self.manager.set_statistical_estimand_strategy(
            self.run_id,
            missing_data_strategy="complete_case_analysis",
            censoring_strategy="exclude_censored_before_analysis",
            censoring_rationale="Re-reviewed: drop censored rows instead.",
            reviewer="Dr. Lee",
            estimation_procedure=self.ESTIMATION_PROCEDURE,
        )
        self.assertTrue((self.run_dir / "statistical_estimand.json").is_file())
        self.assertEqual(
            self.manager.load(self.run_id).gates["statistical_estimand"].status, "pending"
        )
        with self.assertRaisesRegex(RuntimeError, "statistical_estimand gate is not approved"):
            self.manager.run_estimation(self.run_id)

class WorkflowRunEstimationTMLETests(WorkflowRunEstimationTests):
    """`run_estimation` must dispatch to `TMLEAgent` -- not `EstimationAgent`
    -- when the reviewed `statistical_estimand.json` names
    `estimation_procedure="targeted_minimum_loss_based_estimation"`. Same
    fixture as `WorkflowRunEstimationTests` (inherited, including its
    guard-rail tests, which must keep passing unchanged here too), with only
    `ESTIMATION_PROCEDURE` overridden so `setUp`'s `statistical_estimand`
    gate records the TMLE procedure instead of the default plug-in one."""

    ESTIMATION_PROCEDURE = "targeted_minimum_loss_based_estimation"

    def test_writes_an_estimation_artifact(self):
        # Overrides (not just extends) the parent test: `n_bootstrap` is
        # meaningless for TMLE, and the written artifact's `method`/
        # `outcome_model` differ from the plug-in baseline's -- the parent
        # class's version of this same-named test would otherwise also run
        # here unmodified and assert the wrong method string.
        state = self.manager.run_estimation(self.run_id, random_seed=1)
        artifact_path = self.run_dir / "estimation.json"
        self.assertTrue(artifact_path.is_file())
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.assertEqual(artifact["status"], "estimated")
        self.assertEqual(artifact["estimand"]["method"], "targeted_minimum_loss_based_estimation")
        self.assertEqual(artifact["estimand"]["outcome_model"], "super_learner_ensemble")
        self.assertIn("cov_sofa_score", artifact["estimand"]["covariates_adjusted"])
        self.assertIn("estimation.json", state.artifacts)
        self.assertTrue(any(entry["event"] == "estimation_completed" for entry in state.history))

    def test_statistical_estimand_records_the_reviewed_procedure(self):
        statistical_estimand = json.loads(
            (self.run_dir / "statistical_estimand.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            statistical_estimand["estimation_procedure"], "targeted_minimum_loss_based_estimation"
        )
        self.assertEqual(statistical_estimand["outcome_model"], "super_learner_ensemble")
        self.assertEqual(statistical_estimand["schema_version"], "1.2")


class WorkflowCATETests(WorkflowRunEstimationTests):
    """`run_cate`'s guard logic and `set_cate_cutpoint` -- same fixture as
    `WorkflowRunEstimationTests`, which declares the default
    `population_summary="ATE"` (the base protocol's `target_trial.
    primary_estimand`). Most of what matters here is that CATE is never
    runnable unless a study explicitly declared it as the primary
    estimand -- exercising a real `CATEAgent` fit through the full
    governed pipeline is covered directly and far more cheaply in
    `tests/test_cate_agent.py`."""

    def test_rejects_when_population_summary_is_not_cate(self):
        statistical_estimand = json.loads(
            (self.run_dir / "statistical_estimand.json").read_text(encoding="utf-8")
        )
        self.assertEqual(statistical_estimand["population_summary"], "ATE")
        with self.assertRaisesRegex(RuntimeError, "CATE"):
            self.manager.run_cate(self.run_id)

    def test_requires_materialized_dataset(self):
        (self.run_dir / "analytic_dataset.parquet").unlink()
        with self.assertRaises(RuntimeError):
            self.manager.run_cate(self.run_id)

    def test_requires_execution_or_diagnostics_phase(self):
        state = self.manager.load(self.run_id)
        state.phase = "analysis_plan"
        self.manager.save(state)
        with self.assertRaises(RuntimeError):
            self.manager.run_cate(self.run_id)

    def test_set_cate_cutpoint_records_the_value_under_the_slugified_column_name(self):
        state = self.manager.set_cate_cutpoint(self.run_id, "SOFA score", 5.0, "Dr. Lee")
        protocol = json.loads((self.run_dir / "protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(protocol["cate_cutpoints"]["cov_sofa_score"], 5.0)
        self.assertTrue(any(entry["event"] == "cate_cutpoint_set" for entry in state.history))

    def test_set_cate_cutpoint_requires_a_reviewer(self):
        with self.assertRaises(ValueError):
            self.manager.set_cate_cutpoint(self.run_id, "SOFA score", 5.0, "  ")

    def test_set_cate_cutpoint_requires_a_covariate_name(self):
        with self.assertRaises(ValueError):
            self.manager.set_cate_cutpoint(self.run_id, "  ", 5.0, "Dr. Lee")


class DispatchEstimatorCausalForestDMLTests(unittest.TestCase):
    """Regression test for `_dispatch_estimator`'s handling of
    `estimation_procedure="causal_forest_dml"` -- a CATE study's forced
    value (`roadmap.StatisticalEstimand.__post_init__`). Exercised directly
    against `_dispatch_estimator` rather than through
    `WorkflowManager.run_estimation`/`run_cate`'s full governed pipeline:
    fitting a real `CausalForestDML` (`tests/test_cate_agent.py`'s job,
    deliberately not duplicated here or in `WorkflowCATETests`, see that
    class's own docstring) is not what this checks -- this checks that
    `run_estimation`'s dispatch does NOT try to route a `"causal_forest_dml"`
    estimand through `CATEAgent` (whose `CATEEstimate` has no
    `effect_measure`/`point_estimate`/`ci_lower`/`ci_upper` fields and would
    make `run_estimation`'s `state.record(...)` call raise `AttributeError`
    the moment it ran), falling through to the plug-in g-computation
    baseline instead, same as an unset/default `estimation_procedure`."""

    def test_falls_through_to_plug_in_standardization(self):
        dataframe = pd.read_csv(io.StringIO(_synthetic_cohort_csv()))
        statistical_estimand = {
            "estimation_procedure": "causal_forest_dml",
            "censoring_strategy": "ignore_censoring_administrative_waiver",
            "missing_data_strategy": "complete_case_analysis",
        }
        artifact = _dispatch_estimator(
            statistical_estimand, dataframe,
            study_id="dispatch_test", covariate_columns=("age", "cov_sofa_score"),
            n_bootstrap=30, random_seed=1,
        )
        self.assertEqual(artifact.estimand.method, "g_computation_standardization")
        self.assertEqual(artifact.estimand.outcome_model, "logistic_regression_unpenalized")


def _fake_negative_control_llm(prompt: str) -> str:
    return json.dumps({"candidates": [
        {
            "control_type": "negative_control_exposure",
            "label": "Vitamin D supplementation",
            "rationale": "No known mechanism affecting the study outcome.",
        },
    ]})


class WorkflowProposeNegativeControlsTests(WorkflowRunEstimationTests):
    """Negative-control candidate proposal -- same fixture as
    `WorkflowRunEstimationTests` (already through execution phase with a
    materialized dataset and approved statistical_estimand)."""

    def test_proposes_candidates_never_pre_approved(self):
        state = self.manager.propose_negative_controls(self.run_id, _fake_negative_control_llm)
        protocol = json.loads((self.run_dir / "protocol.json").read_text(encoding="utf-8"))
        negative_controls = protocol["negative_controls"]
        self.assertEqual(len(negative_controls), 1)
        entry = next(iter(negative_controls.values()))
        self.assertEqual(entry["mapping_status"], "candidate_pending_human_approval")
        self.assertEqual(entry["provenance"], "llm_proposed")
        self.assertTrue(any(entry["event"] == "negative_controls_proposed" for entry in state.history))

    def test_re_proposing_does_not_overwrite_an_already_approved_candidate(self):
        self.manager.propose_negative_controls(self.run_id, _fake_negative_control_llm)
        protocol = json.loads((self.run_dir / "protocol.json").read_text(encoding="utf-8"))
        control_id = next(iter(protocol["negative_controls"]))
        self.manager.map_negative_control(self.run_id, control_id, "drug", [999], "Dr. Lee", status="approved")

        self.manager.propose_negative_controls(self.run_id, _fake_negative_control_llm)
        protocol = json.loads((self.run_dir / "protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(protocol["negative_controls"][control_id]["mapping_status"], "approved")
        self.assertEqual(protocol["negative_controls"][control_id]["concept_ids"], [999])

    def test_requires_hypothesis_to_have_run(self):
        other = self.manager.start(self.protocol)
        with self.assertRaises(RuntimeError):
            self.manager.propose_negative_controls(other.run_id, _fake_negative_control_llm)


class WorkflowMapNegativeControlTests(WorkflowRunEstimationTests):
    def setUp(self):
        super().setUp()
        self.manager.propose_negative_controls(self.run_id, _fake_negative_control_llm)
        protocol = json.loads((self.run_dir / "protocol.json").read_text(encoding="utf-8"))
        self.control_id = next(iter(protocol["negative_controls"]))

    def test_approves_a_candidate(self):
        state = self.manager.map_negative_control(
            self.run_id, self.control_id, "drug", [999], "Dr. Lee", status="approved",
        )
        protocol = json.loads((self.run_dir / "protocol.json").read_text(encoding="utf-8"))
        entry = protocol["negative_controls"][self.control_id]
        self.assertEqual(entry["mapping_status"], "approved")
        self.assertEqual(entry["concept_ids"], [999])
        self.assertEqual(entry["reviewer"], "Dr. Lee")
        self.assertTrue(any(entry["event"] == "negative_control_mapped" for entry in state.history))

    def test_rejects_a_candidate(self):
        self.manager.map_negative_control(
            self.run_id, self.control_id, "drug", [], "Dr. Lee", status="not_clinically_defensible",
        )
        protocol = json.loads((self.run_dir / "protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(
            protocol["negative_controls"][self.control_id]["mapping_status"], "not_clinically_defensible"
        )

    def test_requires_a_known_control_id(self):
        with self.assertRaises(RuntimeError):
            self.manager.map_negative_control(self.run_id, "not_a_real_id", "drug", [999], "Dr. Lee")

    def test_approved_status_requires_concept_ids(self):
        with self.assertRaises(ValueError):
            self.manager.map_negative_control(
                self.run_id, self.control_id, "drug", [], "Dr. Lee", status="approved",
            )

    def test_rejects_unknown_domain(self):
        with self.assertRaises(ValueError):
            self.manager.map_negative_control(self.run_id, self.control_id, "not_a_domain", [999], "Dr. Lee")


class WorkflowRunNegativeControlEstimationTests(WorkflowRunEstimationTests):
    def setUp(self):
        super().setUp()
        self.manager.propose_negative_controls(self.run_id, _fake_negative_control_llm)
        protocol = json.loads((self.run_dir / "protocol.json").read_text(encoding="utf-8"))
        self.control_id = next(iter(protocol["negative_controls"]))
        self.manager.map_negative_control(
            self.run_id, self.control_id, "drug", [999], "Dr. Lee", status="approved",
        )

    def test_writes_a_negative_control_estimation_artifact(self):
        state = self.manager.run_negative_control_estimation(
            self.run_id, self.control_id, _FakeCohortRowsClient(_synthetic_cohort_csv()),
            n_bootstrap=30, random_seed=1,
        )
        result_path = self.run_dir / f"negative_control_estimation_{self.control_id}.json"
        self.assertTrue(result_path.is_file())
        artifact = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(artifact["status"], "estimated")
        self.assertIn(f"negative_control_estimation_{self.control_id}.json", state.artifacts)
        sql_path = self.run_dir / f"negative_control_{self.control_id}_cohort.sql"
        self.assertTrue(sql_path.is_file())
        self.assertIn("999", sql_path.read_text(encoding="utf-8"))
        self.assertTrue(
            any(entry["event"] == "negative_control_estimation_completed" for entry in state.history)
        )
        # The audit trail is row-count only -- the same discipline
        # `materialize_analytic_dataset` uses for its own history entry.
        completed = next(e for e in state.history if e["event"] == "negative_control_estimation_completed")
        self.assertIn("row(s)", completed["detail"])

    def test_requires_the_control_to_be_approved(self):
        protocol_path = self.run_dir / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["negative_controls"]["pending_one"] = {
            "control_id": "pending_one", "control_type": "negative_control_exposure",
            "label": "Pending", "rationale": "x", "domain": "", "concept_ids": [],
            "mapping_status": "candidate_pending_human_approval", "provenance": "llm_proposed",
        }
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.manager.run_negative_control_estimation(
                self.run_id, "pending_one", _FakeCohortRowsClient(_synthetic_cohort_csv()),
            )

    def test_requires_a_known_control_id(self):
        with self.assertRaises(RuntimeError):
            self.manager.run_negative_control_estimation(
                self.run_id, "not_a_real_id", _FakeCohortRowsClient(_synthetic_cohort_csv()),
            )

    def test_requires_execution_phase(self):
        state = self.manager.load(self.run_id)
        state.phase = "analysis_plan"
        self.manager.save(state)
        with self.assertRaises(RuntimeError):
            self.manager.run_negative_control_estimation(
                self.run_id, self.control_id, _FakeCohortRowsClient(_synthetic_cohort_csv()),
            )

    def test_negative_control_rejects_when_statistical_estimand_gate_invalidated(self):
        self.manager.set_statistical_estimand_strategy(
            self.run_id,
            missing_data_strategy="complete_case_analysis",
            censoring_strategy="exclude_censored_before_analysis",
            censoring_rationale="Re-reviewed: drop censored rows instead.",
            reviewer="Dr. Lee",
        )
        with self.assertRaisesRegex(RuntimeError, "statistical_estimand gate is not approved"):
            self.manager.run_negative_control_estimation(
                self.run_id, self.control_id, _FakeCohortRowsClient(_synthetic_cohort_csv()),
            )

    def test_remains_callable_after_advancing_to_results_review(self):
        """Regression test: `run_negative_control_estimation` used to be
        pinned to `phase == "execution"` only, unlike its siblings
        (`run_estimation`/`run_diagnostics`/`run_sensitivity_analysis`/
        `run_cate`), so a reviewer who approved a negative control late
        could never estimate it once the primary estimate had already
        advanced the run past `diagnostics`."""
        self.manager.run_estimation(self.run_id, n_bootstrap=30, random_seed=1)
        self.manager.advance(self.run_id)
        state = self.manager.advance(self.run_id)
        self.assertEqual(state.phase, "results_review")
        state = self.manager.run_negative_control_estimation(
            self.run_id, self.control_id, _FakeCohortRowsClient(_synthetic_cohort_csv()),
            n_bootstrap=30, random_seed=1,
        )
        result_path = self.run_dir / f"negative_control_estimation_{self.control_id}.json"
        self.assertTrue(result_path.is_file())
        self.assertIn(f"negative_control_estimation_{self.control_id}.json", state.artifacts)

    def test_causal_report_includes_a_negative_controls_section(self):
        self.manager.run_estimation(self.run_id, n_bootstrap=30, random_seed=1)
        self.manager.run_negative_control_estimation(
            self.run_id, self.control_id, _FakeCohortRowsClient(_synthetic_cohort_csv()),
            n_bootstrap=30, random_seed=1,
        )
        self.manager.advance(self.run_id)
        state = self.manager.advance(self.run_id)
        self.assertEqual(state.phase, "results_review")
        report_path = self.manager.build_causal_report(self.run_id)
        text = report_path.read_text(encoding="utf-8")
        self.assertIn("Negative controls", text)
        self.assertIn(self.control_id, text)

    def test_does_not_persist_the_negative_control_dataframe(self):
        self.manager.run_negative_control_estimation(
            self.run_id, self.control_id, _FakeCohortRowsClient(_synthetic_cohort_csv()),
            n_bootstrap=30, random_seed=1,
        )
        self.assertFalse((self.run_dir / f"negative_control_{self.control_id}_dataset.parquet").exists())
        self.assertFalse(any(self.run_dir.glob("negative_control_*.parquet")))


class WorkflowRunDiagnosticsTests(WorkflowRunEstimationTests):
    """`run_diagnostics` needs no fitted model -- same materialized-dataset
    fixture as `WorkflowRunEstimationTests`, but exercised independently of
    `run_estimation` to confirm it does not require it to have run first."""

    def test_writes_a_diagnostics_artifact(self):
        state = self.manager.run_diagnostics(self.run_id)
        artifact_path = self.run_dir / "diagnostics.json"
        self.assertTrue(artifact_path.is_file())
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.assertIn("Austin", artifact["citation"])
        self.assertEqual(artifact["smd_flag_threshold"], 0.1)
        names = {entry["column_name"] for entry in artifact["balance"]}
        self.assertIn("cov_sofa_score", names)
        self.assertTrue(any(name.startswith("gender_concept_id=") for name in names))
        self.assertIn("diagnostics.json", state.artifacts)
        self.assertTrue(any(entry["event"] == "diagnostics_completed" for entry in state.history))

    def test_does_not_require_estimation_to_have_run_first(self):
        self.assertFalse((self.run_dir / "estimation.json").is_file())
        self.manager.run_diagnostics(self.run_id)
        self.assertTrue((self.run_dir / "diagnostics.json").is_file())

    def test_requires_materialized_dataset(self):
        (self.run_dir / "analytic_dataset.parquet").unlink()
        with self.assertRaises(RuntimeError):
            self.manager.run_diagnostics(self.run_id)

    def test_requires_execution_or_diagnostics_phase(self):
        state = self.manager.load(self.run_id)
        state.phase = "analysis_plan"
        self.manager.save(state)
        with self.assertRaises(RuntimeError):
            self.manager.run_diagnostics(self.run_id)


class WorkflowRunSensitivityAnalysisTests(WorkflowRunEstimationTests):
    """`run_sensitivity_analysis` needs an already-computed `estimation.json`
    (unlike `run_diagnostics`, which needs no fitted model) -- same
    materialized-dataset fixture as `WorkflowRunEstimationTests`, carried one
    step further through a real `run_estimation` call."""

    def setUp(self):
        super().setUp()
        self.manager.run_estimation(self.run_id, n_bootstrap=30, random_seed=1)

    def test_writes_a_sensitivity_analysis_artifact(self):
        state = self.manager.run_sensitivity_analysis(self.run_id)
        artifact_path = self.run_dir / "sensitivity_analysis.json"
        self.assertTrue(artifact_path.is_file())
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.assertIn("VanderWeele", artifact["citation"])
        self.assertGreaterEqual(artifact["e_value_point"], 1.0)
        self.assertGreaterEqual(artifact["e_value_ci"], 1.0)
        self.assertIn("sensitivity_analysis.json", state.artifacts)
        self.assertTrue(any(entry["event"] == "sensitivity_analysis_completed" for entry in state.history))

    def test_sensitivity_analysis_artifact_carries_provenance_metadata(self):
        """Regression test: `sensitivity_analysis.json` used to be written
        straight from `EValueResult.to_dict()`, with no `metadata`/
        `study_id` -- unlike every other artifact this workflow writes --
        so it could never appear in `build_causal_report`'s Provenance
        table. `causal.sensitivity.e_value` itself stays a pure function
        (deliberately no `study_id`/metadata surface of its own); the
        wrapping happens here, at the call site, the same way
        `run_results_narrative` wraps `results_narrative.json`."""
        state = self.manager.run_sensitivity_analysis(self.run_id)
        artifact = json.loads((self.run_dir / "sensitivity_analysis.json").read_text(encoding="utf-8"))
        self.assertEqual(artifact["schema_version"], "1.0")
        self.assertEqual(artifact["study_id"], state.study_id)
        self.assertEqual(artifact["metadata"]["agent_name"], "sensitivity_agent")
        self.assertTrue(artifact["metadata"]["generated_at_utc"])

    def test_requires_estimation_to_have_run_first(self):
        (self.run_dir / "estimation.json").unlink()
        with self.assertRaises(RuntimeError):
            self.manager.run_sensitivity_analysis(self.run_id)

    def test_requires_execution_or_diagnostics_phase(self):
        state = self.manager.load(self.run_id)
        state.phase = "analysis_plan"
        self.manager.save(state)
        with self.assertRaises(RuntimeError):
            self.manager.run_sensitivity_analysis(self.run_id)

    def test_causal_report_includes_the_e_value_section(self):
        self.manager.run_sensitivity_analysis(self.run_id)
        self.manager.advance(self.run_id)
        state = self.manager.advance(self.run_id)
        self.assertEqual(state.phase, "results_review")
        report_path = self.manager.build_causal_report(self.run_id)
        report_text = report_path.read_text(encoding="utf-8")
        self.assertIn("Sensitivity analysis (E-value)", report_text)
        self.assertIn("VanderWeele", report_text)


class WorkflowCausalReportTests(WorkflowRunEstimationTests):
    """`build_causal_report` is the auditable causal report (roadmap stage
    11) -- and the first test coverage of the pre-existing `results` gate
    actually being reached and exercised (see `REQUIRED_GATES["results_
    review"] == "results"`, previously unused by any test)."""

    def setUp(self):
        super().setUp()
        self.manager.run_estimation(self.run_id, n_bootstrap=30, random_seed=1)
        self.manager.run_diagnostics(self.run_id)

    def test_requires_results_review_or_report_phase(self):
        with self.assertRaises(RuntimeError):
            self.manager.build_causal_report(self.run_id)

    def test_builds_a_report_with_estimate_and_diagnostics_at_results_review(self):
        self.manager.advance(self.run_id)
        state = self.manager.advance(self.run_id)
        self.assertEqual(state.phase, "results_review")
        report_path = self.manager.build_causal_report(self.run_id)
        text = report_path.read_text(encoding="utf-8")
        self.assertIn("Causal effect estimate", text)
        self.assertIn("g_computation_standardization", text)
        self.assertIn("Covariate balance", text)
        self.assertIn("Assumptions and limitations", text)
        self.assertIn("PROVISIONAL", text)
        self.assertIn("causal_report.md", self.manager.load(self.run_id).artifacts)
        self.assertIn("An average treatment effect has been estimated", text)

    def test_statistical_estimand_section_names_the_reviewed_procedure(self):
        # A plain ATE run: the reviewed procedure and the estimation.json
        # outcome model agree, and the CATE cross-check note is absent.
        self.manager.advance(self.run_id)
        self.manager.advance(self.run_id)
        text = self.manager.build_causal_report(self.run_id).read_text(encoding="utf-8")
        self.assertIn("Estimation procedure: plug_in_standardization", text)
        self.assertNotIn("plug-in ATE cross-check", text)
        self.assertNotIn("secondary cross-check", text)

    def test_results_gate_approves_once_estimation_exists(self):
        self.manager.advance(self.run_id)
        state = self.manager.advance(self.run_id)
        self.assertEqual(state.phase, "results_review")
        state = self.manager.decide_gate(self.run_id, "results", "approved", "Dr. Lee")
        self.assertEqual(state.gates["results"].status, "approved")
        state = self.manager.advance(self.run_id)
        self.assertEqual(state.phase, "report")
        report_path = self.manager.build_causal_report(self.run_id)
        text = report_path.read_text(encoding="utf-8")
        self.assertIn("Results gate approved by Dr. Lee", text)
        self.assertNotIn("PROVISIONAL", text)

    def test_provenance_table_includes_sensitivity_and_cate_when_present(self):
        """Regression test: the Provenance table's artifact list was a
        hard-coded tuple that never included `sensitivity_analysis.json` or
        `cate.json`, even though both are written with the same `metadata`
        block every other listed artifact has. `cate.json` is faked directly
        here rather than run through a real `CATEAgent` fit -- exercising
        the real agent is covered far more cheaply in
        `tests/test_cate_agent.py`; this test only needs an artifact with a
        `metadata` key to exist on disk."""
        self.manager.run_sensitivity_analysis(self.run_id)
        (self.run_dir / "cate.json").write_text(json.dumps({
            "metadata": artifact_metadata("cate_agent", "0.1.0", {"study_id": "test_study"}),
            "schema_version": "1.0",
        }), encoding="utf-8")
        self.manager.advance(self.run_id)
        state = self.manager.advance(self.run_id)
        self.assertEqual(state.phase, "results_review")
        report_path = self.manager.build_causal_report(self.run_id)
        text = report_path.read_text(encoding="utf-8")
        self.assertIn("| sensitivity_analysis.json | sensitivity_agent |", text)
        self.assertIn("| cate.json | cate_agent |", text)

    def test_cate_run_labels_the_plug_in_ate_as_a_cross_check(self):
        # For a CATE study `_dispatch_estimator` falls through to the plug-in
        # g-computation (logistic outcome model) as a cross-check, while
        # `statistical_estimand.json` records `super_learner_ensemble` for
        # the forest -- the report must frame that pair as expected, not as
        # a contradiction.
        estimand_path = self.run_dir / "statistical_estimand.json"
        estimand = json.loads(estimand_path.read_text(encoding="utf-8"))
        estimand["population_summary"] = "CATE"
        estimand["estimation_procedure"] = "causal_forest_dml"
        estimand["outcome_model"] = "super_learner_ensemble"
        estimand_path.write_text(json.dumps(estimand), encoding="utf-8")
        self.manager.advance(self.run_id)
        self.manager.advance(self.run_id)
        text = self.manager.build_causal_report(self.run_id).read_text(encoding="utf-8")
        self.assertIn("Estimation procedure: causal_forest_dml", text)
        self.assertIn("Causal effect estimate (plug-in ATE cross-check)", text)
        self.assertIn("secondary cross-check", text)
        self.assertNotIn("This is the study's estimated average treatment effect", text)

    def test_report_snapshots_the_decision_memory(self):
        protocol_path = self.run_dir / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["rejected_covariates"] = [
            {"label": "baseline_lactate", "reviewer": "Dr. Chen",
             "comment": "post-treatment", "decided_at": "2026-08-20T09:00:00+00:00"}
        ]
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
        self.manager.advance(self.run_id)
        self.manager.advance(self.run_id)

        report_path = self.manager.build_causal_report(self.run_id)
        text = report_path.read_text(encoding="utf-8")
        self.assertIn("## Reviewer decisions", text)
        self.assertIn("baseline_lactate", text)

        snapshot_path = self.run_dir / "decision_memory.json"
        self.assertTrue(snapshot_path.is_file())
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        self.assertEqual(
            [item["label"] for item in snapshot["rejected_covariates"]], ["baseline_lactate"]
        )
        self.assertEqual(snapshot["metadata"]["agent_name"], "decision_memory")
        state = self.manager.load(self.run_id)
        self.assertIn("decision_memory.json", state.artifacts)
        self.assertIn("| decision_memory.json | decision_memory |", text)


class WorkflowCausalReportWithoutEstimateTests(WorkflowMaterializeAnalyticDatasetTests):
    """A report built before `run_estimation` ever runs (e.g. a legacy run,
    or a reviewer previewing early) must not claim an effect was estimated
    -- regression test for a real bug caught by hand-testing against an
    existing run directory: the Interpretation section used to say "An
    average treatment effect has been estimated" unconditionally."""

    def test_report_does_not_claim_an_estimate_exists(self):
        self.manager.advance(self.run_id)
        state = self.manager.advance(self.run_id)
        self.assertEqual(state.phase, "results_review")
        report_path = self.manager.build_causal_report(self.run_id)
        text = report_path.read_text(encoding="utf-8")
        self.assertNotIn("Causal effect estimate", text)
        self.assertNotIn("An average treatment effect has been estimated", text)
        self.assertIn("No causal effect has been estimated for this run yet", text)

    def test_advance_to_report_requires_the_results_gate_approved(self):
        self.manager.advance(self.run_id)
        state = self.manager.advance(self.run_id)
        self.assertEqual(state.phase, "results_review")
        with self.assertRaises(RuntimeError):
            self.manager.advance(self.run_id)

    def test_results_gate_cannot_be_approved_without_estimation(self):
        """Regression test for the governance gap where `results` -- the
        final, most consequential sign-off in the pipeline -- could be
        rubber-stamped "approved" on a run that never called
        `run_estimation` at all (`REQUIRED_GATES` has no gate for the
        `execution`/`diagnostics` phases, so nothing upstream forced it to
        run). See `_block_if_results_incomplete`."""
        self.manager.advance(self.run_id)
        state = self.manager.advance(self.run_id)
        self.assertEqual(state.phase, "results_review")
        with self.assertRaises(ValueError):
            self.manager.decide_gate(self.run_id, "results", "approved", "Dr. Lee")
        self.assertEqual(self.manager.load(self.run_id).gates["results"].status, "pending")


class WorkflowResultsGateDiagnosticsOptionalTests(WorkflowMaterializeAnalyticDatasetTests):
    """`_block_if_results_incomplete` only requires `estimation.json` --
    `run_diagnostics` is explicitly optional elsewhere in this module (it
    "can run before or after `run_estimation`", per its own docstring), so
    the `results` gate must not force it either. A standalone fixture
    (rather than a test method on the shared `WorkflowRunEstimationTests`
    base) because several of that class's subclasses run `run_diagnostics`
    in their own `setUp`, which would make an inherited "diagnostics is
    absent" precondition false for them."""

    def test_results_gate_approves_without_diagnostics(self):
        self.manager.materialize_analytic_dataset(
            self.run_id, _FakeCohortRowsClient(_synthetic_cohort_csv())
        )
        self.manager.run_estimation(self.run_id, n_bootstrap=30, random_seed=1)
        self.assertFalse((self.run_dir / "diagnostics.json").is_file())
        self.manager.advance(self.run_id)
        state = self.manager.advance(self.run_id)
        self.assertEqual(state.phase, "results_review")
        state = self.manager.decide_gate(self.run_id, "results", "approved", "Dr. Lee")
        self.assertEqual(state.gates["results"].status, "approved")


class _ResultsNarrativeFixture:
    """Shared `setUp`/helpers for `WorkflowResultsNarrativeTests` and
    `WorkflowResultsNarrativeReviewTests` -- a plain mixin (not itself a
    `unittest.TestCase`), so each concrete class picks only the test
    *methods* it declares, never the other's. Not a subclass of
    `WorkflowCausalReportTests` either, even though the setup steps largely
    overlap: this fixture's own `setUp` advances two phases further (all the
    way to `results_review`), and subclassing would also inherit its (and
    its ancestors') test methods, most of which assert phase guards that
    only hold at the `execution` phase -- those would spuriously fail
    against a fixture already at `results_review`.
    """

    def setUp(self):
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
                "follow_up_days": 28, "primary_estimand": "ATE",
                **_FULL_TARGET_TRIAL_FIELDS,
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
        self.manager.map_covariate_concepts(self.run_id, "SOFA score", [4301868], "Dr. Lee")
        self.manager.review_covariate_aggregation(
            self.run_id, "SOFA score", "last_value_before_index", "Dr. Lee",
        )
        _approve_pre_mapping_gates(self.manager, self.run_id)
        for _ in range(3):
            state = self.manager.advance(self.run_id)
        self.manager.decide_gate(self.run_id, "clinical_mapping", "approved", "Dr. Lee")
        self.manager.advance(self.run_id)
        self.manager.run_data_science_agent(self.run_id)
        self.manager.advance(self.run_id)
        plan_path = self.run_dir / "analysis_plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan["exploratory_execution_allowed"] = True
        plan["unresolved_issues"] = []
        for phenotype in plan["phenotypes"]:
            phenotype["status"] = "approved"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        self.manager.decide_gate(self.run_id, "analysis_plan", "approved", "Dr. Lee")
        self.manager.advance(self.run_id)
        self.manager.build_cohort_plan(self.run_id, self.datasets)
        self.manager.set_statistical_estimand_strategy(
            self.run_id,
            missing_data_strategy="complete_case_analysis",
            censoring_strategy="ignore_censoring_administrative_waiver",
            censoring_rationale="No informative censoring expected in this synthetic fixture.",
            reviewer="Dr. Lee",
        )
        self.manager.decide_gate(self.run_id, "statistical_estimand", "approved", "Dr. Lee")
        self.manager.advance(self.run_id)
        self.manager.materialize_analytic_dataset(
            self.run_id, _FakeCohortRowsClient(_synthetic_cohort_csv())
        )
        self.manager.run_estimation(self.run_id, n_bootstrap=30, random_seed=1)
        self.manager.run_diagnostics(self.run_id)
        self.manager.advance(self.run_id)
        state = self.manager.advance(self.run_id)
        self.assertEqual(state.phase, "results_review")

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _valid_claims_llm(narrative="Plain language interpretation of the estimate.", claims=None):
        """A structurally-valid (but not necessarily faithful) response --
        for tests that only care that parsing/writing succeeds, not about
        the faithfulness-verification outcome."""
        if claims is None:
            claims = [{
                "text": "See narrative.",
                "claim_type": "effect_direction",
                "source_fields": ["estimation.estimand.point_estimate"],
            }]

        def fake_llm(prompt: str) -> str:
            return json.dumps({"narrative": narrative, "claims": claims})

        return fake_llm

    def _faithful_llm(self):
        """A response built from the run's own already-computed
        `estimation.json`/`diagnostics.json`, so `verify_narrative` grades
        every check `passed` regardless of which way the synthetic cohort's
        randomly-seeded effect happened to point."""
        estimation = json.loads((self.run_dir / "estimation.json").read_text(encoding="utf-8"))
        diagnostics = json.loads((self.run_dir / "diagnostics.json").read_text(encoding="utf-8"))
        estimand = estimation["estimand"]
        direction_word = "an increase" if estimand["point_estimate"] > 0 else "a decrease"
        crosses_null = estimand["ci_lower"] <= 0 <= estimand["ci_upper"]
        uncertainty_phrase = "crosses the null" if crosses_null else "does not cross the null"
        claims = [
            {
                "text": f"The point estimate suggests {direction_word} in risk.",
                "claim_type": "effect_direction",
                "source_fields": ["estimation.estimand.point_estimate"],
            },
            {
                "text": f"The confidence interval {uncertainty_phrase}.",
                "claim_type": "uncertainty",
                "source_fields": ["estimation.estimand.ci_lower", "estimation.estimand.ci_upper"],
            },
            {
                "text": "The analysis is based on the full analytic cohort.",
                "claim_type": "sample_size",
                "source_fields": ["estimation.estimand.n_total"],
            },
            {
                "text": "This analysis assumes no unmeasured confounding.",
                "claim_type": "assumption_or_limitation",
                "source_fields": ["assumptions_and_limitations"],
            },
        ]
        if diagnostics.get("n_covariates_flagged"):
            claims.append({
                "text": "Some covariates show meaningful imbalance between arms.",
                "claim_type": "diagnostic_warning",
                "source_fields": ["diagnostics.n_covariates_flagged"],
            })
        narrative_text = (
            f"The estimated effect suggests {direction_word} in risk, and its interval "
            f"{uncertainty_phrase}."
        )

        def fake_llm(prompt: str) -> str:
            return json.dumps({"narrative": narrative_text, "claims": claims})

        return fake_llm

    @staticmethod
    def _unfaithful_llm():
        """A response with a number nowhere in the artifacts -- deterministically
        fails the faithfulness verifier's number check regardless of the
        run's actual estimate."""
        def fake_llm(prompt: str) -> str:
            return json.dumps({
                "narrative": "The point estimate is 999.999, a huge and fabricated number.",
                "claims": [{
                    "text": "See narrative.",
                    "claim_type": "effect_direction",
                    "source_fields": ["estimation.estimand.point_estimate"],
                }],
            })

        return fake_llm


class WorkflowResultsNarrativeTests(_ResultsNarrativeFixture, unittest.TestCase):
    """`run_results_narrative` is optional and must degrade gracefully."""

    def test_writes_a_results_narrative_artifact(self):
        state = self.manager.run_results_narrative(
            self.run_id, self._valid_claims_llm("The estimated risk difference is small and imprecise."),
            model="test-model", provider="test-provider",
        )
        artifact_path = self.run_dir / "results_narrative.json"
        self.assertTrue(artifact_path.is_file())
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.assertEqual(artifact["narrative"], "The estimated risk difference is small and imprecise.")
        self.assertIn("advisory only", artifact["boundary_statement"])
        self.assertEqual(artifact["model"], "test-model")
        self.assertEqual(artifact["provider"], "test-provider")
        self.assertEqual(artifact["review_status"], "pending")
        self.assertIsNone(artifact["reviewer"])
        self.assertIn(artifact["verification"]["status"], {"passed", "warning", "failed"})
        self.assertEqual(len(artifact["claims"]), 1)
        self.assertIn("results_narrative.json", state.artifacts)
        self.assertTrue(any(e["event"] == "results_narrative_generated" for e in state.history))

    def test_prompt_carries_the_pre_rounded_narrative_display_block(self):
        """The results-narrative prompt must include the deterministic
        `narrative_display` block (rounded values + plain-language
        restatements) so the model quotes it instead of rounding the raw
        floats itself -- which `verify_narrative` would reject."""
        captured: dict[str, str] = {}

        def capturing_llm(prompt: str) -> str:
            captured["prompt"] = prompt
            return json.dumps({
                "narrative": "See claims.",
                "claims": [{
                    "text": "See narrative.", "claim_type": "effect_direction",
                    "source_fields": ["estimation.narrative_display.absolute_effect.plain_statement"],
                }],
            })

        self.manager.run_results_narrative(self.run_id, capturing_llm)
        self.assertIn("narrative_display", captured["prompt"])
        self.assertIn("events_per_n_point", captured["prompt"])
        self.assertIn("number_needed_to_treat", captured["prompt"])

    def test_causal_report_includes_the_narrative_once_approved(self):
        self.manager.run_results_narrative(self.run_id, self._faithful_llm())
        self.manager.review_results_narrative(self.run_id, "approved", "Dr. Lee")
        report_path = self.manager.build_causal_report(self.run_id)
        text = report_path.read_text(encoding="utf-8")
        self.assertIn("LLM interpretive narrative", text)
        self.assertIn("its interval", text)
        self.assertIn("advisory only", text)
        self.assertIn("Approved by Dr. Lee", text)

    def test_causal_report_effect_section_is_rounded_and_has_a_plain_terms_block(self):
        report_path = self.manager.build_causal_report(self.run_id)
        text = report_path.read_text(encoding="utf-8")
        self.assertIn("### In plain terms", text)
        self.assertIn("events per 1000 patients", text)
        self.assertIn("rounded for readability", text)

    def test_report_omits_narrative_section_when_absent(self):
        report_path = self.manager.build_causal_report(self.run_id)
        text = report_path.read_text(encoding="utf-8")
        self.assertIn("LLM interpretive narrative", text)
        self.assertIn("No LLM interpretation was requested for this run.", text)

    def test_an_unapproved_narrative_is_not_included_in_the_report(self):
        """A generated-but-still-`pending` narrative must not leak its text
        into the final report -- only an explicit approval does."""
        self.manager.run_results_narrative(self.run_id, self._faithful_llm())
        report_path = self.manager.build_causal_report(self.run_id)
        text = report_path.read_text(encoding="utf-8")
        self.assertNotIn("The estimated effect suggests", text)
        self.assertIn("pending human review", text)

    def test_requires_estimation_to_have_run(self):
        (self.run_dir / "estimation.json").unlink()
        with self.assertRaises(RuntimeError):
            self.manager.run_results_narrative(self.run_id, self._valid_claims_llm())

    def test_a_failing_llm_call_does_not_write_an_artifact_or_affect_the_run(self):
        """A broken/unavailable LLM must not corrupt the run or block a
        report from being built -- the narrative section is purely
        additive, never required."""
        def broken_llm(prompt: str) -> str:
            raise RuntimeError("network down")

        with self.assertRaises(RuntimeError):
            self.manager.run_results_narrative(self.run_id, broken_llm)
        self.assertFalse((self.run_dir / "results_narrative.json").is_file())
        report_path = self.manager.build_causal_report(self.run_id)
        self.assertTrue(report_path.is_file())
        self.assertIn("No LLM interpretation was requested for this run.", report_path.read_text(encoding="utf-8"))

    def test_narrative_requires_results_review_or_report_phase(self):
        state = self.manager.load(self.run_id)
        state.phase = "execution"
        self.manager.save(state)
        with self.assertRaises(RuntimeError):
            self.manager.run_results_narrative(self.run_id, self._valid_claims_llm())

    def test_a_response_without_claims_is_rejected(self):
        def no_claims_llm(prompt: str) -> str:
            return json.dumps({"narrative": "text only"})

        with self.assertRaises(ValueError):
            self.manager.run_results_narrative(self.run_id, no_claims_llm)
        self.assertFalse((self.run_dir / "results_narrative.json").is_file())


class WorkflowResultsNarrativeReviewTests(_ResultsNarrativeFixture, unittest.TestCase):
    """`review_results_narrative` is the human-review step, entirely
    separate from the mandatory `results` gate (`decide_gate`): approving or
    excluding a narrative never touches `state.gates["results"]`."""

    def test_approves_a_narrative_with_no_failed_checks(self):
        self.manager.run_results_narrative(self.run_id, self._faithful_llm())
        state = self.manager.review_results_narrative(self.run_id, "approved", "Dr. Lee", "Looks faithful.")
        artifact = json.loads((self.run_dir / "results_narrative.json").read_text(encoding="utf-8"))
        self.assertEqual(artifact["review_status"], "approved")
        self.assertEqual(artifact["reviewer"], "Dr. Lee")
        self.assertEqual(artifact["review_note"], "Looks faithful.")
        self.assertIsNotNone(artifact["reviewed_at"])
        self.assertEqual(state.gates["results"].status, "pending")
        self.assertTrue(any(e["event"] == "results_narrative_reviewed" for e in state.history))

    def test_refuses_approval_when_a_check_failed(self):
        self.manager.run_results_narrative(self.run_id, self._unfaithful_llm())
        artifact = json.loads((self.run_dir / "results_narrative.json").read_text(encoding="utf-8"))
        self.assertEqual(artifact["verification"]["status"], "failed")
        with self.assertRaises(ValueError):
            self.manager.review_results_narrative(self.run_id, "approved", "Dr. Lee")
        artifact_after = json.loads((self.run_dir / "results_narrative.json").read_text(encoding="utf-8"))
        self.assertEqual(artifact_after["review_status"], "pending")

    def test_exclusion_always_succeeds_even_with_failed_checks(self):
        self.manager.run_results_narrative(self.run_id, self._unfaithful_llm())
        self.manager.review_results_narrative(self.run_id, "excluded", "Dr. Lee", "Fabricated number.")
        artifact = json.loads((self.run_dir / "results_narrative.json").read_text(encoding="utf-8"))
        self.assertEqual(artifact["review_status"], "excluded")
        self.assertEqual(artifact["review_note"], "Fabricated number.")

    def test_excluded_narrative_lets_the_report_build_without_its_text(self):
        self.manager.run_results_narrative(self.run_id, self._unfaithful_llm())
        self.manager.review_results_narrative(self.run_id, "excluded", "Dr. Lee")
        report_path = self.manager.build_causal_report(self.run_id)
        text = report_path.read_text(encoding="utf-8")
        self.assertIn("The LLM interpretation was excluded from this version of the report.", text)
        self.assertNotIn("999.999", text)

    def test_requires_an_existing_narrative_artifact(self):
        with self.assertRaises(RuntimeError):
            self.manager.review_results_narrative(self.run_id, "approved", "Dr. Lee")

    def test_requires_a_non_empty_reviewer(self):
        self.manager.run_results_narrative(self.run_id, self._faithful_llm())
        with self.assertRaises(ValueError):
            self.manager.review_results_narrative(self.run_id, "approved", "  ")

    def test_rejects_an_unknown_decision(self):
        self.manager.run_results_narrative(self.run_id, self._faithful_llm())
        with self.assertRaises(ValueError):
            self.manager.review_results_narrative(self.run_id, "rejected", "Dr. Lee")

    def test_requires_results_review_or_report_phase(self):
        self.manager.run_results_narrative(self.run_id, self._faithful_llm())
        state = self.manager.load(self.run_id)
        state.phase = "execution"
        self.manager.save(state)
        with self.assertRaises(RuntimeError):
            self.manager.review_results_narrative(self.run_id, "approved", "Dr. Lee")


class WorkflowResultsCompanionTurnTests(_ResultsNarrativeFixture, unittest.TestCase):
    """`run_results_companion_turn` is the LLM-owning half of the
    results-interpretation dialogue; the pure prompt-build/parse is covered
    by `test_companion_agent.py`. `save_results_narrative_draft` persists a
    drafted narrative through the same path as the one-shot step."""

    def _draft_payload(self) -> dict:
        """A structurally valid, faithful draft built from the run's own
        estimation.json -- reuses `_faithful_llm`'s response body."""
        return json.loads(self._faithful_llm()(""))

    def test_returns_a_parsed_reply_only_turn(self):
        llm = _CapturingLLM(json.dumps({"reply": "Let's start with the direction of the effect."}))
        turn = self.manager.run_results_companion_turn(self.run_id, llm, [])
        self.assertEqual(turn.reply, "Let's start with the direction of the effect.")
        self.assertIsNone(turn.drafted_narrative)

    def test_prompt_is_grounded_in_the_estimate_and_history(self):
        estimation = json.loads((self.run_dir / "estimation.json").read_text(encoding="utf-8"))
        n_total = str(estimation["estimand"]["n_total"])
        llm = _CapturingLLM(json.dumps({"reply": "ok"}))
        self.manager.run_results_companion_turn(
            self.run_id, llm, [CompanionMessage("user", "which way does it point?")]
        )
        prompt = llm.prompts[0]
        self.assertIn(n_total, prompt)
        self.assertIn("Reviewer: which way does it point?", prompt)
        self.assertNotIn("analytic_dataset.parquet", prompt)

    def test_turn_can_carry_a_validated_draft(self):
        llm = _CapturingLLM(json.dumps({"reply": "Drafted it.", "drafted_narrative": self._draft_payload()}))
        turn = self.manager.run_results_companion_turn(self.run_id, llm, [])
        self.assertIsNotNone(turn.drafted_narrative)
        self.assertGreaterEqual(len(turn.drafted_narrative.claims), 3)

    def test_a_turn_does_not_mutate_run_state_or_write_an_artifact(self):
        before = self.manager.load(self.run_id)
        llm = _CapturingLLM(json.dumps({"reply": "Drafted it.", "drafted_narrative": self._draft_payload()}))
        self.manager.run_results_companion_turn(self.run_id, llm, [])
        after = self.manager.load(self.run_id)
        self.assertEqual(len(after.history), len(before.history))
        self.assertFalse((self.run_dir / "results_narrative.json").is_file())

    def test_rejects_the_wrong_phase(self):
        state = self.manager.load(self.run_id)
        state.phase = "analysis_plan"
        self.manager.save(state)
        with self.assertRaises(RuntimeError):
            self.manager.run_results_companion_turn(self.run_id, _CapturingLLM("{}"), [])

    def test_rejects_a_missing_estimation_artifact(self):
        (self.run_dir / "estimation.json").unlink()
        with self.assertRaises(RuntimeError):
            self.manager.run_results_companion_turn(self.run_id, _CapturingLLM("{}"), [])

    def test_save_draft_writes_a_pending_artifact_tagged_dialogue(self):
        state = self.manager.save_results_narrative_draft(
            self.run_id, self._draft_payload(), [CompanionMessage("user", "go ahead")],
            model="test-model", provider="test-provider",
        )
        artifact = json.loads((self.run_dir / "results_narrative.json").read_text(encoding="utf-8"))
        self.assertEqual(artifact["source"], "dialogue")
        self.assertEqual(artifact["review_status"], "pending")
        self.assertEqual(artifact["model"], "test-model")
        self.assertIn(artifact["verification"]["status"], {"passed", "warning", "failed"})
        self.assertIn("results_narrative.json", state.artifacts)
        self.assertTrue(any(
            e["event"] == "results_narrative_generated" and "source=dialogue" in e["detail"]
            for e in state.history
        ))

    def test_a_saved_draft_goes_through_the_normal_human_review(self):
        self.manager.save_results_narrative_draft(
            self.run_id, self._draft_payload(), [], model="m", provider="p"
        )
        self.manager.review_results_narrative(self.run_id, "approved", "Dr. Lee", "Faithful to the estimate.")
        artifact = json.loads((self.run_dir / "results_narrative.json").read_text(encoding="utf-8"))
        self.assertEqual(artifact["review_status"], "approved")
        report = self.manager.build_causal_report(self.run_id).read_text(encoding="utf-8")
        self.assertIn("Approved by Dr. Lee", report)

    def test_save_draft_rejects_a_malformed_payload(self):
        with self.assertRaises(ValueError):
            self.manager.save_results_narrative_draft(
                self.run_id, {"narrative": "text", "claims": []}, [], model="m", provider="p"
            )
        self.assertFalse((self.run_dir / "results_narrative.json").is_file())


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


class _FakeCausalSelectClient:
    """Duck-typed stand-in for `OMCPClient`: records every `causal_select`
    call and answers with one generic aggregate CSV row, regardless of which
    audit query was sent -- `run_mapping_audit_mcp` only needs each response
    to parse as zero-or-more CSV rows, it never asserts a specific shape."""

    def __init__(self):
        self.calls: list[tuple[str, str, dict]] = []

    def causal_select(self, query: str, operation: str, causal_context: dict) -> MCPToolResponse:
        self.calls.append((query, operation, causal_context))
        return MCPToolResponse(text="concept_id,row_count\n123,10\n", metadata={"rows": 1})


class WorkflowMappingAuditMcpTests(unittest.TestCase):
    """`run_mapping_audit_mcp` is the wiring that makes `causal.mapping.audit`'s
    SQL diagnostics actually runnable by a reviewer -- before it existed,
    nothing outside `tests/test_mapping_audit.py` ever called those SQL
    builders. Exercises the plumbing (registry lookup -> ExtractionPlan ->
    governed OMCP execution -> written artifact) with a duck-typed fake OMCP
    client, since `OMCPClient` itself requires a real `omcp` checkout."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.protocol_path = root / "protocol.json"
        self.protocol_path.write_text(json.dumps({
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
        state = self.manager.start(self.protocol_path)
        self.run_id = state.run_id
        self.run_dir = root / "runs" / self.run_id
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
        self.datasets = BigQueryDatasets()

    def tearDown(self):
        self.temp.cleanup()

    def test_runs_every_applicable_check_and_writes_an_artifact(self):
        """'sofa_score' is a measurement (VALUE_DOMAINS) covariate, so the
        unit-observation check is included alongside the three checks every
        variable gets."""
        client = _FakeCausalSelectClient()
        state = self.manager.run_mapping_audit_mcp(self.run_id, "sofa_score", client, self.datasets)
        self.assertEqual(len(client.calls), 7)
        self.assertTrue(all(operation == "aggregate_feasibility" for _, operation, _ in client.calls))
        artifact_path = self.run_dir / "mapping_audit_sofa_score.json"
        self.assertTrue(artifact_path.is_file())
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.assertEqual(artifact["variable_id"], "sofa_score")
        self.assertEqual(artifact["status"], "aggregate_mapping_audit_only")
        self.assertEqual(
            set(artifact["checks"]),
            {
                "concept_row_and_person_counts",
                "approved_concepts_missing_from_data",
                "invalid_or_nonstandard_concepts",
                "observed_units",
                "orphan_concepts",
                "index_event_breakdown",
                "incidence_rate",
            },
        )
        self.assertIn("mapping_audit_sofa_score.json", state.artifacts)
        self.assertTrue(any(entry["event"] == "mapping_audit_completed" for entry in state.history))

    def test_omits_observed_units_for_a_presence_only_domain_variable(self):
        """'population' is a condition-domain phenotype (presence_in_window,
        not a VALUE_DOMAINS aggregation) -- units are meaningless for it."""
        client = _FakeCausalSelectClient()
        self.manager.run_mapping_audit_mcp(self.run_id, "population", client, self.datasets)
        artifact = json.loads(
            (self.run_dir / "mapping_audit_population.json").read_text(encoding="utf-8")
        )
        self.assertNotIn("observed_units", artifact["checks"])
        self.assertIn("orphan_concepts", artifact["checks"])
        self.assertIn("index_event_breakdown", artifact["checks"])
        self.assertIn("incidence_rate", artifact["checks"])
        self.assertEqual(len(client.calls), 6)

    def test_requires_registry_json_to_already_exist(self):
        other = self.manager.start(self.protocol_path)
        with self.assertRaises(RuntimeError):
            self.manager.run_mapping_audit_mcp(
                other.run_id, "population", _FakeCausalSelectClient(), self.datasets
            )

    def test_rejects_a_variable_id_not_in_the_registry(self):
        with self.assertRaises(RuntimeError):
            self.manager.run_mapping_audit_mcp(
                self.run_id, "not_a_real_variable", _FakeCausalSelectClient(), self.datasets
            )

    def test_the_causal_context_identifies_the_run_and_variable(self):
        client = _FakeCausalSelectClient()
        self.manager.run_mapping_audit_mcp(self.run_id, "sofa_score", client, self.datasets)
        _, _, context = client.calls[0]
        self.assertEqual(context["run_id"], self.run_id)
        self.assertEqual(context["variable_id"], "sofa_score")


class WorkflowDataSupportAuditTests(unittest.TestCase):
    """`run_positivity_audit_mcp`/`run_covariate_missingness_audit_mcp`
    (data-support roadmap stage 7) and `approve_censoring_definition`
    (stage 7's censoring piece) -- same fixture shape as
    `WorkflowBuildCohortPlanCovariateTests`, a full pipeline through one
    reviewed, mapped, aggregation-reviewed covariate."""

    def setUp(self):
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
                "follow_up_days": 28, "primary_estimand": "ATE",
                **_FULL_TARGET_TRIAL_FIELDS,
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
        _approve_pre_mapping_gates(self.manager, self.run_id)
        for _ in range(3):
            state = self.manager.advance(self.run_id)
        state = self.manager.run_data_science_agent(self.run_id)
        state.phase = "analysis_plan"
        self.manager.save(state)

    def tearDown(self):
        self.temp.cleanup()

    def test_positivity_audit_writes_an_artifact(self):
        client = _FakeCausalSelectClient()
        state = self.manager.run_positivity_audit_mcp(self.run_id, client, self.datasets)
        self.assertEqual(len(client.calls), 1)
        query, operation, _ = client.calls[0]
        self.assertIn("cov_sofa_score_treated_min", query)
        self.assertEqual(operation, "aggregate_feasibility")
        artifact_path = self.run_dir / "positivity_audit.json"
        self.assertTrue(artifact_path.is_file())
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.assertEqual(artifact["status"], "aggregate_observability_audit_only")
        self.assertFalse(artifact["causal_effect_estimated"])
        self.assertIn("positivity_audit.json", state.artifacts)
        self.assertTrue(any(entry["event"] == "positivity_audit_completed" for entry in state.history))

    def test_positivity_audit_requires_analysis_plan_json(self):
        (self.run_dir / "analysis_plan.json").unlink()
        with self.assertRaises(RuntimeError):
            self.manager.run_positivity_audit_mcp(self.run_id, _FakeCausalSelectClient(), self.datasets)

    def test_covariate_missingness_audit_writes_an_artifact(self):
        client = _FakeCausalSelectClient()
        state = self.manager.run_covariate_missingness_audit_mcp(self.run_id, client, self.datasets)
        self.assertEqual(len(client.calls), 1)
        query, _, _ = client.calls[0]
        self.assertIn("cov_sofa_score_missing", query)
        artifact_path = self.run_dir / "covariate_missingness.json"
        self.assertTrue(artifact_path.is_file())
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.assertEqual(artifact["status"], "aggregate_observability_audit_only")
        self.assertIn("covariate_missingness.json", state.artifacts)
        self.assertTrue(
            any(entry["event"] == "covariate_missingness_audit_completed" for entry in state.history)
        )

    def test_covariate_missingness_audit_requires_analysis_plan_json(self):
        (self.run_dir / "analysis_plan.json").unlink()
        with self.assertRaises(RuntimeError):
            self.manager.run_covariate_missingness_audit_mcp(
                self.run_id, _FakeCausalSelectClient(), self.datasets
            )

    def test_approve_censoring_definition_requires_followup_audit_first(self):
        with self.assertRaises(ValueError):
            self.manager.approve_censoring_definition(self.run_id, "Dr. Lee")

    def test_approve_censoring_definition_flips_the_flag(self):
        (self.run_dir / "followup_audit.json").write_text(
            json.dumps({"status": "aggregate_observability_audit_only"}), encoding="utf-8"
        )
        state = self.manager.approve_censoring_definition(self.run_id, "Dr. Lee", "reviewed nominal counts")
        protocol = json.loads((self.run_dir / "protocol.json").read_text(encoding="utf-8"))
        self.assertTrue(protocol["query_settings"]["censoring_definition_approved"])
        event = next(entry for entry in state.history if entry["event"] == "censoring_definition_approved")
        self.assertEqual(event["changes"]["after"], True)

    def test_approve_censoring_definition_invalidates_analysis_plan_gate(self):
        (self.run_dir / "followup_audit.json").write_text(
            json.dumps({"status": "aggregate_observability_audit_only"}), encoding="utf-8"
        )
        self.manager.decide_gate(self.run_id, "analysis_plan", "provisional", "Dr. Lee")
        state = self.manager.approve_censoring_definition(self.run_id, "Dr. Lee")
        self.assertEqual(state.gates["analysis_plan"].status, "pending")

    def test_approved_censoring_definition_produces_a_real_censored_column(self):
        (self.run_dir / "followup_audit.json").write_text(
            json.dumps({"status": "aggregate_observability_audit_only"}), encoding="utf-8"
        )
        self.manager.approve_censoring_definition(self.run_id, "Dr. Lee")
        self.manager.build_cohort_plan(self.run_id, self.datasets)
        sql = (self.run_dir / "cohort.sql").read_text(encoding="utf-8")
        self.assertNotIn("CAST(NULL AS INT64) AS censored", sql)
        manifest = json.loads((self.run_dir / "cohort_manifest.json").read_text(encoding="utf-8"))
        self.assertNotIn(
            "censoring and observation-window completeness are not implemented",
            manifest["unresolved_issues"],
        )


class MappingCoverageRolesTests(unittest.TestCase):
    """`_mapping_coverage_roles` decides which query roles the feasibility
    gate's mapping-coverage check runs against -- must line up exactly with
    which roles `causal.queries.feasibility._cohort_ctes` actually resolves
    to a real OMOP concept set (see `EXPOSURE_CONTRASTS`/the `death`-outcome
    default), or the gate would either miss a role it should check or query
    a concept set that was never meant to be queried."""

    def _specs(self, **overrides):
        specs = {
            "population": ConceptSetSpec(domain="condition", concept_ids=(1,)),
            "treatment": ConceptSetSpec(domain="drug", concept_ids=(2,)),
            "comparator": ConceptSetSpec(domain="drug", concept_ids=(3,)),
            "outcome": ConceptSetSpec(domain="condition", concept_ids=(4,)),
        }
        specs.update(overrides)
        return specs

    def test_checks_all_four_roles_by_default(self):
        roles = _mapping_coverage_roles(self._specs(), "add_on")
        self.assertEqual([role for role, _ in roles], ["population", "treatment", "comparator", "outcome"])

    def test_skips_comparator_under_vs_unexposed(self):
        roles = _mapping_coverage_roles(self._specs(), "vs_unexposed")
        self.assertEqual([role for role, _ in roles], ["population", "treatment", "outcome"])

    def test_skips_a_death_table_outcome(self):
        specs = self._specs(outcome=ConceptSetSpec(domain="death", concept_ids=()))
        roles = _mapping_coverage_roles(specs, "add_on")
        self.assertEqual([role for role, _ in roles], ["population", "treatment", "comparator"])

    def test_skips_a_role_with_no_concept_ids(self):
        specs = self._specs(comparator=ConceptSetSpec(domain="", concept_ids=()))
        roles = _mapping_coverage_roles(specs, "add_on")
        self.assertEqual([role for role, _ in roles], ["population", "treatment", "outcome"])


class CheckMappingCoverageTests(unittest.TestCase):
    """`_check_mapping_coverage` is the actual guardrail decision: the
    thresholds it is judged against (`_MAPPING_COVERAGE_MIN_EVIDENCE_ROWS`/
    `_MAPPING_COVERAGE_MIN_RATIO`) exist to reproduce, without a live
    database, the exact real-world finding that motivated this gate --
    RxNorm "Aspirin" (concept 1112807) matched only 3 `drug_exposure` rows
    database-wide in mimic22_omop531, against ~180K unmapped rows whose
    `drug_source_value` still read "Aspirin ..."."""

    def setUp(self):
        self.spec = ConceptSetSpec(domain="drug", concept_ids=(1112807,))

    def test_blocks_the_aspirin_shaped_finding(self):
        with self.assertRaises(RuntimeError) as ctx:
            _check_mapping_coverage(
                "treatment", self.spec,
                {"mapped_row_count": 3, "unmapped_name_matched_row_count": 180000},
            )
        message = str(ctx.exception)
        self.assertIn("treatment", message)
        self.assertIn("1112807", message)

    def test_passes_when_coverage_is_high(self):
        _check_mapping_coverage(
            "treatment", self.spec,
            {"mapped_row_count": 100000, "unmapped_name_matched_row_count": 50},
        )

    def test_passes_when_there_is_too_little_evidence_to_judge(self):
        """A genuinely rare but well-mapped concept can have very few rows on
        both sides -- below the evidence floor the gate must stay silent
        rather than block on noise."""
        _check_mapping_coverage(
            "treatment", self.spec,
            {"mapped_row_count": 1, "unmapped_name_matched_row_count": 2},
        )

    def test_treats_missing_counts_as_zero(self):
        _check_mapping_coverage("treatment", self.spec, {})


class _FakeSequentialCausalSelectClient:
    """Duck-typed OMCP client returning one queued CSV response per
    `causal_select` call, in call order. Lets a test assert exactly how many/
    which queries `run_feasibility_mcp`'s mapping-coverage pre-check issues,
    and confirms it short-circuits (raises) before consuming any later
    queued response it never reaches."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[str, str, dict]] = []

    def causal_select(self, query: str, operation: str, causal_context: dict) -> MCPToolResponse:
        self.calls.append((query, operation, causal_context))
        text = self._responses.pop(0)
        return MCPToolResponse(text=text, metadata={"rows": 1})


_GOOD_COVERAGE_ROW = "mapped_row_count,unmapped_name_matched_row_count\n1000,0\n"
_BAD_COVERAGE_ROW = "mapped_row_count,unmapped_name_matched_row_count\n3,180000\n"
_FEASIBILITY_ROW = (
    "eligible_persons,any_treatment,any_comparator,treatment_and_comparator,"
    "treatment_only,comparator_only,neither_exposure,outcome_events,"
    "outcome_events_treatment,outcome_events_comparator_only,"
    "outcome_events_during_grace_period\n"
    "500,0,120,0,0,120,380,40,0,40,2\n"
)


class WorkflowFeasibilityMappingGateTests(unittest.TestCase):
    """`run_feasibility_mcp` must run the mapping-coverage check for every
    applicable role *before* the real feasibility query -- turning a would-be
    silent zero-patient cohort (see `causal.mapping.audit.
    concept_mapping_coverage_sql`'s docstring for the MIMIC-IV/OMOP aspirin
    finding that motivated this) into a hard, actionable error instead."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.protocol_path = root / "protocol.json"
        self.protocol_path.write_text(json.dumps({
            "study_id": "test_study",
            "source_question": "Does treatment affect outcome?",
            "target_trial": {
                "population": "population", "treatment": "treatment",
                "comparator": "comparator", "time_zero": "index",
                "grace_period_hours": 24, "outcome": "outcome",
                "follow_up_days": 28, "primary_estimand": "ATE",
                **_FULL_TARGET_TRIAL_FIELDS,
            },
            "data_source": "project.dataset",
            "concept_sets": {
                "population": {
                    "role": "population", "status": "approved",
                    "domain": "condition", "concept_ids": [132797],
                },
                "treatment": {
                    "role": "treatment", "status": "approved",
                    "domain": "drug", "concept_ids": [1112807],
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
        state = self.manager.start(self.protocol_path, "exploratory")
        self.run_id = state.run_id
        self.run_dir = root / "runs" / self.run_id
        self.manager.run_hypothesis_agent(self.run_id)
        _approve_pre_mapping_gates(self.manager, self.run_id)
        for _ in range(3):
            self.manager.advance(self.run_id)
        self.manager.decide_gate(self.run_id, "clinical_mapping", "provisional", "tester")
        state = self.manager.advance(self.run_id)
        self.assertEqual(state.phase, "feasibility")
        (self.run_dir / "analysis_plan.json").write_text(json.dumps({
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
        self.datasets = BigQueryDatasets()

    def tearDown(self):
        self.temp.cleanup()

    def test_blocks_on_a_critically_unmapped_treatment_concept(self):
        client = _FakeSequentialCausalSelectClient([
            _GOOD_COVERAGE_ROW,  # population
            _BAD_COVERAGE_ROW,   # treatment -- the aspirin-shaped finding
            _GOOD_COVERAGE_ROW,  # comparator (never reached)
            _GOOD_COVERAGE_ROW,  # outcome (never reached)
            _FEASIBILITY_ROW,    # main feasibility query (never reached)
        ])
        with self.assertRaises(RuntimeError) as ctx:
            self.manager.run_feasibility_mcp(self.run_id, client, self.datasets)
        self.assertIn("treatment", str(ctx.exception))
        # Only population + treatment were checked before the block fired.
        self.assertEqual(len(client.calls), 2)
        self.assertFalse((self.run_dir / "feasibility.json").is_file())

    def test_proceeds_to_feasibility_when_coverage_is_fine_for_every_role(self):
        client = _FakeSequentialCausalSelectClient([
            _GOOD_COVERAGE_ROW,  # population
            _GOOD_COVERAGE_ROW,  # treatment
            _GOOD_COVERAGE_ROW,  # comparator
            _GOOD_COVERAGE_ROW,  # outcome
            _FEASIBILITY_ROW,    # main feasibility query
        ])
        state = self.manager.run_feasibility_mcp(self.run_id, client, self.datasets)
        self.assertEqual(len(client.calls), 5)
        self.assertTrue(all(op == "aggregate_feasibility" for _, op, _ in client.calls))
        self.assertIn("feasibility.json", state.artifacts)
        feasibility = json.loads((self.run_dir / "feasibility.json").read_text(encoding="utf-8"))
        self.assertEqual(feasibility["counts"]["eligible_persons"], 500)


if __name__ == "__main__":
    unittest.main()
