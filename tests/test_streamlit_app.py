"""Coverage for `causal.ui.streamlit_app`'s pure helper functions.

The module is a Streamlit script (`st.*` calls throughout its `_render_*`
functions), so most of it needs a real app run to exercise. This file only
targets the helpers that don't touch `st.*` -- JSON/CSV parsing, dataclass
construction from a dict, DOT generation, sorting/labeling -- which is where
plain data bugs (a missing `.get()` default, a wrong sort key) would actually
show up. Importing the module itself is also useful signal: it fails fast if
a rename in `causal.agents.hypothesis`/`causal.roadmap`/etc. breaks the app's
imports, something `python -c "import ..."` isn't otherwise checked for by
this test suite.
"""

import inspect
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import streamlit as st

from causal.roadmap import CausalDAG, CausalEdge, CausalNode
from causal.ui import streamlit_app as app


class ReadJsonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_missing_file_returns_none(self):
        self.assertIsNone(app._read_json(self.dir / "missing.json"))

    def test_malformed_json_returns_none_instead_of_raising(self):
        path = self.dir / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        self.assertIsNone(app._read_json(path))

    def test_valid_json_is_parsed(self):
        path = self.dir / "good.json"
        path.write_text(json.dumps({"a": 1}), encoding="utf-8")
        self.assertEqual(app._read_json(path), {"a": 1})


class BadgeTests(unittest.TestCase):
    def test_known_status_uses_its_color(self):
        html = app._badge("approved")
        self.assertIn(app._GATE_COLORS["approved"], html)
        self.assertIn("approved", html)

    def test_unknown_status_falls_back_to_default_color(self):
        html = app._badge("some_new_status")
        self.assertIn("#757575", html)

    def test_status_text_is_html_escaped(self):
        # A gate/phase status is always one of a fixed set of internal
        # strings today, but the badge still escapes it -- so a future
        # status sourced from less trusted input doesn't inject markup into
        # the reviewer's page.
        html = app._badge("<script>alert(1)</script>")
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)


class SourceQuestionAndRunListingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.runs_dir = Path(self.temp.name)
        self._patch = mock.patch.object(app, "RUNS_DIR", self.runs_dir)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self.temp.cleanup()

    def _make_run(self, run_id: str, **artifacts) -> Path:
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "state.json").write_text("{}", encoding="utf-8")
        for name, data in artifacts.items():
            (run_dir / name).write_text(json.dumps(data), encoding="utf-8")
        return run_dir

    def test_source_question_prefers_protocol_over_hypothesis(self):
        run_dir = self._make_run(
            "run1",
            **{
                "protocol.json": {"source_question": "from protocol"},
                "hypothesis.json": {"source_question": "from hypothesis"},
            },
        )
        self.assertEqual(app._source_question(run_dir), "from protocol")

    def test_source_question_falls_back_to_hypothesis(self):
        run_dir = self._make_run("run1", **{"hypothesis.json": {"source_question": "only here"}})
        self.assertEqual(app._source_question(run_dir), "only here")

    def test_source_question_ignores_blank_value(self):
        run_dir = self._make_run("run1", **{"protocol.json": {"source_question": "   "}})
        self.assertIsNone(app._source_question(run_dir))

    def test_source_question_none_when_no_artifacts(self):
        run_dir = self._make_run("run1")
        self.assertIsNone(app._source_question(run_dir))

    def test_list_runs_returns_empty_when_runs_dir_absent(self):
        # RUNS_DIR is patched to a real temp dir that exists here, so exercise
        # the "doesn't exist at all" branch against a sub-path that was never created.
        missing = self.runs_dir / "does_not_exist"
        with mock.patch.object(app, "RUNS_DIR", missing):
            self.assertEqual(app._list_runs(), [])

    def test_list_runs_only_includes_dirs_with_state_json(self):
        self._make_run("run_with_state")
        (self.runs_dir / "not_a_run").mkdir()
        self.assertEqual(app._list_runs(), ["run_with_state"])

    def test_list_runs_sorted_reverse(self):
        self._make_run("run_a")
        self._make_run("run_b")
        self._make_run("run_c")
        self.assertEqual(app._list_runs(), ["run_c", "run_b", "run_a"])

    def test_run_option_label_includes_truncated_question(self):
        self._make_run("run1", **{"protocol.json": {"source_question": "Does X cause Y?"}})
        self.assertEqual(app._run_option_label("run1"), "run1  —  Does X cause Y?")

    def test_run_option_label_falls_back_to_run_id_without_a_question(self):
        self._make_run("run1")
        self.assertEqual(app._run_option_label("run1"), "run1")

    def test_run_option_label_truncates_long_questions(self):
        long_question = "A" * 100
        self._make_run("run1", **{"protocol.json": {"source_question": long_question}})
        label = app._run_option_label("run1")
        self.assertTrue(label.endswith("..."))
        self.assertLessEqual(len(label) - len("run1  —  "), 70)


class HypothesisArtifactFromDictTests(unittest.TestCase):
    _REQUIRED = {
        "study_id": "study_1",
        "population": "adults with sepsis",
        "treatment": "ceftriaxone",
        "comparator": "piperacillin-tazobactam",
        "outcome": "28-day mortality",
        "time_zero": "first_qualifying_administration",
        "grace_period_hours": 24,
        "follow_up_days": 28,
        "primary_estimand": "risk_difference",
    }

    def test_minimal_dict_round_trips_required_fields(self):
        artifact = app._hypothesis_artifact_from_dict(dict(self._REQUIRED))
        self.assertEqual(artifact.study_id, "study_1")
        self.assertEqual(artifact.treatment, "ceftriaxone")
        self.assertEqual(artifact.candidate_dag, ())
        self.assertEqual(artifact.executable, False)

    def test_missing_required_field_raises(self):
        data = dict(self._REQUIRED)
        del data["study_id"]
        with self.assertRaises(KeyError):
            app._hypothesis_artifact_from_dict(data)

    def test_candidate_dag_variable_without_temporal_classification_defaults_to_pre_treatment(self):
        # Older runs predate `temporal_classification`; the UI must still
        # render them rather than raising on a missing key.
        data = dict(self._REQUIRED)
        data["candidate_dag"] = [
            {"label": "Age", "role": "confounder", "rationale": "Older patients differ in risk."}
        ]
        artifact = app._hypothesis_artifact_from_dict(data)
        self.assertEqual(len(artifact.candidate_dag), 1)
        self.assertEqual(artifact.candidate_dag[0].temporal_classification, "pre_treatment")

    def test_covariate_relations_and_rejected_covariates_are_parsed(self):
        data = dict(self._REQUIRED)
        data["covariate_relations"] = [
            {"source": "age", "target": "mortality", "rationale": "Older patients have higher risk."}
        ]
        data["rejected_covariates"] = [
            {"label": "eye_color", "reviewer": "dr_smith", "comment": "Not clinically plausible."}
        ]
        artifact = app._hypothesis_artifact_from_dict(data)
        self.assertEqual(artifact.covariate_relations[0].source, "age")
        self.assertEqual(artifact.rejected_covariates[0].label, "eye_color")


class DagToDotTests(unittest.TestCase):
    def _dag(self) -> CausalDAG:
        return CausalDAG(
            nodes=(
                CausalNode("exposure", "Ceftriaxone", "exposure", "treatment", True),
                CausalNode("outcome", "28-day mortality", "outcome", "outcome", True),
                CausalNode("age", 'Patient "Age"', "confounder", "baseline", True),
            ),
            edges=(
                CausalEdge("age", "exposure", "older patients treated differently", status="expert_approved"),
                CausalEdge("age", "outcome", "older patients have higher risk", status="proposed"),
            ),
            exposure_id="exposure",
            outcome_id="outcome",
        )

    def test_output_is_a_valid_looking_digraph(self):
        dot = app._dag_to_dot(self._dag())
        self.assertTrue(dot.startswith("digraph G {"))
        self.assertTrue(dot.endswith("}"))
        self.assertIn(" -> ", dot)

    def test_every_node_and_edge_is_rendered(self):
        dot = app._dag_to_dot(self._dag())
        for node_id in ("exposure", "outcome", "age"):
            self.assertIn(f'"{node_id}"', dot)
        self.assertIn('"age" -> "exposure"', dot)
        self.assertIn('"age" -> "outcome"', dot)

    def test_edge_status_controls_line_style(self):
        dot = app._dag_to_dot(self._dag())
        self.assertIn('"age" -> "exposure" [style=solid]', dot)
        self.assertIn('"age" -> "outcome" [style=dashed]', dot)

    def test_double_quotes_in_label_are_normalized_to_single_quotes(self):
        # Node labels are interpolated directly into a DOT string literal;
        # an un-escaped double quote would break the generated graph syntax.
        dot = app._dag_to_dot(self._dag())
        self.assertNotIn('"Patient "Age""', dot)
        self.assertIn("Patient 'Age'", dot)


class OmcpLookupDispatchTests(unittest.TestCase):
    def test_condition_domain_calls_lookup_condition(self):
        client = mock.Mock()
        app._omcp_lookup(client, "condition", "sepsis")
        client.lookup_condition.assert_called_once_with("sepsis")
        client.lookup_drug.assert_not_called()
        client.lookup_concept.assert_not_called()

    def test_drug_domain_calls_lookup_drug(self):
        client = mock.Mock()
        app._omcp_lookup(client, "drug", "ceftriaxone")
        client.lookup_drug.assert_called_once_with("ceftriaxone")

    def test_other_domains_fall_back_to_lookup_concept(self):
        client = mock.Mock()
        app._omcp_lookup(client, "measurement", "lactate")
        client.lookup_concept.assert_called_once_with("lactate", "measurement")


class ConceptOptionLabelTests(unittest.TestCase):
    """Leads with the human-readable name -- see `_concept_option_label`'s
    docstring: relative match quality is conveyed by list order
    (`_sorted_by_match_score` sorts candidates best-first before this is
    ever called), not by spelling out the raw edit-distance score inline."""

    def test_formats_name_and_concept_id(self):
        label = app._concept_option_label(
            {"concept_id": 123, "concept_name": "Sepsis", "match_score": 2.0}
        )
        self.assertEqual(label, "Sepsis — OMOP concept 123")

    def test_formats_name_and_concept_id_without_a_score(self):
        label = app._concept_option_label({"concept_id": 123, "concept_name": "Sepsis"})
        self.assertEqual(label, "Sepsis — OMOP concept 123")


class SortedByMatchScoreTests(unittest.TestCase):
    def test_lower_score_sorts_first(self):
        candidates = [{"match_score": 5.0}, {"match_score": 1.0}, {"match_score": 3.0}]
        result = app._sorted_by_match_score(candidates)
        self.assertEqual([c["match_score"] for c in result], [1.0, 3.0, 5.0])

    def test_missing_score_sorts_last(self):
        candidates = [{"match_score": 2.0}, {}, {"match_score": 1.0}]
        result = app._sorted_by_match_score(candidates)
        self.assertEqual(result[0]["match_score"], 1.0)
        self.assertEqual(result[1]["match_score"], 2.0)
        self.assertNotIn("match_score", result[2])


class ParseConceptRowsTests(unittest.TestCase):
    def test_parses_matching_domain_rows(self):
        csv_text = (
            "concept_id,concept_name,vocabulary_id,domain_id,name_edit_distance\r\n"
            "123,Sepsis,SNOMED,Condition,0.0\r\n"
        )
        rows = app._parse_concept_rows(csv_text, "condition")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["concept_id"], 123)
        self.assertEqual(rows[0]["match_score"], 0.0)

    def test_drops_rows_from_a_different_domain(self):
        csv_text = (
            "concept_id,concept_name,vocabulary_id,domain_id\r\n"
            "123,Ceftriaxone,RxNorm,Drug\r\n"
        )
        rows = app._parse_concept_rows(csv_text, "condition")
        self.assertEqual(rows, [])

    def test_drops_rows_with_unparseable_concept_id(self):
        csv_text = (
            "concept_id,concept_name,vocabulary_id,domain_id\r\n"
            "not_an_int,Sepsis,SNOMED,Condition\r\n"
        )
        rows = app._parse_concept_rows(csv_text, "condition")
        self.assertEqual(rows, [])

    def test_missing_name_edit_distance_column_yields_none_match_score(self):
        # Get_Concepts_By_Id responses predate this column.
        csv_text = "concept_id,concept_name,vocabulary_id,domain_id\r\n123,Sepsis,SNOMED,Condition\r\n"
        rows = app._parse_concept_rows(csv_text, "condition")
        self.assertIsNone(rows[0]["match_score"])


class DefaultDomainIndexTests(unittest.TestCase):
    _OPTIONS = ("condition", "drug", "measurement", "observation", "procedure")

    def test_current_domain_already_set_takes_priority(self):
        self.assertEqual(
            app._default_domain_index("treatment", "measurement", self._OPTIONS),
            self._OPTIONS.index("measurement"),
        )

    def test_falls_back_to_role_hint_when_domain_unset(self):
        self.assertEqual(
            app._default_domain_index("treatment", "", self._OPTIONS),
            self._OPTIONS.index("drug"),
        )
        self.assertEqual(
            app._default_domain_index("outcome", "", self._OPTIONS),
            self._OPTIONS.index("condition"),
        )

    def test_unknown_role_falls_back_to_first_option(self):
        self.assertEqual(app._default_domain_index("some_unknown_role", "", self._OPTIONS), 0)


class NarrativeBlocksReportTests(unittest.TestCase):
    """`_narrative_blocks_report` decides whether "Build causal report" is
    disabled -- pure logic, no `st.*` calls, matching this file's "helpers
    only" scope (see module docstring)."""

    def test_no_narrative_never_blocks(self):
        self.assertFalse(app._narrative_blocks_report(None))

    def test_pending_review_blocks(self):
        narrative = {"review_status": "pending", "verification": {"status": "not_run", "checks": []}}
        self.assertTrue(app._narrative_blocks_report(narrative))

    def test_excluded_never_blocks_even_with_failed_checks(self):
        narrative = {
            "review_status": "excluded",
            "verification": {"status": "failed", "checks": [{"status": "failed"}]},
        }
        self.assertFalse(app._narrative_blocks_report(narrative))

    def test_approved_with_clean_verification_does_not_block(self):
        narrative = {
            "review_status": "approved",
            "verification": {"status": "passed", "checks": [{"status": "passed"}]},
        }
        self.assertFalse(app._narrative_blocks_report(narrative))

    def test_approved_with_warning_only_verification_does_not_block(self):
        narrative = {
            "review_status": "approved",
            "verification": {"status": "warning", "checks": [{"status": "warning"}]},
        }
        self.assertFalse(app._narrative_blocks_report(narrative))

    def test_approved_but_a_check_failed_still_blocks(self):
        """Defensive case: a tampered or legacy artifact marked `approved`
        whose verification nonetheless contains a `failed` check must still
        block -- approval alone is never sufficient."""
        narrative = {
            "review_status": "approved",
            "verification": {"status": "passed", "checks": [{"status": "failed"}]},
        }
        self.assertTrue(app._narrative_blocks_report(narrative))

    def test_approved_but_verification_never_ran_blocks(self):
        narrative = {"review_status": "approved", "verification": {"status": "not_run", "checks": []}}
        self.assertTrue(app._narrative_blocks_report(narrative))


_REVIEWER_TEXT_INPUT_KEY_RE = re.compile(
    r'"Reviewer".*?key=f?"([a-zA-Z0-9_]*?_reviewer_)', re.DOTALL,
)


class ReviewerFieldSyncTests(unittest.TestCase):
    """`_sync_reviewer_fields` (wired as the sidebar Reviewer-name field's
    `on_change`, see `main`) only pushes the sidebar name into
    `st.session_state` keys matching `_REVIEWER_FIELD_KEY_PREFIXES` --
    `st.text_input(value=...)` otherwise only seeds a widget's session_state
    the first time its key is created (`_sync_reviewer_fields`'s own
    docstring). A per-form Reviewer field whose key prefix is missing from
    that tuple silently stops following the sidebar name after its first
    render -- exactly the bug this class exists to catch, discovered live
    when a newly added "Reject" tab's own Reviewer field (`causal_reject_
    reviewer_`) turned out not to be registered.

    `test_every_reviewer_field_key_prefix_is_registered` is the regression
    guard against this happening again: it scans the module's own source for
    every `st.text_input("Reviewer", ..., key=...)` call and asserts each
    one's static key prefix is in `_REVIEWER_FIELD_KEY_PREFIXES`, rather than
    hand-listing the current set of forms (which would need updating, and
    could silently drift, every time a new one is added -- the same failure
    mode this test exists to prevent)."""

    def test_every_reviewer_field_key_prefix_is_registered(self):
        source = inspect.getsource(app)
        found_prefixes = set(_REVIEWER_TEXT_INPUT_KEY_RE.findall(source))
        self.assertTrue(found_prefixes, "expected to find at least one Reviewer text_input in the module")
        unregistered = {
            prefix for prefix in found_prefixes
            if not any(prefix.startswith(p) or p.startswith(prefix) for p in app._REVIEWER_FIELD_KEY_PREFIXES)
        }
        self.assertFalse(
            unregistered,
            f"Reviewer field key prefix(es) {unregistered} are not in _REVIEWER_FIELD_KEY_PREFIXES -- "
            "these fields will not follow the sidebar Reviewer name after their first render",
        )

    def test_sync_pushes_the_sidebar_name_into_every_registered_prefix(self):
        keys = [f"{prefix}some_run_id" for prefix in app._REVIEWER_FIELD_KEY_PREFIXES]
        try:
            st.session_state["reviewer_name"] = "Dr. Rivera"
            for key in keys:
                st.session_state[key] = "stale text from an earlier render"
            app._sync_reviewer_fields()
            for key in keys:
                self.assertEqual(st.session_state[key], "Dr. Rivera")
        finally:
            for key in keys + ["reviewer_name"]:
                st.session_state.pop(key, None)

    def test_sync_does_not_touch_an_unrelated_key(self):
        try:
            st.session_state["reviewer_name"] = "Dr. Rivera"
            st.session_state["some_other_widget"] = "untouched"
            app._sync_reviewer_fields()
            self.assertEqual(st.session_state["some_other_widget"], "untouched")
        finally:
            for key in ("reviewer_name", "some_other_widget"):
                st.session_state.pop(key, None)


if __name__ == "__main__":
    unittest.main()
