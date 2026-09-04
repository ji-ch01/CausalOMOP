"""Coverage for `causal.agents.dag_point_revision` -- the pure prompt-build,
strict-parse, and deterministic-apply helpers behind
`WorkflowManager.revise_candidate_dag_from_points`."""

import json
import unittest

from causal.agents.dag_point_revision import (
    DEFAULT_DAG_POINT_REVISION_PROMPT_PATH,
    apply_dag_point_operations,
    build_dag_point_revision_prompt,
    format_current_dag,
    parse_dag_point_revision_response,
)

_TARGET = {
    "population": "ICU adults", "treatment": "drug A", "comparator": "drug B", "outcome": "AKI",
}
_DAG = [
    {"label": "age", "role": "confounder", "temporal_classification": "pre_treatment",
     "rationale": "prognostic", "pmid": "PM1", "concept_ids": [1], "mapping_status": "approved"},
    {"label": "vasopressor use", "role": "confounder", "temporal_classification": "pre_treatment",
     "rationale": "severity", "pmid": ""},
]


class BuildPromptTests(unittest.TestCase):
    def test_all_placeholders_filled(self):
        prompt = build_dag_point_revision_prompt(
            DEFAULT_DAG_POINT_REVISION_PROMPT_PATH,
            target=_TARGET,
            current_dag_text=format_current_dag(_DAG),
            points=["add baseline creatinine as a confounder"],
        )
        self.assertNotIn("{{", prompt)
        self.assertIn("drug A", prompt)
        self.assertIn("1. add baseline creatinine as a confounder", prompt)
        self.assertIn("vasopressor use", prompt)


class ParseResponseTests(unittest.TestCase):
    def test_valid_operations_parse(self):
        raw = json.dumps({
            "operations": [
                {"op": "add", "label": "creatinine", "role": "confounder",
                 "temporal_classification": "pre_treatment", "rationale": "confounds both"},
                {"op": "remove", "label": "vasopressor use", "rationale": "collider"},
            ],
            "acknowledged_errors": ["missed renal function"],
            "unaddressed_points": ["something vague"],
        })
        revision = parse_dag_point_revision_response(raw)
        self.assertEqual([op.op for op in revision.operations], ["add", "remove"])
        self.assertEqual(revision.acknowledged_errors, ("missed renal function",))
        self.assertEqual(revision.unaddressed_points, ("something vague",))

    def test_malformed_ops_are_dropped_but_valid_ones_kept(self):
        raw = json.dumps({"operations": [
            {"op": "add", "label": "creatinine", "role": "not_a_role",
             "temporal_classification": "pre_treatment", "rationale": "x"},
            {"op": "reclassify", "label": "age", "role": "mediator",
             "temporal_classification": "post_treatment", "rationale": "y"},
        ]})
        revision = parse_dag_point_revision_response(raw)
        self.assertEqual([op.label for op in revision.operations], ["age"])

    def test_all_ops_malformed_raises(self):
        raw = json.dumps({"operations": [{"op": "frobnicate", "label": "x", "rationale": "y"}]})
        with self.assertRaises(ValueError):
            parse_dag_point_revision_response(raw)

    def test_non_json_raises(self):
        with self.assertRaises(ValueError):
            parse_dag_point_revision_response("not json")


class ApplyOperationsTests(unittest.TestCase):
    def _revision(self, *operations):
        raw = json.dumps({"operations": list(operations)})
        return parse_dag_point_revision_response(raw)

    def test_add_appends_expert_asserted_variable(self):
        revision = self._revision({
            "op": "add", "label": "creatinine", "role": "confounder",
            "temporal_classification": "pre_treatment", "rationale": "confounds both",
        })
        updated, notes = apply_dag_point_operations(
            {"candidate_dag": _DAG}, revision, "tester", now="2026-09-04T00:00:00Z"
        )
        added = next(v for v in updated["candidate_dag"] if v["label"] == "creatinine")
        self.assertEqual(added["pmid"], "")
        self.assertEqual(added["evidence_level"], "uncited")
        self.assertEqual(notes, [])

    def test_reclassify_preserves_concept_mapping(self):
        revision = self._revision({
            "op": "reclassify", "label": "age", "role": "effect_modifier",
            "temporal_classification": "pre_treatment", "rationale": "modifies the effect",
        })
        updated, _ = apply_dag_point_operations(
            {"candidate_dag": _DAG}, revision, "tester", now="2026-09-04T00:00:00Z"
        )
        age = next(v for v in updated["candidate_dag"] if v["label"] == "age")
        self.assertEqual(age["role"], "effect_modifier")
        self.assertEqual(age["concept_ids"], [1])
        self.assertEqual(age["mapping_status"], "approved")

    def test_remove_moves_to_rejected_and_drops_relations(self):
        protocol = {
            "candidate_dag": _DAG,
            "covariate_relations": [{"source": "age", "target": "vasopressor use", "pmid": "P"}],
        }
        revision = self._revision({"op": "remove", "label": "vasopressor use", "rationale": "collider"})
        updated, _ = apply_dag_point_operations(protocol, revision, "tester", now="2026-09-04T00:00:00Z")
        self.assertEqual([v["label"] for v in updated["candidate_dag"]], ["age"])
        self.assertEqual(updated["rejected_covariates"][0]["label"], "vasopressor use")
        self.assertEqual(updated["covariate_relations"], [])

    def test_unknown_label_is_noted_not_raised(self):
        revision = self._revision({"op": "remove", "label": "ghost", "rationale": "x"})
        updated, notes = apply_dag_point_operations(
            {"candidate_dag": _DAG}, revision, "tester", now="2026-09-04T00:00:00Z"
        )
        self.assertEqual(len(updated["candidate_dag"]), 2)
        self.assertTrue(notes and "ghost" in notes[0])

    def test_input_protocol_is_not_mutated(self):
        protocol = {"candidate_dag": _DAG}
        revision = self._revision({"op": "remove", "label": "age", "rationale": "x"})
        apply_dag_point_operations(protocol, revision, "tester", now="2026-09-04T00:00:00Z")
        self.assertEqual(len(protocol["candidate_dag"]), 2)


if __name__ == "__main__":
    unittest.main()
