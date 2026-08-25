import json
import tempfile
import unittest
from pathlib import Path

from causal.agents.estimation_procedure_suggestion import (
    build_estimation_procedure_suggestion_prompt,
    deterministic_cate_suggestion,
    parse_estimation_procedure_suggestion_response,
)


class ParseEstimationProcedureSuggestionResponseTests(unittest.TestCase):
    def test_rejects_empty_response(self):
        with self.assertRaises(ValueError):
            parse_estimation_procedure_suggestion_response("")

    def test_rejects_non_json(self):
        with self.assertRaises(ValueError):
            parse_estimation_procedure_suggestion_response("not json")

    def test_rejects_non_object(self):
        with self.assertRaises(ValueError):
            parse_estimation_procedure_suggestion_response("[1, 2, 3]")

    def test_rejects_missing_estimation_procedure(self):
        with self.assertRaises(ValueError):
            parse_estimation_procedure_suggestion_response(json.dumps({"rationale": "x"}))

    def test_rejects_invalid_estimation_procedure_value(self):
        raw = json.dumps({"estimation_procedure": "not_a_procedure", "rationale": "x"})
        with self.assertRaises(ValueError):
            parse_estimation_procedure_suggestion_response(raw)

    def test_rejects_missing_rationale(self):
        raw = json.dumps({"estimation_procedure": "plug_in_standardization"})
        with self.assertRaises(ValueError):
            parse_estimation_procedure_suggestion_response(raw)

    def test_rejects_empty_rationale(self):
        raw = json.dumps({"estimation_procedure": "plug_in_standardization", "rationale": "  "})
        with self.assertRaises(ValueError):
            parse_estimation_procedure_suggestion_response(raw)

    def test_accepts_valid_plug_in_standardization_response(self):
        raw = json.dumps({
            "estimation_procedure": "plug_in_standardization",
            "rationale": "Diagnostics give no particular reason to prefer TMLE's robustness here.",
        })
        result = parse_estimation_procedure_suggestion_response(raw)
        self.assertEqual(result.estimation_procedure, "plug_in_standardization")
        self.assertEqual(result.provenance, "llm_proposed")

    def test_accepts_valid_tmle_response(self):
        raw = json.dumps({
            "estimation_procedure": "targeted_minimum_loss_based_estimation",
            "rationale": "Positivity audit shows a zero-cell covariate; TMLE's double "
            "robustness is a better fit.",
        })
        result = parse_estimation_procedure_suggestion_response(raw)
        self.assertEqual(result.estimation_procedure, "targeted_minimum_loss_based_estimation")

    def test_accepts_causal_forest_dml_value_at_parse_layer(self):
        # The pure parser validates against the full ESTIMATION_PROCEDURES set;
        # rejecting causal_forest_dml for a non-CATE study is a workflow-layer
        # cross-check (WorkflowManager.propose_estimation_procedure), not this
        # parser's concern.
        raw = json.dumps({"estimation_procedure": "causal_forest_dml", "rationale": "x"})
        result = parse_estimation_procedure_suggestion_response(raw)
        self.assertEqual(result.estimation_procedure, "causal_forest_dml")


class BuildEstimationProcedureSuggestionPromptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.template_path = Path(self.temp.name) / "estimation_procedure_suggestion.md"

    def tearDown(self):
        self.temp.cleanup()

    def _write_template(self, text: str) -> None:
        self.template_path.write_text(text, encoding="utf-8")

    _FULL_TEMPLATE = (
        "POPULATION_SUMMARY={{POPULATION_SUMMARY}}\n"
        "EFFECT_MEASURE={{EFFECT_MEASURE}}\n"
        "FEASIBILITY={{FEASIBILITY_SUMMARY}}\n"
        "FOLLOWUP={{FOLLOWUP_AUDIT_SUMMARY}}\n"
        "POSITIVITY={{POSITIVITY_AUDIT_SUMMARY}}\n"
        "MISSINGNESS={{COVARIATE_MISSINGNESS_SUMMARY}}\n"
    )

    def _build(self, **overrides):
        kwargs = {
            "population_summary": "ATE",
            "effect_measure": "risk_difference",
            "feasibility": {},
            "followup_audit": {},
            "positivity_audit": {},
            "covariate_missingness": {},
        }
        kwargs.update(overrides)
        return build_estimation_procedure_suggestion_prompt(self.template_path, **kwargs)

    def test_raises_on_missing_required_placeholder(self):
        self._write_template("POPULATION_SUMMARY={{POPULATION_SUMMARY}}")
        with self.assertRaises(ValueError):
            self._build()

    def test_substitutes_all_required_placeholders(self):
        self._write_template(self._FULL_TEMPLATE)
        rendered = self._build(
            population_summary="ATE",
            effect_measure="risk_difference",
            feasibility={"counts": {"eligible_persons": 500}},
        )
        self.assertIn("POPULATION_SUMMARY=ATE", rendered)
        self.assertIn("EFFECT_MEASURE=risk_difference", rendered)
        self.assertIn('"eligible_persons": 500', rendered)

    def test_summarizes_missing_diagnostic_as_not_yet_run(self):
        self._write_template(self._FULL_TEMPLATE)
        rendered = self._build(positivity_audit={})
        self.assertIn("POSITIVITY=Not yet run.", rendered)

    def test_summarizes_present_diagnostic_counts(self):
        self._write_template(self._FULL_TEMPLATE)
        rendered = self._build(
            covariate_missingness={"counts": {"age_missing": 3}}
        )
        self.assertIn('"age_missing": 3', rendered)
        self.assertNotIn("MISSINGNESS=Not yet run.", rendered)


class DeterministicCateSuggestionTests(unittest.TestCase):
    def test_returns_causal_forest_dml_with_deterministic_provenance(self):
        suggestion = deterministic_cate_suggestion(2)
        self.assertEqual(suggestion.estimation_procedure, "causal_forest_dml")
        self.assertEqual(suggestion.provenance, "deterministic_cate_rule")

    def test_rationale_mentions_effect_modifier_count(self):
        suggestion = deterministic_cate_suggestion(3)
        self.assertIn("3 reviewed, ready effect_modifier", suggestion.rationale)


if __name__ == "__main__":
    unittest.main()
