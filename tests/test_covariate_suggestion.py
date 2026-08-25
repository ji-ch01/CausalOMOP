import json
import tempfile
import unittest
from pathlib import Path

from causal.agents.covariate_suggestion import (
    build_covariate_suggestion_prompt,
    parse_covariate_suggestion_response,
)


class ParseCovariateSuggestionResponseTests(unittest.TestCase):
    def test_rejects_empty_response(self):
        with self.assertRaises(ValueError):
            parse_covariate_suggestion_response("", {"age"})

    def test_rejects_non_json(self):
        with self.assertRaises(ValueError):
            parse_covariate_suggestion_response("not json", {"age"})

    def test_rejects_missing_suggestions_key(self):
        with self.assertRaises(ValueError):
            parse_covariate_suggestion_response("{}", {"age"})

    def test_rejects_non_list_suggestions(self):
        with self.assertRaises(ValueError):
            parse_covariate_suggestion_response('{"suggestions": "x"}', {"age"})

    def test_accepts_valid_confounder_suggestion(self):
        raw = json.dumps({
            "suggestions": [{
                "label": "chronic_kidney_disease",
                "decision": "confounder",
                "temporal_classification": "pre_treatment",
                "domain": "condition",
                "rationale": "Plausibly affects both treatment choice and outcome.",
            }]
        })
        result = parse_covariate_suggestion_response(raw, {"chronic_kidney_disease"})
        self.assertEqual(set(result), {"chronic_kidney_disease"})
        suggestion = result["chronic_kidney_disease"]
        self.assertEqual(suggestion.decision, "confounder")
        self.assertEqual(suggestion.temporal_classification, "pre_treatment")
        self.assertEqual(suggestion.domain, "condition")

    def test_accepts_reject_suggestion_without_temporal_classification(self):
        raw = json.dumps({
            "suggestions": [{
                "label": "sex",
                "decision": "reject",
                "rationale": "Already adjusted for via a fixed baseline column.",
            }]
        })
        result = parse_covariate_suggestion_response(raw, {"sex"})
        self.assertEqual(result["sex"].decision, "reject")
        self.assertEqual(result["sex"].temporal_classification, "")

    def test_drops_suggestion_for_unknown_label(self):
        raw = json.dumps({
            "suggestions": [{
                "label": "hallucinated_variable",
                "decision": "reject",
                "rationale": "x",
            }]
        })
        result = parse_covariate_suggestion_response(raw, {"age"})
        self.assertEqual(result, {})

    def test_drops_suggestion_with_missing_rationale(self):
        raw = json.dumps({
            "suggestions": [{"label": "age", "decision": "reject", "rationale": ""}]
        })
        result = parse_covariate_suggestion_response(raw, {"age"})
        self.assertEqual(result, {})

    def test_drops_suggestion_with_invalid_decision(self):
        raw = json.dumps({
            "suggestions": [{"label": "age", "decision": "not_a_role", "rationale": "x"}]
        })
        result = parse_covariate_suggestion_response(raw, {"age"})
        self.assertEqual(result, {})

    def test_drops_suggestion_with_role_temporal_mismatch(self):
        # `mediator` only allows post_treatment (ROLE_ALLOWED_TEMPORAL_CLASSES).
        raw = json.dumps({
            "suggestions": [{
                "label": "age", "decision": "mediator",
                "temporal_classification": "pre_treatment", "rationale": "x",
            }]
        })
        result = parse_covariate_suggestion_response(raw, {"age"})
        self.assertEqual(result, {})

    def test_drops_suggestion_with_invalid_domain_but_keeps_it_domainless(self):
        raw = json.dumps({
            "suggestions": [{
                "label": "age", "decision": "confounder",
                "temporal_classification": "pre_treatment", "domain": "not_a_domain",
                "rationale": "x",
            }]
        })
        result = parse_covariate_suggestion_response(raw, {"age"})
        self.assertEqual(result["age"].domain, "")

    def test_keeps_valid_entries_when_another_entry_is_malformed(self):
        raw = json.dumps({
            "suggestions": [
                {"label": "age", "decision": "reject", "rationale": "x"},
                {"label": "unknown", "decision": "reject", "rationale": "x"},
            ]
        })
        result = parse_covariate_suggestion_response(raw, {"age"})
        self.assertEqual(set(result), {"age"})

    def test_caps_suggestions_at_max(self):
        from causal.agents.covariate_suggestion import _MAX_SUGGESTIONS

        known = {f"v{i}" for i in range(_MAX_SUGGESTIONS + 5)}
        many = [
            {"label": f"v{i}", "decision": "reject", "rationale": "x"}
            for i in range(_MAX_SUGGESTIONS + 5)
        ]
        result = parse_covariate_suggestion_response(json.dumps({"suggestions": many}), known)
        self.assertEqual(len(result), _MAX_SUGGESTIONS)


class BuildCovariateSuggestionPromptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.template_path = Path(self.temp.name) / "covariate_suggestion.md"

    def tearDown(self):
        self.temp.cleanup()

    def _write_template(self, text: str) -> None:
        self.template_path.write_text(text, encoding="utf-8")

    def test_raises_on_missing_required_placeholder(self):
        self._write_template("POPULATION={{POPULATION}}")
        with self.assertRaises(ValueError):
            build_covariate_suggestion_prompt(self.template_path, {"population": "p"}, [])

    def test_substitutes_all_required_placeholders(self):
        self._write_template(
            "POPULATION={{POPULATION}}\nTREATMENT={{TREATMENT}}\nCOMPARATOR={{COMPARATOR}}\n"
            "OUTCOME={{OUTCOME}}\nUNRESOLVED={{UNRESOLVED_COVARIATES}}"
        )
        rendered = build_covariate_suggestion_prompt(
            self.template_path,
            {"population": "P", "treatment": "T", "comparator": "C", "outcome": "O"},
            [{"label": "age", "reason": "no abstract addressed it"}],
        )
        self.assertIn("POPULATION=P", rendered)
        self.assertIn("TREATMENT=T", rendered)
        self.assertIn("COMPARATOR=C", rendered)
        self.assertIn("OUTCOME=O", rendered)
        self.assertIn('"label": "age"', rendered)
        self.assertIn("no abstract addressed it", rendered)


if __name__ == "__main__":
    unittest.main()
