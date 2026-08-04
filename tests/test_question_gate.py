import json
import tempfile
import unittest
from pathlib import Path

from causal.agents.question_gate import build_gate_prompt, parse_gate_response


class ParseGateResponseTests(unittest.TestCase):
    def test_rejects_empty_response(self):
        with self.assertRaises(ValueError):
            parse_gate_response("")

    def test_rejects_non_json(self):
        with self.assertRaises(ValueError):
            parse_gate_response("not json")

    def test_rejects_non_object(self):
        with self.assertRaises(ValueError):
            parse_gate_response("[1, 2, 3]")

    def test_rejects_missing_category(self):
        with self.assertRaises(ValueError):
            parse_gate_response('{"reasoning": "no category given"}')

    def test_rejects_unknown_category(self):
        with self.assertRaises(ValueError):
            parse_gate_response('{"category": "maybe", "reasoning": "x"}')

    def test_rejects_missing_reasoning(self):
        with self.assertRaises(ValueError):
            parse_gate_response('{"category": "causal"}')

    def test_accepts_causal_with_intervention_and_outcome(self):
        result = parse_gate_response(json.dumps({
            "category": "causal",
            "reasoning": "Asks whether starting a treatment changes an outcome.",
            "identified_intervention": "early vasopressors",
            "identified_outcome": "28-day mortality",
            "clarifying_question": "",
        }))
        self.assertEqual(result.category, "causal")
        self.assertEqual(result.identified_intervention, "early vasopressors")
        self.assertEqual(result.identified_outcome, "28-day mortality")
        self.assertEqual(result.clarifying_question, "")

    def test_accepts_descriptive_without_intervention_or_outcome(self):
        result = parse_gate_response(json.dumps({
            "category": "descriptive",
            "reasoning": "Just counts patients matching a concept.",
        }))
        self.assertEqual(result.category, "descriptive")
        self.assertEqual(result.identified_intervention, "")
        self.assertEqual(result.identified_outcome, "")

    def test_needs_clarification_with_no_clarifying_question_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_gate_response(json.dumps({
                "category": "needs_clarification",
                "reasoning": "Associational phrasing without a stated exposure.",
                "clarifying_question": "",
            }))

    def test_needs_clarification_with_clarifying_question_is_accepted(self):
        result = parse_gate_response(json.dumps({
            "category": "needs_clarification",
            "reasoning": "Associational phrasing without a stated exposure.",
            "clarifying_question": "Which specific treatment or policy change should be compared?",
        }))
        self.assertEqual(result.category, "needs_clarification")
        self.assertEqual(
            result.clarifying_question,
            "Which specific treatment or policy change should be compared?",
        )


class BuildGatePromptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.template_path = Path(self.temp.name) / "question_gate.md"

    def tearDown(self):
        self.temp.cleanup()

    def test_rejects_blank_question(self):
        self.template_path.write_text("{{USER_QUESTION}}", encoding="utf-8")
        with self.assertRaises(ValueError):
            build_gate_prompt(self.template_path, "   ")

    def test_raises_on_missing_placeholder(self):
        self.template_path.write_text("no placeholder here", encoding="utf-8")
        with self.assertRaises(ValueError):
            build_gate_prompt(self.template_path, "Does X cause Y?")

    def test_substitutes_question(self):
        self.template_path.write_text("SYSTEM\n{{USER_QUESTION}}", encoding="utf-8")
        rendered = build_gate_prompt(self.template_path, "Does X cause Y?")
        self.assertIn("Does X cause Y?", rendered)


if __name__ == "__main__":
    unittest.main()
