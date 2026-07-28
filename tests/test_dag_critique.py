import tempfile
import unittest
from pathlib import Path

from causal.agents.dag_critique import (
    _MAX_CRITIQUE_CONCERNS,
    build_critique_prompt,
    build_pre_check,
    build_revision_prompt,
    format_structural_findings,
    parse_critique_response,
)


def _valid_protocol() -> dict:
    return {
        "study_id": "test_study",
        "source_question": "Does treatment affect outcome?",
        "target_trial": {
            "population": "population", "treatment": "treatment",
            "comparator": "comparator", "time_zero": "index",
            "grace_period_hours": 24, "outcome": "outcome",
            "follow_up_days": 28, "primary_estimand": "ATE",
        },
        "concept_sets": {"treatment": {"status": "pending"}},
    }


def _confounder(label: str = "Baseline severity") -> dict:
    return {
        "label": label,
        "role": "confounder",
        "rationale": "Affects both treatment choice and outcome.",
        "temporal_classification": "pre_treatment",
        "pmid": "111",
    }


class ParseCritiqueResponseTests(unittest.TestCase):
    def test_rejects_empty_response(self):
        with self.assertRaises(ValueError):
            parse_critique_response("")

    def test_rejects_non_json(self):
        with self.assertRaises(ValueError):
            parse_critique_response("not json")

    def test_rejects_non_object(self):
        with self.assertRaises(ValueError):
            parse_critique_response("[1, 2, 3]")

    def test_rejects_missing_verdict(self):
        with self.assertRaises(ValueError):
            parse_critique_response('{"concerns": []}')

    def test_rejects_unknown_verdict(self):
        with self.assertRaises(ValueError):
            parse_critique_response('{"verdict": "maybe", "concerns": []}')

    def test_accepts_approve_with_empty_concerns(self):
        result = parse_critique_response('{"verdict": "approve", "concerns": []}')
        self.assertEqual(result.verdict, "approve")
        self.assertEqual(result.concerns, ())

    def test_revise_with_no_usable_concerns_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_critique_response('{"verdict": "revise", "concerns": []}')

    def test_revise_with_only_malformed_concerns_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_critique_response('{"verdict": "revise", "concerns": [{"target": "x"}]}')

    def test_drops_malformed_individual_concerns_but_keeps_valid_ones(self):
        result = parse_critique_response(
            '{"verdict": "revise", "concerns": ['
            '{"target": "x", "issue": "a real concern"}, '
            '{"target": "y"}, '
            '"not a dict"'
            ']}'
        )
        self.assertEqual(result.concerns, ({"target": "x", "issue": "a real concern"},))

    def test_caps_concerns_at_max(self):
        many = [{"target": f"v{i}", "issue": f"issue {i}"} for i in range(_MAX_CRITIQUE_CONCERNS + 5)]
        import json
        result = parse_critique_response(json.dumps({"verdict": "revise", "concerns": many}))
        self.assertEqual(len(result.concerns), _MAX_CRITIQUE_CONCERNS)


class BuildCritiquePromptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.template_path = Path(self.temp.name) / "dag_critique.md"

    def tearDown(self):
        self.temp.cleanup()

    def _write_template(self, text: str) -> None:
        self.template_path.write_text(text, encoding="utf-8")

    def test_raises_on_missing_required_placeholder(self):
        self._write_template("POPULATION={{POPULATION}}")
        with self.assertRaises(ValueError):
            build_critique_prompt(
                self.template_path, {"population": "p"}, "articles", "kg", "{}", "findings",
            )

    def test_substitutes_all_required_placeholders(self):
        self._write_template(
            "POPULATION={{POPULATION}}\nTREATMENT={{TREATMENT}}\nCOMPARATOR={{COMPARATOR}}\n"
            "OUTCOME={{OUTCOME}}\nARTICLES={{ARTICLES}}\nPROPOSED_DAG={{PROPOSED_DAG}}\n"
            "STRUCTURAL_FINDINGS={{STRUCTURAL_FINDINGS}}"
        )
        rendered = build_critique_prompt(
            self.template_path,
            {"population": "P", "treatment": "T", "comparator": "C", "outcome": "O"},
            "some articles", "some kg relations", '{"candidate_dag": []}', "no findings",
        )
        self.assertIn("POPULATION=P", rendered)
        self.assertIn("TREATMENT=T", rendered)
        self.assertIn("COMPARATOR=C", rendered)
        self.assertIn("OUTCOME=O", rendered)
        self.assertIn("ARTICLES=some articles", rendered)
        self.assertIn('PROPOSED_DAG={"candidate_dag": []}', rendered)
        self.assertIn("STRUCTURAL_FINDINGS=no findings", rendered)

    def test_kg_relations_placeholder_is_optional(self):
        self._write_template(
            "{{POPULATION}}{{TREATMENT}}{{COMPARATOR}}{{OUTCOME}}{{ARTICLES}}"
            "{{PROPOSED_DAG}}{{STRUCTURAL_FINDINGS}}"
        )
        rendered = build_critique_prompt(
            self.template_path, {}, "", "unused kg text", "{}", "",
        )
        self.assertNotIn("unused kg text", rendered)


class BuildPreCheckTests(unittest.TestCase):
    def test_returns_review_for_a_complete_protocol(self):
        review = build_pre_check(_valid_protocol(), [_confounder()], [])
        self.assertIsNotNone(review)
        self.assertTrue(review.identification.identified_by_backdoor_adjustment)

    def test_returns_none_for_an_incomplete_protocol_instead_of_raising(self):
        broken = {"study_id": "test_study", "source_question": "x"}
        review = build_pre_check(broken, [_confounder()], [])
        self.assertIsNone(review)


class FormatStructuralFindingsTests(unittest.TestCase):
    def test_unavailable_review_message(self):
        self.assertIn("unavailable", format_structural_findings(None))

    def test_empty_dag_notes_are_rendered(self):
        review = build_pre_check(_valid_protocol(), [], [])
        text = format_structural_findings(review)
        self.assertIn("trivially satisfied", text)

    def test_no_concerns_message_when_dag_is_clean(self):
        review = build_pre_check(_valid_protocol(), [_confounder()], [])
        text = format_structural_findings(review)
        self.assertTrue(text)


class BuildRevisionPromptTests(unittest.TestCase):
    def test_includes_prior_proposal_and_concerns(self):
        from causal.agents.dag_critique import CritiqueResult
        critique = CritiqueResult(
            verdict="revise",
            concerns=({"target": "Baseline severity", "issue": "wrong direction"},),
        )
        prompt = build_revision_prompt("BASE PROMPT TEXT", '{"candidate_dag": []}', critique)
        self.assertIn("BASE PROMPT TEXT", prompt)
        self.assertIn('{"candidate_dag": []}', prompt)
        self.assertIn("Baseline severity", prompt)
        self.assertIn("wrong direction", prompt)


if __name__ == "__main__":
    unittest.main()
