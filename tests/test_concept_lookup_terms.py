import json
import tempfile
import unittest
from pathlib import Path

from causal.agents.concept_lookup_terms import (
    DEFAULT_CONCEPT_LOOKUP_TERMS_PROMPT_PATH,
    build_concept_lookup_terms_prompt,
    parse_concept_lookup_terms_response,
)

_TARGET = {
    "population": "adults admitted with acute myocardial infarction",
    "treatment": "low-dose aspirin",
    "comparator": "no aspirin",
    "outcome": "30-day all-cause mortality",
}
_CONCEPT_SETS = [
    {"name": "population", "role": "population", "label": "acute myocardial infarction"},
    {"name": "treatment", "role": "treatment", "label": "low-dose aspirin"},
    {"name": "outcome", "role": "outcome", "label": "30-day mortality"},
]


class BuildConceptLookupTermsPromptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.template = Path(self.temp.name) / "concept_lookup_terms.md"
        self.template.write_text(
            "P {{POPULATION}} T {{TREATMENT}} C {{COMPARATOR}} O {{OUTCOME}}\nSETS {{CONCEPT_SETS}}\n",
            encoding="utf-8",
        )

    def test_fills_every_placeholder(self):
        prompt = build_concept_lookup_terms_prompt(self.template, _TARGET, _CONCEPT_SETS)
        for placeholder in ("{{POPULATION}}", "{{TREATMENT}}", "{{COMPARATOR}}", "{{OUTCOME}}", "{{CONCEPT_SETS}}"):
            self.assertNotIn(placeholder, prompt)
        self.assertIn("acute myocardial infarction", prompt)
        self.assertIn('"role": "treatment"', prompt)

    def test_missing_placeholder_raises(self):
        self.template.write_text("no placeholders", encoding="utf-8")
        with self.assertRaises(ValueError):
            build_concept_lookup_terms_prompt(self.template, _TARGET, _CONCEPT_SETS)

    def test_shipped_template_has_every_placeholder(self):
        prompt = build_concept_lookup_terms_prompt(
            DEFAULT_CONCEPT_LOOKUP_TERMS_PROMPT_PATH, _TARGET, _CONCEPT_SETS
        )
        for placeholder in ("{{POPULATION}}", "{{TREATMENT}}", "{{COMPARATOR}}", "{{OUTCOME}}", "{{CONCEPT_SETS}}"):
            self.assertNotIn(placeholder, prompt)


class ParseConceptLookupTermsResponseTests(unittest.TestCase):
    def test_rejects_empty_response(self):
        with self.assertRaises(ValueError):
            parse_concept_lookup_terms_response("", {"population"})

    def test_rejects_non_json(self):
        with self.assertRaises(ValueError):
            parse_concept_lookup_terms_response("not json", {"population"})

    def test_rejects_missing_terms_key(self):
        with self.assertRaises(ValueError):
            parse_concept_lookup_terms_response("{}", {"population"})

    def test_rejects_non_list_terms(self):
        with self.assertRaises(ValueError):
            parse_concept_lookup_terms_response('{"terms": "x"}', {"population"})

    def test_keeps_valid_entries_and_strips_whitespace(self):
        raw = json.dumps({
            "terms": [
                {"name": "population", "search_term": "  acute myocardial infarction "},
                {"name": "treatment", "search_term": "aspirin"},
            ]
        })
        result = parse_concept_lookup_terms_response(raw, {"population", "treatment", "outcome"})
        self.assertEqual(result, {"population": "acute myocardial infarction", "treatment": "aspirin"})

    def test_drops_unknown_names(self):
        raw = json.dumps({"terms": [{"name": "hallucinated", "search_term": "x"}]})
        self.assertEqual(parse_concept_lookup_terms_response(raw, {"population"}), {})

    def test_drops_blank_and_overlong_terms(self):
        raw = json.dumps({
            "terms": [
                {"name": "population", "search_term": "   "},
                {"name": "treatment", "search_term": "x" * 200},
            ]
        })
        self.assertEqual(parse_concept_lookup_terms_response(raw, {"population", "treatment"}), {})

    def test_first_occurrence_of_a_repeated_name_wins(self):
        raw = json.dumps({
            "terms": [
                {"name": "population", "search_term": "myocardial infarction"},
                {"name": "population", "search_term": "heart attack"},
            ]
        })
        self.assertEqual(
            parse_concept_lookup_terms_response(raw, {"population"}), {"population": "myocardial infarction"}
        )

    def test_one_bad_entry_does_not_discard_the_good_ones(self):
        raw = json.dumps({
            "terms": [
                {"name": "population"},  # missing search_term
                {"name": "outcome", "search_term": "death"},
            ]
        })
        self.assertEqual(parse_concept_lookup_terms_response(raw, {"population", "outcome"}), {"outcome": "death"})


if __name__ == "__main__":
    unittest.main()
