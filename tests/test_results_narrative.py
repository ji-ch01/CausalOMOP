import json
import tempfile
import unittest
from pathlib import Path

from causal.agents.results_narrative import (
    CLAIM_TYPES,
    DEFAULT_NARRATIVE_PROMPT_PATH,
    NARRATIVE_BOUNDARY_STATEMENT,
    Claim,
    NarrativeInput,
    build_narrative_prompt,
    narrative_result_from_payload,
    normalize_narrative_artifact,
    parse_narrative_response,
)

_INPUT = NarrativeInput(
    study_id="test_study",
    question={"population": "sepsis", "treatment": "albumin", "comparator": "crystalloid", "outcome": "death"},
    estimand={"population_summary": "ATE", "effect_measure": "risk_difference"},
    dataset_summary={"row_count": 300, "column_count": 10},
    estimation={"estimand": {"point_estimate": 0.0421, "ci_lower": -0.02, "ci_upper": 0.10, "n_total": 300}},
    diagnostics={"n_covariates_flagged": 1, "smd_flag_threshold": 0.1},
    gate_history=({"name": "results", "status": "pending", "reviewer": None, "decided_at": None},),
    assumptions_and_limitations=("Does not adjust for informative censoring.",),
)

_VALID_CLAIM = {
    "text": "The estimated risk difference is small and imprecise.",
    "claim_type": "effect_direction",
    "source_fields": ["estimation.estimand.point_estimate"],
}


def _response(narrative: str = "The estimated risk difference is small and imprecise.", claims=(_VALID_CLAIM,)) -> str:
    return json.dumps({"narrative": narrative, "claims": list(claims)})


class BuildNarrativePromptTests(unittest.TestCase):
    def test_real_prompt_template_has_the_placeholder(self):
        # Exercises the actual shipped template, not just a synthetic one.
        prompt = build_narrative_prompt(DEFAULT_NARRATIVE_PROMPT_PATH, _INPUT)
        self.assertNotIn("{{RESULTS_JSON}}", prompt)

    def test_every_number_in_the_input_appears_verbatim_in_the_prompt(self):
        prompt = build_narrative_prompt(DEFAULT_NARRATIVE_PROMPT_PATH, _INPUT)
        for needle in ("0.0421", "-0.02", "300", "0.1"):
            self.assertIn(needle, prompt)

    def test_rejects_a_template_missing_the_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.md"
            path.write_text("no placeholder here", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_narrative_prompt(path, _INPUT)

    def test_never_includes_the_word_parquet_or_a_dataframe_repr(self):
        """Sanity check that NarrativeInput's JSON-safe fields alone cannot
        accidentally carry the patient-level artifact through the prompt."""
        prompt = build_narrative_prompt(DEFAULT_NARRATIVE_PROMPT_PATH, _INPUT)
        self.assertNotIn("analytic_dataset.parquet", prompt)

    def test_prompt_lists_every_allowed_claim_type(self):
        prompt = build_narrative_prompt(DEFAULT_NARRATIVE_PROMPT_PATH, _INPUT)
        for claim_type in CLAIM_TYPES:
            self.assertIn(claim_type, prompt)


class ParseNarrativeResponseTests(unittest.TestCase):
    def test_parses_a_well_formed_response(self):
        result = parse_narrative_response(_response())
        self.assertEqual(result.narrative_text, "The estimated risk difference is small and imprecise.")
        self.assertEqual(len(result.claims), 1)
        claim = result.claims[0]
        self.assertIsInstance(claim, Claim)
        self.assertEqual(claim.claim_type, "effect_direction")
        self.assertEqual(claim.source_fields, ("estimation.estimand.point_estimate",))

    def test_round_trips_through_a_mocked_plain_callable_llm(self):
        """`LLMCallable = Callable[[str], str]` (causal.llm.base) -- a plain
        function is a valid stand-in, no network access involved."""
        def fake_llm(prompt: str) -> str:
            self.assertIn("RESULTS JSON", prompt)
            return _response("Plain language summary.")

        prompt = build_narrative_prompt(DEFAULT_NARRATIVE_PROMPT_PATH, _INPUT)
        result = parse_narrative_response(fake_llm(prompt))
        self.assertEqual(result.narrative_text, "Plain language summary.")

    def test_rejects_empty_response(self):
        with self.assertRaises(ValueError):
            parse_narrative_response("")

    def test_rejects_non_json_response(self):
        with self.assertRaises(ValueError):
            parse_narrative_response("not json")

    def test_rejects_a_json_array(self):
        with self.assertRaises(ValueError):
            parse_narrative_response(json.dumps([1, 2, 3]))

    def test_rejects_a_missing_narrative_field(self):
        with self.assertRaises(ValueError):
            parse_narrative_response(json.dumps({"claims": [_VALID_CLAIM]}))

    def test_rejects_a_blank_narrative_field(self):
        with self.assertRaises(ValueError):
            parse_narrative_response(json.dumps({"narrative": "   ", "claims": [_VALID_CLAIM]}))

    def test_does_not_trust_a_model_supplied_boundary_statement(self):
        """The caller always appends NARRATIVE_BOUNDARY_STATEMENT itself;
        parsing must not read or rely on any disclaimer text the model
        happens to include in its own narrative."""
        raw = json.dumps(
            {"narrative": "Some interpretation.", "claims": [_VALID_CLAIM], "boundary": "trust me, I'm advisory"}
        )
        result = parse_narrative_response(raw)
        self.assertEqual(result.narrative_text, "Some interpretation.")
        self.assertNotIn("trust me", NARRATIVE_BOUNDARY_STATEMENT)

    def test_rejects_a_missing_claims_field(self):
        with self.assertRaises(ValueError):
            parse_narrative_response(json.dumps({"narrative": "text only, no claims"}))

    def test_rejects_an_empty_claims_list(self):
        with self.assertRaises(ValueError):
            parse_narrative_response(json.dumps({"narrative": "text", "claims": []}))

    def test_rejects_a_claim_missing_text(self):
        bad_claim = {"claim_type": "effect_direction", "source_fields": ["estimation.estimand.point_estimate"]}
        with self.assertRaises(ValueError):
            parse_narrative_response(_response(claims=(bad_claim,)))

    def test_rejects_an_unknown_claim_type(self):
        bad_claim = dict(_VALID_CLAIM, claim_type="clinical_recommendation")
        with self.assertRaises(ValueError):
            parse_narrative_response(_response(claims=(bad_claim,)))

    def test_rejects_a_claim_with_no_source_fields(self):
        bad_claim = dict(_VALID_CLAIM, source_fields=[])
        with self.assertRaises(ValueError):
            parse_narrative_response(_response(claims=(bad_claim,)))

    def test_rejects_a_claim_whose_source_fields_is_not_a_list_of_strings(self):
        bad_claim = dict(_VALID_CLAIM, source_fields=[123])
        with self.assertRaises(ValueError):
            parse_narrative_response(_response(claims=(bad_claim,)))

    def test_rejects_claims_that_is_not_a_list(self):
        with self.assertRaises(ValueError):
            parse_narrative_response(json.dumps({"narrative": "text", "claims": "not a list"}))

    def test_parses_multiple_claims_of_different_types(self):
        claims = (
            _VALID_CLAIM,
            {
                "text": "The confidence interval crosses zero.",
                "claim_type": "uncertainty",
                "source_fields": ["estimation.estimand.ci_lower", "estimation.estimand.ci_upper"],
            },
        )
        result = parse_narrative_response(_response(claims=claims))
        self.assertEqual([c.claim_type for c in result.claims], ["effect_direction", "uncertainty"])


class NarrativeResultFromPayloadTests(unittest.TestCase):
    """`narrative_result_from_payload` is the shared validator behind both
    `parse_narrative_response` and the results-dialogue's drafted narrative
    (`causal.companion.companion.parse_results_companion_turn`)."""

    def test_validates_an_already_parsed_object(self):
        result = narrative_result_from_payload({"narrative": "A summary.", "claims": [_VALID_CLAIM]})
        self.assertEqual(result.narrative_text, "A summary.")
        self.assertEqual(result.claims[0].claim_type, "effect_direction")

    def test_missing_narrative_raises_with_the_given_context(self):
        with self.assertRaisesRegex(ValueError, "drafted_narrative"):
            narrative_result_from_payload({"claims": [_VALID_CLAIM]}, context="drafted_narrative")

    def test_bad_claim_type_raises_with_the_given_context(self):
        bad_claim = dict(_VALID_CLAIM, claim_type="clinical_recommendation")
        with self.assertRaisesRegex(ValueError, "drafted_narrative"):
            narrative_result_from_payload(
                {"narrative": "text", "claims": [bad_claim]}, context="drafted_narrative"
            )

    def test_empty_claims_list_raises(self):
        with self.assertRaises(ValueError):
            narrative_result_from_payload({"narrative": "text", "claims": []})

    def test_round_trips_a_serialized_claim(self):
        payload = {
            "narrative": "text",
            "claims": [Claim("t", "sample_size", ("estimation.estimand.n_total",)).to_dict()],
        }
        result = narrative_result_from_payload(payload)
        self.assertEqual(result.claims[0].source_fields, ("estimation.estimand.n_total",))


class NormalizeNarrativeArtifactTests(unittest.TestCase):
    def test_legacy_artifact_gets_tolerant_defaults(self):
        legacy = {
            "schema_version": "1.0",
            "narrative": "An old narrative.",
            "boundary_statement": NARRATIVE_BOUNDARY_STATEMENT,
            "model": "gpt-x",
            "provider": "openrouter",
            "prompt_sha256": "abc123",
        }
        normalized = normalize_narrative_artifact(legacy)
        self.assertEqual(normalized["claims"], [])
        self.assertEqual(normalized["verification"]["status"], "not_run")
        self.assertEqual(normalized["review_status"], "pending")
        self.assertIsNone(normalized["reviewer"])
        self.assertIsNone(normalized["reviewed_at"])
        self.assertIsNone(normalized["review_note"])
        # Original fields survive untouched.
        self.assertEqual(normalized["narrative"], "An old narrative.")

    def test_new_style_artifact_is_left_untouched(self):
        current = {
            "narrative": "text",
            "claims": [_VALID_CLAIM],
            "verification": {"status": "passed", "checks": [], "checked_at_utc": "2026-08-11T00:00:00+00:00"},
            "review_status": "approved",
            "reviewer": "Clinician",
            "reviewed_at": "2026-08-11T00:00:00+00:00",
            "review_note": "Looks good.",
            "source": "one_shot",
        }
        normalized = normalize_narrative_artifact(current)
        self.assertEqual(normalized, current)

    def test_legacy_artifact_defaults_source_to_one_shot(self):
        normalized = normalize_narrative_artifact({"narrative": "An old narrative."})
        self.assertEqual(normalized["source"], "one_shot")


if __name__ == "__main__":
    unittest.main()
