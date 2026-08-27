"""Coverage for `causal.companion.companion` -- prompt-build and strict parse
for one study-companion turn. Pure module (no LLM, no I/O beyond the prompt
template), so it is fully exercised here with a temp template and canned
LLM output strings."""

import json
import tempfile
import unittest
from pathlib import Path

from causal.companion.actions import ProposedAction
from causal.companion.companion import (
    CompanionMessage,
    CompanionTurn,
    EstimandDialogueContext,
    build_estimand_dialogue_prompt,
    parse_companion_turn,
)

_PLACEHOLDERS = (
    "{{CONVERSATION}}",
    "{{PRIMARY_ESTIMAND}}",
    "{{ADJUSTMENT_SET}}",
    "{{CENSORING_DEFINITION_APPROVED}}",
    "{{CURRENT_SETTINGS}}",
    "{{FEASIBILITY_SUMMARY}}",
    "{{FOLLOWUP_AUDIT_SUMMARY}}",
    "{{POSITIVITY_AUDIT_SUMMARY}}",
    "{{COVARIATE_MISSINGNESS_SUMMARY}}",
)

_VALID_ACTION = {
    "action": "set_statistical_estimand_strategy",
    "args": {
        "missing_data_strategy": "complete_case_analysis",
        "censoring_strategy": "ipcw_administrative_censoring",
        "estimation_procedure": "targeted_minimum_loss_based_estimation",
        "censoring_rationale": "Administrative censoring at 28 days, reweighted rather than dropped.",
    },
}


class CompanionMessageTests(unittest.TestCase):
    def test_valid_message(self):
        self.assertEqual(CompanionMessage("user", "hi").role, "user")

    def test_bad_role_raises(self):
        with self.assertRaises(ValueError):
            CompanionMessage("event", "a run was created")

    def test_empty_text_raises(self):
        with self.assertRaises(ValueError):
            CompanionMessage("assistant", "   ")


class BuildEstimandDialoguePromptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.template = Path(self.temp.name) / "estimand_dialogue.md"
        self.template.write_text(
            "HISTORY:\n{{CONVERSATION}}\n"
            "ESTIMAND: {{PRIMARY_ESTIMAND}}\n"
            "ADJ: {{ADJUSTMENT_SET}}\n"
            "CENSOR_OK: {{CENSORING_DEFINITION_APPROVED}}\n"
            "SETTINGS: {{CURRENT_SETTINGS}}\n"
            "FEAS: {{FEASIBILITY_SUMMARY}}\n"
            "FOLLOWUP: {{FOLLOWUP_AUDIT_SUMMARY}}\n"
            "POS: {{POSITIVITY_AUDIT_SUMMARY}}\n"
            "MISS: {{COVARIATE_MISSINGNESS_SUMMARY}}\n",
            encoding="utf-8",
        )

    def _context(self, **overrides) -> EstimandDialogueContext:
        return EstimandDialogueContext(**overrides)

    def test_all_placeholders_are_filled(self):
        prompt = build_estimand_dialogue_prompt(
            self.template, history=[], context=self._context()
        )
        for placeholder in _PLACEHOLDERS:
            self.assertNotIn(placeholder, prompt)

    def test_missing_placeholder_in_template_raises(self):
        self.template.write_text("no placeholders here", encoding="utf-8")
        with self.assertRaises(ValueError):
            build_estimand_dialogue_prompt(self.template, history=[], context=self._context())

    def test_history_is_labelled_by_speaker(self):
        prompt = build_estimand_dialogue_prompt(
            self.template,
            history=[
                CompanionMessage("user", "what about IPCW?"),
                CompanionMessage("assistant", "IPCW keeps the estimate over the eligible population."),
            ],
            context=self._context(),
        )
        self.assertIn("Reviewer: what about IPCW?", prompt)
        self.assertIn("Companion: IPCW keeps the estimate", prompt)

    def test_adjustment_set_comes_from_baseline_covariate_columns(self):
        context = self._context(
            cohort_manifest={
                "columns": [
                    {"name": "age", "role": "baseline_covariate"},
                    {"name": "treated", "role": "treatment"},
                    {"name": "sofa", "role": "baseline_covariate"},
                ]
            }
        )
        prompt = build_estimand_dialogue_prompt(self.template, history=[], context=context)
        self.assertIn("ADJ: age, sofa", prompt)

    def test_counts_summary_distinguishes_absent_from_present(self):
        context = self._context(followup_audit={"counts": {"censored": 41, "total": 1000}})
        prompt = build_estimand_dialogue_prompt(self.template, history=[], context=context)
        self.assertIn("FEAS: Not yet run.", prompt)
        self.assertIn('"censored": 41', prompt)

    def test_censoring_definition_flag_renders_yes_no(self):
        yes = build_estimand_dialogue_prompt(
            self.template, history=[], context=self._context(censoring_definition_approved=True)
        )
        self.assertIn("CENSOR_OK: yes", yes)
        no = build_estimand_dialogue_prompt(
            self.template, history=[], context=self._context(censoring_definition_approved=False)
        )
        self.assertIn("CENSOR_OK: no", no)


class ParseCompanionTurnTests(unittest.TestCase):
    def test_reply_only_turn(self):
        turn = parse_companion_turn(json.dumps({"reply": "  Let's start with censoring.  "}))
        self.assertIsInstance(turn, CompanionTurn)
        self.assertEqual(turn.reply, "Let's start with censoring.")
        self.assertIsNone(turn.proposed_action)

    def test_turn_with_a_valid_action(self):
        turn = parse_companion_turn(
            json.dumps({"reply": "Here is the strategy I'd record.", "proposed_action": _VALID_ACTION})
        )
        self.assertIsInstance(turn.proposed_action, ProposedAction)
        self.assertEqual(
            turn.proposed_action.args["censoring_strategy"], "ipcw_administrative_censoring"
        )

    def test_null_proposed_action_is_fine(self):
        turn = parse_companion_turn(json.dumps({"reply": "ok", "proposed_action": None}))
        self.assertIsNone(turn.proposed_action)

    def test_empty_response_raises(self):
        with self.assertRaises(ValueError):
            parse_companion_turn("   ")

    def test_non_json_raises(self):
        with self.assertRaises(ValueError):
            parse_companion_turn("Let's talk about censoring.")

    def test_non_object_raises(self):
        with self.assertRaises(ValueError):
            parse_companion_turn(json.dumps(["reply"]))

    def test_missing_or_empty_reply_raises(self):
        with self.assertRaises(ValueError):
            parse_companion_turn(json.dumps({"proposed_action": None}))
        with self.assertRaises(ValueError):
            parse_companion_turn(json.dumps({"reply": "  "}))

    def test_malformed_action_fails_the_whole_parse(self):
        bad = {
            "reply": "recording it",
            "proposed_action": {
                "action": "set_statistical_estimand_strategy",
                "args": {**_VALID_ACTION["args"], "censoring_strategy": "delete_rows"},
            },
        }
        with self.assertRaises(ValueError):
            parse_companion_turn(json.dumps(bad))

    def test_action_cannot_smuggle_a_reviewer(self):
        bad = {
            "reply": "recording it",
            "proposed_action": {
                "action": "set_statistical_estimand_strategy",
                "args": {**_VALID_ACTION["args"], "reviewer": "dr_who"},
            },
        }
        with self.assertRaises(ValueError):
            parse_companion_turn(json.dumps(bad))


if __name__ == "__main__":
    unittest.main()
