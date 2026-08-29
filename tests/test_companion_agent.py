"""Coverage for `causal.companion.companion` -- prompt-build and strict parse
for one study-companion turn. Pure module (no LLM, no I/O beyond the prompt
template), so it is fully exercised here with a temp template and canned
LLM output strings."""

import json
import tempfile
import unittest
from pathlib import Path

from causal.agents.results_narrative import NarrativeInput, NarrativeResult
from causal.companion.actions import ProposedAction
from causal.companion.companion import (
    DEFAULT_ESTIMAND_DIALOGUE_PROMPT_PATH,
    DEFAULT_RESULTS_DIALOGUE_PROMPT_PATH,
    CompanionMessage,
    CompanionTurn,
    EstimandDialogueContext,
    ResultsCompanionTurn,
    build_estimand_dialogue_prompt,
    build_results_dialogue_prompt,
    parse_companion_turn,
    parse_results_companion_turn,
)
from causal.companion.decision_memory import DecisionMemory, RejectedCovariate
from causal.companion.reviewer_memory import ReviewerMemory, RememberedExclusion

_PLACEHOLDERS = (
    "{{CONVERSATION}}",
    "{{PRIMARY_ESTIMAND}}",
    "{{ADJUSTMENT_SET}}",
    "{{CENSORING_DEFINITION_APPROVED}}",
    "{{DECISION_MEMORY}}",
    "{{REVIEWER_MEMORY}}",
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
            "DECISIONS: {{DECISION_MEMORY}}\n"
            "CONVENTIONS: {{REVIEWER_MEMORY}}\n"
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

    def test_decision_memory_renders_excluded_covariates(self):
        context = self._context(
            decision_memory=DecisionMemory(
                rejected_covariates=(
                    RejectedCovariate(
                        "baseline_lactate", "Dr. Chen", "post-treatment", "2026-08-20T09:00:00+00:00"
                    ),
                )
            )
        )
        prompt = build_estimand_dialogue_prompt(self.template, history=[], context=context)
        self.assertIn("DECISIONS:", prompt)
        self.assertIn("baseline_lactate", prompt)
        self.assertIn("Covariates the reviewer has excluded:", prompt)

    def test_decision_memory_defaults_to_the_empty_sentence(self):
        prompt = build_estimand_dialogue_prompt(
            self.template, history=[], context=self._context()
        )
        self.assertIn("No reviewer decisions have been recorded", prompt)

    def test_reviewer_memory_renders_cross_study_conventions(self):
        context = self._context(
            reviewer_memory=ReviewerMemory(
                reviewer_id="dr_chen",
                exclusions=(
                    RememberedExclusion(
                        "baseline_lactate", "post-treatment", "aspirin_ami",
                        "2026-08-01T00:00:00+00:00",
                    ),
                ),
            )
        )
        prompt = build_estimand_dialogue_prompt(self.template, history=[], context=context)
        self.assertIn("CONVENTIONS:", prompt)
        self.assertIn("excluded in earlier studies", prompt)
        self.assertIn("aspirin_ami", prompt)

    def test_reviewer_memory_defaults_to_the_empty_sentence(self):
        prompt = build_estimand_dialogue_prompt(
            self.template, history=[], context=self._context()
        )
        self.assertIn("No cross-study conventions recorded", prompt)

    def test_censoring_definition_flag_renders_yes_no(self):
        yes = build_estimand_dialogue_prompt(
            self.template, history=[], context=self._context(censoring_definition_approved=True)
        )
        self.assertIn("CENSOR_OK: yes", yes)
        no = build_estimand_dialogue_prompt(
            self.template, history=[], context=self._context(censoring_definition_approved=False)
        )
        self.assertIn("CENSOR_OK: no", no)

    def test_shipped_prompt_template_has_every_placeholder(self):
        # The default template is loaded in production; a missing placeholder
        # would raise only at call time otherwise.
        prompt = build_estimand_dialogue_prompt(
            DEFAULT_ESTIMAND_DIALOGUE_PROMPT_PATH, history=[], context=self._context()
        )
        for placeholder in _PLACEHOLDERS:
            self.assertNotIn(placeholder, prompt)
        for procedure in ("plug_in_standardization", "targeted_minimum_loss_based_estimation"):
            self.assertIn(procedure, prompt)


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


_RESULTS_PLACEHOLDERS = ("{{CONVERSATION}}", "{{RESULTS_JSON}}")

_NARRATIVE_INPUT = NarrativeInput(
    study_id="test_study",
    question={"population": "sepsis", "treatment": "albumin", "comparator": "crystalloid", "outcome": "death"},
    estimand={"population_summary": "ATE", "effect_measure": "risk_difference"},
    dataset_summary={"row_count": 300, "column_count": 10},
    estimation={"estimand": {"point_estimate": 0.0421, "ci_lower": -0.02, "ci_upper": 0.10, "n_total": 300}},
    diagnostics={"n_covariates_flagged": 1},
    gate_history=({"name": "results", "status": "pending", "reviewer": None, "decided_at": None},),
    assumptions_and_limitations=("Does not adjust for informative censoring.",),
)

_VALID_DRAFT = {
    "narrative": "The estimated risk difference is small and its interval crosses the null.",
    "claims": [
        {
            "text": "The point estimate is 0.0421.",
            "claim_type": "effect_direction",
            "source_fields": ["estimation.estimand.point_estimate"],
        },
        {
            "text": "The interval runs from -0.02 to 0.10 and crosses the null.",
            "claim_type": "uncertainty",
            "source_fields": ["estimation.estimand.ci_lower", "estimation.estimand.ci_upper"],
        },
        {
            "text": "The analysis includes 300 patients.",
            "claim_type": "sample_size",
            "source_fields": ["estimation.estimand.n_total"],
        },
    ],
}


class BuildResultsDialoguePromptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.template = Path(self.temp.name) / "results_dialogue.md"
        self.template.write_text("HISTORY:\n{{CONVERSATION}}\nRESULTS:\n{{RESULTS_JSON}}\n", encoding="utf-8")

    def test_all_placeholders_are_filled(self):
        prompt = build_results_dialogue_prompt(
            self.template, history=[], narrative_input=_NARRATIVE_INPUT
        )
        for placeholder in _RESULTS_PLACEHOLDERS:
            self.assertNotIn(placeholder, prompt)

    def test_missing_placeholder_in_template_raises(self):
        self.template.write_text("no placeholders", encoding="utf-8")
        with self.assertRaises(ValueError):
            build_results_dialogue_prompt(self.template, history=[], narrative_input=_NARRATIVE_INPUT)

    def test_numbers_appear_verbatim_from_the_narrative_input(self):
        prompt = build_results_dialogue_prompt(
            self.template, history=[], narrative_input=_NARRATIVE_INPUT
        )
        for needle in ("0.0421", "-0.02", "300"):
            self.assertIn(needle, prompt)

    def test_history_is_labelled_by_speaker(self):
        prompt = build_results_dialogue_prompt(
            self.template,
            history=[
                CompanionMessage("user", "does the interval cross the null?"),
                CompanionMessage("assistant", "yes, it runs from -0.02 to 0.10."),
            ],
            narrative_input=_NARRATIVE_INPUT,
        )
        self.assertIn("Reviewer: does the interval cross the null?", prompt)
        self.assertIn("Companion: yes, it runs", prompt)

    def test_shipped_template_has_every_placeholder(self):
        prompt = build_results_dialogue_prompt(
            DEFAULT_RESULTS_DIALOGUE_PROMPT_PATH, history=[], narrative_input=_NARRATIVE_INPUT
        )
        for placeholder in _RESULTS_PLACEHOLDERS:
            self.assertNotIn(placeholder, prompt)

    def test_shipped_template_never_carries_the_patient_level_artifact(self):
        prompt = build_results_dialogue_prompt(
            DEFAULT_RESULTS_DIALOGUE_PROMPT_PATH, history=[], narrative_input=_NARRATIVE_INPUT
        )
        self.assertNotIn("analytic_dataset.parquet", prompt)


class ParseResultsCompanionTurnTests(unittest.TestCase):
    def test_reply_only_turn(self):
        turn = parse_results_companion_turn(json.dumps({"reply": "  Let's start with direction.  "}))
        self.assertIsInstance(turn, ResultsCompanionTurn)
        self.assertEqual(turn.reply, "Let's start with direction.")
        self.assertIsNone(turn.drafted_narrative)

    def test_null_drafted_narrative_is_fine(self):
        turn = parse_results_companion_turn(json.dumps({"reply": "ok", "drafted_narrative": None}))
        self.assertIsNone(turn.drafted_narrative)

    def test_turn_with_a_valid_draft(self):
        turn = parse_results_companion_turn(
            json.dumps({"reply": "Drafted it for your review.", "drafted_narrative": _VALID_DRAFT})
        )
        self.assertIsInstance(turn.drafted_narrative, NarrativeResult)
        self.assertEqual(turn.drafted_narrative.narrative_text.split()[0], "The")
        self.assertEqual(len(turn.drafted_narrative.claims), 3)

    def test_empty_response_raises(self):
        with self.assertRaises(ValueError):
            parse_results_companion_turn("   ")

    def test_non_json_raises(self):
        with self.assertRaises(ValueError):
            parse_results_companion_turn("Let's talk about the estimate.")

    def test_non_object_raises(self):
        with self.assertRaises(ValueError):
            parse_results_companion_turn(json.dumps(["reply"]))

    def test_missing_or_empty_reply_raises(self):
        with self.assertRaises(ValueError):
            parse_results_companion_turn(json.dumps({"drafted_narrative": None}))
        with self.assertRaises(ValueError):
            parse_results_companion_turn(json.dumps({"reply": "  "}))

    def test_malformed_draft_fails_the_whole_parse(self):
        bad = {
            "reply": "drafting it",
            "drafted_narrative": {"narrative": "text", "claims": []},
        }
        with self.assertRaises(ValueError):
            parse_results_companion_turn(json.dumps(bad))

    def test_draft_with_an_unknown_claim_type_fails(self):
        bad_claim = dict(_VALID_DRAFT["claims"][0], claim_type="clinical_recommendation")
        bad = {"reply": "drafting it", "drafted_narrative": {**_VALID_DRAFT, "claims": [bad_claim]}}
        with self.assertRaises(ValueError):
            parse_results_companion_turn(json.dumps(bad))

    def test_draft_that_is_not_an_object_raises(self):
        with self.assertRaises(ValueError):
            parse_results_companion_turn(
                json.dumps({"reply": "drafting it", "drafted_narrative": "a string"})
            )


if __name__ == "__main__":
    unittest.main()
