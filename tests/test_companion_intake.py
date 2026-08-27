"""Coverage for `causal.companion.intake` -- the question -> gate ->
{FastOMOP | clarify | draft} routing shared by the CLI and the Streamlit
conversation panel. Real `WorkflowManager` against a temp dir, fake LLM,
patched FastOMOP."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from causal.agents.user_question import UserQuestionHypothesisService
from causal.companion.intake import (
    ClarificationNeeded,
    DescriptiveAnswer,
    DraftOptions,
    QuestionIntake,
    StudyDrafted,
    combine_clarifications,
)
from causal.fastomop_bridge import FastOMOPAnswer
from causal.llm import ResolvedLLM
from causal.orchestrator.workflow import WorkflowManager

QUESTION = "Does treatment A reduce 28-day mortality compared with treatment B?"

_GATE_MARKER = "GATE-PROMPT "


def _valid_protocol_response(source_question: str = QUESTION) -> dict:
    return {
        "study_id": "treatment_a_mortality",
        "source_question": source_question,
        "target_trial": {
            "population": "Eligible adults",
            "treatment": "Treatment A",
            "comparator": "Treatment B",
            "time_zero": "Treatment initiation",
            "grace_period_hours": 0,
            "outcome": "All-cause mortality",
            "follow_up_days": 28,
            "primary_estimand": "ATE",
            "secondary_estimands": [],
            "eligibility_criteria": ["adult", "meets trial eligibility"],
            "assignment_procedure": "Emulated baseline assignment",
            "censoring_strategy": "Administrative censoring at end of follow-up",
            "analysis_strategy": "Intention-to-treat analogue",
            "effect_measure": "risk_difference",
            "treatment_value": "treatment_a",
            "comparator_value": "treatment_b",
        },
        "literature_search": {
            "population_terms": ["critical illness"],
            "treatment_terms": ["treatment A"],
            "comparator_terms": ["treatment B"],
            "outcome_terms": ["mortality", "death"],
        },
        "first_pass_covariates": ["age"],
        "design_constraints": [],
        "assumptions_requiring_review": ["Eligibility is not fully specified"],
    }


class _FakeLLM:
    """Returns a canned gate verdict for the gate prompt and a fixed valid
    protocol for anything else. Records every prompt it saw."""

    def __init__(self, gate_response: dict | str):
        self.gate_response = gate_response
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if prompt.startswith(_GATE_MARKER):
            return self.gate_response if isinstance(self.gate_response, str) else json.dumps(self.gate_response)
        return json.dumps(_valid_protocol_response())

    @property
    def gate_calls(self) -> int:
        return sum(1 for prompt in self.prompts if prompt.startswith(_GATE_MARKER))


class QuestionIntakeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.runs_dir = base / "runs"
        self.manager = WorkflowManager(runs_dir=self.runs_dir)

        self.gate_prompt = base / "gate.md"
        self.gate_prompt.write_text(_GATE_MARKER + "{{USER_QUESTION}}", encoding="utf-8")
        hypothesis_prompt = base / "hypothesis.md"
        hypothesis_prompt.write_text("DRAFT\n{{USER_QUESTION}}", encoding="utf-8")
        dag_prompt = base / "dag.md"
        dag_prompt.write_text(
            "POP={{POPULATION}} TREAT={{TREATMENT}} COMP={{COMPARATOR}} "
            "OUT={{OUTCOME}} ART={{ARTICLES}}",
            encoding="utf-8",
        )
        dag_critique_prompt = base / "dag_critique.md"
        dag_critique_prompt.write_text("CRITIQUE", encoding="utf-8")
        self.service = UserQuestionHypothesisService(hypothesis_prompt, dag_prompt, dag_critique_prompt)

    def _intake(self, gate_response: dict | str) -> tuple[QuestionIntake, _FakeLLM]:
        llm = _FakeLLM(gate_response)
        intake = QuestionIntake(
            manager=self.manager,
            resolved_llm=ResolvedLLM(call=llm, provider="openrouter", model="test-model"),
            gate_prompt_path=self.gate_prompt,
            service=self.service,
        )
        return intake, llm

    def test_empty_question_is_rejected(self):
        intake, _ = self._intake({"category": "causal", "reasoning": "x"})
        with self.assertRaises(ValueError):
            intake.run("   ")

    def test_descriptive_question_is_answered_by_fastomop_without_a_run(self):
        intake, _ = self._intake(
            {"category": "descriptive", "reasoning": "asks for a count, not an effect"}
        )
        fake_answer = FastOMOPAnswer(answer="42 patients", sql_query="SELECT COUNT(*) FROM person")
        with mock.patch("causal.companion.intake.ask_fastomop", return_value=fake_answer) as ask:
            outcome = intake.run("How many patients are in the cohort?")
        ask.assert_called_once()
        self.assertIsInstance(outcome, DescriptiveAnswer)
        self.assertEqual(outcome.answer, "42 patients")
        self.assertEqual(outcome.sql, "SELECT COUNT(*) FROM person")
        self.assertFalse(self.runs_dir.exists() and any(self.runs_dir.iterdir()))

    def test_needs_clarification_returns_the_clarifying_question_without_a_run(self):
        intake, _ = self._intake(
            {
                "category": "needs_clarification",
                "reasoning": "no comparator given",
                "clarifying_question": "Compared with what alternative treatment?",
            }
        )
        outcome = intake.run("Is treatment A good?")
        self.assertIsInstance(outcome, ClarificationNeeded)
        self.assertEqual(outcome.clarifying_question, "Compared with what alternative treatment?")
        self.assertFalse(self.runs_dir.exists() and any(self.runs_dir.iterdir()))

    def test_causal_question_drafts_a_governed_run_with_all_gates_pending(self):
        intake, llm = self._intake({"category": "causal", "reasoning": "clear intervention and outcome"})
        outcome = intake.run(QUESTION)
        self.assertIsInstance(outcome, StudyDrafted)
        self.assertEqual(outcome.provider, "openrouter")
        self.assertEqual(outcome.state.phase, "target_trial_review")
        self.assertTrue(all(gate.status == "pending" for gate in outcome.state.gates.values()))
        self.assertTrue((self.runs_dir / outcome.state.run_id / "protocol.json").is_file())
        self.assertEqual(llm.gate_calls, 1)

    def test_skip_question_gate_goes_straight_to_drafting(self):
        intake, llm = self._intake("this should never be parsed")
        outcome = intake.run(QUESTION, DraftOptions(skip_question_gate=True))
        self.assertIsInstance(outcome, StudyDrafted)
        self.assertEqual(llm.gate_calls, 0)

    def test_malformed_gate_response_raises(self):
        intake, _ = self._intake("not json at all")
        with self.assertRaises(ValueError):
            intake.run(QUESTION)

    def test_on_stage_receives_progress_breadcrumbs(self):
        intake, _ = self._intake({"category": "causal", "reasoning": "ok"})
        seen: list[str] = []
        intake.run(QUESTION, on_stage=seen.append)
        self.assertTrue(seen)
        self.assertTrue(any("Drafting" in message for message in seen))

    def test_on_gate_result_fires_once_with_the_parsed_verdict(self):
        intake, _ = self._intake(
            {
                "category": "needs_clarification",
                "reasoning": "no comparator",
                "clarifying_question": "Versus what?",
            }
        )
        seen = []
        intake.run("Is A good?", on_gate_result=seen.append)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].category, "needs_clarification")

    def test_on_gate_result_is_not_called_when_the_gate_is_skipped(self):
        intake, _ = self._intake("unused")
        seen = []
        intake.run(QUESTION, DraftOptions(skip_question_gate=True), on_gate_result=seen.append)
        self.assertEqual(seen, [])


class CombineClarificationsTests(unittest.TestCase):
    def test_no_rounds_returns_the_stripped_original(self):
        self.assertEqual(combine_clarifications("  the question  ", []), "the question")

    def test_one_round_appends_a_numbered_q_and_a_block(self):
        combined = combine_clarifications("original", [("Which comparator?", "Placebo")])
        self.assertIn("Clarification 1", combined)
        self.assertIn("Q: Which comparator?", combined)
        self.assertIn("A: Placebo", combined)
        self.assertTrue(combined.startswith("original"))

    def test_multiple_rounds_are_numbered_in_order(self):
        combined = combine_clarifications(
            "original",
            [("Which comparator?", "Placebo"), ("Which population?", "ICU adults")],
        )
        self.assertLess(combined.index("Clarification 1"), combined.index("Clarification 2"))
        self.assertIn("A: ICU adults", combined)


if __name__ == "__main__":
    unittest.main()
