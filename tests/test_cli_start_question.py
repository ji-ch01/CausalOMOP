"""Regression guard for `workflow start-question`.

Exercises the real `causal.cli.main` entry point so the CLI's question
routing keeps going through `causal.companion.intake.QuestionIntake` -- the
one place the gate -> {FastOMOP | clarify | draft} decision lives, shared
with the Streamlit conversation panel (same reason `test_cli_dag_export.py`
and `test_cli_decision_memory.py` exist). The routing itself is covered in
depth by `test_companion_intake.py`; this checks the three outcomes reach
the terminal correctly (exit code, stdout/stderr)."""

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from causal import cli
from causal.companion.intake import StudyDrafted
from causal.fastomop_bridge import FastOMOPAnswer
from causal.llm import ResolvedLLM
from causal.orchestrator import WorkflowManager


class _FakeLLM:
    def __init__(self, gate_response: dict):
        self._gate_response = gate_response

    def __call__(self, prompt: str) -> str:
        return json.dumps(self._gate_response)


def _resolved(gate_response: dict) -> ResolvedLLM:
    return ResolvedLLM(call=_FakeLLM(gate_response), provider="openrouter", model="test-model")


class StartQuestionCliTests(unittest.TestCase):
    def _run(self, *argv_tail: str, resolved_llm: ResolvedLLM) -> tuple[int, str, str]:
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowManager(Path(tmp) / "runs")
            argv = ["causalomop", "workflow", "start-question", *argv_tail]
            out, err = io.StringIO(), io.StringIO()
            with patch.object(sys, "argv", argv), \
                    patch("causal.cli.WorkflowManager", return_value=manager), \
                    patch("causal.cli.build_hypothesis_llm", return_value=resolved_llm), \
                    patch("sys.stdout", out), patch("sys.stderr", err):
                exit_code = cli.main()
        return exit_code, out.getvalue(), err.getvalue()

    def test_descriptive_question_is_answered_by_fastomop(self):
        resolved = _resolved({"category": "descriptive", "reasoning": "asks for a count"})
        answer = FastOMOPAnswer(answer="4,210 patients", sql_query="SELECT COUNT(*) FROM person")
        with patch("causal.companion.intake.ask_fastomop", return_value=answer) as ask:
            exit_code, out, err = self._run(
                "How many sepsis patients are there?", "--skip-literature", resolved_llm=resolved
            )
        ask.assert_called_once()
        self.assertEqual(exit_code, 0)
        self.assertIn("4,210 patients", out)
        self.assertIn("SELECT COUNT(*) FROM person", out)
        self.assertIn("routed to FastOMOP", err)

    def test_ambiguous_question_returns_a_clarification_and_exit_2(self):
        resolved = _resolved(
            {
                "category": "needs_clarification",
                "reasoning": "no comparator given",
                "clarifying_question": "Compared with what?",
            }
        )
        exit_code, out, err = self._run(
            "Is albumin good?", "--skip-literature", resolved_llm=resolved
        )
        self.assertEqual(exit_code, 2)
        self.assertEqual(out, "")
        self.assertIn("Compared with what?", err)

    def test_causal_question_prints_the_new_run_state(self):
        resolved = _resolved({"category": "causal", "reasoning": "well-formed effect question"})

        class _Drafted:
            run_id = "generated-run-1"

            def to_dict(self):
                return {"run_id": self.run_id, "phase": "target_trial"}

        drafted = StudyDrafted(state=_Drafted(), provider="openrouter", model="test-model")
        with patch("causal.cli.QuestionIntake") as intake_cls:
            intake_cls.return_value.run.return_value = drafted
            exit_code, out, err = self._run(
                "Does albumin plus crystalloids cut 28-day mortality vs crystalloids alone?",
                "--skip-literature",
                resolved_llm=resolved,
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(out)["run_id"], "generated-run-1")

    def test_skip_question_gate_still_drafts(self):
        resolved = _resolved({"category": "descriptive", "reasoning": "would be routed away"})

        class _Drafted:
            def to_dict(self):
                return {"run_id": "r", "phase": "target_trial"}

        with patch("causal.cli.QuestionIntake") as intake_cls:
            intake_cls.return_value.run.return_value = StudyDrafted(
                state=_Drafted(), provider="openrouter", model="test-model"
            )
            exit_code, out, _ = self._run(
                "count the women", "--skip-question-gate", "--skip-literature", resolved_llm=resolved
            )
        options = intake_cls.return_value.run.call_args.args[1]
        self.assertTrue(options.skip_question_gate)
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
