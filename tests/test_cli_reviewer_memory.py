"""Regression guard for `workflow remember-run` / `workflow reviewer-memory`
(`causal.companion.reviewer_memory`). Drives `causal.cli.main` end to end,
like `test_cli_decision_memory.py`."""

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from causal import cli
from causal.orchestrator import WorkflowManager


def _protocol_payload() -> dict:
    return {
        "study_id": "albumin_sepsis",
        "source_question": "Does albumin affect mortality?",
        "target_trial": {
            "population": "Adults with sepsis",
            "treatment": "Albumin plus crystalloids",
            "comparator": "Crystalloids alone",
            "time_zero": "ICU admission",
            "grace_period_hours": 24,
            "outcome": "28-day mortality",
            "follow_up_days": 28,
            "primary_estimand": "ATE",
            "secondary_estimands": [],
        },
        "first_pass_covariates": ["SOFA score"],
        "concept_sets": {"sepsis": {"status": "approved", "concept_ids": [132797]}},
        "candidate_dag": [],
        "covariate_relations": [],
        "rejected_covariates": [
            {
                "label": "baseline_lactate",
                "reviewer": "Dr. Chen",
                "comment": "post-treatment",
                "decided_at": "2026-08-01T00:00:00+00:00",
            }
        ],
        "major_confounders": [],
        "statistical_estimand_settings": {
            "missing_data_strategy": "complete_case_analysis",
            "censoring_strategy": "ignore_censoring_administrative_waiver",
            "estimation_procedure": "plug_in_standardization",
        },
    }


class ReviewerMemoryCliTests(unittest.TestCase):
    def _run(self, manager: WorkflowManager, *argv: str) -> str:
        stdout = StringIO()
        with patch.object(sys, "argv", ["causalomop", "workflow", *argv]), \
                patch("causal.cli.WorkflowManager", return_value=manager), \
                redirect_stdout(stdout):
            exit_code = cli.main()
        self.assertEqual(exit_code, 0)
        return stdout.getvalue()

    def test_remember_run_then_show_reviewer_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowManager(Path(tmp) / "runs")
            protocol_path = Path(tmp) / "protocol.json"
            protocol_path.write_text(json.dumps(_protocol_payload()), encoding="utf-8")
            state = manager.start(protocol_path)

            self._run(manager, "remember-run", state.run_id, "--reviewer", "Dr. Chen")
            text = self._run(manager, "reviewer-memory", "Dr. Chen")
            self.assertIn("baseline_lactate", text)
            self.assertIn("excluded in earlier studies", text)

            payload = json.loads(self._run(manager, "reviewer-memory", "Dr. Chen", "--json"))
            self.assertEqual(payload["reviewer_id"], "dr_chen")
            self.assertEqual(len(payload["estimand_choices"]), 1)

    def test_unknown_reviewer_prints_the_empty_sentence(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowManager(Path(tmp) / "runs")
            text = self._run(manager, "reviewer-memory", "Nobody")
            self.assertIn("No cross-study conventions recorded", text)


if __name__ == "__main__":
    unittest.main()
