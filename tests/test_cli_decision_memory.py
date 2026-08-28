"""Regression guard for `workflow decision-memory`.

Exercises the real `causal.cli.main` entry point end to end against a temp
run directory so a rename/refactor in `WorkflowManager.decision_memory` or
`causal.companion.decision_memory` fails loudly here instead of shipping a
broken CLI command (the same reason `test_cli_dag_export.py` exists)."""

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
                "comment": "measured after the exposure; post-treatment",
                "decided_at": "2026-08-20T09:00:00+00:00",
            }
        ],
        "major_confounders": [],
    }


class DecisionMemoryCliTests(unittest.TestCase):
    def _run(self, *extra: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            manager = WorkflowManager(Path(tmp) / "runs")
            protocol_path = Path(tmp) / "protocol.json"
            protocol_path.write_text(json.dumps(_protocol_payload()), encoding="utf-8")
            state = manager.start(protocol_path)

            argv = ["causalomop", "workflow", "decision-memory", state.run_id, *extra]
            stdout = StringIO()
            with patch.object(sys, "argv", argv), \
                    patch("causal.cli.WorkflowManager", return_value=manager), \
                    redirect_stdout(stdout):
                exit_code = cli.main()
        self.assertEqual(exit_code, 0)
        return stdout.getvalue()

    def test_prints_the_digest_text(self):
        out = self._run()
        self.assertIn("Covariates the reviewer has excluded:", out)
        self.assertIn("baseline_lactate", out)

    def test_json_flag_emits_a_parseable_object(self):
        payload = json.loads(self._run("--json"))
        self.assertEqual(
            [item["label"] for item in payload["rejected_covariates"]], ["baseline_lactate"]
        )
        self.assertEqual(payload["schema_version"], "1.0")


if __name__ == "__main__":
    unittest.main()
