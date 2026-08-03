import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
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
        "concept_sets": {
            "sepsis": {"status": "approved", "concept_ids": [132797]},
        },
        "candidate_dag": [
            {
                "label": "SOFA score",
                "role": "confounder",
                "pmid": "111",
                "temporal_classification": "pre_treatment",
                "rationale": "SOFA score influences both treatment choice and the outcome.",
            }
        ],
        "covariate_relations": [],
        "rejected_covariates": [],
        "major_confounders": [],
    }


class DagExportCliTests(unittest.TestCase):
    """Regression guard for `workflow dag-export`.

    This command previously raised `AttributeError` at runtime
    (`DAGIdentificationResult.warnings` does not exist -- the field is
    `technical_notes`) because it had zero test coverage; the bug was only
    found by reading the code, not by a failing test. This exercises the
    real `causal.cli.main` entry point end to end against a temp run
    directory so a future rename/refactor here fails loudly instead of
    shipping silently.
    """

    def test_dag_export_runs_without_raising_and_prints_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            manager = WorkflowManager(runs_dir)
            protocol_path = Path(tmp) / "protocol.json"
            protocol_path.write_text(json.dumps(_protocol_payload()), encoding="utf-8")
            state = manager.start(protocol_path)

            argv = ["causalomop", "workflow", "dag-export", state.run_id, "--highlight-adjustment-set"]
            stdout, stderr = StringIO(), StringIO()
            with patch.object(sys, "argv", argv), \
                    patch("causal.cli.WorkflowManager", return_value=manager), \
                    redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = cli.main()

        self.assertEqual(exit_code, 0)
        stdout_text = stdout.getvalue()
        self.assertIn("dag {", stdout_text)
        self.assertIn("adjusted", stdout_text)

        stderr_text = stderr.getvalue()
        self.assertIn("identified_by_backdoor_adjustment=True", stderr_text)
        self.assertIn("confounder_count=1", stderr_text)
        self.assertIn("adjustment_sets=1", stderr_text)
        self.assertIn("core=['SOFA score']", stderr_text)
        self.assertIn("evidence: literature_only=1", stderr_text)
        self.assertIn("cited_fraction=1.00", stderr_text)
        self.assertIn("// caveat: No published or expert reference DAG was checked", stderr_text)


if __name__ == "__main__":
    unittest.main()
