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


def _protocol_payload(candidate_dag: list[dict]) -> dict:
    return {
        "study_id": "demo_study",
        "source_question": "Does the drug affect mortality?",
        "target_trial": {
            "population": "ICU adults",
            "treatment": "The drug",
            "comparator": "No drug",
            "time_zero": "ICU admission",
            "grace_period_hours": 24,
            "outcome": "In-hospital mortality",
            "follow_up_days": 28,
            "primary_estimand": "ATE",
            "secondary_estimands": [],
        },
        "first_pass_covariates": ["age"],
        "concept_sets": {"the_drug": {"status": "approved", "concept_ids": [1]}},
        "candidate_dag": candidate_dag,
        "covariate_relations": [],
        "rejected_covariates": [],
        "major_confounders": [],
    }


def _confounder(label: str) -> dict:
    return {
        "label": label,
        "role": "confounder",
        "pmid": "",
        "temporal_classification": "pre_treatment",
        "rationale": f"{label} is a common cause of treatment and outcome.",
    }


class CompareReferenceDagCliTests(unittest.TestCase):
    """`workflow compare-reference-dag`: the generated-vs-reference DAG score
    with `causal.agents.dag_metrics` label alignment. Exercises the real
    `causal.cli.main` entry point end to end against temp run / reference
    directories."""

    def _run(
        self,
        extra_argv: list[str],
        *,
        run_candidate: list[dict] | None = None,
        reference_variables: list[dict] | None = None,
    ) -> tuple[int, str, str]:
        run_candidate = run_candidate or [_confounder("age"), _confounder("sepsis")]
        reference_variables = reference_variables or [
            _confounder("Age at ICU admission"),
            _confounder("Sepsis"),
            _confounder("Baseline illness severity"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_dir = root / "runs"
            manager = WorkflowManager(runs_dir)

            run_protocol = root / "protocol.json"
            run_protocol.write_text(
                json.dumps(_protocol_payload(run_candidate)), encoding="utf-8"
            )
            state = manager.start(run_protocol)

            reference_dir = root / "reference"
            reference_dir.mkdir()
            (reference_dir / "protocol.json").write_text(
                json.dumps(_protocol_payload([])), encoding="utf-8"
            )
            (reference_dir / "reference_dag.json").write_text(
                json.dumps({"variables": reference_variables, "relations": []}), encoding="utf-8"
            )

            argv = [
                "causalomop", "workflow", "compare-reference-dag", state.run_id,
                "--reference", str(reference_dir), *extra_argv,
            ]
            stdout, stderr = StringIO(), StringIO()
            with patch.object(sys, "argv", argv), \
                    patch("causal.cli.WorkflowManager", return_value=manager), \
                    redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = cli.main()
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_text_output_reports_alignment_and_distances(self):
        exit_code, stdout_text, _ = self._run([])
        self.assertEqual(exit_code, 0)
        # "age" aligns to "Age at ICU admission" despite the phrasing gap.
        self.assertIn("'age'  ->  'Age at ICU admission'", stdout_text)
        self.assertIn("'sepsis'  ->  'Sepsis'", stdout_text)
        # The reference confounder the run never proposed is surfaced for review.
        self.assertIn("Baseline illness severity", stdout_text)
        self.assertIn("SHD (structural)", stdout_text)
        self.assertIn("SID (generated vs reference-as-true)", stdout_text)

    def test_json_output_is_machine_readable(self):
        exit_code, stdout_text, _ = self._run(["--json"])
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout_text)
        self.assertEqual(
            set(payload),
            {
                "alignment", "shared_label_count", "union_label_count", "shd",
                "sid_generated_vs_reference", "sid_reference_vs_generated",
                "structural_agreement", "causal_agreement",
            },
        )
        self.assertEqual(payload["alignment"]["unmatched_reference"], ["Baseline illness severity"])
        self.assertGreater(payload["sid_generated_vs_reference"], 0)

    def test_manual_alias_forces_a_pair(self):
        exit_code, stdout_text, _ = self._run(
            ["--alias", "sepsis=Baseline illness severity", "--json"]
        )
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout_text)
        manual = [m for m in payload["alignment"]["matches"] if m["method"] == "manual"]
        self.assertEqual(len(manual), 1)
        self.assertEqual(manual[0]["reference_label"], "Baseline illness severity")

    def test_malformed_alias_is_rejected(self):
        exit_code, _, stderr_text = self._run(["--alias", "missing-separator"])
        self.assertEqual(exit_code, 1)
        self.assertIn("CANDIDATE=REFERENCE", stderr_text)

    def test_merge_collapses_a_covariate_family_before_matching(self):
        exit_code, stdout_text, _ = self._run(
            [
                "--merge", "Nephrotoxin cluster=NSAID exposure|contrast exposure|aminoglycoside exposure",
                "--json",
            ],
            run_candidate=[
                _confounder("NSAID exposure"),
                _confounder("contrast exposure"),
                _confounder("aminoglycoside exposure"),
            ],
            reference_variables=[_confounder("Nephrotoxin cluster")],
        )
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout_text)
        # The three members collapsed to one node that then matched the
        # reference's single cluster node -- nothing left unmatched on either side.
        self.assertEqual(payload["alignment"]["unmatched_reference"], [])
        self.assertEqual(payload["alignment"]["unmatched_candidate"], [])
        self.assertEqual(payload["shd"], 0)

    def test_malformed_merge_is_rejected(self):
        exit_code, _, stderr_text = self._run(["--merge", "no-separator-here"])
        self.assertEqual(exit_code, 1)
        self.assertIn("--merge", stderr_text)

    def test_missing_reference_dir_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = WorkflowManager(root / "runs")
            run_protocol = root / "protocol.json"
            run_protocol.write_text(json.dumps(_protocol_payload([])), encoding="utf-8")
            state = manager.start(run_protocol)
            argv = [
                "causalomop", "workflow", "compare-reference-dag", state.run_id,
                "--reference", str(root / "does_not_exist"),
            ]
            stdout, stderr = StringIO(), StringIO()
            with patch.object(sys, "argv", argv), \
                    patch("causal.cli.WorkflowManager", return_value=manager), \
                    redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = cli.main()
        self.assertEqual(exit_code, 1)
        self.assertIn("No reference_dag.json", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
