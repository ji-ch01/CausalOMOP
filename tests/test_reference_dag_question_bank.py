"""Sanity checks on the question-bank studies' `reference_dag.json` files
(`docs/validation_question_bank.md`): Q1 `ppi_vs_h2ra_sup_mortality`,
Q8 `corticosteroids_severe_cap_mortality`, Q5 `vanco_piptazo_vs_cefepime_aki`.

This is not a test of production code; it guards the transcribed *data*
against silently breaking (a typo that stops the JSON parsing into a
`CausalDAG`, or an edit that changes the confounder count without updating
`analysis_feature_reconciliation`).

Each of the three is constructed from the anchor trial's covariates plus
clinical reasoning rather than transcribed from a published DAG figure (PEPTIC
/ CAPE COD / ACORN publish none) -- see each file's `source.description`. Each
carries explicitly unobserved confounders, so the full observed-confounder set
must still leave an open backdoor path.

Builds via `build_causal_dag` directly, not `build_pre_check`: for a DAG with
16-20 confounders and no valid adjustment set (unobserved nodes),
`assess_identification`'s exhaustive `minimum_adjustment_sets` search cannot
short-circuit. SID/SHD need only the structure.
"""

import json
import unittest
from pathlib import Path

from causal.agents.dag_identification import build_causal_dag
from causal.agents.dag_metrics import structural_intervention_distance
from causal.agents.hypothesis import HypothesisAgent
from causal.roadmap import CausalDAG

_STUDIES_DIR = Path(__file__).resolve().parents[1] / "studies"

_CASES = {
    "ppi_vs_h2ra_sup_mortality": 18,
    "corticosteroids_severe_cap_mortality": 20,
    "vanco_piptazo_vs_cefepime_aki": 16,
}


def _load_reference_dag(study: str) -> CausalDAG:
    study_dir = _STUDIES_DIR / study
    protocol = json.loads((study_dir / "protocol.json").read_text(encoding="utf-8"))
    reference = json.loads((study_dir / "reference_dag.json").read_text(encoding="utf-8"))
    draft = {
        **protocol,
        "candidate_dag": reference["variables"],
        "covariate_relations": reference["relations"],
    }
    return build_causal_dag(HypothesisAgent().run(draft))


class ReferenceDagQuestionBankTests(unittest.TestCase):
    def test_each_builds_a_valid_causal_dag_with_the_recorded_confounder_count(self):
        for study, expected_confounders in _CASES.items():
            with self.subTest(study=study):
                dag = _load_reference_dag(study)
                confounder_count = sum(1 for node in dag.nodes if node.role == "confounder")
                self.assertEqual(confounder_count, expected_confounders)

    def test_confounder_count_matches_analysis_feature_reconciliation(self):
        for study in _CASES:
            with self.subTest(study=study):
                reference = json.loads(
                    (_STUDIES_DIR / study / "reference_dag.json").read_text(encoding="utf-8")
                )
                recon = reference["analysis_feature_reconciliation"]
                declared = recon["observed_nodes"] + recon["unobserved_nodes"]
                transcribed = len(reference["variables"])
                self.assertEqual(declared, transcribed)

    def test_each_has_zero_sid_from_itself(self):
        for study in _CASES:
            with self.subTest(study=study):
                dag = _load_reference_dag(study)
                self.assertEqual(structural_intervention_distance(dag, dag), 0)

    def test_full_observed_confounder_set_still_leaves_an_open_backdoor_path(self):
        # Every case transcribes at least one confounder as observed=False
        # (institutional prescribing culture, goals-of-care limitation,
        # radiographic extent, ...), so conditioning on every observed
        # confounder cannot block the backdoor path through the unobserved
        # ones.
        for study in _CASES:
            with self.subTest(study=study):
                dag = _load_reference_dag(study)
                observed_confounders = tuple(
                    node.node_id
                    for node in dag.nodes
                    if node.role == "confounder" and node.observed
                )
                assessment = dag.assess_adjustment_set(observed_confounders)
                self.assertFalse(assessment.valid)
                self.assertIn("open_backdoor_path", assessment.blockers)


if __name__ == "__main__":
    unittest.main()
