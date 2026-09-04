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
from causal.agents.dag_metrics import (
    compare_generated_to_reference,
    structural_hamming_distance,
    structural_intervention_distance,
)
from causal.agents.hypothesis import HypothesisAgent
from causal.roadmap import CausalDAG

_STUDIES_DIR = Path(__file__).resolve().parents[1] / "studies"

_CASES = {
    "ppi_vs_h2ra_sup_mortality": 18,
    "corticosteroids_severe_cap_mortality": 20,
    "vanco_piptazo_vs_cefepime_aki": 16,
}


def _load_protocol(study: str) -> dict:
    return json.loads((_STUDIES_DIR / study / "protocol.json").read_text(encoding="utf-8"))


def _load_reference_dag(study: str) -> CausalDAG:
    study_dir = _STUDIES_DIR / study
    protocol = _load_protocol(study)
    reference = json.loads((study_dir / "reference_dag.json").read_text(encoding="utf-8"))
    draft = {
        **protocol,
        "candidate_dag": reference["variables"],
        "covariate_relations": reference["relations"],
    }
    return build_causal_dag(HypothesisAgent().run(draft))


def _first_pass_dag(study: str) -> CausalDAG:
    """A stand-in for a pipeline-drafted candidate DAG until a real run
    exists: each `protocol["first_pass_covariates"]` label as a plain
    pre-treatment confounder. Its labels are the LLM's own first-pass
    phrasing ("age", "baseline illness-severity score (SOFA ...)"), so it
    exercises `align_labels` against genuinely mismatched free text -- not
    the reference DAG's own labels."""
    protocol = _load_protocol(study)
    draft = {
        **protocol,
        "candidate_dag": [
            {
                "label": label,
                "role": "confounder",
                "temporal_classification": "pre_treatment",
                "rationale": f"{label} is a plausible common cause of treatment and outcome.",
                "pmid": "",
                "observed": True,
            }
            for label in protocol["first_pass_covariates"]
        ],
        "covariate_relations": [],
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


class ReferenceDagLabelAlignmentTests(unittest.TestCase):
    """`compare_generated_to_reference` on the real study artifacts. Until a
    pipeline run drafts a `candidate_dag` for these questions, the "generated"
    side is `_first_pass_dag` -- but the point here is the label-alignment
    step, which sees the same free-text mismatch a real draft would
    (`docs/dag_generation_audit.md` SS9: the comparison "is not done yet").
    """

    def test_alignment_recovers_the_bulk_of_the_reference_labels(self):
        for study in _CASES:
            with self.subTest(study=study):
                reference = _load_reference_dag(study)
                generated = _first_pass_dag(study)
                comparison = compare_generated_to_reference(reference, generated)
                reference_label_count = len({node.label for node in reference.nodes})
                matched = len(comparison.alignment.matches)
                # A first-pass covariate list is not expected to name every
                # unobserved confounder, but it should align to a clear
                # majority of the reference nodes -- otherwise the aligner is
                # too strict to be useful for the evaluation.
                self.assertGreaterEqual(matched, int(0.6 * reference_label_count))

    def test_alignment_collapses_the_label_only_distance_inflation(self):
        # The whole reason this layer exists: without alignment, "age" vs
        # "Age at ICU admission" (etc.) makes every distance meaningless.
        for study in _CASES:
            with self.subTest(study=study):
                reference = _load_reference_dag(study)
                generated = _first_pass_dag(study)
                bare_shd = structural_hamming_distance(reference, generated)
                comparison = compare_generated_to_reference(reference, generated)
                self.assertLess(comparison.shd, bare_shd)
                self.assertLessEqual(comparison.structural_agreement, 1.0)
                self.assertGreaterEqual(comparison.structural_agreement, 0.0)

    def test_unmatched_reference_labels_are_reported_for_review(self):
        # The unmatched list is the audit surface -- a reviewer reads it to
        # confirm the generated DAG really is missing those concepts rather
        # than just phrasing them unrecognisably.
        reference = _load_reference_dag("ppi_vs_h2ra_sup_mortality")
        generated = _first_pass_dag("ppi_vs_h2ra_sup_mortality")
        comparison = compare_generated_to_reference(reference, generated)
        unmatched = set(comparison.alignment.unmatched_reference)
        # The reconciliation note flags these as the ones an LLM draft misses;
        # the first-pass list here does not name them either.
        self.assertTrue(unmatched)
        self.assertEqual(comparison.alignment.unmatched_candidate, ())


if __name__ == "__main__":
    unittest.main()
