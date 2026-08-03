"""Sanity checks on `studies/albumin_sepsis_28d/reference_dag.json` -- the
expert DAG transcribed from the published MIMIC-IV study's own S6 Fig
(Peters & Buhlmann-style validation-metric use case, see
docs/dag_generation_audit.md SS0f). This is not a test of production code;
it guards the transcribed *data* against silently breaking (a typo that
makes the JSON fail to parse into a `CausalDAG`, or an edit that changes the
confounder count without updating `source.transcription_notes`).

Deliberately builds the `CausalDAG` via `build_causal_dag` directly, *not*
`causal.agents.dag_critique.build_pre_check`: `build_pre_check` also runs
`assess_identification` (default `max_candidates=24`), which for a DAG this
size (20 observed baseline confounders) tries `CausalDAG.
minimum_adjustment_sets`'s exhaustive `combinations` search across every
subset size from 0 to 20. Because 3 of this DAG's confounders are
transcribed as unobserved (matching the paper's own gray "not controlled
for" nodes), no valid adjustment set exists at *any* size -- the search
cannot short-circuit and must exhaust all 2**20 (~1e6) subsets before
concluding failure, which does not finish in practical time. SID/SHD need
only the graph structure, not an identification verdict, so
`build_causal_dag` (a single pass, no combinatorial search) is not just
faster but the actually-correct level of the API for this use case.
"""

import json
import unittest
from pathlib import Path

from causal.agents.dag_identification import build_causal_dag
from causal.agents.dag_metrics import structural_intervention_distance
from causal.agents.hypothesis import HypothesisAgent
from causal.roadmap import CausalDAG

_STUDY_DIR = Path(__file__).resolve().parents[1] / "studies" / "albumin_sepsis_28d"


def _load_reference_dag() -> CausalDAG:
    protocol = json.loads((_STUDY_DIR / "protocol.json").read_text(encoding="utf-8"))
    reference = json.loads((_STUDY_DIR / "reference_dag.json").read_text(encoding="utf-8"))
    draft = {**protocol, "candidate_dag": reference["variables"], "covariate_relations": reference["relations"]}
    return build_causal_dag(HypothesisAgent().run(draft))


class ReferenceDagAlbuminSepsisTests(unittest.TestCase):
    def test_builds_a_valid_causal_dag_with_every_transcribed_confounder(self):
        dag = _load_reference_dag()
        confounder_count = sum(1 for node in dag.nodes if node.role == "confounder")
        self.assertEqual(confounder_count, 23)

    def test_has_zero_sid_from_itself(self):
        dag = _load_reference_dag()
        self.assertEqual(structural_intervention_distance(dag, dag), 0)

    def test_full_observed_confounder_set_still_leaves_an_open_backdoor_path(self):
        # Three confounders (Socio-economical factor, Comorbidity, lifestyle)
        # are transcribed as observed=False, matching the paper's own gray
        # "not controlled for" nodes -- conditioning on *every* observed
        # confounder simultaneously still cannot block the backdoor path
        # running through any of the three unobserved ones. A single
        # `assess_adjustment_set` call confirms this without the exhaustive
        # (and here, computationally infeasible) minimum-adjustment-set
        # search across every subset size -- see module docstring.
        dag = _load_reference_dag()
        observed_confounders = tuple(
            node.node_id for node in dag.nodes if node.role == "confounder" and node.observed
        )
        assessment = dag.assess_adjustment_set(observed_confounders)
        self.assertFalse(assessment.valid)
        self.assertIn("open_backdoor_path", assessment.blockers)


if __name__ == "__main__":
    unittest.main()
