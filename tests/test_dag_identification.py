import unittest

from causal.agents.dag_identification import (
    DAGIdentificationResult,
    EvidenceSourceSummary,
    adjustment_set_summary,
    assess_clinical_completeness,
    assess_identification,
    build_causal_dag,
    evidence_summary,
    grounding_caveat,
    review_dag,
)
from causal.agents.hypothesis import HypothesisAgent
from causal.roadmap import CausalDAG, CausalEdge, CausalNode


def protocol(
    candidate_dag=(),
    first_pass_covariates=(),
    covariate_relations=(),
    rejected_covariates=(),
    major_confounders=(),
):
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
        "first_pass_covariates": list(first_pass_covariates),
        "concept_sets": {
            "sepsis": {"status": "approved", "concept_ids": [132797]},
        },
        "candidate_dag": list(candidate_dag),
        "covariate_relations": list(covariate_relations),
        "rejected_covariates": list(rejected_covariates),
        "major_confounders": list(major_confounders),
    }


def confounder(
    label,
    pmid="111",
    temporal_classification="pre_treatment",
    kg_evidence_id="",
    certainty="established",
    evidence_level=None,
):
    entry = {
        "label": label,
        "role": "confounder",
        "pmid": pmid,
        "temporal_classification": temporal_classification,
        "rationale": f"{label} influences both treatment choice and the outcome.",
        "kg_evidence_id": kg_evidence_id,
        "certainty": certainty,
    }
    if evidence_level is not None:
        entry["evidence_level"] = evidence_level
    return entry


class DAGIdentificationTests(unittest.TestCase):
    def test_empty_candidate_dag_is_trivially_identified_with_a_note(self):
        hypothesis = HypothesisAgent().run(protocol())
        result = assess_identification(hypothesis)
        self.assertEqual(result.confounder_count, 0)
        self.assertTrue(result.identified_by_backdoor_adjustment)
        self.assertEqual(result.minimum_adjustment_sets, ((),))
        self.assertTrue(any("No confounders were proposed" in w for w in result.technical_notes))

    def test_single_confounder_requires_adjustment_to_block_backdoor_path(self):
        hypothesis = HypothesisAgent().run(protocol([confounder("SOFA score")]))
        result = assess_identification(hypothesis)
        self.assertEqual(result.confounder_count, 1)
        self.assertTrue(result.identified_by_backdoor_adjustment)
        self.assertEqual(len(result.minimum_adjustment_sets), 1)
        adjustment_set = result.minimum_adjustment_sets[0]
        self.assertEqual(len(adjustment_set), 1)
        node = result.dag.node_map[adjustment_set[0]]
        self.assertEqual(node.label, "SOFA score")
        self.assertEqual(node.role, "confounder")
        self.assertFalse(result.technical_notes)

    def test_distinct_labels_produce_distinct_nodes(self):
        hypothesis = HypothesisAgent().run(protocol([
            confounder("SOFA score", pmid="111"),
            confounder("Baseline lactate", pmid="222"),
        ]))
        dag = build_causal_dag(hypothesis)
        confounder_nodes = [node for node in dag.nodes if node.role == "confounder"]
        self.assertEqual(len(confounder_nodes), 2)
        self.assertEqual(len({node.node_id for node in confounder_nodes}), 2)

    def test_dag_represents_the_active_comparator(self):
        hypothesis = HypothesisAgent().run(protocol())
        dag = build_causal_dag(hypothesis)
        self.assertEqual(dag.comparator, "Crystalloids alone")

    def test_edges_carry_pmid_as_evidence(self):
        hypothesis = HypothesisAgent().run(protocol([confounder("SOFA score", pmid="111")]))
        dag = build_causal_dag(hypothesis)
        confounder_edges = [edge for edge in dag.edges if edge.source != "exposure"]
        self.assertEqual(len(confounder_edges), 2)
        self.assertTrue(all(edge.evidence_ids == ("111",) for edge in confounder_edges))
        self.assertTrue(all(edge.status == "evidence_supported" for edge in confounder_edges))
        self.assertEqual({edge.target for edge in confounder_edges}, {"exposure", "outcome"})

    def test_hypothesis_edge_always_links_exposure_to_outcome(self):
        hypothesis = HypothesisAgent().run(protocol())
        dag = build_causal_dag(hypothesis)
        hypothesis_edges = [
            edge for edge in dag.edges if edge.source == "exposure" and edge.target == "outcome"
        ]
        self.assertEqual(len(hypothesis_edges), 1)
        self.assertEqual(hypothesis_edges[0].status, "proposed")
        self.assertEqual(hypothesis_edges[0].evidence_ids, ())

    def test_treatment_cause_only_gets_single_edge_to_exposure(self):
        hypothesis = HypothesisAgent().run(protocol([
            {
                "label": "local prescribing pattern",
                "role": "treatment_cause_only",
                "pmid": "",
                "temporal_classification": "pre_treatment",
                "rationale": "Institutional practice affects albumin use but not mortality directly.",
            }
        ]))
        dag = build_causal_dag(hypothesis)
        node = next(n for n in dag.nodes if n.label == "local prescribing pattern")
        self.assertEqual(node.role, "other")
        self.assertEqual(node.timing, "baseline")
        node_edges = [edge for edge in dag.edges if edge.source == node.node_id or edge.target == node.node_id]
        self.assertEqual(len(node_edges), 1)
        self.assertEqual((node_edges[0].source, node_edges[0].target), (node.node_id, "exposure"))

    def test_outcome_cause_only_gets_single_edge_to_outcome_as_prognostic(self):
        hypothesis = HypothesisAgent().run(protocol([
            {
                "label": "baseline frailty index",
                "role": "outcome_cause_only",
                "pmid": "",
                "temporal_classification": "pre_treatment",
                "rationale": "Frailty predicts mortality but does not influence albumin prescribing here.",
            }
        ]))
        dag = build_causal_dag(hypothesis)
        node = next(n for n in dag.nodes if n.label == "baseline frailty index")
        self.assertEqual(node.role, "prognostic")
        self.assertEqual(node.timing, "baseline")
        node_edges = [edge for edge in dag.edges if edge.source == node.node_id or edge.target == node.node_id]
        self.assertEqual(len(node_edges), 1)
        self.assertEqual((node_edges[0].source, node_edges[0].target), (node.node_id, "outcome"))

    def test_mediator_sits_on_the_causal_pathway_and_is_never_in_adjustment_sets(self):
        hypothesis = HypothesisAgent().run(protocol([
            confounder("SOFA score"),
            {
                "label": "vasopressor use after admission",
                "role": "mediator",
                "pmid": "",
                "temporal_classification": "post_treatment",
                "rationale": "Albumin plus crystalloids changes downstream vasopressor requirements, which affect mortality.",
            },
        ]))
        dag = build_causal_dag(hypothesis)
        node = next(n for n in dag.nodes if n.label == "vasopressor use after admission")
        self.assertEqual(node.role, "mediator")
        self.assertEqual(node.timing, "post_treatment")
        node_edges = [edge for edge in dag.edges if edge.source == node.node_id or edge.target == node.node_id]
        self.assertEqual(len(node_edges), 2)
        self.assertEqual({(e.source, e.target) for e in node_edges}, {("exposure", node.node_id), (node.node_id, "outcome")})

        result = assess_identification(hypothesis)
        for adjustment_set in result.minimum_adjustment_sets:
            self.assertNotIn(node.node_id, adjustment_set)

    def test_collider_is_never_in_adjustment_sets(self):
        hypothesis = HypothesisAgent().run(protocol([
            confounder("SOFA score"),
            {
                "label": "ICU length of stay",
                "role": "collider",
                "pmid": "",
                "temporal_classification": "post_treatment",
                "rationale": "Length of stay is jointly determined by treatment response and survival to discharge.",
            },
        ]))
        dag = build_causal_dag(hypothesis)
        node = next(n for n in dag.nodes if n.label == "ICU length of stay")
        self.assertEqual(node.role, "collider")
        self.assertEqual(node.timing, "outcome")
        node_edges = [edge for edge in dag.edges if edge.source == node.node_id or edge.target == node.node_id]
        self.assertEqual({(e.source, e.target) for e in node_edges}, {("exposure", node.node_id), ("outcome", node.node_id)})

        result = assess_identification(hypothesis)
        for adjustment_set in result.minimum_adjustment_sets:
            self.assertNotIn(node.node_id, adjustment_set)

    def test_instrument_gets_single_edge_to_exposure_and_is_never_in_adjustment_sets(self):
        hypothesis = HypothesisAgent().run(protocol([
            confounder("SOFA score"),
            {
                "label": "regional albumin supply shock",
                "role": "instrument",
                "pmid": "",
                "temporal_classification": "pre_treatment",
                "rationale": "Supply availability affects albumin use but has no plausible direct path to mortality.",
            },
        ]))
        dag = build_causal_dag(hypothesis)
        node = next(n for n in dag.nodes if n.label == "regional albumin supply shock")
        self.assertEqual(node.role, "instrument")
        self.assertEqual(node.timing, "baseline")
        node_edges = [edge for edge in dag.edges if edge.source == node.node_id or edge.target == node.node_id]
        self.assertEqual(len(node_edges), 1)
        self.assertEqual((node_edges[0].source, node_edges[0].target), (node.node_id, "exposure"))

        result = assess_identification(hypothesis)
        for adjustment_set in result.minimum_adjustment_sets:
            self.assertNotIn(node.node_id, adjustment_set)

    def test_effect_modifier_gets_single_edge_to_outcome(self):
        hypothesis = HypothesisAgent().run(protocol([
            {
                "label": "baseline renal function",
                "role": "effect_modifier",
                "pmid": "",
                "temporal_classification": "pre_treatment",
                "rationale": "The abstract reports a stronger albumin-mortality association in patients with impaired renal function.",
            }
        ]))
        dag = build_causal_dag(hypothesis)
        node = next(n for n in dag.nodes if n.label == "baseline renal function")
        self.assertEqual(node.role, "effect_modifier")
        self.assertEqual(node.timing, "baseline")
        node_edges = [edge for edge in dag.edges if edge.source == node.node_id or edge.target == node.node_id]
        self.assertEqual(len(node_edges), 1)
        self.assertEqual((node_edges[0].source, node_edges[0].target), (node.node_id, "outcome"))

    def test_selection_is_never_in_adjustment_sets_and_blocks_covariate_relations(self):
        hypothesis = HypothesisAgent().run(protocol([
            confounder("SOFA score"),
            {
                "label": "ICU survival to discharge assessment",
                "role": "selection",
                "pmid": "",
                "temporal_classification": "post_treatment",
                "rationale": "Being assessed at discharge depends jointly on receiving albumin and on surviving.",
            },
        ]))
        dag = build_causal_dag(hypothesis)
        node = next(n for n in dag.nodes if n.label == "ICU survival to discharge assessment")
        self.assertEqual(node.role, "selection")
        self.assertEqual(node.timing, "outcome")
        node_edges = [edge for edge in dag.edges if edge.source == node.node_id or edge.target == node.node_id]
        self.assertEqual({(e.source, e.target) for e in node_edges}, {("exposure", node.node_id), ("outcome", node.node_id)})

        result = assess_identification(hypothesis)
        for adjustment_set in result.minimum_adjustment_sets:
            self.assertNotIn(node.node_id, adjustment_set)

        hypothesis_with_bad_relation = HypothesisAgent().run(protocol(
            candidate_dag=[
                confounder("SOFA score"),
                {
                    "label": "ICU survival to discharge assessment",
                    "role": "selection",
                    "pmid": "",
                    "temporal_classification": "post_treatment",
                    "rationale": "Being assessed at discharge depends jointly on receiving albumin and on surviving.",
                },
            ],
            covariate_relations=[
                {
                    "source": "SOFA score",
                    "target": "ICU survival to discharge assessment",
                    "pmid": "",
                    "rationale": "Higher SOFA scores reduce the chance of being assessed at discharge.",
                }
            ],
        ))
        with self.assertRaises(ValueError):
            build_causal_dag(hypothesis_with_bad_relation)

    def test_contested_edge_propagates_and_surfaces_an_advisory_note(self):
        hypothesis = HypothesisAgent().run(protocol([
            {
                "label": "SOFA score",
                "role": "confounder",
                "pmid": "111",
                "temporal_classification": "pre_treatment",
                "rationale": "Different cited studies disagree on this association's direction.",
                "certainty": "contested",
            },
        ]))
        dag = build_causal_dag(hypothesis)
        confounder_edges = [edge for edge in dag.edges if edge.source != "exposure" or edge.target != "outcome"]
        self.assertTrue(all(edge.certainty == "contested" for edge in confounder_edges))

        # Identification is purely graph-theoretic and does not know about certainty.
        identification = assess_identification(hypothesis)
        self.assertFalse(any("contested" in note for note in identification.technical_notes))

        clinical = assess_clinical_completeness(hypothesis)
        self.assertTrue(any("contested" in note and "SOFA score" in note for note in clinical.advisory_notes))
        self.assertTrue(clinical.approved_for_dag_gate)

    def test_covariate_relation_certainty_propagates_to_its_edge(self):
        hypothesis = HypothesisAgent().run(protocol(
            candidate_dag=[
                confounder("renal function"),
                confounder("SOFA score"),
            ],
            covariate_relations=[
                {
                    "source": "renal function",
                    "target": "SOFA score",
                    "pmid": "",
                    "rationale": "Direction of causation between the two is debated.",
                    "certainty": "unknown_direction",
                }
            ],
        ))
        dag = build_causal_dag(hypothesis)
        renal_id = next(n.node_id for n in dag.nodes if n.label == "renal function")
        sofa_id = next(n.node_id for n in dag.nodes if n.label == "SOFA score")
        relation_edge = next(edge for edge in dag.edges if edge.source == renal_id and edge.target == sofa_id)
        self.assertEqual(relation_edge.certainty, "unknown_direction")

    def test_unobserved_confounder_blocks_identification_when_it_is_the_only_blocker(self):
        hypothesis = HypothesisAgent().run(protocol([
            {
                "label": "unmeasured frailty",
                "role": "confounder",
                "pmid": "",
                "temporal_classification": "pre_treatment",
                "rationale": "Frailty plausibly affects both treatment choice and mortality.",
                "observed": False,
            }
        ]))
        result = assess_identification(hypothesis)
        self.assertEqual(result.confounder_count, 1)
        self.assertFalse(result.identified_by_backdoor_adjustment)
        self.assertEqual(result.minimum_adjustment_sets, ())

    def test_covariate_relation_edge_is_added_between_declared_variables(self):
        hypothesis = HypothesisAgent().run(protocol(
            candidate_dag=[
                confounder("renal function"),
                confounder("SOFA score"),
            ],
            covariate_relations=[
                {
                    "source": "renal function",
                    "target": "SOFA score",
                    "pmid": "",
                    "rationale": "Renal function is a component of the SOFA score.",
                }
            ],
        ))
        dag = build_causal_dag(hypothesis)
        renal_id = next(n.node_id for n in dag.nodes if n.label == "renal function")
        sofa_id = next(n.node_id for n in dag.nodes if n.label == "SOFA score")
        relation_edges = [edge for edge in dag.edges if edge.source == renal_id and edge.target == sofa_id]
        self.assertEqual(len(relation_edges), 1)

    def test_covariate_relation_participates_in_backdoor_moralization(self):
        # renal function and SOFA score are both parents of a collider; a covariate
        # relation between them must still be respected by the moralized graph used
        # for backdoor-path blocking (no engine change, just wiring the extra edge in).
        hypothesis = HypothesisAgent().run(protocol(
            candidate_dag=[
                confounder("renal function"),
                confounder("SOFA score"),
            ],
            covariate_relations=[
                {
                    "source": "renal function",
                    "target": "SOFA score",
                    "pmid": "",
                    "rationale": "Renal function is a component of the SOFA score.",
                }
            ],
        ))
        result = assess_identification(hypothesis)
        self.assertTrue(result.identified_by_backdoor_adjustment)

    def test_role_temporal_mismatch_is_rejected_before_it_reaches_the_dag(self):
        with self.assertRaises(ValueError):
            HypothesisAgent().run(protocol([
                {
                    "label": "hospital-acquired resistant organism",
                    "role": "confounder",
                    "pmid": "",
                    # Contradiction: a role that implies pre/at-treatment timing
                    # combined with a variable the classifier itself calls
                    # post-treatment. This is exactly the failure mode that let
                    # CRPA infection / resistant Serratia enter as confounders.
                    "temporal_classification": "post_treatment",
                    "rationale": "Organism recovered on culture after antibiotic exposure began.",
                }
            ]))

    def test_unclassified_first_pass_covariate_is_unresolved_not_silently_dropped(self):
        hypothesis = HypothesisAgent().run(protocol(
            candidate_dag=[confounder("SOFA score")],
            first_pass_covariates=["SOFA score", "age", "mechanical ventilation"],
        ))
        labels_in_dag = {node.label for node in build_causal_dag(hypothesis).nodes}
        self.assertNotIn("age", labels_in_dag)
        self.assertNotIn("mechanical ventilation", labels_in_dag)
        self.assertEqual(set(hypothesis.unresolved_covariates), {"age", "mechanical ventilation"})

        clinical = assess_clinical_completeness(hypothesis)
        self.assertTrue(any("'age'" in n and "unresolved" in n for n in clinical.advisory_notes))
        self.assertTrue(any("'mechanical ventilation'" in n for n in clinical.advisory_notes))
        self.assertFalse(clinical.blocking_issues)
        self.assertTrue(clinical.approved_for_dag_gate)

    def test_unresolved_major_confounder_blocks_the_dag_gate(self):
        hypothesis = HypothesisAgent().run(protocol(
            candidate_dag=[confounder("SOFA score")],
            first_pass_covariates=["SOFA score", "age", "presence of septic shock"],
            major_confounders=["age", "presence of septic shock"],
        ))
        self.assertFalse(hypothesis.executable)
        self.assertEqual(set(hypothesis.unresolved_major_confounders), {"age", "presence of septic shock"})

        review = review_dag(hypothesis)
        self.assertFalse(review.clinical.approved_for_dag_gate)
        self.assertTrue(any("age" in issue for issue in review.clinical.blocking_issues))
        self.assertTrue(any("presence of septic shock" in issue for issue in review.clinical.blocking_issues))
        # Graph-theoretic identification is unaffected by the clinical block:
        # SOFA score alone still identifies the effect by backdoor adjustment.
        self.assertTrue(review.identification.identified_by_backdoor_adjustment)

    def test_classifying_a_major_confounder_unblocks_the_gate(self):
        hypothesis = HypothesisAgent().run(protocol(
            candidate_dag=[confounder("SOFA score"), confounder("age")],
            first_pass_covariates=["SOFA score", "age"],
            major_confounders=["age"],
        ))
        self.assertEqual(hypothesis.unresolved_major_confounders, ())
        self.assertTrue(assess_clinical_completeness(hypothesis).approved_for_dag_gate)

    def test_rejected_major_confounder_unblocks_the_gate_with_an_audit_trail(self):
        hypothesis = HypothesisAgent().run(protocol(
            candidate_dag=[confounder("SOFA score")],
            first_pass_covariates=["SOFA score", "age"],
            major_confounders=["age"],
            rejected_covariates=[
                {
                    "label": "age",
                    "reviewer": "Dr. Chen",
                    "comment": "Cohort is age-restricted at enrollment; no residual confounding by age.",
                }
            ],
        ))
        self.assertEqual(hypothesis.unresolved_major_confounders, ())
        clinical = assess_clinical_completeness(hypothesis)
        self.assertTrue(clinical.approved_for_dag_gate)
        self.assertTrue(
            any(
                "'age'" in n and "reviewed and excluded by Dr. Chen" in n and "age-restricted" in n
                for n in clinical.advisory_notes
            )
        )

    def test_zero_confounders_with_unresolved_covariates_is_advisory_not_blocking(self):
        """An empty confounder set trivially satisfies backdoor identification
        (no path left to block) -- this must never be silently read as "no
        confounding". The reviewer gets a loud advisory note, but this alone
        must not block approval: only a human-declared `major_confounders`
        entry is allowed to do that (see
        `causal.agents.hypothesis._major_confounders` and
        `WorkflowIdentificationGateTests.test_empty_dag_is_trivially_identified_and_not_blocked_here`
        in test_workflow.py).
        """
        hypothesis = HypothesisAgent().run(protocol(
            first_pass_covariates=["age", "SOFA score"],
        ))
        clinical = assess_clinical_completeness(hypothesis)
        self.assertTrue(
            any(
                "no candidate_dag variable has been classified as a confounder" in note
                and "'age'" in note and "'SOFA score'" in note
                for note in clinical.advisory_notes
            )
        )
        self.assertFalse(clinical.blocking_issues)
        self.assertTrue(clinical.approved_for_dag_gate)

    def test_zero_confounders_note_is_skipped_when_at_least_one_is_classified(self):
        hypothesis = HypothesisAgent().run(protocol(
            candidate_dag=[confounder("SOFA score")],
            first_pass_covariates=["SOFA score", "age"],
        ))
        clinical = assess_clinical_completeness(hypothesis)
        self.assertFalse(
            any("no candidate_dag variable has been classified as a confounder" in note
                for note in clinical.advisory_notes)
        )

    def test_major_confounders_must_be_declared_first_pass_covariates(self):
        with self.assertRaises(ValueError):
            HypothesisAgent().run(protocol(
                first_pass_covariates=["SOFA score"],
                major_confounders=["age"],
            ))


class EvidenceSummaryTests(unittest.TestCase):
    """`evidence_summary` classifies every candidate_dag/covariate_relations
    item into exactly one of four grounding buckets (see its docstring)."""

    def test_classifies_each_grounding_bucket(self):
        hypothesis = HypothesisAgent().run(protocol(candidate_dag=[
            confounder("literature only", pmid="111"),
            confounder("kg only", pmid="optimuskg:X1|associated_with|X2"),
            confounder(
                "corroborated", pmid="111", kg_evidence_id="optimuskg:X3|associated_with|X4"
            ),
            {
                "label": "uncited",
                "role": "confounder",
                "pmid": "",
                "temporal_classification": "pre_treatment",
                "rationale": "Expert-asserted without a citation.",
            },
        ]))
        summary = evidence_summary(hypothesis)
        self.assertEqual(summary.total_items, 4)
        self.assertEqual(summary.literature_only, 1)
        self.assertEqual(summary.kg_only, 1)
        self.assertEqual(summary.corroborated, 1)
        self.assertEqual(summary.uncited, 1)
        self.assertAlmostEqual(summary.cited_fraction, 0.75)
        self.assertAlmostEqual(summary.corroborated_fraction, 0.25)

    def test_covariate_relations_count_toward_the_same_summary(self):
        hypothesis = HypothesisAgent().run(protocol(
            candidate_dag=[confounder("a"), confounder("b")],
            covariate_relations=[
                {"source": "a", "target": "b", "pmid": "111", "rationale": "a precedes b."}
            ],
        ))
        summary = evidence_summary(hypothesis)
        self.assertEqual(summary.total_items, 3)
        self.assertEqual(summary.literature_only, 3)

    def test_empty_dag_reports_zero_fractions_not_a_division_error(self):
        hypothesis = HypothesisAgent().run(protocol())
        summary = evidence_summary(hypothesis)
        self.assertEqual(summary.total_items, 0)
        self.assertEqual(summary.cited_fraction, 0.0)
        self.assertEqual(summary.corroborated_fraction, 0.0)


def _node(node_id, role, timing="baseline", observed=True):
    return CausalNode(node_id, node_id.replace("_", " ").title(), role, timing, observed)


def _edge(source, target):
    return CausalEdge(source, target, f"Assumed causal relation {source}->{target}")


class AdjustmentSetSummaryTests(unittest.TestCase):
    """Synthetic DAG built directly from `causal.roadmap` primitives (not
    through the LLM/HypothesisArtifact bridge) so every category
    `adjustment_set_summary` reports can be exercised in one graph:

    - `confounder1`/`confounder2` each sit on their own independent backdoor
      path (a direct edge to both exposure and outcome) -- neither is
      optional, so both must appear in *every* minimum adjustment set (core).
    - `node_a`/`node_b` sit on a single shared backdoor path as a chain
      (`node_a -> node_b -> exposure`, `node_a -> outcome`): conditioning on
      either one blocks that path, so they are interchangeable
      (alternatives), never both required together.
    - `unused_confounder` carries the `confounder` role but has no edge to
      exposure at all, so it never lies on any backdoor path and can never
      be required by identification, however it is labeled.
    """

    def _build_dag(self) -> CausalDAG:
        return CausalDAG(
            nodes=(
                _node("exposure", "exposure", "treatment"),
                _node("outcome", "outcome", "outcome"),
                _node("confounder1", "confounder"),
                _node("confounder2", "confounder"),
                _node("node_a", "prognostic"),
                _node("node_b", "other"),
                _node("unused_confounder", "confounder"),
            ),
            edges=(
                _edge("exposure", "outcome"),
                _edge("confounder1", "exposure"),
                _edge("confounder1", "outcome"),
                _edge("confounder2", "exposure"),
                _edge("confounder2", "outcome"),
                _edge("node_a", "outcome"),
                _edge("node_a", "node_b"),
                _edge("node_b", "exposure"),
            ),
            exposure_id="exposure",
            outcome_id="outcome",
        )

    def _identification(self) -> DAGIdentificationResult:
        dag = self._build_dag()
        sets = dag.minimum_adjustment_sets()
        return DAGIdentificationResult(
            dag=dag,
            confounder_count=sum(1 for node in dag.nodes if node.role == "confounder"),
            identified_by_backdoor_adjustment=bool(sets),
            minimum_adjustment_sets=sets,
            technical_notes=(),
        )

    def test_core_alternative_and_unused_are_correctly_partitioned(self):
        identification = self._identification()
        # Sanity check on the fixture itself before trusting the summary of it.
        self.assertEqual(
            set(identification.minimum_adjustment_sets),
            {
                tuple(sorted(("confounder1", "confounder2", "node_a"))),
                tuple(sorted(("confounder1", "confounder2", "node_b"))),
            },
        )

        summary = adjustment_set_summary(identification)
        self.assertEqual(summary.set_count, 2)
        self.assertEqual(summary.set_size, 3)
        self.assertEqual(set(summary.core_variable_labels), {"Confounder1", "Confounder2"})
        self.assertEqual(set(summary.alternative_variable_labels), {"Node A", "Node B"})
        self.assertEqual(summary.unused_confounder_labels, ("Unused Confounder",))

    def test_unidentified_dag_reports_all_confounders_as_unused(self):
        dag = CausalDAG(
            nodes=(
                _node("exposure", "exposure", "treatment"),
                _node("outcome", "outcome", "outcome"),
                _node("latent_confounder", "confounder", observed=False),
            ),
            edges=(
                _edge("exposure", "outcome"),
                _edge("latent_confounder", "exposure"),
                _edge("latent_confounder", "outcome"),
            ),
            exposure_id="exposure",
            outcome_id="outcome",
        )
        identification = DAGIdentificationResult(
            dag=dag, confounder_count=1, identified_by_backdoor_adjustment=False,
            minimum_adjustment_sets=(), technical_notes=(),
        )
        summary = adjustment_set_summary(identification)
        self.assertEqual(summary.set_count, 0)
        self.assertEqual(summary.set_size, 0)
        self.assertEqual(summary.core_variable_labels, ())
        self.assertEqual(summary.alternative_variable_labels, ())
        self.assertEqual(summary.unused_confounder_labels, ("Latent Confounder",))


class GroundingCaveatTests(unittest.TestCase):
    """`grounding_caveat` is a fixed disclaimer meant to stop 'identified' /
    'approved for DAG gate' from being misread as clinical validation -- see
    its own docstring for why. These tests only check it stays present,
    reports the coverage numbers it claims to, and never omits the
    human-review reminder."""

    def test_empty_dag_gets_the_no_items_variant(self):
        summary = EvidenceSourceSummary(
            total_items=0, literature_only=0, kg_only=0, corroborated=0, uncited=0
        )
        message = grounding_caveat(summary)
        self.assertIn("No published or expert reference DAG", message)
        self.assertIn("Human covariate review", message)
        self.assertNotIn("%", message)

    def test_caveat_reports_the_coverage_fractions(self):
        summary = EvidenceSourceSummary(
            total_items=4, literature_only=2, kg_only=0, corroborated=1, uncited=1
        )
        message = grounding_caveat(summary)
        self.assertIn("No published or expert reference DAG", message)
        self.assertIn("75%", message)
        self.assertIn("25%", message)
        self.assertIn("Human covariate review", message)


class ReviewDagIncludesMetricsTests(unittest.TestCase):
    def test_review_dag_bundles_evidence_and_adjustment_summaries(self):
        hypothesis = HypothesisAgent().run(protocol([confounder("SOFA score")]))
        review = review_dag(hypothesis)
        self.assertEqual(review.evidence.total_items, 1)
        self.assertEqual(review.evidence.literature_only, 1)
        self.assertEqual(review.adjustment.set_count, 1)
        self.assertEqual(review.adjustment.core_variable_labels, ("SOFA score",))
        self.assertIn("evidence", review.to_dict())
        self.assertIn("adjustment", review.to_dict())
        self.assertEqual(review.caveat, grounding_caveat(review.evidence))
        self.assertIn("caveat", review.to_dict())


if __name__ == "__main__":
    unittest.main()
