import unittest

from causal.roadmap import (
    CausalDAG,
    CausalEdge,
    CausalEstimand,
    CausalNode,
    CausalQuestion,
    CausalStudySpecification,
    TargetTrialProtocol,
)


def node(node_id, role, timing="baseline", observed=True):
    return CausalNode(node_id, node_id, role, timing, observed)


def edge(source, target):
    return CausalEdge(source, target, f"Assumed causal relation {source}->{target}")


class CausalRoadmapTests(unittest.TestCase):
    def test_complete_specification_precedes_omop_mapping(self):
        dag = CausalDAG(
            nodes=(
                node("severity", "confounder"),
                node("albumin", "exposure", "treatment"),
                node("mortality", "outcome", "outcome"),
            ),
            edges=(edge("severity", "albumin"), edge("severity", "mortality")),
            exposure_id="albumin",
            outcome_id="mortality",
        )
        specification = CausalStudySpecification(
            schema_version="1.0",
            study_id="albumin_sepsis_28d",
            source_question="Does albumin affect 28-day mortality in ICU sepsis?",
            question=CausalQuestion(
                population="Adults with sepsis in ICU",
                treatment="Albumin plus crystalloids",
                comparator="Crystalloids alone",
                outcome="28-day all-cause mortality",
            ),
            target_trial=TargetTrialProtocol(
                eligibility_criteria=("adult", "ICU admission", "sepsis at baseline"),
                treatment_strategy="Initiate albumin plus crystalloids within 24 hours",
                comparator_strategy="Use crystalloids without albumin within 24 hours",
                assignment_procedure="Emulated baseline assignment",
                time_zero="ICU admission",
                grace_period_hours=24,
                follow_up_days=28,
                outcome_definition="All-cause death by day 28",
                censoring_strategy="Censor only at verified loss of mortality follow-up",
                analysis_strategy="Intention-to-treat analogue",
            ),
            estimand=CausalEstimand(
                population_summary="ATE",
                effect_measure="risk_difference",
                treatment_value="albumin_plus_crystalloids",
                comparator_value="crystalloids_alone",
                outcome="death",
                horizon_days=28,
            ),
            dag=dag,
        )

        artifact = specification.to_dict()

        self.assertTrue(artifact["identification"]["identified_by_backdoor_adjustment"])
        self.assertEqual(artifact["identification"]["minimum_adjustment_sets"], (("severity",),))
        self.assertNotIn("concept_sets", artifact)

    def test_finds_minimum_backdoor_adjustment_set(self):
        dag = CausalDAG(
            nodes=(
                node("severity", "confounder"),
                node("albumin", "exposure", "treatment"),
                node("mortality", "outcome", "outcome"),
            ),
            edges=(edge("severity", "albumin"), edge("severity", "mortality"), edge("albumin", "mortality")),
            exposure_id="albumin",
            outcome_id="mortality",
        )

        self.assertFalse(dag.assess_adjustment_set(()).valid)
        self.assertTrue(dag.assess_adjustment_set(("severity",)).valid)
        self.assertEqual(dag.minimum_adjustment_sets(), (("severity",),))

    def test_rejects_adjustment_for_post_treatment_mediator(self):
        dag = CausalDAG(
            nodes=(
                node("albumin", "exposure", "treatment"),
                node("fluid_balance", "mediator", "post_treatment"),
                node("mortality", "outcome", "outcome"),
            ),
            edges=(edge("albumin", "fluid_balance"), edge("fluid_balance", "mortality")),
            exposure_id="albumin",
            outcome_id="mortality",
        )

        assessment = dag.assess_adjustment_set(("fluid_balance",))

        self.assertFalse(assessment.valid)
        self.assertIn("forbidden_role:fluid_balance:mediator", assessment.blockers)
        self.assertIn("descendant_of_exposure:fluid_balance", assessment.blockers)
        self.assertEqual(dag.minimum_adjustment_sets(), ((),))

    def test_reports_unobserved_confounding_as_not_identified(self):
        dag = CausalDAG(
            nodes=(
                node("latent_severity", "confounder", observed=False),
                node("albumin", "exposure", "treatment"),
                node("mortality", "outcome", "outcome"),
            ),
            edges=(edge("latent_severity", "albumin"), edge("latent_severity", "mortality")),
            exposure_id="albumin",
            outcome_id="mortality",
        )

        self.assertEqual(dag.minimum_adjustment_sets(), ())
        self.assertIn("open_backdoor_path", dag.assess_adjustment_set(()).blockers)

    def test_rejects_cycles(self):
        with self.assertRaisesRegex(ValueError, "acyclic"):
            CausalDAG(
                nodes=(
                    node("baseline_a", "confounder"),
                    node("baseline_b", "confounder"),
                    node("albumin", "exposure", "treatment"),
                    node("mortality", "outcome", "outcome"),
                ),
                edges=(
                    edge("baseline_a", "baseline_b"),
                    edge("baseline_b", "baseline_a"),
                    edge("albumin", "mortality"),
                ),
                exposure_id="albumin",
                outcome_id="mortality",
            )

    def test_comparator_defaults_to_empty_and_can_be_set_explicitly(self):
        bare = CausalDAG(
            nodes=(node("albumin", "exposure", "treatment"), node("mortality", "outcome", "outcome")),
            edges=(edge("albumin", "mortality"),),
            exposure_id="albumin",
            outcome_id="mortality",
        )
        self.assertEqual(bare.comparator, "")

        with_comparator = CausalDAG(
            nodes=(node("albumin", "exposure", "treatment"), node("mortality", "outcome", "outcome")),
            edges=(edge("albumin", "mortality"),),
            exposure_id="albumin",
            outcome_id="mortality",
            comparator="Crystalloids alone",
        )
        self.assertEqual(with_comparator.comparator, "Crystalloids alone")

    def test_rejects_temporally_reversed_edge(self):
        with self.assertRaisesRegex(ValueError, "temporal order"):
            CausalDAG(
                nodes=(
                    node("albumin", "exposure", "treatment"),
                    node("mortality", "outcome", "outcome"),
                ),
                edges=(edge("mortality", "albumin"),),
                exposure_id="albumin",
                outcome_id="mortality",
            )


if __name__ == "__main__":
    unittest.main()
