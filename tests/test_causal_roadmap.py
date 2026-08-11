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


def _estimand(**overrides):
    fields = dict(
        population_summary="ATE",
        effect_measure="risk_difference",
        treatment_value="albumin_plus_crystalloids",
        comparator_value="crystalloids_alone",
        outcome="death",
        horizon_days=28,
    )
    fields.update(overrides)
    return CausalEstimand(**fields)


def _target_trial(**overrides):
    fields = dict(
        eligibility_criteria=("adult", "ICU admission"),
        treatment_strategy="Initiate albumin plus crystalloids within 24 hours",
        comparator_strategy="Use crystalloids without albumin within 24 hours",
        assignment_procedure="Emulated baseline assignment",
        time_zero="ICU admission",
        grace_period_hours=24,
        follow_up_days=28,
        outcome_definition="All-cause death by day 28",
        censoring_strategy="Censor only at verified loss of mortality follow-up",
        analysis_strategy="Intention-to-treat analogue",
    )
    fields.update(overrides)
    return TargetTrialProtocol(**fields)


class CausalQuestionValidationTests(unittest.TestCase):
    def test_rejects_blank_field(self):
        with self.assertRaisesRegex(ValueError, "treatment"):
            CausalQuestion(population="Adults", treatment="  ", comparator="none", outcome="death")

    def test_strips_whitespace(self):
        question = CausalQuestion(population=" Adults ", treatment="A", comparator="B", outcome="death")
        self.assertEqual(question.population, "Adults")


class CausalEstimandValidationTests(unittest.TestCase):
    def test_rejects_unknown_population_summary(self):
        with self.assertRaisesRegex(ValueError, "population_summary"):
            _estimand(population_summary="incidence_rate")

    def test_accepts_ate_att_cate(self):
        for summary in ("ATE", "ATT", "CATE"):
            self.assertEqual(_estimand(population_summary=summary).population_summary, summary)

    def test_rejects_unsupported_effect_measure(self):
        with self.assertRaisesRegex(ValueError, "effect measure"):
            _estimand(effect_measure="hazard_ratio")

    def test_rejects_blank_treatment_value(self):
        with self.assertRaisesRegex(ValueError, "treatment_value"):
            _estimand(treatment_value="")

    def test_rejects_non_positive_horizon(self):
        with self.assertRaisesRegex(ValueError, "horizon_days"):
            _estimand(horizon_days=0)

    def test_counterfactual_contrast_string(self):
        contrast = _estimand().counterfactual_contrast
        self.assertIn("risk_difference", contrast)
        self.assertIn("albumin_plus_crystalloids", contrast)
        self.assertIn("day 28", contrast)


class TargetTrialProtocolValidationTests(unittest.TestCase):
    def test_rejects_empty_eligibility_criteria(self):
        with self.assertRaisesRegex(ValueError, "eligibility criteria"):
            _target_trial(eligibility_criteria=())

    def test_rejects_blank_eligibility_criterion(self):
        with self.assertRaisesRegex(ValueError, "eligibility criteria"):
            _target_trial(eligibility_criteria=("adult", "  "))

    def test_rejects_blank_assignment_procedure(self):
        with self.assertRaisesRegex(ValueError, "assignment_procedure"):
            _target_trial(assignment_procedure="")

    def test_rejects_negative_grace_period(self):
        with self.assertRaisesRegex(ValueError, "grace_period_hours"):
            _target_trial(grace_period_hours=-1)

    def test_rejects_non_positive_follow_up(self):
        with self.assertRaisesRegex(ValueError, "follow_up_days"):
            _target_trial(follow_up_days=0)


class CausalStudySpecificationValidationTests(unittest.TestCase):
    def test_rejects_horizon_mismatch_with_follow_up(self):
        dag = CausalDAG(
            nodes=(node("albumin", "exposure", "treatment"), node("mortality", "outcome", "outcome")),
            edges=(edge("albumin", "mortality"),),
            exposure_id="albumin",
            outcome_id="mortality",
        )
        with self.assertRaisesRegex(ValueError, "horizon"):
            CausalStudySpecification(
                schema_version="1.0",
                study_id="albumin_sepsis_28d",
                source_question="Does albumin affect 28-day mortality in ICU sepsis?",
                question=CausalQuestion(
                    population="Adults with sepsis in ICU",
                    treatment="Albumin plus crystalloids",
                    comparator="Crystalloids alone",
                    outcome="28-day all-cause mortality",
                ),
                target_trial=_target_trial(follow_up_days=28),
                estimand=_estimand(horizon_days=14),
                dag=dag,
            )


if __name__ == "__main__":
    unittest.main()
