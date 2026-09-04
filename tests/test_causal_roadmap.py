import unittest

from causal.roadmap import (
    CausalDAG,
    CausalEdge,
    CausalEstimand,
    CausalNode,
    CausalQuestion,
    CausalStudySpecification,
    StatisticalEstimand,
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

    def test_isolated_prognostic_nodes_are_pruned_from_the_search(self):
        # `troponin` etc. are `prognostic` (outcome_cause_only): a single
        # fixed edge to `mortality` and nothing else, exactly like a
        # literature-grounded covariate the DAG-drafting LLM proposes beyond
        # the reviewer's first-pass list. None of them can sit on a backdoor
        # path (no edge into `albumin` at all), so they must not count
        # against `max_candidates` or be searched over.
        isolated_prognostic = tuple(f"prognostic_{i}" for i in range(20))
        dag = CausalDAG(
            nodes=(
                node("severity", "confounder"),
                node("albumin", "exposure", "treatment"),
                node("mortality", "outcome", "outcome"),
                *(node(label, "prognostic") for label in isolated_prognostic),
            ),
            edges=(
                edge("severity", "albumin"),
                edge("severity", "mortality"),
                edge("albumin", "mortality"),
                *(edge(label, "mortality") for label in isolated_prognostic),
            ),
            exposure_id="albumin",
            outcome_id="mortality",
        )

        # 21 adjustment-role nodes (1 confounder + 20 prognostic) would trip
        # a max_candidates=16 guard if the isolated ones were not pruned.
        self.assertEqual(dag.minimum_adjustment_sets(max_candidates=16), (("severity",),))

    def test_prognostic_node_linked_into_the_confounder_network_is_not_pruned(self):
        # Same `prognostic` role as above, but this one carries an extra
        # edge into `severity` (the `covariate_relations` case) -- so it is
        # no longer degree-one and must stay in the search rather than being
        # silently assumed irrelevant.
        dag = CausalDAG(
            nodes=(
                node("severity", "confounder"),
                node("albumin", "exposure", "treatment"),
                node("mortality", "outcome", "outcome"),
                node("linked_prognostic", "prognostic"),
            ),
            edges=(
                edge("severity", "albumin"),
                edge("severity", "mortality"),
                edge("albumin", "mortality"),
                edge("linked_prognostic", "mortality"),
                edge("linked_prognostic", "severity"),
            ),
            exposure_id="albumin",
            outcome_id="mortality",
        )

        pruned = dag._isolated_non_confounder_candidates(("severity", "linked_prognostic"))
        self.assertEqual(pruned, set())

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


def _statistical_estimand(**overrides):
    fields = dict(
        schema_version="1.0",
        study_id="albumin_sepsis_28d",
        identifying_functional="g_formula_standardization",
        outcome_model="logistic_regression_unpenalized",
        adjustment_set=("severity",),
        effect_measure="risk_difference",
        missing_data_strategy="complete_case_analysis",
        censoring_strategy="ignore_censoring_administrative_waiver",
        censoring_rationale="Administrative censoring reviewed and judged non-informative.",
    )
    fields.update(overrides)
    return StatisticalEstimand(**fields)


class StatisticalEstimandValidationTests(unittest.TestCase):
    def test_distinct_from_causal_estimand(self):
        # Same study, two different, independently persistable contracts --
        # a StatisticalEstimand is not just CausalEstimand plus a couple of
        # fields; it has its own schema_version/study_id and validation.
        causal_estimand = _estimand()
        statistical_estimand = _statistical_estimand()
        self.assertNotIsInstance(statistical_estimand, type(causal_estimand))
        self.assertEqual(statistical_estimand.effect_measure, causal_estimand.effect_measure)

    def test_rejects_unsupported_identifying_functional(self):
        with self.assertRaisesRegex(ValueError, "identifying functional"):
            _statistical_estimand(identifying_functional="tmle")

    def test_rejects_unsupported_outcome_model(self):
        with self.assertRaisesRegex(ValueError, "outcome model"):
            _statistical_estimand(outcome_model="random_forest")

    def test_rejects_unsupported_missing_data_strategy(self):
        with self.assertRaisesRegex(ValueError, "missing-data strategy"):
            _statistical_estimand(missing_data_strategy="multiple_imputation")

    def test_rejects_unsupported_censoring_strategy(self):
        with self.assertRaisesRegex(ValueError, "censoring strategy"):
            _statistical_estimand(censoring_strategy="ipcw")

    def test_accepts_both_implemented_censoring_strategies(self):
        for strategy in ("exclude_censored_before_analysis", "ignore_censoring_administrative_waiver"):
            self.assertEqual(_statistical_estimand(censoring_strategy=strategy).censoring_strategy, strategy)

    def test_accepts_ipcw_with_plug_in_standardization(self):
        estimand = _statistical_estimand(
            censoring_strategy="ipcw_administrative_censoring",
            estimation_procedure="plug_in_standardization",
        )
        self.assertEqual(estimand.censoring_strategy, "ipcw_administrative_censoring")

    def test_rejects_tmle_with_ipcw_administrative_censoring(self):
        # TMLEAgent only implements uncensored-data TMLE and raises the same
        # ValueError at fit time -- catching the pair here means the
        # `statistical_estimand` gate cannot be approved for a combination
        # that would crash mid-`execution`.
        with self.assertRaisesRegex(ValueError, "ipcw_administrative_censoring"):
            _statistical_estimand(
                censoring_strategy="ipcw_administrative_censoring",
                estimation_procedure="targeted_minimum_loss_based_estimation",
                outcome_model="super_learner_ensemble",
            )

    def test_rejects_blank_censoring_rationale(self):
        with self.assertRaisesRegex(ValueError, "censoring_rationale"):
            _statistical_estimand(censoring_rationale="   ")

    def test_rejects_duplicate_adjustment_set_columns(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _statistical_estimand(adjustment_set=("age", "age"))

    def test_accepts_empty_adjustment_set(self):
        self.assertEqual(_statistical_estimand(adjustment_set=()).adjustment_set, ())

    def test_rejects_unsupported_effect_measure(self):
        with self.assertRaisesRegex(ValueError, "effect measure"):
            _statistical_estimand(effect_measure="hazard_ratio")

    def test_rejects_invalid_study_id(self):
        with self.assertRaisesRegex(ValueError, "study_id"):
            _statistical_estimand(study_id="not a valid id!")

    def test_to_dict_round_trips(self):
        as_dict = _statistical_estimand().to_dict()
        self.assertEqual(as_dict["identifying_functional"], "g_formula_standardization")
        self.assertEqual(as_dict["adjustment_set"], ("severity",))

    def test_population_summary_defaults_to_ate(self):
        self.assertEqual(_statistical_estimand().population_summary, "ATE")

    def test_rejects_unsupported_population_summary(self):
        with self.assertRaisesRegex(ValueError, "population_summary"):
            _statistical_estimand(population_summary="not_a_real_summary")

    def test_accepts_cate_with_causal_forest_dml(self):
        estimand = _statistical_estimand(
            population_summary="CATE", estimation_procedure="causal_forest_dml",
            outcome_model="super_learner_ensemble",
        )
        self.assertEqual(estimand.population_summary, "CATE")

    def test_cate_requires_causal_forest_dml_procedure(self):
        with self.assertRaisesRegex(ValueError, "causal_forest_dml"):
            _statistical_estimand(
                population_summary="CATE", estimation_procedure="plug_in_standardization",
            )

    def test_accepts_schema_version_1_2(self):
        self.assertEqual(_statistical_estimand(schema_version="1.2").schema_version, "1.2")

    def test_rejects_unsupported_estimation_procedure(self):
        with self.assertRaisesRegex(ValueError, "estimation procedure"):
            _statistical_estimand(estimation_procedure="not_a_real_procedure")


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
