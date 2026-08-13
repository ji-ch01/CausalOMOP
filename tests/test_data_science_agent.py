import unittest

from causal.agents import DataScienceAgent


class DataScienceAgentTests(unittest.TestCase):
    def test_builds_provisional_plan_without_enabling_final_execution(self):
        hypothesis = {
            "schema_version": "1.0", "study_id": "study", "time_zero": "ICU admission",
            "grace_period_hours": 24, "follow_up_days": 28, "primary_estimand": "ATE",
            "secondary_estimands": ["CATE by age"], "candidate_confounders": ["age", "SOFA"],
            "unresolved_issues": ["concept_set:albumin:unmapped"],
        }
        protocol = {
            "data_source": "project.dataset",
            "concept_sets": {
                "sepsis": {
                    "role": "population", "domain": "condition",
                    "status": "candidate", "concept_ids": [132797],
                },
                "septic_shock": {
                    "role": "effect_modifier", "domain": "condition",
                    "status": "candidate", "concept_ids": [196236],
                },
                "albumin": {
                    "role": "treatment", "domain": "drug",
                    "status": "unmapped", "concept_ids": [],
                    "source_value_rule": r"(?i)^Albumin(?:, Human)?\b",
                },
                "crystalloids": {
                    "role": "treatment_and_comparator", "domain": "drug",
                    "status": "partial", "concept_ids": [],
                    "source_value_rule": r"(?i)(Lactated Ringers|Sodium Chloride).*(Bag|Bottle|Injection)$",
                    "lookup_term": "sodium chloride",
                },
            },
        }
        result = DataScienceAgent().run(hypothesis, protocol)
        albumin = next(item for item in result.phenotypes if item.name == "albumin")
        self.assertEqual(albumin.mode, "source_value_provisional")
        self.assertFalse(result.exploratory_execution_allowed)
        self.assertFalse(result.final_execution_allowed)
        self.assertIn("death", result.required_tables)
        self.assertEqual(result.metadata["agent_name"], "data_science_agent")
        # The two OMOP-standard scaffolds are still added because this
        # protocol declares no concept_sets entry with role="outcome"/"time_zero".
        self.assertIn("all_cause_death", [item.name for item in result.phenotypes])
        self.assertIn("index_event", [item.name for item in result.phenotypes])
        self.assertIn("albumin exposure within grace period", result.feasibility_checks)
        self.assertIn("crystalloids exposure within grace period", result.feasibility_checks)
        self.assertEqual(result.design, "add_on_target_trial_emulation")

    def test_mcp_candidates_are_review_only_and_domain_filtered(self):
        class Response:
            text = (
                "concept_id,concept_name,concept_code,vocabulary_id,domain_id\n"
                "1,Albumin,x,RxNorm,Drug\n2,Wrong domain,x,SNOMED,Condition\n"
            )

        class Lookup:
            def lookup_drug(self, term):
                return Response()

            def lookup_condition(self, term):
                response = Response()
                response.text = response.text.replace("Albumin", "Sepsis").replace(
                    "RxNorm,Drug", "SNOMED,Condition"
                )
                return response

        hypothesis = {
            "schema_version": "1.0", "study_id": "study", "time_zero": "ICU admission",
            "grace_period_hours": 24, "follow_up_days": 28, "primary_estimand": "ATE",
        }
        protocol = {
            "data_source": "project.dataset",
            "concept_sets": {
                "sepsis": {"role": "population", "domain": "condition", "status": "pending", "concept_ids": []},
                "septic_shock": {
                    "role": "effect_modifier", "domain": "condition", "status": "pending", "concept_ids": [],
                },
                "albumin": {"role": "treatment", "domain": "drug", "status": "pending", "concept_ids": []},
                "crystalloids": {
                    "role": "treatment_and_comparator", "domain": "drug", "status": "pending", "concept_ids": [],
                },
            },
        }

        result = DataScienceAgent().run(hypothesis, protocol, concept_lookup=Lookup())
        albumin = next(item for item in result.phenotypes if item.name == "albumin")

        self.assertEqual([item.concept_id for item in albumin.concept_candidates], [1])
        self.assertEqual(albumin.concept_ids, ())
        self.assertEqual(albumin.status, "pending")
        self.assertFalse(result.final_execution_allowed)

    def test_generalizes_to_an_unrelated_study_without_touching_python_code(self):
        """The whole point of deriving phenotypes from concept_sets: a study
        about a completely different clinical domain, with a mutually
        exclusive active-comparator design and its own outcome, works
        without any hardcoded "sepsis"/"albumin" name in the agent."""
        hypothesis = {
            "schema_version": "1.0", "study_id": "study", "time_zero": "diagnosis date",
            "grace_period_hours": 720, "follow_up_days": 365, "primary_estimand": "ATE",
            "candidate_confounders": ["age"],
        }
        protocol = {
            "data_source": "project.dataset",
            "query_settings": {"exposure_contrast": "mutually_exclusive"},
            "concept_sets": {
                "type_2_diabetes": {
                    "role": "population", "domain": "condition",
                    "status": "approved", "concept_ids": [201826],
                },
                "metformin": {
                    "role": "treatment", "domain": "drug",
                    "status": "approved", "concept_ids": [1503297],
                },
                "sulfonylurea": {
                    "role": "comparator", "domain": "drug",
                    "status": "approved", "concept_ids": [1560171],
                },
                "myocardial_infarction": {
                    "role": "outcome", "domain": "condition",
                    "status": "approved", "concept_ids": [4329847],
                },
            },
        }
        result = DataScienceAgent().run(hypothesis, protocol)
        names = {item.name for item in result.phenotypes}
        self.assertIn("metformin", names)
        self.assertIn("sulfonylurea", names)
        self.assertIn("myocardial_infarction", names)
        # A protocol-declared outcome overrides the all_cause_death default,
        # and role="time_zero" is not declared here so the default still applies.
        self.assertNotIn("all_cause_death", names)
        self.assertIn("index_event", names)
        self.assertEqual(result.design, "active_comparator_target_trial_emulation")
        self.assertIn("metformin exposure within grace period", result.feasibility_checks)
        self.assertIn("sulfonylurea exposure within grace period", result.feasibility_checks)
        self.assertIn("mutually exclusive treatment assignment", result.feasibility_checks)

    def test_concept_set_without_role_is_rejected(self):
        hypothesis = {
            "schema_version": "1.0", "study_id": "study", "time_zero": "index",
            "grace_period_hours": 24, "follow_up_days": 28, "primary_estimand": "ATE",
        }
        protocol = {
            "data_source": "project.dataset",
            "concept_sets": {"sepsis": {"domain": "condition", "status": "pending", "concept_ids": []}},
        }
        with self.assertRaises(ValueError):
            DataScienceAgent().run(hypothesis, protocol)

    def test_concept_set_with_unknown_domain_is_rejected(self):
        hypothesis = {
            "schema_version": "1.0", "study_id": "study", "time_zero": "index",
            "grace_period_hours": 24, "follow_up_days": 28, "primary_estimand": "ATE",
        }
        protocol = {
            "data_source": "project.dataset",
            "concept_sets": {
                "sepsis": {"role": "population", "domain": "not_a_domain", "status": "pending", "concept_ids": []},
            },
        }
        with self.assertRaises(ValueError):
            DataScienceAgent().run(hypothesis, protocol)


class DataScienceAgentVsUnexposedComparatorTests(unittest.TestCase):
    """A comparator resolved as "no_standard_concept" is the correct, final
    answer under a "vs_unexposed" design (e.g. "aspirin vs no aspirin") --
    there is no OMOP concept for "absence of a drug" -- so it must not keep
    the analysis plan permanently unresolved the way the same status would
    under every other exposure_contrast."""

    def _hypothesis(self) -> dict:
        return {
            "schema_version": "1.0", "study_id": "study", "time_zero": "ICU admission",
            "grace_period_hours": 24, "follow_up_days": 30, "primary_estimand": "CATE",
        }

    def _protocol(self, exposure_contrast: str) -> dict:
        return {
            "data_source": "project.dataset",
            "query_settings": {"exposure_contrast": exposure_contrast},
            "concept_sets": {
                "ami": {
                    "role": "population", "domain": "condition",
                    "status": "approved", "concept_ids": [4329847],
                },
                "aspirin": {
                    "role": "treatment", "domain": "drug",
                    "status": "approved", "concept_ids": [1112807],
                },
                "no_aspirin": {
                    "role": "comparator", "domain": "drug",
                    "status": "no_standard_concept", "concept_ids": [],
                },
                "in_hospital_mortality": {
                    "role": "outcome", "domain": "death",
                    "status": "approved", "concept_ids": [],
                },
                # Declared so the auto-scaffolded "index_event" phenotype
                # (always definition_pending, unrelated to this test) does
                # not confound the final_execution_allowed assertion below.
                "icu_admission": {
                    "role": "time_zero", "domain": "procedure",
                    "status": "approved", "concept_ids": [4],
                },
            },
        }

    def test_no_standard_concept_comparator_does_not_block_final_execution(self):
        result = DataScienceAgent().run(self._hypothesis(), self._protocol("vs_unexposed"))
        self.assertNotIn("phenotype:no_aspirin:no_standard_concept", result.unresolved_issues)
        self.assertTrue(result.final_execution_allowed)

    def test_comparator_phenotype_does_not_require_approval_under_vs_unexposed(self):
        """The root-cause fix: `approval_required=False` on the comparator
        `PhenotypePlan` itself, not just a filter on the resulting
        `unresolved_issues` strings -- `ExecutionAgent.analysis_plan_blockers`
        re-derives its own `incomplete_phenotypes` list straight from this
        field (see `causal.agents.execution`), independently of
        `unresolved_issues`, so a downstream-only filter would have left
        `ExecutionAgent` still blocking on `phenotype_not_approved:no_aspirin`.
        """
        result = DataScienceAgent().run(self._hypothesis(), self._protocol("vs_unexposed"))
        comparator = next(item for item in result.phenotypes if item.name == "no_aspirin")
        self.assertFalse(comparator.approval_required)

    def test_other_roles_still_require_approval_under_vs_unexposed(self):
        result = DataScienceAgent().run(self._hypothesis(), self._protocol("vs_unexposed"))
        treatment = next(item for item in result.phenotypes if item.name == "aspirin")
        self.assertTrue(treatment.approval_required)

    def test_same_status_still_blocks_final_execution_under_add_on(self):
        result = DataScienceAgent().run(self._hypothesis(), self._protocol("add_on"))
        self.assertIn("phenotype:no_aspirin:no_standard_concept", result.unresolved_issues)
        self.assertFalse(result.final_execution_allowed)


class DataScienceAgentCovariateTests(unittest.TestCase):
    """Covariates are sourced from `candidate_dag` role classifications (see
    `causal.agents.data_science._covariate` and the module-level role
    partition), not the raw `candidate_confounders` list. These tests cover
    that behavior directly, independent of the phenotype/concept_set
    handling already covered above.
    """

    @staticmethod
    def _hypothesis(**overrides):
        base = {
            "schema_version": "1.0", "study_id": "study", "time_zero": "index",
            "grace_period_hours": 24, "follow_up_days": 28, "primary_estimand": "ATE",
        }
        base.update(overrides)
        return base

    @staticmethod
    def _protocol(**overrides):
        base = {"data_source": "project.dataset", "concept_sets": {}}
        base.update(overrides)
        return base

    def test_pre_treatment_confounder_gets_a_wide_low_risk_window(self):
        hypothesis = self._hypothesis(candidate_dag=[
            {"label": "chronic_kidney_disease", "role": "confounder", "temporal_classification": "pre_treatment"},
        ])
        result = DataScienceAgent().run(hypothesis, self._protocol())
        covariate = next(c for c in result.covariates if c.name == "chronic_kidney_disease")
        self.assertEqual(covariate.causal_role, "confounder")
        self.assertEqual(covariate.window_start_hours, -876000)
        self.assertEqual(covariate.leakage_risk, "low")

    def test_reviewer_window_override_takes_precedence_over_temporal_classification(self):
        hypothesis = self._hypothesis(candidate_dag=[
            {
                "label": "lactate", "role": "confounder", "temporal_classification": "pre_treatment",
                "window_start_hours": -6, "window_end_hours": 0,
            },
        ])
        result = DataScienceAgent().run(hypothesis, self._protocol())
        covariate = next(c for c in result.covariates if c.name == "lactate")
        # Without the override, pre_treatment would get the wide -876000h window.
        self.assertEqual(covariate.window_start_hours, -6)
        self.assertEqual(covariate.window_end_hours, 0)

    def test_at_treatment_confounder_gets_a_narrow_review_required_window(self):
        hypothesis = self._hypothesis(candidate_dag=[
            {"label": "septic_shock_severity", "role": "confounder", "temporal_classification": "at_treatment"},
        ])
        result = DataScienceAgent().run(hypothesis, self._protocol())
        covariate = next(c for c in result.covariates if c.name == "septic_shock_severity")
        self.assertEqual(covariate.window_start_hours, -24)
        self.assertEqual(covariate.leakage_risk, "review_required")

    def test_effect_modifier_becomes_a_covariate_without_an_adjustment_role(self):
        hypothesis = self._hypothesis(candidate_dag=[
            {"label": "baseline_severity", "role": "effect_modifier", "temporal_classification": "pre_treatment"},
        ])
        result = DataScienceAgent().run(hypothesis, self._protocol())
        covariate = next(c for c in result.covariates if c.name == "baseline_severity")
        self.assertEqual(covariate.causal_role, "effect_modifier")
        self.assertEqual(covariate.adjustment_role, "")

    def test_mediator_and_collider_are_excluded_not_planned_as_covariates(self):
        hypothesis = self._hypothesis(candidate_dag=[
            {"label": "hospital_acquired_infection", "role": "mediator", "temporal_classification": "post_treatment"},
            {"label": "icu_readmission", "role": "collider", "temporal_classification": "post_treatment"},
        ])
        result = DataScienceAgent().run(hypothesis, self._protocol())
        self.assertNotIn("hospital_acquired_infection", [c.name for c in result.covariates])
        self.assertNotIn("icu_readmission", [c.name for c in result.covariates])
        excluded_text = " ".join(result.excluded_from_adjustment)
        self.assertIn("hospital_acquired_infection", excluded_text)
        self.assertIn("icu_readmission", excluded_text)

    def test_rejected_covariate_is_excluded_with_reviewer_attribution(self):
        hypothesis = self._hypothesis(rejected_covariates=[
            {"label": "lactate", "reviewer": "Dr. Smith", "comment": "not a plausible confounder"},
        ])
        result = DataScienceAgent().run(hypothesis, self._protocol())
        self.assertNotIn("lactate", [c.name for c in result.covariates])
        excluded_text = " ".join(result.excluded_from_adjustment)
        self.assertIn("lactate", excluded_text)
        self.assertIn("Dr. Smith", excluded_text)

    def test_unclassified_covariate_blocks_final_execution_instead_of_silently_planning(self):
        hypothesis = self._hypothesis(candidate_confounders=["SOFA"])
        result = DataScienceAgent().run(hypothesis, self._protocol())
        self.assertNotIn("SOFA", [c.name for c in result.covariates])
        self.assertIn("covariate:SOFA:unclassified_by_dag", result.unresolved_issues)
        self.assertFalse(result.final_execution_allowed)

    def test_unobserved_confounder_is_flagged_rather_than_definition_pending(self):
        hypothesis = self._hypothesis(candidate_dag=[
            {
                "label": "unmeasured_frailty", "role": "confounder",
                "temporal_classification": "pre_treatment", "observed": False,
            },
        ])
        result = DataScienceAgent().run(hypothesis, self._protocol())
        covariate = next(c for c in result.covariates if c.name == "unmeasured_frailty")
        self.assertEqual(covariate.status, "unobserved")

    def test_adjustment_summary_tags_core_alternative_and_unused_confounders(self):
        hypothesis = self._hypothesis(candidate_dag=[
            {"label": "age", "role": "confounder", "temporal_classification": "pre_treatment"},
            {"label": "weight", "role": "confounder", "temporal_classification": "pre_treatment"},
            {"label": "smoking_status", "role": "confounder", "temporal_classification": "pre_treatment"},
            {"label": "baseline_severity", "role": "effect_modifier", "temporal_classification": "pre_treatment"},
        ])
        adjustment = {
            "set_count": 2, "set_size": 2,
            "core_variable_labels": ["age"],
            "alternative_variable_labels": ["weight"],
            "unused_confounder_labels": ["smoking_status"],
        }
        result = DataScienceAgent().run(hypothesis, self._protocol(), adjustment=adjustment)
        by_name = {c.name: c for c in result.covariates}
        self.assertEqual(by_name["age"].adjustment_role, "core")
        self.assertEqual(by_name["weight"].adjustment_role, "alternative")
        self.assertEqual(by_name["smoking_status"].adjustment_role, "unused")
        # Only confounder-role nodes get an adjustment_role from AdjustmentSetSummary.
        self.assertEqual(by_name["baseline_severity"].adjustment_role, "")

    def test_demographic_stays_ready_regardless_of_temporal_classification(self):
        hypothesis = self._hypothesis(candidate_dag=[
            {"label": "age", "role": "confounder", "temporal_classification": "at_treatment"},
        ])
        result = DataScienceAgent().run(hypothesis, self._protocol())
        covariate = next(c for c in result.covariates if c.name == "age")
        self.assertEqual(covariate.status, "ready")
        self.assertEqual(covariate.leakage_risk, "low")
        self.assertEqual(covariate.window_start_hours, -876000)

    class _FakeLookup:
        """Records which method was called with which term/domain, and
        returns one concept row tagged with the domain that was searched --
        enough to prove `DataScienceAgent._lookup_candidates` dispatched to
        the right `ConceptLookup` method without needing a real OMCP server.
        """

        def __init__(self):
            self.calls: list[tuple[str, str]] = []

        def _response(self, domain_id: str):
            class Response:
                text = (
                    "concept_id,concept_name,concept_code,vocabulary_id,domain_id\n"
                    f"1,Found,x,VOCAB,{domain_id}\n"
                )
            return Response()

        def lookup_drug(self, term):
            self.calls.append(("lookup_drug", term))
            return self._response("Drug")

        def lookup_condition(self, term):
            self.calls.append(("lookup_condition", term))
            return self._response("Condition")

        def lookup_concept(self, term, domain):
            self.calls.append((f"lookup_concept:{domain}", term))
            return self._response(domain.capitalize())

    def test_measurement_domain_covariate_uses_the_generalized_lookup(self):
        lookup = self._FakeLookup()
        hypothesis = self._hypothesis(candidate_dag=[
            {
                "label": "sofa_score", "role": "confounder",
                "temporal_classification": "pre_treatment", "domain": "measurement",
            },
        ])
        result = DataScienceAgent().run(hypothesis, self._protocol(), concept_lookup=lookup)
        covariate = next(c for c in result.covariates if c.name == "sofa_score")
        self.assertEqual([c.concept_id for c in covariate.concept_candidates], [1])
        self.assertIn(("lookup_concept:measurement", "sofa score"), lookup.calls)

    def test_condition_domain_covariate_still_uses_the_existing_lookup_condition(self):
        lookup = self._FakeLookup()
        hypothesis = self._hypothesis(candidate_dag=[
            {
                "label": "chronic_kidney_disease", "role": "confounder",
                "temporal_classification": "pre_treatment", "domain": "condition",
            },
        ])
        result = DataScienceAgent().run(hypothesis, self._protocol(), concept_lookup=lookup)
        covariate = next(c for c in result.covariates if c.name == "chronic_kidney_disease")
        self.assertEqual([c.concept_id for c in covariate.concept_candidates], [1])
        self.assertIn(("lookup_condition", "chronic kidney disease"), lookup.calls)
        self.assertFalse(any(call[0].startswith("lookup_concept") for call in lookup.calls))

    def test_approved_concept_mapping_surfaces_on_the_covariate_plan(self):
        hypothesis = self._hypothesis(candidate_dag=[
            {
                "label": "sofa_score", "role": "confounder",
                "temporal_classification": "pre_treatment", "domain": "measurement",
                "concept_ids": [4301868], "mapping_status": "approved",
            },
        ])
        result = DataScienceAgent().run(hypothesis, self._protocol())
        covariate = next(c for c in result.covariates if c.name == "sofa_score")
        self.assertEqual(covariate.concept_ids, (4301868,))
        self.assertEqual(covariate.mapping_status, "approved")

    def test_domain_surfaces_on_the_covariate_plan(self):
        hypothesis = self._hypothesis(candidate_dag=[
            {
                "label": "sofa_score", "role": "confounder",
                "temporal_classification": "pre_treatment", "domain": "measurement",
            },
        ])
        result = DataScienceAgent().run(hypothesis, self._protocol())
        covariate = next(c for c in result.covariates if c.name == "sofa_score")
        self.assertEqual(covariate.domain, "measurement")

    def test_approved_mapping_alone_is_not_enough_to_reach_ready(self):
        """Knowing the concept_id answers "which rows", not "how to combine
        them" -- an approved mapping with no aggregation review must still
        block extraction (see the wiring in
        `causal.orchestrator.workflow.build_cohort_plan`)."""
        hypothesis = self._hypothesis(candidate_dag=[
            {
                "label": "sofa_score", "role": "confounder",
                "temporal_classification": "pre_treatment", "domain": "measurement",
                "concept_ids": [4301868], "mapping_status": "approved",
            },
        ])
        result = DataScienceAgent().run(hypothesis, self._protocol())
        covariate = next(c for c in result.covariates if c.name == "sofa_score")
        self.assertEqual(covariate.status, "definition_pending")
        self.assertEqual(covariate.aggregation, "definition_pending")

    def test_reviewed_aggregation_without_approved_mapping_is_not_ready(self):
        hypothesis = self._hypothesis(candidate_dag=[
            {
                "label": "sofa_score", "role": "confounder",
                "temporal_classification": "pre_treatment", "domain": "measurement",
                "aggregation": "last_value_before_index",
            },
        ])
        result = DataScienceAgent().run(hypothesis, self._protocol())
        covariate = next(c for c in result.covariates if c.name == "sofa_score")
        self.assertEqual(covariate.status, "definition_pending")

    def test_approved_mapping_and_reviewed_aggregation_reach_ready(self):
        hypothesis = self._hypothesis(candidate_dag=[
            {
                "label": "sofa_score", "role": "confounder",
                "temporal_classification": "pre_treatment", "domain": "measurement",
                "concept_ids": [4301868], "mapping_status": "approved",
                "aggregation": "last_value_before_index",
            },
        ])
        result = DataScienceAgent().run(hypothesis, self._protocol())
        covariate = next(c for c in result.covariates if c.name == "sofa_score")
        self.assertEqual(covariate.status, "ready")

    def test_not_ready_confounder_blocks_final_execution(self):
        """A confounder that is not `status == "ready"` (unmapped, no
        aggregation reviewed, etc.) must show up in `unresolved_issues` and
        block `final_execution_allowed` -- otherwise a study could run in
        final mode without ever adjusting for this confounder, silently
        biasing the estimate instead of surfacing a blocking warning."""
        hypothesis = self._hypothesis(candidate_dag=[
            {
                "label": "sofa_score", "role": "confounder",
                "temporal_classification": "pre_treatment", "domain": "measurement",
            },
        ])
        result = DataScienceAgent().run(hypothesis, self._protocol())
        covariate = next(c for c in result.covariates if c.name == "sofa_score")
        self.assertEqual(covariate.status, "definition_pending")
        self.assertIn("covariate:sofa_score:not_ready", result.unresolved_issues)
        self.assertFalse(result.final_execution_allowed)

    def test_unobserved_confounder_is_also_flagged_as_not_ready(self):
        hypothesis = self._hypothesis(candidate_dag=[
            {
                "label": "unmeasured_frailty", "role": "confounder",
                "temporal_classification": "pre_treatment", "observed": False,
            },
        ])
        result = DataScienceAgent().run(hypothesis, self._protocol())
        self.assertIn("covariate:unmeasured_frailty:not_ready", result.unresolved_issues)

    def test_ready_confounder_is_not_flagged_as_not_ready(self):
        hypothesis = self._hypothesis(candidate_dag=[
            {
                "label": "sofa_score", "role": "confounder",
                "temporal_classification": "pre_treatment", "domain": "measurement",
                "concept_ids": [4301868], "mapping_status": "approved",
                "aggregation": "last_value_before_index",
            },
        ])
        result = DataScienceAgent().run(hypothesis, self._protocol())
        self.assertNotIn("covariate:sofa_score:not_ready", result.unresolved_issues)

    def test_not_ready_non_confounder_covariate_does_not_block_final_execution(self):
        """Only `role == "confounder"` covariates are adjustment-critical in
        the "biases the effect estimate if skipped" sense this check guards
        against; an effect_modifier/outcome_cause_only/treatment_cause_only
        left not-ready is a missed refinement, not a silent bias risk, so it
        intentionally does not gain this extra blocking check."""
        hypothesis = self._hypothesis(candidate_dag=[
            {"label": "baseline_severity", "role": "effect_modifier", "temporal_classification": "pre_treatment"},
        ])
        result = DataScienceAgent().run(hypothesis, self._protocol())
        covariate = next(c for c in result.covariates if c.name == "baseline_severity")
        self.assertEqual(covariate.status, "definition_pending")
        self.assertNotIn("covariate:baseline_severity:not_ready", result.unresolved_issues)

    def test_covariate_without_a_domain_is_left_unmapped(self):
        lookup = self._FakeLookup()
        hypothesis = self._hypothesis(candidate_dag=[
            {"label": "weight", "role": "confounder", "temporal_classification": "pre_treatment"},
        ])
        result = DataScienceAgent().run(hypothesis, self._protocol(), concept_lookup=lookup)
        covariate = next(c for c in result.covariates if c.name == "weight")
        self.assertEqual(covariate.concept_candidates, ())
        self.assertEqual(lookup.calls, [])

    def test_covariate_without_a_domain_is_flagged_as_an_unresolved_issue(self):
        """A domain-less covariate is deliberately never guessed (see
        `test_covariate_without_a_domain_is_left_unmapped`), but that used to
        leave it silently stuck at `definition_pending` with nothing telling
        a reviewer it needs a domain assigned -- it must now show up in
        `unresolved_issues`, the same way an unclassified covariate already
        does."""
        lookup = self._FakeLookup()
        hypothesis = self._hypothesis(candidate_dag=[
            {"label": "weight", "role": "confounder", "temporal_classification": "pre_treatment"},
        ])
        result = DataScienceAgent().run(hypothesis, self._protocol(), concept_lookup=lookup)
        self.assertIn("covariate:weight:missing_domain", result.unresolved_issues)

    def test_covariate_with_a_domain_is_not_flagged_as_missing_domain(self):
        lookup = self._FakeLookup()
        hypothesis = self._hypothesis(candidate_dag=[
            {
                "label": "sofa_score", "role": "confounder",
                "temporal_classification": "pre_treatment", "domain": "measurement",
            },
        ])
        result = DataScienceAgent().run(hypothesis, self._protocol(), concept_lookup=lookup)
        self.assertNotIn("covariate:sofa_score:missing_domain", result.unresolved_issues)


if __name__ == "__main__":
    unittest.main()
