import unittest

from causal.agents.data_science import EXCLUDED_FROM_ADJUSTMENT_ROLES
from causal.config import BigQueryDatasets
from causal.data_builder import AnalyticCohortBuilder
from causal.mapping.registry import (
    MAPPING_STATUS_APPROVED,
    OMOPVariableDefinition,
    OMOPVariableRegistry,
    build_extraction_plan,
    default_missingness_rule,
    extraction_plan_to_concept_set_spec,
    extraction_plan_to_covariate_spec,
    fixed_domain_to_concept_set_spec,
    is_fixed_domain,
    is_sql_executable,
    parse_variable,
    variable_from_candidate_dag_entry,
    variable_from_concept_set,
)
from causal.queries import ConceptSetSpec, feasibility_sql


def _covariate_dict(**overrides):
    base = {
        "variable_id": "sofa",
        "dag_node_label": "SOFA",
        "variable_kind": "covariate",
        "causal_role": "confounder",
        "observed": True,
        "domain": "measurement",
        "root_concept_ids": (3004249,),
        "mapping_status": MAPPING_STATUS_APPROVED,
        "aggregation": "last_value_before_index",
        "window_start_hours": -24,
        "window_end_hours": 0,
        "reviewer": "dr_smith",
        "provenance": "human_authored",
    }
    base.update(overrides)
    return base


def _candidate_dag_entry(**overrides):
    entry = {
        "label": "SOFA",
        "role": "confounder",
        "rationale": "severity at baseline",
        "temporal_classification": "pre_treatment",
        "observed": True,
        "domain": "measurement",
        "concept_ids": [3004249],
        "mapping_status": MAPPING_STATUS_APPROVED,
        "aggregation": "last_value_before_index",
        "window_start_hours": -24,
        "window_end_hours": 0,
    }
    entry.update(overrides)
    return entry


class SerializationTests(unittest.TestCase):
    def test_variable_round_trips_through_parse_variable(self):
        variable = parse_variable(_covariate_dict())
        restored = parse_variable(variable.to_dict())
        self.assertEqual(variable, restored)

    def test_registry_round_trips_to_and_from_dict(self):
        registry = OMOPVariableRegistry(
            schema_version="1.0",
            study_id="demo_study",
            variables=(parse_variable(_covariate_dict()),),
        )
        data = registry.to_dict()
        restored = OMOPVariableRegistry.from_dict(data)
        self.assertEqual(restored.to_dict(), data)
        self.assertEqual(restored.study_id, "demo_study")
        self.assertEqual(len(restored.variables), 1)


class ValidationRejectionTests(unittest.TestCase):
    def test_rejects_approved_mapping_with_no_concept_ids(self):
        with self.assertRaises(ValueError):
            parse_variable(_covariate_dict(root_concept_ids=()))

    def test_rejects_unknown_domain(self):
        with self.assertRaises(ValueError):
            parse_variable(_covariate_dict(domain="not_a_real_domain"))

    def test_rejects_incompatible_table_for_domain(self):
        with self.assertRaises(ValueError):
            parse_variable(_covariate_dict(omop_tables=("drug_exposure",)))

    def test_rejects_incompatible_aggregation_for_domain(self):
        with self.assertRaises(ValueError):
            parse_variable(_covariate_dict(aggregation="presence_in_window"))

    def test_rejects_covariate_role_outside_dag_variable_roles(self):
        with self.assertRaises(ValueError):
            parse_variable(_covariate_dict(causal_role="not_a_real_role"))

    def test_rejects_malformed_window(self):
        with self.assertRaises(ValueError):
            parse_variable(_covariate_dict(window_start_hours=24, window_end_hours=0))

    def test_rejects_one_sided_window(self):
        with self.assertRaises(ValueError):
            parse_variable(_covariate_dict(window_start_hours=-24, window_end_hours=None))

    def test_rejects_missingness_rule_inconsistent_with_aggregation(self):
        with self.assertRaises(ValueError):
            parse_variable(_covariate_dict(missingness_rule="zero_when_absent"))

    def test_rejects_llm_provenance_approval_without_reviewer(self):
        with self.assertRaises(ValueError):
            parse_variable(_covariate_dict(provenance="llm_proposed", reviewer=""))

    def test_accepts_llm_provenance_candidate_with_no_reviewer_when_not_approved(self):
        variable = parse_variable(
            _covariate_dict(mapping_status="unmapped", root_concept_ids=(), provenance="llm_proposed", reviewer="")
        )
        self.assertEqual(variable.mapping_status, "unmapped")


class UnobservedVariableTests(unittest.TestCase):
    def test_rejects_approved_mapping_on_unobserved_variable(self):
        with self.assertRaises(ValueError):
            parse_variable(_covariate_dict(observed=False))

    def test_unobserved_variable_is_not_sql_executable(self):
        variable = parse_variable(
            _covariate_dict(observed=False, mapping_status="unmapped", root_concept_ids=())
        )
        self.assertFalse(is_sql_executable(variable))

    def test_unobserved_variable_cannot_build_extraction_plan(self):
        variable = parse_variable(
            _covariate_dict(observed=False, mapping_status="unmapped", root_concept_ids=())
        )
        with self.assertRaises(ValueError):
            build_extraction_plan(variable)

    def test_unobserved_variable_still_registers_and_is_not_silently_dropped(self):
        """A latent/unobservable confounder stays in the registry as a
        documented identification limitation -- it must not disappear just
        because it has no OMOP mapping (see docs/causal_roadmap.md)."""
        entry = _candidate_dag_entry(
            label="unmeasured_frailty", observed=False, mapping_status="unmapped", concept_ids=[],
            domain="", aggregation="definition_pending", window_start_hours=None, window_end_hours=None,
        )
        variable = variable_from_candidate_dag_entry(entry)
        self.assertFalse(variable.observed)
        self.assertFalse(is_sql_executable(variable))
        self.assertEqual(variable.dag_node_label, "unmeasured_frailty")


class PostTreatmentLeakageTests(unittest.TestCase):
    def test_forbidden_roles_match_data_science_agent_partition(self):
        self.assertEqual(EXCLUDED_FROM_ADJUSTMENT_ROLES, {"mediator", "collider", "selection", "instrument"})

    def test_mediator_rejected_as_adjustment_covariate(self):
        entry = _candidate_dag_entry(
            label="AKI", role="mediator", temporal_classification="post_treatment",
        )
        variable = variable_from_candidate_dag_entry(entry)
        plan = build_extraction_plan(variable)
        with self.assertRaises(ValueError):
            extraction_plan_to_covariate_spec(variable, plan, "cov_aki")

    def test_collider_rejected_as_adjustment_covariate(self):
        entry = _candidate_dag_entry(
            label="icu_readmission", role="collider", temporal_classification="post_treatment",
        )
        variable = variable_from_candidate_dag_entry(entry)
        plan = build_extraction_plan(variable)
        with self.assertRaises(ValueError):
            extraction_plan_to_covariate_spec(variable, plan, "cov_readmit")

    def test_confounder_is_not_rejected(self):
        variable = variable_from_candidate_dag_entry(_candidate_dag_entry())
        plan = build_extraction_plan(variable)
        spec = extraction_plan_to_covariate_spec(variable, plan, "cov_sofa")
        self.assertEqual(spec.column_name, "cov_sofa")


class DescendantExpansionTests(unittest.TestCase):
    def test_include_descendants_defaults_true_and_expands_in_sql(self):
        entry = _candidate_dag_entry(
            label="diabetes", role="confounder", domain="condition", concept_ids=[201820],
            aggregation="presence_in_window",
        )
        variable = variable_from_candidate_dag_entry(entry)
        plan = build_extraction_plan(variable)
        self.assertTrue(plan.include_descendants)
        spec = extraction_plan_to_covariate_spec(variable, plan, "cov_diabetes")
        sql, _ = AnalyticCohortBuilder().build(
            "study", BigQueryDatasets(),
            population=ConceptSetSpec(domain="condition", concept_ids=(132797,)),
            treatment=ConceptSetSpec(domain="drug", concept_ids=(1,)),
            comparator=ConceptSetSpec(domain="drug", concept_ids=(2,)),
            outcome=ConceptSetSpec(domain="death"),
            execution_allowed=False, covariates=(spec,),
        )
        self.assertIn("concept_ancestor", sql)
        self.assertIn("cov_diabetes_concepts AS (", sql)

    def test_include_descendants_false_excludes_ancestor_expansion(self):
        entry = _candidate_dag_entry(
            label="diabetes", role="confounder", domain="condition", concept_ids=[201820],
            aggregation="presence_in_window", include_descendants=False,
        )
        variable = variable_from_candidate_dag_entry(entry)
        plan = build_extraction_plan(variable)
        self.assertFalse(plan.include_descendants)
        spec = extraction_plan_to_covariate_spec(variable, plan, "cov_diabetes")
        sql, _ = AnalyticCohortBuilder().build(
            "study", BigQueryDatasets(),
            population=ConceptSetSpec(domain="condition", concept_ids=(132797,)),
            treatment=ConceptSetSpec(domain="drug", concept_ids=(1,)),
            comparator=ConceptSetSpec(domain="drug", concept_ids=(2,)),
            outcome=ConceptSetSpec(domain="death"),
            execution_allowed=False, covariates=(spec,),
        )
        covariate_cte = sql.split("cov_diabetes_concepts AS (")[1].split("\n    )")[0]
        self.assertNotIn("concept_ancestor", covariate_cte)
        self.assertIn("UNNEST([201820])", covariate_cte)


class DescendantExclusionTests(unittest.TestCase):
    def test_excluded_concept_ids_are_removed_after_expansion(self):
        entry = _candidate_dag_entry(
            label="diabetes", role="confounder", domain="condition", concept_ids=[201820],
            aggregation="presence_in_window", excluded_concept_ids=[999999],
        )
        variable = variable_from_candidate_dag_entry(entry)
        plan = build_extraction_plan(variable)
        self.assertEqual(plan.excluded_concept_ids, (999999,))
        spec = extraction_plan_to_covariate_spec(variable, plan, "cov_diabetes")
        sql, _ = AnalyticCohortBuilder().build(
            "study", BigQueryDatasets(),
            population=ConceptSetSpec(domain="condition", concept_ids=(132797,)),
            treatment=ConceptSetSpec(domain="drug", concept_ids=(1,)),
            comparator=ConceptSetSpec(domain="drug", concept_ids=(2,)),
            outcome=ConceptSetSpec(domain="death"),
            execution_allowed=False, covariates=(spec,),
        )
        self.assertIn("NOT IN (999999)", sql)

    def test_exclusion_wins_even_when_descendants_are_included(self):
        spec = ConceptSetSpec(domain="condition", concept_ids=(201820,), excluded_concept_ids=(999999,))
        self.assertTrue(spec.include_descendants)
        sql = feasibility_sql(
            BigQueryDatasets(),
            population=spec,
            treatment=ConceptSetSpec(domain="drug", concept_ids=(1,)),
            comparator=ConceptSetSpec(domain="drug", concept_ids=(2,)),
            outcome=ConceptSetSpec(domain="death"),
        )
        self.assertIn("concept_ancestor", sql)
        self.assertIn("NOT IN (999999)", sql)


class ConceptSetConversionTests(unittest.TestCase):
    def test_converts_an_approved_concept_set(self):
        definition = {"role": "population", "domain": "condition", "status": "approved", "concept_ids": [132797]}
        variable = variable_from_concept_set("sepsis", definition)
        self.assertEqual(variable.variable_kind, "phenotype")
        self.assertEqual(variable.mapping_status, MAPPING_STATUS_APPROVED)
        self.assertEqual(variable.root_concept_ids, (132797,))
        self.assertEqual(variable.candidate_concept_ids, ())
        self.assertTrue(is_sql_executable(variable))

    def test_converts_a_death_domain_outcome_with_no_concept_ids(self):
        definition = {"role": "outcome", "domain": "death", "status": "approved", "concept_ids": []}
        variable = variable_from_concept_set("mortality", definition)
        self.assertEqual(variable.domain, "death")
        self.assertTrue(is_sql_executable(variable))

    def test_does_not_auto_approve_a_pending_concept_set(self):
        """The registry must never upgrade a proposed status to approved --
        this is the concrete, testable form of 'no LLM/tool ever
        auto-approves a clinical mapping'."""
        definition = {
            "role": "population", "domain": "condition",
            "status": "candidate_pending_human_approval", "concept_ids": [132797],
        }
        variable = variable_from_concept_set("sepsis", definition)
        self.assertNotEqual(variable.mapping_status, MAPPING_STATUS_APPROVED)
        self.assertEqual(variable.candidate_concept_ids, (132797,))
        self.assertEqual(variable.root_concept_ids, ())
        self.assertFalse(is_sql_executable(variable))

    def test_converts_a_measurement_domain_phenotype_without_covariate_aggregation_rules(self):
        """A measurement/observation-domain outcome is still ascertained by
        presence, not by a value aggregate -- it must not be rejected by the
        covariate-only aggregation/domain compatibility rule."""
        definition = {"role": "outcome", "domain": "measurement", "status": "approved", "concept_ids": [3016723]}
        variable = variable_from_concept_set("lab_outcome", definition)
        self.assertTrue(is_sql_executable(variable))

    def test_real_study_protocol_converts_without_error(self):
        import json
        from pathlib import Path

        protocol = json.loads(
            Path("studies/albumin_sepsis_28d/protocol.json").read_text(encoding="utf-8")
        )
        registry = OMOPVariableRegistry.from_protocol(protocol)
        self.assertEqual(len(registry.variables), len(protocol["concept_sets"]))
        for variable in registry.variables:
            self.assertNotEqual(variable.mapping_status, MAPPING_STATUS_APPROVED)


class CandidateDagConversionTests(unittest.TestCase):
    def test_converts_an_approved_covariate(self):
        variable = variable_from_candidate_dag_entry(_candidate_dag_entry())
        self.assertEqual(variable.variable_kind, "covariate")
        self.assertEqual(variable.causal_role, "confounder")
        self.assertTrue(is_sql_executable(variable))

    def test_converts_an_unmapped_covariate_without_raising(self):
        entry = _candidate_dag_entry(
            domain="", concept_ids=[], mapping_status="unmapped",
            aggregation="definition_pending", window_start_hours=None, window_end_hours=None,
        )
        variable = variable_from_candidate_dag_entry(entry)
        self.assertEqual(variable.domain, "")
        self.assertFalse(is_sql_executable(variable))

    def test_carries_adjustment_role_into_review_comment(self):
        variable = variable_from_candidate_dag_entry(_candidate_dag_entry(), adjustment_role="core")
        self.assertIn("adjustment_role=core", variable.review_comment)


class ExtractionPlanSqlStabilityTests(unittest.TestCase):
    """Guards that ExtractionPlan -> ConceptSetSpec/CovariateSpec conversion
    produces the exact same SQL shape the pre-registry hand-built specs did
    -- the registry must not silently change what already-working SQL
    generation does."""

    def test_concept_set_spec_conversion_matches_a_hand_built_spec(self):
        definition = {"role": "population", "domain": "condition", "status": "approved", "concept_ids": [132797]}
        variable = variable_from_concept_set("sepsis", definition)
        plan = build_extraction_plan(variable)
        converted = extraction_plan_to_concept_set_spec(plan)
        hand_built = ConceptSetSpec(domain="condition", concept_ids=(132797,))
        self.assertEqual(converted, hand_built)

    def test_covariate_spec_sql_is_identical_to_a_hand_built_covariate_spec(self):
        from causal.queries import CovariateSpec

        variable = variable_from_candidate_dag_entry(_candidate_dag_entry())
        plan = build_extraction_plan(variable)
        converted = extraction_plan_to_covariate_spec(variable, plan, "cov_sofa")
        hand_built = CovariateSpec(
            column_name="cov_sofa", domain="measurement", concept_ids=(3004249,),
            window_start_hours=-24, window_end_hours=0, aggregation="last_value_before_index",
        )
        datasets = BigQueryDatasets()
        common_kwargs = dict(
            population=ConceptSetSpec(domain="condition", concept_ids=(132797,)),
            treatment=ConceptSetSpec(domain="drug", concept_ids=(1,)),
            comparator=ConceptSetSpec(domain="drug", concept_ids=(2,)),
            outcome=ConceptSetSpec(domain="death"),
            execution_allowed=False,
        )
        sql_from_registry, _ = AnalyticCohortBuilder().build(
            "study", datasets, covariates=(converted,), **common_kwargs
        )
        sql_hand_built, _ = AnalyticCohortBuilder().build(
            "study", datasets, covariates=(hand_built,), **common_kwargs
        )
        self.assertEqual(sql_from_registry, sql_hand_built)


class MissingnessRuleTests(unittest.TestCase):
    def test_presence_aggregation_defaults_to_zero_when_absent(self):
        self.assertEqual(default_missingness_rule("condition", "presence_in_window"), "zero_when_absent")

    def test_value_aggregation_defaults_to_null_when_missing(self):
        self.assertEqual(default_missingness_rule("measurement", "last_value_before_index"), "null_when_missing")

    def test_definition_pending_has_no_missingness_rule_yet(self):
        self.assertEqual(default_missingness_rule("measurement", "definition_pending"), "")


class ResolveRoleTests(unittest.TestCase):
    """`OMOPVariableRegistry.resolve_role` is the role-indexed lookup
    `causal.orchestrator.workflow._load_concept_specs` is migrated onto --
    same contract the removed `_resolve_concept_spec` helper had, but over
    `OMOPVariableDefinition.causal_role` instead of a raw `concept_sets`
    dict."""

    def _registry(self, *variables: OMOPVariableDefinition) -> OMOPVariableRegistry:
        return OMOPVariableRegistry(schema_version="1.0", study_id="test_study", variables=variables)

    def test_resolves_an_exact_role_match(self):
        population = variable_from_concept_set(
            "sepsis", {"role": "population", "domain": "condition", "status": "approved", "concept_ids": [1]},
        )
        registry = self._registry(population)
        self.assertEqual(registry.resolve_role("population").variable_id, population.variable_id)

    def test_falls_back_to_the_second_role_when_the_first_is_absent(self):
        backbone = variable_from_concept_set(
            "crystalloids",
            {"role": "treatment_and_comparator", "domain": "drug", "status": "approved", "concept_ids": [2]},
        )
        registry = self._registry(backbone)
        matched = registry.resolve_role("treatment", "treatment_and_comparator")
        self.assertEqual(matched.variable_id, backbone.variable_id)

    def test_an_exact_role_match_takes_priority_over_a_fallback(self):
        exact = variable_from_concept_set(
            "albumin", {"role": "treatment", "domain": "drug", "status": "approved", "concept_ids": [1]},
        )
        backbone = variable_from_concept_set(
            "crystalloids",
            {"role": "treatment_and_comparator", "domain": "drug", "status": "approved", "concept_ids": [2]},
        )
        registry = self._registry(exact, backbone)
        matched = registry.resolve_role("treatment", "treatment_and_comparator")
        self.assertEqual(matched.variable_id, exact.variable_id)

    def test_returns_none_when_nothing_declares_any_candidate_role(self):
        registry = self._registry()
        self.assertIsNone(registry.resolve_role("population"))
        self.assertIsNone(registry.resolve_role("treatment", "treatment_and_comparator"))

    def test_raises_on_more_than_one_entry_declaring_the_same_role(self):
        first = variable_from_concept_set(
            "sepsis_v1", {"role": "population", "domain": "condition", "status": "approved", "concept_ids": [1]},
        )
        second = variable_from_concept_set(
            "sepsis_v2", {"role": "population", "domain": "condition", "status": "approved", "concept_ids": [2]},
        )
        registry = self._registry(first, second)
        with self.assertRaises(RuntimeError):
            registry.resolve_role("population")

    def test_resolution_does_not_require_an_approved_mapping(self):
        """`resolve_role` is a pure lookup, not a gate check -- it returns a
        pending-approval entry too, leaving the approval decision to the
        caller (`fixed_domain_to_concept_set_spec`/`build_extraction_plan`,
        both of which raise on an unapproved mapping)."""
        pending = variable_from_concept_set(
            "sepsis",
            {"role": "population", "domain": "condition", "status": "candidate_pending_human_approval", "concept_ids": [1]},
        )
        registry = self._registry(pending)
        matched = registry.resolve_role("population")
        self.assertIsNotNone(matched)
        self.assertNotEqual(matched.mapping_status, MAPPING_STATUS_APPROVED)


class FixedDomainConceptSetSpecTests(unittest.TestCase):
    """`fixed_domain_to_concept_set_spec` is `build_extraction_plan`'s
    sibling for the two roles with no concept-based domain (`death`,
    `standard`) -- Gap A from the OMOP-registry-unification plan:
    `build_extraction_plan` rejects these domains outright (it requires a
    concept-based domain and `root_concept_ids`), but the default outcome
    path depends on `domain="death"` being resolvable."""

    def test_is_fixed_domain_identifies_death_and_standard(self):
        self.assertTrue(is_fixed_domain("death"))
        self.assertTrue(is_fixed_domain("standard"))
        self.assertFalse(is_fixed_domain("condition"))
        self.assertFalse(is_fixed_domain(""))

    def test_converts_an_approved_death_domain_variable_with_no_root_concept_ids(self):
        definition = OMOPVariableDefinition(
            variable_id="outcome_death", dag_node_label="outcome", variable_kind="phenotype",
            causal_role="outcome", observed=True, domain="death", mapping_status=MAPPING_STATUS_APPROVED,
        )
        spec = fixed_domain_to_concept_set_spec(definition)
        self.assertEqual(spec, ConceptSetSpec(domain="death", concept_ids=()))

    def test_rejects_an_unapproved_mapping(self):
        definition = OMOPVariableDefinition(
            variable_id="outcome_death", dag_node_label="outcome", variable_kind="phenotype",
            causal_role="outcome", observed=True, domain="death", mapping_status="unmapped",
        )
        with self.assertRaisesRegex(ValueError, "must be approved"):
            fixed_domain_to_concept_set_spec(definition)

    def test_rejects_an_unobserved_variable(self):
        definition = OMOPVariableDefinition(
            variable_id="outcome_death", dag_node_label="outcome", variable_kind="phenotype",
            causal_role="outcome", observed=False, domain="death", mapping_status=MAPPING_STATUS_APPROVED,
        )
        with self.assertRaisesRegex(ValueError, "not observed"):
            fixed_domain_to_concept_set_spec(definition)

    def test_rejects_a_concept_based_domain(self):
        """A real concept-based domain must go through `build_extraction_plan`
        instead -- `fixed_domain_to_concept_set_spec` is only for `death`/
        `standard`, not a shortcut around root_concept_ids for anything else."""
        definition = OMOPVariableDefinition(
            variable_id="sepsis", dag_node_label="sepsis", variable_kind="phenotype",
            causal_role="population", observed=True, domain="condition",
            mapping_status=MAPPING_STATUS_APPROVED, root_concept_ids=(132797,),
        )
        with self.assertRaisesRegex(ValueError, "fixed_domain_to_concept_set_spec requires"):
            fixed_domain_to_concept_set_spec(definition)


class ComparatorDomainLessCarveOutTests(unittest.TestCase):
    """Gap B: `variable_from_concept_set` must tolerate `domain == ""` for a
    `role="comparator"` entry whose `status` is `"no_standard_concept"` or
    unset -- the `exposure_contrast="vs_unexposed"` design's correct, final
    state (there is no OMOP concept for "absence of a drug"), not every
    other role/status combination."""

    def test_accepts_a_domain_less_no_standard_concept_comparator(self):
        definition = {"role": "comparator", "status": "no_standard_concept", "domain": "", "concept_ids": []}
        variable = variable_from_concept_set("no_albumin", definition)
        self.assertEqual(variable.domain, "")
        self.assertEqual(variable.causal_role, "comparator")

    def test_accepts_a_domain_less_comparator_with_unset_status(self):
        definition = {"role": "comparator", "domain": "", "concept_ids": []}
        variable = variable_from_concept_set("no_albumin", definition)
        self.assertEqual(variable.domain, "")

    def test_still_rejects_a_domain_less_population_entry(self):
        definition = {"role": "population", "status": "no_standard_concept", "domain": "", "concept_ids": []}
        with self.assertRaises(ValueError):
            variable_from_concept_set("sepsis", definition)

    def test_still_rejects_a_domain_less_approved_comparator(self):
        """The carve-out is only for the two genuinely-domain-less statuses
        -- an entry claiming approved must still have a real domain."""
        definition = {"role": "comparator", "status": "approved", "domain": "", "concept_ids": []}
        with self.assertRaises(ValueError):
            variable_from_concept_set("no_albumin", definition)


if __name__ == "__main__":
    unittest.main()
