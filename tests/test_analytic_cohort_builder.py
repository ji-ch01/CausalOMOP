import unittest

from causal.config import BigQueryDatasets
from causal.data_builder import AnalyticCohortBuilder
from causal.queries import ConceptSetSpec, CovariateSpec

SEPSIS = ConceptSetSpec(domain="condition", concept_ids=(132797,))
ALBUMIN = ConceptSetSpec(domain="drug", concept_ids=(1,))
CRYSTALLOID = ConceptSetSpec(domain="drug", concept_ids=(2,))
DEATH = ConceptSetSpec(domain="death")


class AnalyticCohortBuilderTests(unittest.TestCase):
    def test_builds_non_executable_person_level_contract(self):
        sql, manifest = AnalyticCohortBuilder().build(
            "study", BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            execution_allowed=False,
        )
        self.assertIn("CASE WHEN a.has_treatment THEN 1 ELSE 0 END AS A", sql)
        self.assertIn("CASE WHEN a.outcome_event THEN 1 ELSE 0 END AS Y", sql)
        self.assertIn("CAST(NULL AS INT64) AS censored", sql)
        self.assertTrue(manifest.contains_person_level_data)
        self.assertFalse(manifest.delivery_to_llm_allowed)
        self.assertFalse(manifest.execution_allowed)
        self.assertEqual(manifest.status, "draft_not_executable")
        self.assertEqual(manifest.exposure_contrast, "add_on")

    def test_add_on_contrast_requires_the_comparator_backbone(self):
        """Default design (e.g. crystalloids +/- albumin): everyone in the
        cohort must have received the comparator; A marks the add-on."""
        sql, manifest = AnalyticCohortBuilder().build(
            "study", BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            exposure_contrast="add_on", execution_allowed=False,
        )
        self.assertIn("WHERE a.has_comparator", sql)
        self.assertEqual(manifest.exposure_contrast, "add_on")

    def test_mutually_exclusive_contrast_excludes_both_and_neither(self):
        """Active-comparator design (drug A vs drug B): only persons with
        exactly one of treatment/comparator are kept."""
        sql, manifest = AnalyticCohortBuilder().build(
            "study", BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            exposure_contrast="mutually_exclusive", execution_allowed=False,
        )
        self.assertIn("WHERE a.has_treatment != a.has_comparator", sql)
        self.assertNotIn("WHERE a.has_comparator\n", sql)
        self.assertEqual(manifest.exposure_contrast, "mutually_exclusive")

    def test_rejects_unknown_exposure_contrast(self):
        with self.assertRaises(ValueError):
            AnalyticCohortBuilder().build(
                "study", BigQueryDatasets(),
                population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
                exposure_contrast="head_to_head", execution_allowed=False,
            )

    def test_profile_is_aggregate_only(self):
        sql = AnalyticCohortBuilder().profile_sql(
            "study", BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
        )
        self.assertIn("COUNT(*) AS cohort_persons", sql)
        self.assertIn("COUNTIF(censored IS NULL) AS unresolved_censoring", sql)
        self.assertNotIn("SELECT *", sql.upper())

    def test_followup_audit_is_aggregate_and_uses_observation_period(self):
        sql = AnalyticCohortBuilder().followup_audit_sql(
            "study", BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
        )

        self.assertIn("observation_period", sql)
        self.assertIn("nominal_followup_complete", sql)
        self.assertIn("DATE_ADD(index_date, INTERVAL 28 DAY)", sql)
        self.assertIn("COUNT(*) AS cohort_persons", sql)
        self.assertNotIn("SELECT *", sql.upper())

    def test_builder_generalizes_to_a_different_question(self):
        """The whole point of the generic builder: a different domain mix
        (measurement population, procedure outcome) works without touching
        any Python code, just different ConceptSetSpecs."""
        population = ConceptSetSpec(domain="measurement", concept_ids=(3004249,))
        treatment = ConceptSetSpec(domain="procedure", concept_ids=(4230167,))
        comparator = ConceptSetSpec(domain="procedure", concept_ids=(4163872,))
        outcome = ConceptSetSpec(domain="condition", concept_ids=(316866,))
        sql, manifest = AnalyticCohortBuilder().build(
            "study", BigQueryDatasets(),
            population=population, treatment=treatment, comparator=comparator, outcome=outcome,
            exposure_contrast="mutually_exclusive", execution_allowed=False,
        )
        self.assertIn("measurement_concept_id", sql)
        self.assertIn("procedure_occurrence", sql)
        self.assertIn("condition_occurrence", sql)
        # This is exactly the design the default "add_on" contrast gets wrong:
        # two mutually exclusive procedures, so both arms must be reachable
        # as A=1, not silently dropped by an add_on-style backbone filter.
        self.assertIn("WHERE a.has_treatment != a.has_comparator", sql)
        self.assertEqual(manifest.exposure_contrast, "mutually_exclusive")

    def test_demographics_only_call_is_unaffected(self):
        """Regression: omitting `covariates` entirely must behave exactly
        like before this feature existed."""
        sql, manifest = AnalyticCohortBuilder().build(
            "study", BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            execution_allowed=False,
        )
        self.assertNotIn("non-demographic baseline covariates are not implemented", manifest.unresolved_issues)
        self.assertEqual(len(manifest.columns), 9)

    def test_presence_covariate_generates_a_windowed_flag_column(self):
        covariate = CovariateSpec(
            column_name="cov_diabetes",
            domain="condition",
            concept_ids=(201820,),
            window_start_hours=-876000,
            window_end_hours=0,
            aggregation="presence_in_window",
        )
        sql, manifest = AnalyticCohortBuilder().build(
            "study", BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            execution_allowed=False, covariates=(covariate,),
        )
        self.assertIn("cov_diabetes_concepts AS (", sql)
        self.assertIn("CASE WHEN cov_diabetes_events.event_count > 0 THEN 1 ELSE 0 END AS cov_diabetes", sql)
        self.assertIn("LEFT JOIN cov_diabetes_events USING (person_id)", sql)
        self.assertIn(
            "DATETIME_ADD(e.index_datetime, INTERVAL -876000 HOUR)", sql,
        )
        self.assertTrue(any(c.name == "cov_diabetes" and c.data_type == "INT64" for c in manifest.columns))

    def test_count_covariate_coalesces_missing_to_zero(self):
        covariate = CovariateSpec(
            column_name="cov_prior_admissions",
            domain="procedure",
            concept_ids=(4230167,),
            window_start_hours=-24,
            window_end_hours=0,
            aggregation="count_in_window",
        )
        sql, _ = AnalyticCohortBuilder().build(
            "study", BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            execution_allowed=False, covariates=(covariate,),
        )
        self.assertIn(
            "COALESCE(cov_prior_admissions_events.event_count, 0) AS cov_prior_admissions", sql,
        )

    def test_last_value_covariate_uses_row_number_not_coalesce(self):
        covariate = CovariateSpec(
            column_name="cov_sofa_score",
            domain="measurement",
            concept_ids=(3004249,),
            window_start_hours=-24,
            window_end_hours=0,
            aggregation="last_value_before_index",
        )
        sql, manifest = AnalyticCohortBuilder().build(
            "study", BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            execution_allowed=False, covariates=(covariate,),
        )
        self.assertIn("value_as_number", sql)
        self.assertIn(
            "QUALIFY ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY event_dt DESC) = 1", sql,
        )
        self.assertIn("cov_sofa_score_agg.agg_value AS cov_sofa_score", sql)
        # Missingness must never be imputed for a value aggregation.
        self.assertNotIn("COALESCE(cov_sofa_score", sql)
        self.assertTrue(any(c.name == "cov_sofa_score" and c.data_type == "FLOAT64" for c in manifest.columns))

    def test_max_min_mean_in_window_use_the_matching_sql_function(self):
        for aggregation, sql_fn in (
            ("max_in_window", "MAX"), ("min_in_window", "MIN"), ("mean_in_window", "AVG"),
        ):
            covariate = CovariateSpec(
                column_name="cov_lactate",
                domain="measurement",
                concept_ids=(3004249,),
                window_start_hours=-24,
                window_end_hours=0,
                aggregation=aggregation,
            )
            sql, _ = AnalyticCohortBuilder().build(
                "study", BigQueryDatasets(),
                population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
                execution_allowed=False, covariates=(covariate,),
            )
            self.assertIn(f"{sql_fn}(event_value) AS agg_value", sql)

    def test_rejects_unknown_aggregation_for_domain(self):
        covariate = CovariateSpec(
            column_name="cov_diabetes",
            domain="condition",
            concept_ids=(201820,),
            window_start_hours=-24,
            window_end_hours=0,
            aggregation="max_in_window",
        )
        with self.assertRaises(ValueError):
            AnalyticCohortBuilder().build(
                "study", BigQueryDatasets(),
                population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
                execution_allowed=False, covariates=(covariate,),
            )

    def test_rejects_column_name_colliding_with_a_fixed_column(self):
        covariate = CovariateSpec(
            column_name="person_id",
            domain="condition",
            concept_ids=(201820,),
            window_start_hours=-24,
            window_end_hours=0,
            aggregation="presence_in_window",
        )
        with self.assertRaises(ValueError):
            AnalyticCohortBuilder().build(
                "study", BigQueryDatasets(),
                population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
                execution_allowed=False, covariates=(covariate,),
            )

    def test_rejects_two_covariates_colliding_on_column_name(self):
        covariate = CovariateSpec(
            column_name="cov_diabetes",
            domain="condition",
            concept_ids=(201820,),
            window_start_hours=-24,
            window_end_hours=0,
            aggregation="presence_in_window",
        )
        with self.assertRaises(ValueError):
            AnalyticCohortBuilder().build(
                "study", BigQueryDatasets(),
                population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
                execution_allowed=False, covariates=(covariate, covariate),
            )

    def test_rejects_unsafe_column_identifier(self):
        covariate = CovariateSpec(
            column_name="cov; DROP TABLE person",
            domain="condition",
            concept_ids=(201820,),
            window_start_hours=-24,
            window_end_hours=0,
            aggregation="presence_in_window",
        )
        with self.assertRaises(ValueError):
            AnalyticCohortBuilder().build(
                "study", BigQueryDatasets(),
                population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
                execution_allowed=False, covariates=(covariate,),
            )

    def test_censoring_stays_null_when_not_approved(self):
        sql, manifest = AnalyticCohortBuilder().build(
            "study", BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            execution_allowed=False,
        )
        self.assertIn("CAST(NULL AS INT64) AS censored", sql)
        self.assertIn("censoring and observation-window completeness are not implemented", manifest.unresolved_issues)

    def test_censoring_computed_when_approved(self):
        sql, manifest = AnalyticCohortBuilder().build(
            "study", BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            execution_allowed=False, censoring_definition_approved=True,
        )
        self.assertNotIn("CAST(NULL AS INT64) AS censored", sql)
        self.assertIn("observation_period_end_date", sql)
        self.assertIn("death_date", sql)
        self.assertNotIn(
            "censoring and observation-window completeness are not implemented", manifest.unresolved_issues
        )
        self.assertIn("effect modifiers are not implemented in this cohort query", manifest.unresolved_issues)


class PositivityAuditSqlTests(unittest.TestCase):
    def test_reports_presence_covariate_overlap_by_arm(self):
        covariate = CovariateSpec(
            column_name="cov_diabetes",
            domain="condition",
            concept_ids=(201820,),
            window_start_hours=-876000,
            window_end_hours=0,
            aggregation="presence_in_window",
        )
        sql = AnalyticCohortBuilder().positivity_audit_sql(
            "study", BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            covariates=(covariate,),
        )
        self.assertIn("COUNTIF(A = 1 AND cov_diabetes > 0) AS cov_diabetes_treated_present", sql)
        self.assertIn("COUNTIF(A = 0 AND cov_diabetes > 0) AS cov_diabetes_comparator_present", sql)
        self.assertIn("COUNTIF(A = 1) AS treated_persons", sql)
        self.assertNotIn("SELECT *", sql.upper())

    def test_reports_value_covariate_range_by_arm(self):
        covariate = CovariateSpec(
            column_name="cov_lactate",
            domain="measurement",
            concept_ids=(3004249,),
            window_start_hours=-24,
            window_end_hours=0,
            aggregation="mean_in_window",
        )
        sql = AnalyticCohortBuilder().positivity_audit_sql(
            "study", BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            covariates=(covariate,),
        )
        self.assertIn("MIN(IF(A = 1, cov_lactate, NULL)) AS cov_lactate_treated_min", sql)
        self.assertIn("MAX(IF(A = 0, cov_lactate, NULL)) AS cov_lactate_comparator_max", sql)
        self.assertNotIn("cov_lactate_treated_present", sql)

    def test_no_covariates_still_reports_arm_counts(self):
        sql = AnalyticCohortBuilder().positivity_audit_sql(
            "study", BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
        )
        self.assertIn("COUNT(*) AS cohort_persons", sql)
        self.assertIn("COUNTIF(A = 1) AS treated_persons", sql)


class CovariateMissingnessSqlTests(unittest.TestCase):
    def test_reports_missing_count_per_covariate(self):
        covariate = CovariateSpec(
            column_name="cov_sofa_score",
            domain="measurement",
            concept_ids=(3004249,),
            window_start_hours=-24,
            window_end_hours=0,
            aggregation="last_value_before_index",
        )
        sql = AnalyticCohortBuilder().covariate_missingness_sql(
            "study", BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            covariates=(covariate,),
        )
        self.assertIn("COUNTIF(cov_sofa_score IS NULL) AS cov_sofa_score_missing", sql)
        self.assertIn("COUNT(*) AS cohort_persons", sql)
        self.assertNotIn("SELECT *", sql.upper())

    def test_no_covariates_still_returns_cohort_count(self):
        sql = AnalyticCohortBuilder().covariate_missingness_sql(
            "study", BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
        )
        self.assertIn("COUNT(*) AS cohort_persons", sql)


COPD = ConceptSetSpec(domain="condition", concept_ids=(255573,))


class AdditionalPopulationPassThroughTests(unittest.TestCase):
    """`additional_population` is a thin pass-through on every method here
    (no logic of its own -- see `causal.queries.feasibility` for the actual
    AND-ing); these just confirm each method actually forwards it rather
    than silently dropping it."""

    def test_build_forwards_additional_population(self):
        sql, _ = AnalyticCohortBuilder().build(
            "study", BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            additional_population=(COPD,), execution_allowed=False,
        )
        self.assertIn("population_additional_0_concepts", sql)

    def test_profile_sql_forwards_additional_population(self):
        sql = AnalyticCohortBuilder().profile_sql(
            "study", BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            additional_population=(COPD,),
        )
        self.assertIn("population_additional_0_concepts", sql)

    def test_followup_audit_sql_forwards_additional_population(self):
        sql = AnalyticCohortBuilder().followup_audit_sql(
            "study", BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            additional_population=(COPD,),
        )
        self.assertIn("population_additional_0_concepts", sql)

    def test_positivity_audit_sql_forwards_additional_population(self):
        sql = AnalyticCohortBuilder().positivity_audit_sql(
            "study", BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            additional_population=(COPD,),
        )
        self.assertIn("population_additional_0_concepts", sql)

    def test_covariate_missingness_sql_forwards_additional_population(self):
        sql = AnalyticCohortBuilder().covariate_missingness_sql(
            "study", BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            additional_population=(COPD,),
        )
        self.assertIn("population_additional_0_concepts", sql)


if __name__ == "__main__":
    unittest.main()
