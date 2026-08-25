import unittest

from causal.config import BigQueryDatasets
from causal.queries import ConceptSetSpec, analytic_cohort_sql, feasibility_sql

SEPSIS = ConceptSetSpec(domain="condition", concept_ids=(132797,))
ALBUMIN = ConceptSetSpec(domain="drug", concept_ids=(1,))
CRYSTALLOID = ConceptSetSpec(domain="drug", concept_ids=(2,))
DEATH = ConceptSetSpec(domain="death")


class FeasibilityQueryTests(unittest.TestCase):
    def test_query_contains_no_unbound_parameters(self):
        sql = feasibility_sql(
            BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
        )
        self.assertNotIn("@", sql)
        self.assertIn("omop_cdm", sql)
        self.assertIn("COUNT(*) AS eligible_persons", sql)

    def test_non_death_outcome_uses_its_own_domain_table(self):
        outcome = ConceptSetSpec(domain="measurement", concept_ids=(3016723,))
        sql = feasibility_sql(
            BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=outcome,
        )
        self.assertIn("measurement_concept_id", sql)
        self.assertNotIn("death_date", sql)

    def test_rejects_unmapped_concept_ids(self):
        unmapped = ConceptSetSpec(domain="drug", concept_ids=())
        with self.assertRaises(ValueError):
            feasibility_sql(
                BigQueryDatasets(),
                population=SEPSIS, treatment=unmapped, comparator=CRYSTALLOID, outcome=DEATH,
            )

    def test_rejects_unknown_domain(self):
        bad = ConceptSetSpec(domain="not_a_domain", concept_ids=(1,))
        with self.assertRaises(ValueError):
            feasibility_sql(
                BigQueryDatasets(),
                population=bad, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            )

    def test_reports_outcome_events_during_grace_period_for_death_outcome(self):
        sql = feasibility_sql(
            BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            grace_period_hours=24,
        )
        self.assertIn("outcome_event_during_grace_period", sql)
        self.assertIn("COUNTIF(outcome_event_during_grace_period) AS outcome_events_during_grace_period", sql)
        self.assertIn("INTERVAL 24 HOUR", sql)

    def test_reports_outcome_events_during_grace_period_for_domain_outcome(self):
        outcome = ConceptSetSpec(domain="measurement", concept_ids=(3016723,))
        sql = feasibility_sql(
            BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=outcome,
        )
        self.assertIn("outcome_event_during_grace_period", sql)

    def test_optional_care_site_filter(self):
        sql = feasibility_sql(
            BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            care_site_names=("Medical Intensive Care Unit (MICU)",),
        )
        self.assertIn("care_site_name IN", sql)
        self.assertIn("Medical Intensive Care Unit (MICU)", sql)

    def test_rejects_unknown_exposure_contrast(self):
        with self.assertRaises(ValueError):
            feasibility_sql(
                BigQueryDatasets(),
                population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
                exposure_contrast="not_a_real_mode",
            )


class VsUnexposedExposureContrastTests(unittest.TestCase):
    """"vs_unexposed" (e.g. "aspirin vs no aspirin"): there is no OMOP
    concept for "absence of a drug", so `comparator` must not be required to
    resolve to one -- unlike every other exposure_contrast, where an empty
    concept set is rejected (`test_rejects_unmapped_concept_ids` above)."""

    NO_COMPARATOR = ConceptSetSpec(domain="", concept_ids=())

    def test_feasibility_sql_does_not_require_a_comparator_concept(self):
        sql = feasibility_sql(
            BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=self.NO_COMPARATOR, outcome=DEATH,
            exposure_contrast="vs_unexposed",
        )
        self.assertNotIn("comparator_concepts", sql)

    def test_analytic_cohort_sql_does_not_require_a_comparator_concept(self):
        sql = analytic_cohort_sql(
            BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=self.NO_COMPARATOR, outcome=DEATH,
            exposure_contrast="vs_unexposed",
        )
        self.assertNotIn("comparator_concepts", sql)

    def test_analytic_cohort_sql_has_no_inclusion_filter(self):
        # Every other mode filters the cohort with a WHERE clause on
        # has_treatment/has_comparator; "vs_unexposed" keeps everyone
        # eligible, split into A=1/A=0 by treatment alone.
        sql = analytic_cohort_sql(
            BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=self.NO_COMPARATOR, outcome=DEATH,
            exposure_contrast="vs_unexposed",
        )
        self.assertNotIn("WHERE a.has_comparator", sql)
        self.assertNotIn("WHERE a.has_treatment != a.has_comparator", sql)

    def test_a_real_comparator_still_works_if_one_happens_to_be_mapped(self):
        # A reviewer picking "vs_unexposed" is not forced to leave comparator
        # unmapped -- the mode just does not require it.
        sql = analytic_cohort_sql(
            BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            exposure_contrast="add_on",
        )
        self.assertIn("comparator_concepts", sql)


DUCKDB_DATASETS = BigQueryDatasets(backend="duckdb", duckdb_path="data/mimiciv_omop.duckdb", duckdb_schema="main")


class DuckDBFeasibilityQueryTests(unittest.TestCase):
    """Same SQL generator, `backend="duckdb"` -- must drop every
    BigQuery-only construct (`DATETIME_ADD`, backtick-qualified
    `project.dataset.table` identifiers) that DuckDB's parser rejects."""

    def test_no_datetime_add_or_backtick_identifiers(self):
        sql = feasibility_sql(
            DUCKDB_DATASETS,
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
        )
        self.assertNotIn("DATETIME_ADD", sql)
        self.assertNotIn("`", sql)
        self.assertIn('"main"."person"', sql)
        self.assertIn("INTERVAL (24) HOUR", sql)

    def test_negative_covariate_window_hours_are_parenthesized(self):
        from causal.queries.feasibility import CovariateSpec, analytic_cohort_sql

        covariate = CovariateSpec(
            column_name="diabetes",
            domain="condition",
            concept_ids=(201820,),
            window_start_hours=-876000,
            window_end_hours=0,
            aggregation="presence_in_window",
        )
        sql = analytic_cohort_sql(
            DUCKDB_DATASETS,
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            covariates=(covariate,),
        )
        self.assertIn("INTERVAL (-876000) HOUR", sql)
        self.assertNotIn("DATETIME_ADD", sql)

    def test_bigquery_output_is_unchanged_by_the_duckdb_branch(self):
        sql = feasibility_sql(
            BigQueryDatasets(),
            population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
        )
        self.assertIn("DATETIME_ADD(e.index_datetime, INTERVAL 24 HOUR)", sql)


COPD = ConceptSetSpec(domain="condition", concept_ids=(255573,))
T2DM = ConceptSetSpec(domain="condition", concept_ids=(201826,))


class AdditionalPopulationConceptSetTests(unittest.TestCase):
    """A population defined by more than one independently-required
    diagnosis (e.g. "type 2 diabetes AND COPD") ANDs each extra concept set
    onto `population` via `additional_population` -- see `_cohort_ctes`."""

    def test_defaults_to_unchanged_behavior(self):
        with_default = analytic_cohort_sql(
            BigQueryDatasets(), population=T2DM, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
        )
        explicit_empty = analytic_cohort_sql(
            BigQueryDatasets(), population=T2DM, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            additional_population=(),
        )
        self.assertEqual(with_default, explicit_empty)
        self.assertNotIn("EXISTS", with_default)

    def test_adds_one_exists_semi_join_per_additional_concept_set(self):
        sql = analytic_cohort_sql(
            BigQueryDatasets(), population=T2DM, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            additional_population=(COPD,),
        )
        self.assertIn("population_additional_0_concepts", sql)
        self.assertEqual(sql.count("EXISTS ("), 1)
        # A semi-join, not a plain JOIN, so a person with multiple qualifying
        # COPD rows is not counted more than once.
        self.assertIn("WHERE ap.person_id = el.person_id", sql)
        self.assertIn("ap.condition_start_datetime <= el.index_datetime", sql)

    def test_two_additional_concept_sets_are_both_anded_in(self):
        second = ConceptSetSpec(domain="measurement", concept_ids=(3004410,))
        sql = analytic_cohort_sql(
            BigQueryDatasets(), population=T2DM, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            additional_population=(COPD, second),
        )
        self.assertIn("population_additional_0_concepts", sql)
        self.assertIn("population_additional_1_concepts", sql)
        self.assertEqual(sql.count("EXISTS ("), 2)
        self.assertIn(" AND EXISTS (", sql)

    def test_combines_with_min_age_years_filter(self):
        sql = analytic_cohort_sql(
            BigQueryDatasets(), population=T2DM, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            additional_population=(COPD,), min_age_years=18,
        )
        self.assertIn("EXTRACT(YEAR FROM DATE(el.index_datetime)) - p.year_of_birth >= 18 AND EXISTS (", sql)

    def test_rejects_unknown_domain(self):
        bad = ConceptSetSpec(domain="not_a_domain", concept_ids=(1,))
        with self.assertRaises(ValueError):
            analytic_cohort_sql(
                BigQueryDatasets(), population=T2DM, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
                additional_population=(bad,),
            )

    def test_feasibility_sql_also_supports_additional_population(self):
        sql = feasibility_sql(
            BigQueryDatasets(), population=T2DM, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            additional_population=(COPD,),
        )
        self.assertIn("population_additional_0_concepts", sql)
        self.assertIn("EXISTS (", sql)


if __name__ == "__main__":
    unittest.main()
