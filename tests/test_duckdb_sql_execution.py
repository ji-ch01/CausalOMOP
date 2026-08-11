"""Execute every OMCPClient-fed SQL generator against a real, empty DuckDB
database with `backend="duckdb"`.

String-matching tests (`tests/test_feasibility_query.py`,
`tests/test_mapping_audit.py`) check SQL *shape*; they cannot catch a
dialect mismatch DuckDB's parser/binder would actually reject at execution
time (e.g. `UNNEST(list) AS alias` naming the table instead of the column in
DuckDB, unlike BigQuery, or a `GROUP BY` term ambiguous against a joined
column of the same name). This module is the execution-level check: it
builds a minimal fake OMOP schema and runs every generated query through a
real `duckdb.connect()`, so a construct DuckDB cannot parse/bind fails a
test here instead of surfacing only against a real local DuckDB export.
"""

from __future__ import annotations

import unittest

import duckdb

from causal.config import BigQueryDatasets
from causal.data_builder.analytic import AnalyticCohortBuilder
from causal.mapping import audit
from causal.mapping.registry import build_extraction_plan, variable_from_candidate_dag_entry
from causal.queries.feasibility import ConceptSetSpec, CovariateSpec, analytic_cohort_sql, feasibility_sql

DATASETS = BigQueryDatasets(backend="duckdb", duckdb_path=":memory:", duckdb_schema="main")

SEPSIS = ConceptSetSpec(domain="condition", concept_ids=(132797,))
ALBUMIN = ConceptSetSpec(domain="drug", concept_ids=(1,))
CRYSTALLOID = ConceptSetSpec(domain="drug", concept_ids=(2,))
DEATH = ConceptSetSpec(domain="death")

_CONDITION_ENTRY = {
    "label": "diabetes", "role": "confounder", "rationale": "comorbidity",
    "temporal_classification": "pre_treatment", "observed": True,
    "domain": "condition", "concept_ids": [201820], "mapping_status": "approved",
    "aggregation": "presence_in_window", "window_start_hours": -876000, "window_end_hours": 0,
}
_MEASUREMENT_ENTRY = {
    "label": "SOFA", "role": "confounder", "rationale": "severity at baseline",
    "temporal_classification": "pre_treatment", "observed": True,
    "domain": "measurement", "concept_ids": [3004249], "mapping_status": "approved",
    "aggregation": "last_value_before_index", "window_start_hours": -24, "window_end_hours": 0,
}

_SCHEMA_DDL = (
    "CREATE TABLE person (person_id BIGINT, year_of_birth BIGINT, "
    "gender_concept_id BIGINT, race_concept_id BIGINT)",
    "CREATE TABLE condition_occurrence (person_id BIGINT, visit_occurrence_id BIGINT, "
    "condition_concept_id BIGINT, condition_source_value VARCHAR, condition_start_datetime TIMESTAMP)",
    "CREATE TABLE drug_exposure (person_id BIGINT, visit_occurrence_id BIGINT, "
    "drug_concept_id BIGINT, drug_exposure_start_datetime TIMESTAMP)",
    "CREATE TABLE death (person_id BIGINT, death_date DATE)",
    "CREATE TABLE concept_ancestor (ancestor_concept_id BIGINT, descendant_concept_id BIGINT, "
    "min_levels_of_separation INT)",
    "CREATE TABLE concept (concept_id BIGINT, concept_name VARCHAR, domain_id VARCHAR, "
    "standard_concept VARCHAR, invalid_reason VARCHAR)",
    "CREATE TABLE observation_period (person_id BIGINT, observation_period_start_date DATE, "
    "observation_period_end_date DATE)",
    "CREATE TABLE measurement (person_id BIGINT, visit_occurrence_id BIGINT, "
    "measurement_concept_id BIGINT, measurement_datetime TIMESTAMP, value_as_number DOUBLE, "
    "unit_concept_id BIGINT)",
    "CREATE TABLE visit_detail (person_id BIGINT, visit_occurrence_id BIGINT, care_site_id BIGINT, "
    "visit_detail_concept_id BIGINT, visit_detail_start_datetime TIMESTAMP, "
    "visit_detail_end_datetime TIMESTAMP)",
    "CREATE TABLE care_site (care_site_id BIGINT, care_site_name VARCHAR, care_site_source_value VARCHAR)",
)


class DuckDBGeneratedSqlExecutesTests(unittest.TestCase):
    def setUp(self):
        self.con = duckdb.connect()
        for statement in _SCHEMA_DDL:
            self.con.execute(statement)
        self.addCleanup(self.con.close)

    def _plan(self, entry):
        return build_extraction_plan(variable_from_candidate_dag_entry(entry))

    def test_feasibility_sql_executes(self):
        sql = feasibility_sql(DATASETS, population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH)
        self.con.execute(sql).fetchall()

    def test_vs_unexposed_sql_executes_without_a_comparator_concept(self):
        # "vs_unexposed" (e.g. "aspirin vs no aspirin") has no real
        # comparator exposure to query -- there is no OMOP concept for
        # "absence of a drug" -- so an entirely empty/unset ConceptSetSpec
        # must still produce SQL DuckDB can actually run.
        no_comparator = ConceptSetSpec(domain="", concept_ids=())
        feasibility = feasibility_sql(
            DATASETS, population=SEPSIS, treatment=ALBUMIN, comparator=no_comparator, outcome=DEATH,
            exposure_contrast="vs_unexposed",
        )
        self.con.execute(feasibility).fetchall()
        cohort = analytic_cohort_sql(
            DATASETS, population=SEPSIS, treatment=ALBUMIN, comparator=no_comparator, outcome=DEATH,
            exposure_contrast="vs_unexposed",
        )
        self.con.execute(cohort).fetchall()

    def test_analytic_cohort_sql_with_covariate_executes(self):
        covariate = CovariateSpec(
            column_name="diabetes", domain="condition", concept_ids=(201820,),
            window_start_hours=-876000, window_end_hours=0, aggregation="presence_in_window",
        )
        sql = analytic_cohort_sql(
            DATASETS, population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            covariates=(covariate,),
        )
        self.con.execute(sql).fetchall()

    def test_analytic_cohort_builder_profile_and_followup_sql_execute(self):
        builder = AnalyticCohortBuilder()
        profile_sql = builder.profile_sql(
            "study1", DATASETS, population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH
        )
        self.con.execute(profile_sql).fetchall()
        followup_sql = builder.followup_audit_sql(
            "study1", DATASETS, population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH
        )
        self.con.execute(followup_sql).fetchall()

    def test_analytic_cohort_sql_with_approved_censoring_executes(self):
        sql = analytic_cohort_sql(
            DATASETS, population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            censoring_definition_approved=True,
        )
        rows = self.con.execute(sql).fetchall()
        self.assertEqual(rows, [])

    def test_analytic_cohort_builder_positivity_and_missingness_sql_execute(self):
        covariate = CovariateSpec(
            column_name="diabetes", domain="condition", concept_ids=(201820,),
            window_start_hours=-876000, window_end_hours=0, aggregation="presence_in_window",
        )
        value_covariate = CovariateSpec(
            column_name="sofa", domain="measurement", concept_ids=(3004249,),
            window_start_hours=-24, window_end_hours=0, aggregation="last_value_before_index",
        )
        builder = AnalyticCohortBuilder()
        positivity_sql = builder.positivity_audit_sql(
            "study1", DATASETS, population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            covariates=(covariate, value_covariate),
        )
        self.con.execute(positivity_sql).fetchall()
        missingness_sql = builder.covariate_missingness_sql(
            "study1", DATASETS, population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
            covariates=(covariate, value_covariate),
        )
        self.con.execute(missingness_sql).fetchall()

    def test_all_mapping_audit_generators_execute(self):
        condition_plan = self._plan(_CONDITION_ENTRY)
        measurement_plan = self._plan(_MEASUREMENT_ENTRY)
        queries = {
            "concept_row_and_person_counts_sql": audit.concept_row_and_person_counts_sql(DATASETS, condition_plan),
            "approved_concepts_missing_from_data_sql": audit.approved_concepts_missing_from_data_sql(
                DATASETS, condition_plan
            ),
            "concept_zero_rate_sql": audit.concept_zero_rate_sql(DATASETS, "condition"),
            "unmapped_source_values_sql": audit.unmapped_source_values_sql(DATASETS, "condition"),
            "invalid_or_nonstandard_concepts_sql": audit.invalid_or_nonstandard_concepts_sql(
                DATASETS, condition_plan
            ),
            "observed_units_sql": audit.observed_units_sql(DATASETS, measurement_plan),
            "descendant_expansion_coverage_sql": audit.descendant_expansion_coverage_sql(DATASETS, condition_plan),
            "orphan_concepts_sql": audit.orphan_concepts_sql(DATASETS, condition_plan),
            "index_event_breakdown_sql": audit.index_event_breakdown_sql(DATASETS, condition_plan),
            "incidence_rate_sql": audit.incidence_rate_sql(DATASETS, condition_plan),
        }
        for name, sql in queries.items():
            with self.subTest(query=name):
                self.con.execute(sql).fetchall()

    def test_approved_concepts_missing_from_data_returns_scalar_ids_not_structs(self):
        # Regression: DuckDB's `UNNEST(list) AS alias` names the *table*, not
        # the column (unlike BigQuery) -- an unqualified fix would silently
        # return one-field STRUCT rows instead of plain concept_id scalars.
        sql = audit.approved_concepts_missing_from_data_sql(DATASETS, self._plan(_CONDITION_ENTRY))
        rows = self.con.execute(sql).fetchall()
        self.assertEqual(rows, [(201820,)])


if __name__ == "__main__":
    unittest.main()
