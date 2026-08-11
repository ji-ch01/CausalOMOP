import re
import unittest

from causal.config import BigQueryDatasets
from causal.mapping import audit
from causal.mapping.registry import build_extraction_plan, variable_from_candidate_dag_entry
from causal.queries.feasibility import ConceptSetSpec

DATASETS = BigQueryDatasets()

_MEASUREMENT_ENTRY = {
    "label": "SOFA", "role": "confounder", "rationale": "severity at baseline",
    "temporal_classification": "pre_treatment", "observed": True,
    "domain": "measurement", "concept_ids": [3004249], "mapping_status": "approved",
    "aggregation": "last_value_before_index", "window_start_hours": -24, "window_end_hours": 0,
}

_CONDITION_ENTRY = {
    "label": "diabetes", "role": "confounder", "rationale": "comorbidity",
    "temporal_classification": "pre_treatment", "observed": True,
    "domain": "condition", "concept_ids": [201820], "mapping_status": "approved",
    "aggregation": "presence_in_window", "window_start_hours": -876000, "window_end_hours": 0,
}

# A bare, row-level person_id column projected by a final SELECT -- what none
# of these audit queries may ever contain. Matches on "person_id" not
# immediately preceded by "DISTINCT " or wrapped in COUNT(...); a JOIN
# predicate like "t.person_id = e.person_id" is fine (never returned), only
# a projected column would leak identifiers.
_BARE_PERSON_ID_PROJECTION = re.compile(r"SELECT\s+[a-zA-Z0-9_.]*person_id\b(?!\s*[)=])", re.IGNORECASE)


def _measurement_plan():
    return build_extraction_plan(variable_from_candidate_dag_entry(_MEASUREMENT_ENTRY))


def _condition_plan():
    return build_extraction_plan(variable_from_candidate_dag_entry(_CONDITION_ENTRY))


class NoPatientIdentifiersTests(unittest.TestCase):
    """Every audit query must be aggregate-only: no raw person_id column may
    ever appear in a final projection, only inside COUNT(DISTINCT ...) or as
    a join predicate that is never itself selected."""

    def _assert_aggregate_only(self, sql: str) -> None:
        self.assertNotRegex(sql, _BARE_PERSON_ID_PROJECTION)

    def test_concept_row_and_person_counts_is_aggregate_only(self):
        sql = audit.concept_row_and_person_counts_sql(DATASETS, _measurement_plan())
        self._assert_aggregate_only(sql)
        self.assertIn("COUNT(DISTINCT t.person_id) AS person_count", sql)

    def test_approved_concepts_missing_from_data_is_aggregate_only(self):
        sql = audit.approved_concepts_missing_from_data_sql(DATASETS, _measurement_plan())
        self._assert_aggregate_only(sql)
        self.assertNotIn("person_id", sql)

    def test_concept_zero_rate_is_aggregate_only(self):
        sql = audit.concept_zero_rate_sql(DATASETS, "measurement")
        self._assert_aggregate_only(sql)
        self.assertNotIn("person_id", sql)

    def test_unmapped_source_values_is_aggregate_only(self):
        sql = audit.unmapped_source_values_sql(DATASETS, "measurement")
        self._assert_aggregate_only(sql)
        self.assertNotIn("person_id", sql)

    def test_invalid_or_nonstandard_concepts_is_aggregate_only(self):
        sql = audit.invalid_or_nonstandard_concepts_sql(DATASETS, _measurement_plan())
        self._assert_aggregate_only(sql)
        self.assertNotIn("person_id", sql)

    def test_observed_units_is_aggregate_only(self):
        sql = audit.observed_units_sql(DATASETS, _measurement_plan())
        self._assert_aggregate_only(sql)
        self.assertNotIn("person_id", sql)

    def test_descendant_expansion_coverage_is_aggregate_only(self):
        sql = audit.descendant_expansion_coverage_sql(DATASETS, _measurement_plan())
        self._assert_aggregate_only(sql)
        self.assertIn("persons_root_only", sql)
        self.assertIn("persons_with_descendants", sql)

    def test_orphan_concepts_is_aggregate_only(self):
        sql = audit.orphan_concepts_sql(DATASETS, _condition_plan())
        self._assert_aggregate_only(sql)
        self.assertIn("COUNT(DISTINCT t.person_id) AS person_count", sql)

    def test_index_event_breakdown_is_aggregate_only(self):
        sql = audit.index_event_breakdown_sql(DATASETS, _condition_plan())
        self._assert_aggregate_only(sql)

    def test_incidence_rate_is_aggregate_only(self):
        sql = audit.incidence_rate_sql(DATASETS, _condition_plan())
        self._assert_aggregate_only(sql)

    def test_concept_mapping_coverage_is_aggregate_only(self):
        spec = ConceptSetSpec(domain="drug", concept_ids=(1112807,))
        sql = audit.concept_mapping_coverage_sql(DATASETS, spec, "treatment")
        self._assert_aggregate_only(sql)
        self.assertNotIn("t.person_id", sql)


class AuditSqlShapeTests(unittest.TestCase):
    def test_concept_row_and_person_counts_expands_descendants(self):
        sql = audit.concept_row_and_person_counts_sql(DATASETS, _condition_plan())
        self.assertIn("concept_ancestor", sql)
        self.assertIn("GROUP BY t.condition_concept_id", sql)

    def test_concept_zero_rate_rejects_unknown_domain(self):
        with self.assertRaises(ValueError):
            audit.concept_zero_rate_sql(DATASETS, "not_a_domain")

    def test_unmapped_source_values_uses_domain_specific_source_column(self):
        sql = audit.unmapped_source_values_sql(DATASETS, "condition", top_n=5)
        self.assertIn("condition_source_value", sql)
        self.assertIn("LIMIT 5", sql)

    def test_observed_units_rejects_non_value_domain(self):
        with self.assertRaises(ValueError):
            audit.observed_units_sql(DATASETS, _condition_plan())

    def test_observed_units_accepts_measurement_domain(self):
        sql = audit.observed_units_sql(DATASETS, _measurement_plan())
        self.assertIn("unit_concept_id", sql)

    def test_descendant_expansion_coverage_compares_root_only_and_expanded(self):
        sql = audit.descendant_expansion_coverage_sql(DATASETS, _condition_plan())
        self.assertIn("root_only_concepts AS (", sql)
        self.assertIn("expanded_concepts AS (", sql)
        self.assertIn("concept_ancestor", sql)

    def test_concept_mapping_coverage_rejects_unknown_domain(self):
        spec = ConceptSetSpec(domain="not_a_domain", concept_ids=(1,))
        with self.assertRaises(ValueError):
            audit.concept_mapping_coverage_sql(DATASETS, spec, "treatment")

    def test_concept_mapping_coverage_uses_domain_specific_source_column(self):
        spec = ConceptSetSpec(domain="drug", concept_ids=(1112807,))
        sql = audit.concept_mapping_coverage_sql(DATASETS, spec, "treatment")
        self.assertIn("drug_source_value", sql)
        self.assertIn("drug_concept_id = 0", sql)
        self.assertIn("mapped_row_count", sql)
        self.assertIn("unmapped_name_matched_row_count", sql)
        self.assertIn("root_names AS (", sql)


class CohortEntryDiagnosticsTests(unittest.TestCase):
    """`orphan_concepts_sql`/`index_event_breakdown_sql`/`incidence_rate_sql`
    -- the simplified CohortDiagnostics-inspired diagnostics (see
    `causal.mapping.audit` module docstring for the citation and the two
    documented simplifications)."""

    def test_orphan_concepts_excludes_already_included_concepts(self):
        sql = audit.orphan_concepts_sql(DATASETS, _condition_plan())
        self.assertIn("sibling_parents AS (", sql)
        self.assertIn("min_levels_of_separation = 1", sql)
        self.assertIn("LEFT JOIN included_concepts ic ON sc.concept_id = ic.concept_id", sql)
        self.assertIn("WHERE ic.concept_id IS NULL", sql)

    def test_orphan_concepts_filters_by_the_domain_id_of_the_plan(self):
        sql = audit.orphan_concepts_sql(DATASETS, _condition_plan())
        self.assertIn("con.domain_id = 'Condition'", sql)
        sql = audit.orphan_concepts_sql(DATASETS, _measurement_plan())
        self.assertIn("con.domain_id = 'Measurement'", sql)

    def test_orphan_concepts_respects_top_n(self):
        sql = audit.orphan_concepts_sql(DATASETS, _condition_plan(), top_n=5)
        self.assertIn("LIMIT 5", sql)

    def test_index_event_breakdown_groups_by_concept_and_orders_by_person_count(self):
        sql = audit.index_event_breakdown_sql(DATASETS, _condition_plan())
        self.assertIn("GROUP BY fe.concept_id, con.concept_name", sql)
        self.assertIn("ORDER BY person_count DESC", sql)
        self.assertIn("QUALIFY ROW_NUMBER() OVER (", sql)

    def test_index_event_breakdown_breaks_ties_deterministically_by_concept_id(self):
        sql = audit.index_event_breakdown_sql(DATASETS, _condition_plan())
        self.assertIn("ORDER BY t.condition_start_datetime, t.condition_concept_id", sql)

    def test_incidence_rate_divides_by_observation_period_person_years(self):
        sql = audit.incidence_rate_sql(DATASETS, _condition_plan())
        self.assertIn("observation_period", sql)
        self.assertIn("/ 365.25", sql)
        self.assertIn("SAFE_DIVIDE(", sql)
        self.assertIn("incidence_rate_per_1000_person_years", sql)


DUCKDB_DATASETS = BigQueryDatasets(backend="duckdb", duckdb_path="data/mimiciv_omop.duckdb", duckdb_schema="main")


class DuckDBAuditSqlTests(unittest.TestCase):
    """Same SQL generators, `backend="duckdb"` -- must drop every
    BigQuery-only construct (`SAFE_DIVIDE`, `DATE_DIFF`'s BigQuery argument
    order, backtick identifiers) that DuckDB's parser/catalog rejects, while
    leaving the default BigQuery output byte-identical (see
    `test_incidence_rate_divides_by_observation_period_person_years` above)."""

    def test_concept_zero_rate_has_no_safe_divide_or_backticks(self):
        sql = audit.concept_zero_rate_sql(DUCKDB_DATASETS, "condition")
        self.assertNotIn("SAFE_DIVIDE", sql)
        self.assertNotIn("`", sql)
        self.assertIn("CASE WHEN", sql)
        self.assertIn('"main"."condition_occurrence"', sql)

    def test_incidence_rate_uses_duckdb_date_diff_argument_order(self):
        sql = audit.incidence_rate_sql(DUCKDB_DATASETS, _condition_plan())
        self.assertNotIn("SAFE_DIVIDE", sql)
        self.assertNotIn("`", sql)
        self.assertIn(
            "date_diff('day', observation_period_start_date, observation_period_end_date)", sql
        )
        self.assertIn("incidence_rate_per_1000_person_years", sql)

    def test_concept_mapping_coverage_has_no_backticks(self):
        spec = ConceptSetSpec(domain="drug", concept_ids=(1112807,))
        sql = audit.concept_mapping_coverage_sql(DUCKDB_DATASETS, spec, "treatment")
        self.assertNotIn("`", sql)
        self.assertIn('"main"."drug_exposure"', sql)


if __name__ == "__main__":
    unittest.main()
