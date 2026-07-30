import re
import unittest

from causal.config import BigQueryDatasets
from causal.mapping import audit
from causal.mapping.registry import build_extraction_plan, variable_from_candidate_dag_entry

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


class AuditSqlShapeTests(unittest.TestCase):
    def test_concept_row_and_person_counts_expands_descendants(self):
        sql = audit.concept_row_and_person_counts_sql(DATASETS, _condition_plan())
        self.assertIn("concept_ancestor", sql)
        self.assertIn("GROUP BY concept_id", sql)

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


if __name__ == "__main__":
    unittest.main()
