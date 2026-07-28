import unittest
from types import SimpleNamespace

from causal.bigquery import BigQueryOMOPGateway, REQUIRED_OMOP_TABLES
from causal.config import BigQueryDatasets
from causal.queries import ConceptSetSpec


class FakeClient:
    def get_dataset(self, dataset):
        return dataset

    def list_tables(self, dataset):
        return [SimpleNamespace(table_id=name) for name in REQUIRED_OMOP_TABLES]


class FoundationTests(unittest.TestCase):
    def test_dataset_roles(self):
        cfg = BigQueryDatasets()
        self.assertEqual(cfg.fq_dataset("omop"), "example-gcp-project.omop_cdm")
        self.assertEqual(cfg.table("metrics", "run_log"), "`example-gcp-project.omop_metrics.run_log`")

    def test_preflight_contract(self):
        result = BigQueryOMOPGateway(BigQueryDatasets(), FakeClient()).preflight()
        self.assertTrue(result.ready)

    def test_concept_search_rejects_bad_limits_before_query(self):
        gateway = BigQueryOMOPGateway(BigQueryDatasets(), FakeClient())
        with self.assertRaises(ValueError):
            gateway.search_concepts("albumin", 101)
        with self.assertRaises(ValueError):
            gateway.search_used_concepts("measurement", "lactate")
        with self.assertRaises(ValueError):
            gateway.search_source_values("procedure", "albumin")
        with self.assertRaises(ValueError):
            gateway.audit_visit_details(0)
        with self.assertRaises(ValueError):
            gateway.feasibility(
                population=ConceptSetSpec(domain="condition", concept_ids=(132797,)),
                treatment=ConceptSetSpec(domain="drug", concept_ids=()),
                comparator=ConceptSetSpec(domain="drug", concept_ids=(2,)),
                outcome=ConceptSetSpec(domain="death"),
            )


if __name__ == "__main__":
    unittest.main()
