import unittest

from causal.config import BigQueryDatasets
from causal.queries import albumin_sepsis_feasibility_sql


class AlbuminSepsisQueryTests(unittest.TestCase):
    def test_mcp_query_contains_no_unbound_parameters(self):
        sql = albumin_sepsis_feasibility_sql(BigQueryDatasets())
        self.assertNotIn("@sepsis_seed", sql)
        self.assertNotIn("@grace", sql)
        self.assertIn("omop_cdm", sql)
        self.assertIn("COUNT(*) AS eligible_persons", sql)


if __name__ == "__main__":
    unittest.main()
