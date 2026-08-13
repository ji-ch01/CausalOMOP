import tempfile
import unittest
from pathlib import Path

import pandas as pd

from causal.data_builder.analytic import AnalyticColumn
from causal.data_builder.materialize import (
    DATASET_FILENAME,
    parse_cohort_csv_to_dataframe,
    write_analytic_dataset,
)

_COLUMNS = (
    AnalyticColumn("person_id", "identifier", "INT64", "index_linkage_only"),
    AnalyticColumn("A", "treatment", "INT64", "within_grace_period"),
    AnalyticColumn("Y", "outcome", "INT64", "within_follow_up"),
    AnalyticColumn("age", "baseline_covariate", "INT64", "at_index"),
    AnalyticColumn("cov_sofa_score", "baseline_covariate", "FLOAT64", "24h_to_0h_relative_to_index"),
)


class ParseCohortCsvTests(unittest.TestCase):
    def test_parses_columns_in_the_declared_order_with_declared_dtypes(self):
        csv_text = "cov_sofa_score,Y,A,age,person_id\n4.5,1,1,60,101\n,0,0,45,102\n"
        frame = parse_cohort_csv_to_dataframe(csv_text, _COLUMNS)
        self.assertEqual(list(frame.columns), ["person_id", "A", "Y", "age", "cov_sofa_score"])
        self.assertEqual(frame["person_id"].tolist(), [101, 102])
        self.assertTrue(pd.isna(frame["cov_sofa_score"].iloc[1]))

    def test_missing_expected_column_raises(self):
        csv_text = "A,Y,age,person_id\n1,1,60,101\n"
        with self.assertRaises(ValueError):
            parse_cohort_csv_to_dataframe(csv_text, _COLUMNS)

    def test_empty_response_raises(self):
        with self.assertRaises(ValueError):
            parse_cohort_csv_to_dataframe("", _COLUMNS)


class WriteAnalyticDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_writes_parquet_and_returns_a_safe_manifest(self):
        frame = pd.DataFrame({"person_id": [101, 102], "A": [1, 0], "Y": [1, 0]})
        manifest = write_analytic_dataset(
            frame, self.run_dir,
            study_id="test_study", run_id="run123", source_query_sha256="abc123",
        )
        self.assertTrue((self.run_dir / DATASET_FILENAME).is_file())
        self.assertEqual(manifest.row_count, 2)
        self.assertEqual(manifest.column_count, 3)
        self.assertEqual(manifest.columns, ("person_id", "A", "Y"))
        self.assertTrue(manifest.parquet_contains_person_level_data)
        self.assertEqual(len(manifest.parquet_sha256), 64)
        roundtrip = pd.read_parquet(self.run_dir / DATASET_FILENAME)
        self.assertEqual(len(roundtrip), 2)

    def test_manifest_never_contains_a_patient_value(self):
        """The manifest -- unlike the parquet file -- is meant to be safe to
        report or hand to an LLM, so it must never carry a cell value. Uses
        6+ digit sentinel values -- a short one (e.g. a plausible "age" like
        37) has a non-negligible chance of appearing as a coincidental
        substring inside one of the manifest's own SHA256 hashes, which
        would be a false positive, not a real leak."""
        frame = pd.DataFrame({"person_id": [999999], "A": [1], "Y": [1], "age": [867530]})
        manifest = write_analytic_dataset(
            frame, self.run_dir,
            study_id="test_study", run_id="run123", source_query_sha256="abc123",
        )
        serialized = str(manifest.to_dict())
        self.assertNotIn("999999", serialized)
        self.assertNotIn("867530", serialized)

    def test_empty_dataframe_is_recorded_as_a_warning_not_an_error(self):
        frame = pd.DataFrame({"person_id": [], "A": [], "Y": []})
        manifest = write_analytic_dataset(
            frame, self.run_dir,
            study_id="test_study", run_id="run123", source_query_sha256="abc123",
        )
        self.assertEqual(manifest.row_count, 0)
        self.assertTrue(any("zero rows" in warning for warning in manifest.warnings))


if __name__ == "__main__":
    unittest.main()
