import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pyarrow as pa

from scripts.bigquery_etl.config import BigQueryDatasets
from scripts.bigquery_etl.export_duckdb import export_dataset_to_duckdb, list_dataset_tables


class FakeField:
    def __init__(self, name: str, field_type: str, mode: str = "NULLABLE"):
        self.name = name
        self.field_type = field_type
        self.mode = mode


class FakeRowIterator:
    def __init__(self, batches: list, schema: list):
        self._batches = batches
        self.schema = schema

    def to_arrow_iterable(self, bqstorage_client=None):
        return iter(self._batches)


class FakeRow:
    def __init__(self, n: int):
        self.n = n


class FakeQueryResult:
    def __init__(self, rows: list):
        self._rows = rows

    def result(self):
        return self._rows


class FakeBigQueryClient:
    """Duck-typed stand-in for google.cloud.bigquery.Client."""

    def __init__(self):
        self._tables: dict[str, tuple[list, list]] = {}
        self._source_count_overrides: dict[str, int] = {}

    def add_table(self, name: str, arrow_table: pa.Table, chunk_size: int = 1000) -> None:
        fields = [
            FakeField(f.name, "STRING", "NULLABLE") for f in arrow_table.schema
        ]
        batches = arrow_table.to_batches(max_chunksize=chunk_size)
        self._tables[name] = (batches, fields)

    def add_empty_table(self, name: str, fields: list) -> None:
        self._tables[name] = ([], fields)

    def force_source_count(self, name: str, count: int) -> None:
        self._source_count_overrides[name] = count

    def get_dataset(self, dataset_name: str):
        return dataset_name

    def list_tables(self, dataset):
        return [SimpleNamespace(table_id=name) for name in self._tables]

    def list_rows(self, table_ref: str, page_size: int = 100_000):
        table_name = table_ref.rsplit(".", 1)[-1]
        batches, fields = self._tables[table_name]
        return FakeRowIterator(batches, fields)

    def query(self, sql: str):
        table_name = sql.split("FROM `")[1].split("`")[0].rsplit(".", 1)[-1]
        if table_name in self._source_count_overrides:
            n = self._source_count_overrides[table_name]
        else:
            batches, _ = self._tables[table_name]
            n = sum(batch.num_rows for batch in batches)
        return FakeQueryResult([FakeRow(n)])


def _person_table() -> pa.Table:
    return pa.table({
        "person_id": pa.array([1, 2, 3], type=pa.int64()),
        "gender_concept_id": pa.array([8507, 8532, 8507], type=pa.int64()),
    })


class ExportDuckDBTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.datasets = BigQueryDatasets()

    def _output_path(self, name: str = "export.duckdb") -> str:
        return str(Path(self.tmpdir.name) / name)

    def test_discovers_and_exports_every_table_with_verification(self):
        client = FakeBigQueryClient()
        client.add_table("person", _person_table(), chunk_size=2)
        client.add_empty_table("note", [FakeField("note_id", "INTEGER"), FakeField("note_text", "STRING")])

        output = self._output_path()
        summary = export_dataset_to_duckdb(client, self.datasets, output, bqstorage_client=None)

        self.assertEqual(set(list_dataset_tables(client, self.datasets)), {"person", "note"})
        self.assertTrue(summary.all_verified)
        self.assertEqual(summary.mismatched_tables, ())

        con = duckdb.connect(output, read_only=True)
        try:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM person").fetchone()[0], 3)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM note").fetchone()[0], 0)
            row = con.execute("SELECT person_id, gender_concept_id FROM person ORDER BY person_id").fetchall()
            self.assertEqual(row, [(1, 8507), (2, 8532), (3, 8507)])
        finally:
            con.close()

    def test_tables_argument_restricts_export(self):
        client = FakeBigQueryClient()
        client.add_table("person", _person_table())
        client.add_empty_table("note", [FakeField("note_id", "INTEGER")])

        output = self._output_path()
        summary = export_dataset_to_duckdb(
            client, self.datasets, output, tables=["person"], bqstorage_client=None
        )

        self.assertEqual([r.table for r in summary.results], ["person"])
        con = duckdb.connect(output, read_only=True)
        try:
            tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        finally:
            con.close()
        self.assertEqual(tables, {"person"})

    def test_row_count_mismatch_is_reported_not_raised(self):
        client = FakeBigQueryClient()
        client.add_table("person", _person_table())
        client.force_source_count("person", 999)

        summary = export_dataset_to_duckdb(
            client, self.datasets, self._output_path(), tables=["person"], bqstorage_client=None
        )

        self.assertFalse(summary.all_verified)
        self.assertEqual(summary.mismatched_tables, ("person",))
        self.assertEqual(summary.results[0].source_row_count, 999)
        self.assertEqual(summary.results[0].dest_row_count, 3)

    def test_existing_table_requires_explicit_overwrite(self):
        client = FakeBigQueryClient()
        client.add_table("person", _person_table())
        output = self._output_path()

        export_dataset_to_duckdb(client, self.datasets, output, tables=["person"], bqstorage_client=None)
        with self.assertRaises(FileExistsError):
            export_dataset_to_duckdb(client, self.datasets, output, tables=["person"], bqstorage_client=None)

        summary = export_dataset_to_duckdb(
            client, self.datasets, output, tables=["person"], overwrite=True, bqstorage_client=None
        )
        self.assertTrue(summary.all_verified)

    def test_rejects_unsafe_identifiers(self):
        client = FakeBigQueryClient()
        client.add_table("person", _person_table())
        with self.assertRaises(ValueError):
            export_dataset_to_duckdb(
                client, self.datasets, self._output_path(),
                tables=["person"], schema="bad; schema", bqstorage_client=None,
            )
        with self.assertRaises(ValueError):
            export_dataset_to_duckdb(
                client, self.datasets, self._output_path(),
                tables=["person; DROP TABLE person"], bqstorage_client=None,
            )


if __name__ == "__main__":
    unittest.main()
