import os
import unittest
from pathlib import Path
from unittest import mock

from causal.config import BigQueryDatasets

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_without_causalomop_vars(**overrides) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("CAUSALOMOP_")}
    env.update(overrides)
    return env


class BigQueryBackendTests(unittest.TestCase):
    def test_defaults_to_bigquery_backend(self):
        datasets = BigQueryDatasets()
        self.assertEqual(datasets.backend, "bigquery")

    def test_table_uses_backtick_qualified_bigquery_identifier(self):
        datasets = BigQueryDatasets(project="proj", omop="omop_cdm")
        self.assertEqual(datasets.table("omop", "person"), "`proj.omop_cdm.person`")

    def test_validated_rejects_unknown_backend(self):
        with self.assertRaises(ValueError):
            BigQueryDatasets(backend="postgres").validated()

    def test_validated_rejects_malformed_bigquery_identifier(self):
        with self.assertRaises(ValueError):
            BigQueryDatasets(project="bad project").validated()


class DuckDBBackendTests(unittest.TestCase):
    def _datasets(self, **overrides) -> BigQueryDatasets:
        fields = {"backend": "duckdb", "duckdb_path": "data/mimiciv_omop.duckdb", "duckdb_schema": "main"}
        fields.update(overrides)
        return BigQueryDatasets(**fields)

    def test_table_uses_double_quoted_schema_qualified_identifier_regardless_of_role(self):
        datasets = self._datasets()
        self.assertEqual(datasets.table("omop", "person"), '"main"."person"')
        self.assertEqual(datasets.table("vocabulary", "concept"), '"main"."concept"')

    def test_table_rejects_unknown_role(self):
        with self.assertRaises(ValueError):
            self._datasets().table("not_a_role", "person")

    def test_table_rejects_unsafe_table_name(self):
        with self.assertRaises(ValueError):
            self._datasets().table("omop", "person; DROP TABLE person")

    def test_validated_requires_duckdb_path(self):
        with self.assertRaises(ValueError):
            self._datasets(duckdb_path="").validated()

    def test_validated_rejects_unsafe_duckdb_schema(self):
        with self.assertRaises(ValueError):
            self._datasets(duckdb_schema="bad; schema").validated()

    def test_validated_ignores_bigquery_only_fields(self):
        # project/omop/etc keep their (unused) defaults and must not block
        # validation for the duckdb backend.
        self._datasets(project="").validated()

    def test_resolved_duckdb_path_is_absolute_and_relative_to_repo_root(self):
        datasets = self._datasets(duckdb_path="data/mimiciv_omop.duckdb")
        resolved = datasets.resolved_duckdb_path()
        self.assertTrue(resolved.is_absolute())
        self.assertEqual(resolved, (_PROJECT_ROOT / "data/mimiciv_omop.duckdb").resolve())

    def test_resolved_duckdb_path_leaves_an_already_absolute_path_untouched(self):
        absolute = str((_PROJECT_ROOT / "elsewhere" / "custom.duckdb").resolve())
        datasets = self._datasets(duckdb_path=absolute)
        self.assertEqual(datasets.resolved_duckdb_path(), Path(absolute))

    def test_describe_reports_the_local_file_and_schema(self):
        datasets = self._datasets()
        description = datasets.describe()
        self.assertIn(str(datasets.resolved_duckdb_path()), description)
        self.assertIn("main", description)


class FromEnvPlaceholderGuardTests(unittest.TestCase):
    """`from_env()` is the config path every real entry point (CLI, workflow,
    Streamlit) uses (see `causal.cli`/`causal.orchestrator.workflow`/
    `causal.ui.streamlit_app`, all `BigQueryDatasets.from_env()`). Without
    this guard, a missing/misplaced repo-root `.env` silently produces a
    regex-valid client pointed at the placeholder `example-gcp-project`
    default -- which only fails later, deep inside a BigQuery API call, with
    no hint that the real MIMIC-IV OMOP dataset config never actually
    loaded."""

    def test_refuses_to_run_against_the_placeholder_bigquery_project(self):
        env = _env_without_causalomop_vars()
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("causal.config._load_local_env", lambda *a, **k: None):
            with self.assertRaises(RuntimeError):
                BigQueryDatasets.from_env()

    def test_duckdb_backend_is_exempt_from_the_placeholder_guard(self):
        env = _env_without_causalomop_vars(CAUSALOMOP_DB_BACKEND="duckdb")
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("causal.config._load_local_env", lambda *a, **k: None):
            datasets = BigQueryDatasets.from_env()
        self.assertEqual(datasets.backend, "duckdb")

    def test_a_real_project_env_var_is_accepted(self):
        env = _env_without_causalomop_vars(CAUSALOMOP_BQ_PROJECT="real-project-123")
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("causal.config._load_local_env", lambda *a, **k: None):
            datasets = BigQueryDatasets.from_env()
        self.assertEqual(datasets.project, "real-project-123")


if __name__ == "__main__":
    unittest.main()
