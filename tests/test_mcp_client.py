import tempfile
import unittest
from pathlib import Path
from unittest import mock

from causal.config import BigQueryDatasets
from causal.mcp_client import (
    MCPToolResponse,
    OMCPClient,
    _normalize_tool_response,
    _omcp_db_overrides,
    _sign_causal_context,
    _subprocess_environment,
)

BIGQUERY_DATASETS = BigQueryDatasets(project="project", omop="dataset")
DUCKDB_DATASETS = BigQueryDatasets(backend="duckdb", duckdb_path="data/mimiciv_omop.duckdb", duckdb_schema="main")


class OMCPClientTests(unittest.TestCase):
    def test_requires_existing_server_directory(self):
        with tempfile.TemporaryDirectory() as root:
            client = OMCPClient(BIGQUERY_DATASETS, omcp_dir=root)
            self.assertTrue(client.omcp_dir.is_absolute())

    def test_rejects_missing_server_directory(self):
        with self.assertRaises(FileNotFoundError):
            OMCPClient(BIGQUERY_DATASETS, omcp_dir="missing-omcp")

    def test_unwraps_legacy_call_tool_result_json(self):
        response = _normalize_tool_response(
            '{"_meta":{"run_id":"run"},"content":[{"type":"text","text":"n\\r\\n3\\r\\n"}]}',
            {},
        )
        self.assertEqual(response.text, "n\r\n3\r\n")
        self.assertEqual(response.metadata["run_id"], "run")

    def test_subprocess_environment_does_not_leak_unrelated_secrets(self):
        with mock.patch.dict(
            "os.environ",
            {
                "OPENROUTER_API_KEY": "sk-should-not-leak",
                "SOME_OTHER_SERVICE_TOKEN": "should-not-leak",
                "PATH": "/usr/bin",
                "GOOGLE_APPLICATION_CREDENTIALS": "/creds.json",
                "LANGFUSE_SECRET_KEY": "lf-secret",
            },
            clear=True,
        ):
            environment = _subprocess_environment({"DB_TYPE": "bigquery"})
        self.assertNotIn("OPENROUTER_API_KEY", environment)
        self.assertNotIn("SOME_OTHER_SERVICE_TOKEN", environment)
        self.assertEqual(environment["PATH"], "/usr/bin")
        self.assertEqual(environment["GOOGLE_APPLICATION_CREDENTIALS"], "/creds.json")
        self.assertEqual(environment["LANGFUSE_SECRET_KEY"], "lf-secret")
        self.assertEqual(environment["DB_TYPE"], "bigquery")

    def test_subprocess_environment_overrides_win_over_inherited_values(self):
        with mock.patch.dict("os.environ", {"PATH": "/usr/bin"}, clear=True):
            environment = _subprocess_environment({"PATH": "/overridden"})
        self.assertEqual(environment["PATH"], "/overridden")

    def test_db_overrides_for_bigquery_backend(self):
        overrides, extra = _omcp_db_overrides(BIGQUERY_DATASETS)
        self.assertEqual(extra, "bigquery")
        self.assertEqual(overrides["DB_TYPE"], "bigquery")
        self.assertEqual(overrides["DB_PROJECT"], "project")
        self.assertEqual(overrides["DB_DATASET"], "dataset")
        self.assertEqual(overrides["CDM_SCHEMA"], "dataset")
        self.assertEqual(overrides["VOCAB_SCHEMA"], "dataset")
        self.assertEqual(overrides["DB_READ_ONLY"], "true")
        self.assertNotIn("DB_PATH", overrides)

    def test_db_overrides_for_duckdb_backend(self):
        overrides, extra = _omcp_db_overrides(DUCKDB_DATASETS)
        self.assertEqual(extra, "duckdb")
        self.assertEqual(overrides["DB_TYPE"], "duckdb")
        self.assertEqual(overrides["CDM_SCHEMA"], "main")
        self.assertEqual(overrides["VOCAB_SCHEMA"], "main")
        self.assertEqual(overrides["DB_READ_ONLY"], "true")
        self.assertTrue(Path(overrides["DB_PATH"]).is_absolute())
        self.assertNotIn("DB_PROJECT", overrides)
        self.assertNotIn("DB_DATASET", overrides)

    def test_duckdb_client_accepts_relative_omcp_dir_without_requiring_bigquery_fields(self):
        with tempfile.TemporaryDirectory() as root:
            client = OMCPClient(DUCKDB_DATASETS, omcp_dir=root)
            self.assertEqual(client.datasets.backend, "duckdb")

    def test_causal_select_signs_context_and_scopes_secret_to_the_subprocess_env(self):
        """`causal_select` must attach a signature OMCP can verify (see
        `omcp.governance.verify_causal_context_signature`) and pass the
        matching secret only to that one subprocess spawn -- never a fixed
        or globally-set secret, which would let a stale value keep working
        across calls."""
        context = {
            "run_id": "run-1",
            "study_id": "study-1",
            "mode": "exploratory",
            "clinical_mapping_status": "provisional",
            "analysis_plan_status": "pending",
            "aggregate_feasibility_allowed": True,
            "analytic_dataset_build_allowed": False,
            "causal_estimation_allowed": False,
        }
        with tempfile.TemporaryDirectory() as root:
            client = OMCPClient(BIGQUERY_DATASETS, omcp_dir=root)
            with mock.patch.object(
                client, "_call_tool", new=mock.AsyncMock(return_value=MCPToolResponse("", {}))
            ) as call_tool:
                client.causal_select("SELECT 1", "aggregate_feasibility", context)

        (tool_name, arguments), kwargs = call_tool.call_args
        self.assertEqual(tool_name, "Causal_Select_Query")
        sent_context = arguments["causal_context"]
        secret = kwargs["env_overrides"]["OMCP_CONTEXT_SECRET"]
        self.assertEqual(
            sent_context["signature"], _sign_causal_context(context, secret)
        )
        # The signature travels with the context; the original dict passed
        # in by the caller is untouched.
        self.assertNotIn("signature", context)


if __name__ == "__main__":
    unittest.main()
