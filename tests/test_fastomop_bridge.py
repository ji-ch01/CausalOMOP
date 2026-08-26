import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from causal.config import BigQueryDatasets
from causal.fastomop_bridge import FastOMOPError, ask_fastomop

DUCKDB_DATASETS = BigQueryDatasets(backend="duckdb", duckdb_path="data/mimiciv_omop.duckdb", duckdb_schema="main")


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class AskFastOMOPTests(unittest.TestCase):
    def setUp(self):
        self._fastomop_dir = tempfile.TemporaryDirectory()
        self._omcp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._fastomop_dir.cleanup)
        self.addCleanup(self._omcp_dir.cleanup)

    def test_rejects_missing_fastomop_directory(self):
        with self.assertRaises(FileNotFoundError):
            ask_fastomop(
                "how many patients?",
                datasets=DUCKDB_DATASETS,
                fastomop_dir="missing-fastomop",
                omcp_dir=self._omcp_dir.name,
            )

    def test_rejects_missing_omcp_directory(self):
        with self.assertRaises(FileNotFoundError):
            ask_fastomop(
                "how many patients?",
                datasets=DUCKDB_DATASETS,
                fastomop_dir=self._fastomop_dir.name,
                omcp_dir="missing-omcp",
            )

    def test_parses_result_line_among_debug_noise(self):
        stdout = (
            "some agno debug_mode console output\n"
            'FASTOMOP_JSON:{"answer": "42 patients", "sql_query": "SELECT COUNT(*) FROM person"}\n'
            "trailing noise\n"
        )
        with mock.patch("causal.fastomop_bridge.subprocess.run", return_value=_completed(stdout=stdout)):
            result = ask_fastomop(
                "how many patients?",
                datasets=DUCKDB_DATASETS,
                fastomop_dir=self._fastomop_dir.name,
                omcp_dir=self._omcp_dir.name,
            )
        self.assertEqual(result.answer, "42 patients")
        self.assertEqual(result.sql_query, "SELECT COUNT(*) FROM person")

    def test_command_and_environment_passed_to_subprocess(self):
        stdout = 'FASTOMOP_JSON:{"answer": "ok", "sql_query": null}\n'
        with mock.patch.dict("os.environ", {"PATH": "/usr/bin", "SOME_LLM_KEY": "keep-me"}, clear=True):
            with mock.patch(
                "causal.fastomop_bridge.subprocess.run", return_value=_completed(stdout=stdout)
            ) as run:
                ask_fastomop(
                    "how many patients?",
                    datasets=DUCKDB_DATASETS,
                    fastomop_dir=self._fastomop_dir.name,
                    omcp_dir=self._omcp_dir.name,
                )
        (command,), kwargs = run.call_args
        self.assertEqual(
            command,
            [
                "uv", "run", "--directory", str(Path(self._fastomop_dir.name).resolve()),
                "agno_fastomop", "ask", "how many patients?",
            ],
        )
        environment = kwargs["env"]
        # Unlike the narrow OMCP-subprocess gateway, FastOMOP is itself an LLM
        # agent and needs its full environment (provider keys, OLLAMA_HOST, ...).
        self.assertEqual(environment["SOME_LLM_KEY"], "keep-me")
        self.assertEqual(environment["DB_TYPE"], "duckdb")
        self.assertEqual(environment["DB_READ_ONLY"], "true")
        self.assertEqual(environment["OMCP_SERVER_DIR"], str(Path(self._omcp_dir.name).resolve()))

    def test_nonzero_exit_raises_with_stderr(self):
        with mock.patch(
            "causal.fastomop_bridge.subprocess.run",
            return_value=_completed(stderr="boom", returncode=1),
        ):
            with self.assertRaisesRegex(FastOMOPError, "boom"):
                ask_fastomop(
                    "q",
                    datasets=DUCKDB_DATASETS,
                    fastomop_dir=self._fastomop_dir.name,
                    omcp_dir=self._omcp_dir.name,
                )

    def test_missing_result_line_raises(self):
        with mock.patch(
            "causal.fastomop_bridge.subprocess.run", return_value=_completed(stdout="no marker here\n")
        ):
            with self.assertRaises(FastOMOPError):
                ask_fastomop(
                    "q",
                    datasets=DUCKDB_DATASETS,
                    fastomop_dir=self._fastomop_dir.name,
                    omcp_dir=self._omcp_dir.name,
                )

    def test_malformed_json_raises(self):
        with mock.patch(
            "causal.fastomop_bridge.subprocess.run",
            return_value=_completed(stdout="FASTOMOP_JSON:{not json\n"),
        ):
            with self.assertRaises(FastOMOPError):
                ask_fastomop(
                    "q",
                    datasets=DUCKDB_DATASETS,
                    fastomop_dir=self._fastomop_dir.name,
                    omcp_dir=self._omcp_dir.name,
                )

    def test_error_payload_raises(self):
        stdout = 'FASTOMOP_JSON:{"error": "workflow blew up"}\n'
        with mock.patch("causal.fastomop_bridge.subprocess.run", return_value=_completed(stdout=stdout)):
            with self.assertRaisesRegex(FastOMOPError, "workflow blew up"):
                ask_fastomop(
                    "q",
                    datasets=DUCKDB_DATASETS,
                    fastomop_dir=self._fastomop_dir.name,
                    omcp_dir=self._omcp_dir.name,
                )

    def test_timeout_raises_fastomop_error(self):
        with mock.patch(
            "causal.fastomop_bridge.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="uv", timeout=1.0),
        ):
            with self.assertRaises(FastOMOPError):
                ask_fastomop(
                    "q",
                    datasets=DUCKDB_DATASETS,
                    fastomop_dir=self._fastomop_dir.name,
                    omcp_dir=self._omcp_dir.name,
                    timeout_seconds=1.0,
                )

    def test_missing_uv_raises_fastomop_error(self):
        with mock.patch(
            "causal.fastomop_bridge.subprocess.run", side_effect=FileNotFoundError("uv"),
        ):
            with self.assertRaises(FastOMOPError):
                ask_fastomop(
                    "q",
                    datasets=DUCKDB_DATASETS,
                    fastomop_dir=self._fastomop_dir.name,
                    omcp_dir=self._omcp_dir.name,
                )


if __name__ == "__main__":
    unittest.main()
