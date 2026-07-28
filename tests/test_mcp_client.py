import tempfile
import unittest
from unittest import mock

from causal.mcp_client import OMCPClient, _normalize_tool_response, _subprocess_environment


class OMCPClientTests(unittest.TestCase):
    def test_requires_existing_server_directory(self):
        with tempfile.TemporaryDirectory() as root:
            client = OMCPClient(project="project", dataset="dataset", omcp_dir=root)
            self.assertTrue(client.omcp_dir.is_absolute())

    def test_rejects_missing_server_directory(self):
        with self.assertRaises(FileNotFoundError):
            OMCPClient(project="project", dataset="dataset", omcp_dir="missing-omcp")

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


if __name__ == "__main__":
    unittest.main()
