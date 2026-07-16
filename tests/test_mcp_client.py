import tempfile
import unittest

from causal.mcp_client import OMCPClient, _normalize_tool_response


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


if __name__ == "__main__":
    unittest.main()
