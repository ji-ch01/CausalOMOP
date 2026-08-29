"""Real subprocess/console-boundary integration test for ``causal.fastomop_bridge``.

The rest of the bridge's coverage (``tests/test_fastomop_bridge.py``) mocks
``subprocess.run`` and feeds the parser a Python ``str`` directly. That leaves
exactly the part that broke in production untested: commit e57bea7 fixed a
``UnicodeDecodeError`` that only occurs when real bytes cross a real OS pipe
and get decoded against the wrong codepage -- a mock cannot reproduce it, and
neither can a decode that never happens.

These tests spawn a real child process and drive ``ask_fastomop`` from
``subprocess.run`` through payload parsing for real:

* a real OS pipe + the real UTF-8 decode (``encoding="utf-8"``,
  ``errors="replace"``);
* the real marker scan over real ``agno`` ``debug_mode`` console noise,
  including the leading-newline contract with ``agno_fastomop`` ``ask.main``;
* a real non-zero exit carrying real (non-ASCII) stderr text;
* a real ``subprocess.TimeoutExpired`` from a child that actually sleeps;
* a real ``FileNotFoundError`` from a command that actually is not on PATH;
* the real environment hand-off -- the child reports back which ``DB_*`` /
  ``OMCP_SERVER_DIR`` values it was actually given.

What is *not* real is the program on the far end. Running the actual
``agno_fastomop ask`` (agno + an LLM provider + an OMCP subprocess + DuckDB)
is non-deterministic and slow, and it already has its own test suite plus
live-verification notes. ``subprocess.run`` is patched here only to swap the
``uv run ... agno_fastomop ask <question>`` argv for
``python tests/_fastomop_ask_stub.py <question>``; the patched call then goes
straight into the real ``subprocess.run`` with the bridge's own
``cwd`` / ``env`` / ``encoding`` / ``errors`` / ``timeout`` kwargs untouched.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from causal.config import BigQueryDatasets
from causal.fastomop_bridge import FastOMOPError, ask_fastomop

_STUB = Path(__file__).resolve().parent / "_fastomop_ask_stub.py"
_REAL_SUBPROCESS_RUN = subprocess.run

DUCKDB_DATASETS = BigQueryDatasets(
    backend="duckdb", duckdb_path="data/mimiciv_omop.duckdb", duckdb_schema="main"
)


def _redirect_to_stub(command, /, **kwargs):
    """Stand in for ``subprocess.run`` inside the bridge: replace the
    ``uv run --directory <dir> agno_fastomop ask <question>`` argv with the
    local stub, then hand off to the *real* ``subprocess.run`` with every
    other kwarg (cwd, env, capture_output, encoding, errors, timeout) as the
    bridge passed it."""
    question = command[-1]
    return _REAL_SUBPROCESS_RUN([sys.executable, str(_STUB), question], **kwargs)


class FastOMOPConsoleBoundaryTests(unittest.TestCase):
    def setUp(self):
        self._fastomop_dir = tempfile.TemporaryDirectory()
        self._omcp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._fastomop_dir.cleanup)
        self.addCleanup(self._omcp_dir.cleanup)
        patcher = mock.patch("causal.fastomop_bridge.subprocess.run", _redirect_to_stub)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _ask(self, question="how many patients?", *, mode, env=None, **kwargs):
        overrides = {"FASTOMOP_STUB_MODE": mode, **(env or {})}
        with mock.patch.dict("os.environ", overrides):
            return ask_fastomop(
                question,
                datasets=DUCKDB_DATASETS,
                fastomop_dir=self._fastomop_dir.name,
                omcp_dir=self._omcp_dir.name,
                **kwargs,
            )

    def test_marker_parsed_over_real_agno_debug_noise(self):
        result = self._ask(mode="ok")
        self.assertEqual(result.answer, "1234 patients")
        self.assertEqual(result.sql_query, "SELECT COUNT(*) FROM person")

    def test_non_ascii_pipe_content_is_decoded_as_utf8(self):
        # Regression guard for e57bea7. The child writes UTF-8 debug noise
        # containing byte 0x9D (U+201D), which is undefined in cp1252; the
        # bridge pins encoding="utf-8" so the read succeeds and the marker
        # after that noise still parses. Reverting to text=True fails this on
        # any non-UTF-8-locale host (Windows).
        result = self._ask("does aspirin help?", mode="utf8_noise")
        self.assertEqual(result.answer, "réponse pour: does aspirin help?")
        self.assertIsNone(result.sql_query)

    def test_invalid_utf8_in_noise_does_not_crash_the_read(self):
        # errors="replace" guard -- fails on every platform if dropped, since
        # \xff is invalid UTF-8 under the default strict error handler.
        result = self._ask(mode="bad_utf8_noise")
        self.assertEqual(result.answer, "recovered past the bad bytes")

    def test_invalid_utf8_without_marker_raises_fastomop_error(self):
        with self.assertRaises(FastOMOPError):
            self._ask(mode="bad_utf8_no_marker")

    def test_marker_recovered_when_glued_to_a_debug_line(self):
        result = self._ask(mode="glued_marker")
        self.assertEqual(result.answer, "unglued")

    def test_error_payload_from_real_child_raises(self):
        with self.assertRaisesRegex(FastOMOPError, "no concept_code"):
            self._ask(mode="error_payload")

    def test_real_nonzero_exit_surfaces_child_stderr(self):
        with self.assertRaisesRegex(FastOMOPError, "modèle indisponible"):
            self._ask(mode="stderr_nonzero")

    def test_no_child_output_raises(self):
        with self.assertRaises(FastOMOPError):
            self._ask(mode="no_output")

    def test_real_timeout_terminates_child_and_raises_promptly(self):
        start = time.monotonic()
        with self.assertRaises(FastOMOPError):
            self._ask(mode="slow", timeout_seconds=1.0)
        self.assertLess(time.monotonic() - start, 20)

    def test_environment_overrides_reach_the_child_process(self):
        result = self._ask("q", mode="echo", env={"INHERITED_MARKER": "passed-through"})
        seen = json.loads(result.answer)
        self.assertEqual(seen["question"], "q")
        self.assertEqual(seen["DB_TYPE"], "duckdb")
        self.assertEqual(seen["DB_READ_ONLY"], "true")
        self.assertEqual(seen["OMCP_SERVER_DIR"], str(Path(self._omcp_dir.name).resolve()))
        self.assertEqual(seen["INHERITED_MARKER"], "passed-through")
        self.assertTrue(seen["DB_PATH"].endswith(".duckdb"))


class FastOMOPMissingCommandTests(unittest.TestCase):
    def test_missing_uv_binary_raises_fastomop_error(self):
        def _run_missing_binary(command, /, **kwargs):
            return _REAL_SUBPROCESS_RUN(["fastomop-uv-not-a-real-binary-a9f3c1"], **kwargs)

        with tempfile.TemporaryDirectory() as fastomop_dir, tempfile.TemporaryDirectory() as omcp_dir:
            with mock.patch("causal.fastomop_bridge.subprocess.run", _run_missing_binary):
                with self.assertRaisesRegex(FastOMOPError, "uv is not installed"):
                    ask_fastomop(
                        "q",
                        datasets=DUCKDB_DATASETS,
                        fastomop_dir=fastomop_dir,
                        omcp_dir=omcp_dir,
                    )


if __name__ == "__main__":
    unittest.main()
