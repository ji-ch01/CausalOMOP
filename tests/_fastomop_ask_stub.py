"""Deterministic stand-in for ``agno_fastomop ask`` -- see
``tests/test_fastomop_bridge_integration.py``.

Not a test module (no ``test_`` prefix), so pytest does not collect it. It is
invoked as a real child process by the integration test to exercise the
``causal.fastomop_bridge`` stdout/stderr parser against real OS pipes.

It reproduces the parts of ``agno_fastomop/src/agno_fastomop/ask.py`` plus
``Workflow(debug_mode=True)`` console behaviour that the bridge has to
survive: interleaved debug lines on stdout, the ``FASTOMOP_JSON:`` marker
printed with a leading newline, and non-ASCII / invalid bytes mixed into that
noise. Output is written straight to ``stdout.buffer`` / ``stderr.buffer`` so
what crosses the pipe does not depend on this child's own console locale.

Scenario is selected by the ``FASTOMOP_STUB_MODE`` environment variable; the
question is ``argv[1]``.
"""

from __future__ import annotations

import json
import os
import sys
import time

PREFIX = b"FASTOMOP_JSON:"


def _out(data: bytes) -> None:
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def _err(data: bytes) -> None:
    sys.stderr.buffer.write(data)
    sys.stderr.buffer.flush()


def _emit(payload: dict) -> None:
    # Mirrors ask.main(): a leading newline guarantees the marker starts its
    # own line even if the last debug write had no trailing newline.
    _out(b"\n" + PREFIX + json.dumps(payload).encode("utf-8") + b"\n")


def main() -> int:
    mode = os.environ.get("FASTOMOP_STUB_MODE", "ok")
    question = sys.argv[1] if len(sys.argv) > 1 else ""

    if mode == "ok":
        _out(b"INFO agno.workflow starting session abc123\n")
        _out(b"DEBUG semantic_agent resolved 2 concepts\n")
        _emit({"answer": "1234 patients", "sql_query": "SELECT COUNT(*) FROM person"})
        return 0

    if mode == "utf8_noise":
        # agno debug_mode echoes prompt/response text, which carries the kind
        # of non-ASCII punctuation an LLM emits. The right double quote below
        # is U+201D -> UTF-8 byte 0x9D, which is undefined in cp1252: decoding
        # this pipe as anything but UTF-8 raises UnicodeDecodeError on Windows.
        _out('DEBUG "user_query": "does aspirin help” — café 中文"\n'.encode("utf-8"))
        _emit({"answer": "réponse pour: " + question, "sql_query": None})
        return 0

    if mode == "bad_utf8_noise":
        _out(b"DEBUG raw bytes \xff\xfe \x81\x8d\x90 not valid utf-8\n")
        _emit({"answer": "recovered past the bad bytes", "sql_query": None})
        return 0

    if mode == "bad_utf8_no_marker":
        _out(b"DEBUG \xff\xfe\x81 broken bytes and no result marker at all\n")
        return 0

    if mode == "glued_marker":
        # Last debug write with no trailing newline -- the leading "\n" in
        # _emit() is the only thing keeping the marker on its own line.
        _out(b'DEBUG workflow step 3 field "user_query": ')
        _emit({"answer": "unglued", "sql_query": None})
        return 0

    if mode == "error_payload":
        _out(b"DEBUG workflow raised in semantic step\n")
        _emit({"error": "semantic step returned no concept_code"})
        return 0

    if mode == "stderr_nonzero":
        _err("Traceback (most recent call last):\nRuntimeError: modèle indisponible\n".encode("utf-8"))
        return 3

    if mode == "slow":
        time.sleep(30)
        _emit({"answer": "too late", "sql_query": None})
        return 0

    if mode == "no_output":
        return 0

    if mode == "echo":
        _emit(
            {
                "answer": json.dumps(
                    {
                        "question": question,
                        "DB_TYPE": os.environ.get("DB_TYPE"),
                        "DB_READ_ONLY": os.environ.get("DB_READ_ONLY"),
                        "DB_PATH": os.environ.get("DB_PATH"),
                        "OMCP_SERVER_DIR": os.environ.get("OMCP_SERVER_DIR"),
                        "INHERITED_MARKER": os.environ.get("INHERITED_MARKER"),
                    }
                ),
                "sql_query": None,
            }
        )
        return 0

    _err(f"unknown FASTOMOP_STUB_MODE={mode!r}\n".encode("utf-8"))
    return 2


if __name__ == "__main__":
    sys.exit(main())
