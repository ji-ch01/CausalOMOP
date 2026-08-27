"""Persistent, append-only transcript of the study-companion conversation.

One `ConversationEntry` per turn, stored as one JSON object per line in
`artifacts/runs/<run_id>/conversation.jsonl`.

This is a *separate* stream from `RunState.history` (the workflow audit log
in `causal.orchestrator.state`): the transcript also carries the reviewer's
natural-language turns, the causal-question-gate outcome, and any FastOMOP
answer that was shown inline. It is advisory context only -- nothing written
here can decide a gate or advance a phase.

The store is deliberately tiny and pure: no Streamlit, no `WorkflowManager`.
The UI reads the whole file on every rerun rather than caching entries in
session state, so the file on disk is always the single source of truth for
what was said. It is written under `artifacts/runs/` and therefore inherits
that tree's `.gitignore` and "review before sharing" handling, the same as
`llm_outputs.json`.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONVERSATION_FILENAME = "conversation.jsonl"

#: Who a turn is attributed to. `event` is the workflow speaking (a run was
#: created, a gate was decided); it is not a chat participant.
ENTRY_ROLES = frozenset({"user", "assistant", "system", "event"})

#: The `kind` values the phase-1 conversation panel knows how to render.
#: Later phases (the estimand dialogue, decision-memory notes, gate action
#: cards) extend this set. An entry whose kind is not listed here still
#: round-trips through the store unchanged -- it just falls back to a plain
#: text rendering until a dedicated renderer exists.
ENTRY_KINDS = frozenset(
    {
        "message",
        "question_gate_result",
        "clarification_request",
        "fastomop_answer",
        "run_created",
    }
)

_REQUIRED_FIELDS = ("role", "kind", "text")
_SERIALIZED_FIELDS = ("id", "at", "role", "kind", "phase", "text", "data")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ConversationEntry:
    """One turn in the transcript.

    `id` and `at` are filled automatically for a new entry and preserved
    verbatim when one is reloaded from disk. `phase` is the run phase the
    turn happened in, or `None` for intake turns that predate the run.
    `data` carries kind-specific payload (e.g. the SQL for a
    `fastomop_answer`, the `run_id` for a `run_created`).
    """

    role: str
    kind: str
    text: str
    phase: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        if self.role not in ENTRY_ROLES:
            raise ValueError(f"Unknown conversation role: {self.role!r}")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("conversation entry kind is required")
        if not isinstance(self.text, str):
            raise ValueError("conversation entry text must be a string")
        if self.phase is not None and (not isinstance(self.phase, str) or not self.phase.strip()):
            raise ValueError("conversation entry phase must be a non-empty string or None")
        if not isinstance(self.data, dict):
            raise ValueError("conversation entry data must be a dict")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _SERIALIZED_FIELDS}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConversationEntry":
        missing = [name for name in _REQUIRED_FIELDS if name not in value]
        if missing:
            raise ValueError(f"conversation entry is missing required field(s): {missing}")
        return cls(**{name: value[name] for name in _SERIALIZED_FIELDS if name in value})


class ConversationStore:
    """Append-only reader/writer for one run's `conversation.jsonl`.

    Pure file I/O over a single run directory. Safe to construct against a
    run whose transcript does not exist yet: `read()` returns `[]` until the
    first `append()`.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.path = Path(run_dir) / CONVERSATION_FILENAME

    def append(self, entry: ConversationEntry) -> None:
        self.extend((entry,))

    def extend(self, entries: Iterable[ConversationEntry]) -> None:
        entries = list(entries)
        if not entries:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def read(self) -> list[ConversationEntry]:
        if not self.path.is_file():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        entries: list[ConversationEntry] = []
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                # A process killed mid-`extend()` can leave a truncated final
                # line. Tolerate exactly that; a malformed line anywhere
                # earlier is real corruption of an audit artifact and must
                # surface rather than be silently dropped.
                if index == len(lines) - 1:
                    break
                raise ValueError(f"{self.path}: malformed JSON on line {index + 1}") from None
            entries.append(ConversationEntry.from_dict(payload))
        return entries


def flush_intake(run_dir: str | Path, entries: Iterable[ConversationEntry]) -> None:
    """Write the buffered pre-run intake turns into a new run's transcript.

    The intake exchange (the clinical question, the causal-question-gate
    result, any clarification rounds) happens before
    `WorkflowManager.start_generated` mints a run id, so the UI buffers those
    turns and calls this once the run directory exists, preserving order.
    """
    ConversationStore(run_dir).extend(entries)
