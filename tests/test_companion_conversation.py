"""Coverage for `causal.companion.conversation` -- the append-only transcript
store. Pure file I/O and dataclass validation, no Streamlit, so it is fully
exercised here (unlike the UI panel that consumes it)."""

import json
import re
import tempfile
import unittest
from pathlib import Path

from causal.companion.conversation import (
    CONVERSATION_FILENAME,
    ConversationEntry,
    ConversationStore,
    flush_intake,
)

_HEX32 = re.compile(r"\A[0-9a-f]{32}\Z")


def _entry(text: str = "hello", **overrides) -> ConversationEntry:
    fields = {"role": "user", "kind": "message", "text": text}
    fields.update(overrides)
    return ConversationEntry(**fields)


class ConversationEntryTests(unittest.TestCase):
    def test_new_entry_autofills_id_and_timestamp(self):
        entry = _entry()
        self.assertRegex(entry.id, _HEX32)
        # datetime.isoformat() with a UTC tzinfo always ends in +00:00
        self.assertTrue(entry.at.endswith("+00:00"))

    def test_to_dict_from_dict_round_trip_preserves_id_and_timestamp(self):
        original = _entry("original text", phase="dag_review", data={"n": 3})
        restored = ConversationEntry.from_dict(original.to_dict())
        self.assertEqual(restored, original)
        self.assertEqual(restored.id, original.id)
        self.assertEqual(restored.at, original.at)

    def test_from_dict_missing_required_field_raises(self):
        payload = _entry().to_dict()
        del payload["text"]
        with self.assertRaises(ValueError):
            ConversationEntry.from_dict(payload)

    def test_unknown_role_is_rejected(self):
        with self.assertRaises(ValueError):
            _entry(role="moderator")

    def test_blank_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            _entry(kind="   ")

    def test_non_string_text_is_rejected(self):
        with self.assertRaises(ValueError):
            _entry(text=42)

    def test_non_dict_data_is_rejected(self):
        with self.assertRaises(ValueError):
            _entry(data=["not", "a", "dict"])

    def test_empty_phase_string_is_rejected_but_none_is_allowed(self):
        self.assertIsNone(_entry(phase=None).phase)
        with self.assertRaises(ValueError):
            _entry(phase="")

    def test_unknown_kind_still_round_trips(self):
        # Later phases add kinds; an entry written by a newer build must not
        # blow up an older reader.
        entry = _entry(role="assistant", kind="estimand_dialogue_step")
        self.assertEqual(ConversationEntry.from_dict(entry.to_dict()), entry)


class ConversationStoreTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.run_dir = Path(self._temp.name)

    def test_read_missing_file_returns_empty_list(self):
        self.assertEqual(ConversationStore(self.run_dir).read(), [])

    def test_append_then_read_round_trips(self):
        store = ConversationStore(self.run_dir)
        entry = _entry("what is the effect of X on Y?")
        store.append(entry)
        self.assertEqual(store.read(), [entry])

    def test_entries_are_returned_in_write_order(self):
        store = ConversationStore(self.run_dir)
        first = _entry("first")
        second = _entry("second", role="assistant", kind="question_gate_result")
        third = _entry("third")
        store.append(first)
        store.extend([second, third])
        self.assertEqual([e.text for e in store.read()], ["first", "second", "third"])

    def test_non_ascii_text_survives_the_round_trip(self):
        # Guards the same failure class as commit e57bea7 (UnicodeDecodeError
        # on subprocess output): the file is utf-8 and json is not ascii-escaped.
        store = ConversationStore(self.run_dir)
        entry = _entry("Chez l'adulte en sepsis à l'USI — mortalité à 28 jours ? 中文 🧵")
        store.append(entry)
        self.assertEqual(store.read()[0].text, entry.text)
        self.assertIn("à l'USI", (self.run_dir / CONVERSATION_FILENAME).read_text(encoding="utf-8"))

    def test_blank_lines_are_ignored(self):
        path = self.run_dir / CONVERSATION_FILENAME
        path.write_text(
            json.dumps(_entry("a").to_dict()) + "\n\n   \n" + json.dumps(_entry("b").to_dict()) + "\n",
            encoding="utf-8",
        )
        self.assertEqual([e.text for e in ConversationStore(self.run_dir).read()], ["a", "b"])

    def test_truncated_final_line_is_tolerated(self):
        path = self.run_dir / CONVERSATION_FILENAME
        path.write_text(json.dumps(_entry("complete").to_dict()) + '\n{"role": "user", "kind"', encoding="utf-8")
        self.assertEqual([e.text for e in ConversationStore(self.run_dir).read()], ["complete"])

    def test_malformed_earlier_line_raises(self):
        path = self.run_dir / CONVERSATION_FILENAME
        path.write_text('{"role": broken\n' + json.dumps(_entry("after").to_dict()) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            ConversationStore(self.run_dir).read()

    def test_extend_with_no_entries_does_not_create_the_file(self):
        ConversationStore(self.run_dir).extend([])
        self.assertFalse((self.run_dir / CONVERSATION_FILENAME).exists())

    def test_append_creates_a_missing_run_directory(self):
        nested = self.run_dir / "runs" / "study_abc_123"
        store = ConversationStore(nested)
        store.append(_entry("hi"))
        self.assertTrue((nested / CONVERSATION_FILENAME).is_file())


class FlushIntakeTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.run_dir = Path(self._temp.name)

    def test_flush_writes_buffered_turns_in_order(self):
        buffered = [
            _entry("the clinical question"),
            _entry("looks causal", role="assistant", kind="question_gate_result"),
            _entry("draft ready", role="event", kind="run_created", data={"run_id": "x"}),
        ]
        flush_intake(self.run_dir, buffered)
        restored = ConversationStore(self.run_dir).read()
        self.assertEqual(restored, buffered)

    def test_flush_appends_after_existing_entries(self):
        store = ConversationStore(self.run_dir)
        store.append(_entry("already here"))
        flush_intake(self.run_dir, [_entry("added later")])
        self.assertEqual([e.text for e in store.read()], ["already here", "added later"])


if __name__ == "__main__":
    unittest.main()
