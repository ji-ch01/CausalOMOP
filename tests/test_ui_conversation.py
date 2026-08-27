"""Coverage for the pure helpers in `causal.ui.conversation`.

Same scope rule as `tests/test_streamlit_app.py`: only the functions that
don't touch `st.*` (entry -> chat shape, form values -> DraftOptions,
clarification bookkeeping). Importing the module is itself signal -- it
fails fast if a rename in `causal.companion` breaks the panel's imports."""

import unittest

from causal.companion.conversation import ConversationEntry
from causal.companion.intake import MAX_CLARIFICATION_ROUNDS, DraftOptions
from causal.ui import conversation as panel


class EntryChatShapeTests(unittest.TestCase):
    def test_user_turn_maps_to_the_user_side(self):
        entry = ConversationEntry(role="user", kind="message", text="hi")
        self.assertEqual(panel.entry_chat_role(entry), "user")

    def test_event_and_system_turns_render_on_the_assistant_side(self):
        for role in ("assistant", "event", "system"):
            entry = ConversationEntry(role=role, kind="message", text="x")
            self.assertEqual(panel.entry_chat_role(entry), "assistant")

    def test_avatar_is_role_specific_with_an_assistant_fallback(self):
        self.assertEqual(
            panel.entry_avatar(ConversationEntry(role="event", kind="run_created", text="x")),
            panel._ROLE_AVATARS["event"],
        )
        # A role the avatar map does not know still resolves to something.
        self.assertEqual(
            panel.entry_avatar(ConversationEntry(role="assistant", kind="message", text="x")),
            panel._ROLE_AVATARS["assistant"],
        )


class DraftOptionsFromFormTests(unittest.TestCase):
    def test_full_form_maps_through(self):
        options = panel.draft_options_from_form(
            {
                "mode": "final",
                "skip_question_gate": True,
                "skip_critique": False,
                "max_critique_rounds": 3,
                "tournament_size": 4,
            }
        )
        self.assertEqual(
            options,
            DraftOptions(
                mode="final",
                skip_question_gate=True,
                enable_critique=True,
                max_critique_rounds=3,
                tournament_size=4,
            ),
        )

    def test_skip_critique_flips_enable_critique(self):
        self.assertFalse(panel.draft_options_from_form({"skip_critique": True}).enable_critique)

    def test_missing_keys_fall_back_to_defaults(self):
        options = panel.draft_options_from_form({})
        self.assertEqual(options.mode, "exploratory")
        self.assertFalse(options.skip_question_gate)
        self.assertTrue(options.enable_critique)


class ClarificationBookkeepingTests(unittest.TestCase):
    def test_no_state_yields_no_rounds(self):
        self.assertEqual(panel.appended_rounds(None, "an answer"), [])

    def test_first_answer_pairs_with_the_last_clarifying_question(self):
        state = {"clarifying_question": "Versus what?", "rounds": []}
        self.assertEqual(panel.appended_rounds(state, "Placebo"), [("Versus what?", "Placebo")])

    def test_prior_rounds_are_kept_and_normalized_to_tuples(self):
        state = {
            "clarifying_question": "Which population?",
            "rounds": [["Versus what?", "Placebo"]],
        }
        self.assertEqual(
            panel.appended_rounds(state, "ICU adults"),
            [("Versus what?", "Placebo"), ("Which population?", "ICU adults")],
        )

    def test_exhausted_only_past_the_cap(self):
        at_cap = [("q", "a")] * MAX_CLARIFICATION_ROUNDS
        self.assertFalse(panel.clarification_exhausted(at_cap))
        self.assertTrue(panel.clarification_exhausted([*at_cap, ("q", "a")]))


if __name__ == "__main__":
    unittest.main()
