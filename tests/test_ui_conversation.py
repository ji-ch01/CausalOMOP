"""Coverage for `causal.ui.conversation`.

Mostly the pure helpers (entry -> chat shape, form values -> DraftOptions,
clarification bookkeeping, dialogue history, pending-card lookup). The
`EstimandDialogueRenderTests` class is the exception -- it drives the panel's
Streamlit code through `AppTest` to guard the render path against a crash."""

import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from causal.companion.companion import CompanionMessage
from causal.companion.conversation import ConversationEntry, ConversationStore
from causal.companion.intake import MAX_CLARIFICATION_ROUNDS, DraftOptions
from causal.ui import conversation as panel


def _entry(kind: str, *, role: str = "assistant", phase: str = "estimand_review", text: str = "x", **data):
    return ConversationEntry(role=role, kind=kind, phase=phase, text=text, data=data)


def _estimand_dialogue_script(runs_dir: str, run_id: str) -> None:
    """Run as an AppTest script: render the estimand dialogue for one run."""
    import streamlit as st

    from causal.orchestrator.workflow import WorkflowManager
    from causal.ui.conversation import _render_estimand_dialogue

    st.session_state["reviewer_name"] = "Dr. Test"
    _render_estimand_dialogue(WorkflowManager(runs_dir), run_id)


def _conversation_panel_script(runs_dir: str, run_id: str) -> None:
    """AppTest script: render the whole panel for an open run so the study
    progress header (the moved "Study summary" content) is exercised."""
    import json
    from pathlib import Path

    from causal.orchestrator.state import ApprovalGate, RunState
    from causal.orchestrator.workflow import WorkflowManager
    from causal.ui.conversation import render_conversation_panel

    manager = WorkflowManager(runs_dir)
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "protocol.json").write_text(
        json.dumps(
            {
                "source_question": "Does aspirin reduce 30-day mortality after MI?",
                "target_trial": {
                    "population": "adults admitted with MI",
                    "treatment": "aspirin",
                    "comparator": "no aspirin",
                    "outcome": "30-day mortality",
                    "effect_measure": "risk_difference",
                },
            }
        ),
        encoding="utf-8",
    )
    manager.save(
        RunState(
            run_id=run_id,
            study_id="aspirin_mi",
            protocol_path=str(run_dir / "protocol.json"),
            phase="dag_review",
            gates={"target_trial": ApprovalGate("target_trial", status="approved", reviewer="r")},
        )
    )
    render_conversation_panel(manager, run_id)


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


class EstimandDialogueHistoryTests(unittest.TestCase):
    def test_keeps_only_estimand_phase_user_and_assistant_turns(self):
        entries = [
            _entry("message", role="user", phase="target_trial_review", text="the original question"),
            _entry("message", role="user", text="what does IPCW do?"),
            _entry("message", role="assistant", text="IPCW reweights rather than drops."),
            _entry("action_taken", role="event", text="strategy saved"),
            _entry("proposed_action", role="assistant", text="here is what I'd record"),
        ]
        history = panel.estimand_dialogue_history(entries)
        self.assertEqual(
            [(m.role, m.text) for m in history],
            [
                ("user", "what does IPCW do?"),
                ("assistant", "IPCW reweights rather than drops."),
                ("assistant", "here is what I'd record"),
            ],
        )
        self.assertTrue(all(isinstance(m, CompanionMessage) for m in history))

    def test_blank_turns_are_dropped(self):
        entries = [_entry("message", role="user", text="   ")]
        self.assertEqual(panel.estimand_dialogue_history(entries), [])


class LatestPendingActionTests(unittest.TestCase):
    def test_none_when_no_proposal(self):
        self.assertIsNone(panel.latest_pending_action([_entry("message", role="user", text="hi")]))

    def test_returns_the_unresolved_proposal(self):
        proposal = _entry("proposed_action", text="proposal A")
        self.assertIs(panel.latest_pending_action([proposal]), proposal)

    def test_confirmed_or_dismissed_clears_it(self):
        proposal = _entry("proposed_action", text="proposal A")
        for resolver in ("action_taken", "action_dismissed"):
            self.assertIsNone(
                panel.latest_pending_action([proposal, _entry(resolver, role="event", text="done")])
            )

    def test_a_newer_proposal_supersedes_an_older_one(self):
        first = _entry("proposed_action", text="proposal A")
        second = _entry("proposed_action", text="proposal B")
        self.assertIs(panel.latest_pending_action([first, second]), second)

    def test_ignores_proposals_from_other_phases(self):
        self.assertIsNone(
            panel.latest_pending_action(
                [_entry("proposed_action", phase="results_review", text="not this one")]
            )
        )


class EstimandDialogueRenderTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.runs_dir = Path(self._temp.name)

    def _app(self, run_id: str) -> AppTest:
        harness = AppTest.from_function(
            _estimand_dialogue_script,
            kwargs={"runs_dir": str(self.runs_dir), "run_id": run_id},
            default_timeout=30,
        )
        harness.run()
        return harness

    def test_renders_the_thread_and_a_pending_action_card(self):
        run_dir = self.runs_dir / "run1"
        run_dir.mkdir(parents=True)
        ConversationStore(run_dir).extend(
            [
                _entry("message", role="assistant", text="Let's start with the censoring strategy."),
                _entry("message", role="user", text="use IPCW"),
                _entry(
                    "proposed_action",
                    role="assistant",
                    text="Recording IPCW plus TMLE.",
                    action="set_statistical_estimand_strategy",
                    args={
                        "missing_data_strategy": "complete_case_analysis",
                        "censoring_strategy": "ipcw_administrative_censoring",
                        "estimation_procedure": "targeted_minimum_loss_based_estimation",
                        "censoring_rationale": "Reweight rather than drop.",
                    },
                    summary="Save statistical-estimand strategy",
                ),
            ]
        )
        harness = self._app("run1")
        self.assertEqual(list(harness.exception), [])
        self.assertTrue(any("Confirm" in button.label for button in harness.button))
        self.assertEqual(len(harness.chat_input), 1)

    def test_degrades_without_a_crash_when_the_companion_cannot_run(self):
        # An empty run dir: run_companion_turn's own load() raises, the panel
        # catches it and shows the fallback rather than blowing up.
        (self.runs_dir / "run2").mkdir(parents=True)
        harness = self._app("run2")
        self.assertEqual(list(harness.exception), [])
        self.assertTrue(any("unavailable" in warning.value.lower() for warning in harness.warning))


class StudyProgressRenderTests(unittest.TestCase):
    def test_progress_header_renders_for_an_open_run(self):
        with tempfile.TemporaryDirectory() as temp:
            harness = AppTest.from_function(
                _conversation_panel_script,
                kwargs={"runs_dir": str(Path(temp) / "runs"), "run_id": "run1"},
                default_timeout=30,
            )
            harness.run()
        self.assertEqual(list(harness.exception), [])
        prose = " ".join(block.value for block in harness.markdown)
        self.assertIn("aspirin", prose)
        self.assertTrue(any("Gates approved" in metric.label for metric in harness.metric))
        # the "Workflow & evidence" expander body lists every phase label
        self.assertIn("causal DAG review", prose)


if __name__ == "__main__":
    unittest.main()
