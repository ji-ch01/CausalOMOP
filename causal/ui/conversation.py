"""Left-column conversation panel for the Streamlit review app.

Phase 1 of the conversation-first rework: the clinical-question intake and
the clarification loop live here as a persistent chat thread instead of the
old top-right popover, and every turn is written to the run's
`conversation.jsonl` (`causal.companion.conversation`) the moment
`WorkflowManager.start_generated` creates the run.

Before a run exists the thread is buffered in `st.session_state`
(`_intake_thread`); once a run is open the panel shows that run's persisted
transcript, read-only -- the in-thread study companion arrives in a later
phase. Nothing here decides a gate: it only classifies the question and,
on the causal path, opens a run with every gate still pending.

The pure helpers (entry -> chat shape, form values -> `DraftOptions`,
clarification bookkeeping) are kept free of `st.*` so
`tests/test_ui_conversation.py` can exercise them; the `_render_*` /
`handle_*` functions are the thin Streamlit layer over them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import streamlit as st

from causal.companion.conversation import ConversationEntry, ConversationStore, flush_intake
from causal.companion.intake import (
    MAX_CLARIFICATION_ROUNDS,
    ClarificationNeeded,
    ClarificationRound,
    DescriptiveAnswer,
    DraftOptions,
    QuestionIntake,
    StudyDrafted,
    combine_clarifications,
)
from causal.config import BigQueryDatasets
from causal.fastomop_bridge import FastOMOPError
from causal.literature import LiteratureOrchestrator, OpenAlexClient, PubMedClient
from causal.llm import build_hypothesis_llm
from causal.orchestrator.workflow import WorkflowManager

_INTAKE_THREAD_KEY = "_intake_thread"
_CLARIFICATION_KEY = "_clarification"
_THREAD_HEIGHT = 460

_CATEGORY_LABELS = {
    "causal": "Causal question",
    "descriptive": "Descriptive question",
    "needs_clarification": "Needs clarification",
}

_ROLE_AVATARS = {
    "user": ":material/person:",
    "assistant": ":material/neurology:",
    "event": ":material/flag:",
    "system": ":material/info:",
}


# --------------------------------------------------------------------------- #
# Pure helpers (no st.*)
# --------------------------------------------------------------------------- #
def entry_chat_role(entry: ConversationEntry) -> str:
    """`st.chat_message` only accepts a participant name; map the transcript's
    four roles onto "user" / "assistant" (an `event`/`system` turn is shown
    on the assistant side with its own avatar)."""
    return "user" if entry.role == "user" else "assistant"


def entry_avatar(entry: ConversationEntry) -> str:
    return _ROLE_AVATARS.get(entry.role, _ROLE_AVATARS["assistant"])


def draft_options_from_form(form: Mapping[str, Any]) -> DraftOptions:
    """Translate the "Draft options" expander's raw widget values into the
    `DraftOptions` contract. `skip_literature` / `model` are handled by the
    caller (they shape the injected collaborators, not the draft knobs)."""
    return DraftOptions(
        mode=form.get("mode", "exploratory"),
        skip_question_gate=bool(form.get("skip_question_gate", False)),
        enable_critique=not bool(form.get("skip_critique", False)),
        max_critique_rounds=int(form.get("max_critique_rounds", 2)),
        tournament_size=int(form.get("tournament_size", 2)),
    )


def appended_rounds(
    clarification_state: Mapping[str, Any] | None, answer: str
) -> list[ClarificationRound]:
    """The clarification rounds so far plus the current answer paired with
    the assistant's last clarifying question. `[]` -> `[(q, answer)]`."""
    if not clarification_state:
        return []
    prior: Sequence[ClarificationRound] = clarification_state.get("rounds", ())
    clarifying_question = clarification_state.get("clarifying_question", "")
    return [*(tuple(item) for item in prior), (clarifying_question, answer)]


def clarification_exhausted(rounds: Sequence[ClarificationRound]) -> bool:
    return len(rounds) > MAX_CLARIFICATION_ROUNDS


def reset_intake() -> None:
    """Clear the buffered intake thread and any pending clarification -- wired
    to the sidebar "New study" action so a fresh thread starts cleanly."""
    st.session_state.pop(_INTAKE_THREAD_KEY, None)
    st.session_state.pop(_CLARIFICATION_KEY, None)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _render_entry(entry: ConversationEntry) -> None:
    with st.chat_message(entry_chat_role(entry), avatar=entry_avatar(entry)):
        if entry.kind == "question_gate_result":
            category = entry.data.get("category", "")
            st.markdown(f"**{_CATEGORY_LABELS.get(category, 'Question classified')}**")
            if entry.text:
                st.caption(entry.text)
        elif entry.kind == "clarification_request":
            st.markdown(entry.text)
            if entry.data.get("reasoning"):
                st.caption(entry.data["reasoning"])
        elif entry.kind == "fastomop_answer":
            if entry.data.get("reasoning"):
                st.caption(entry.data["reasoning"])
            st.markdown(entry.text)
            if entry.data.get("sql"):
                st.code(entry.data["sql"], language="sql")
        elif entry.kind == "run_created":
            st.markdown(f":material/task_alt: {entry.text}")
            model = entry.data.get("model")
            if model:
                st.caption(f"Draft model: {entry.data.get('provider', 'unknown')}:{model}")
        else:  # "message" and any kind a later build introduced
            st.markdown(entry.text or "_(empty)_")


def _render_thread(entries: Sequence[ConversationEntry]) -> None:
    with st.container(height=_THREAD_HEIGHT, border=False):
        if not entries:
            st.chat_message("assistant", avatar=_ROLE_AVATARS["assistant"]).markdown(
                "Ask a clinical question to start a causal study. A descriptive question "
                "(a count, a distribution) is answered inline via FastOMOP; an ambiguous "
                "one gets a clarifying question first."
            )
            return
        for entry in entries:
            _render_entry(entry)


def _render_draft_options() -> dict[str, Any]:
    with st.expander("Draft options", icon=":material/tune:"):
        mode = st.selectbox("Mode", ("exploratory", "final"), key="intake_mode")
        model = st.text_input("Model override (optional)", key="intake_model")
        skip_question_gate = st.checkbox(
            "Assume the question is already causal (skip the classification gate)",
            key="intake_skip_gate",
        )
        skip_literature = st.checkbox(
            "Skip literature grounding (PubMed / OpenAlex)", key="intake_skip_lit"
        )
        skip_critique = st.checkbox("Skip automated DAG review", key="intake_skip_critique")
        left, right = st.columns(2)
        max_critique_rounds = left.number_input(
            "Review rounds", min_value=1, max_value=5, value=2,
            disabled=skip_critique, key="intake_max_critique_rounds",
        )
        tournament_size = right.number_input(
            "Candidate DAGs", min_value=1, max_value=5, value=2, key="intake_tournament_size",
        )
    return {
        "mode": mode,
        "model": model,
        "skip_question_gate": skip_question_gate,
        "skip_literature": skip_literature,
        "skip_critique": skip_critique,
        "max_critique_rounds": max_critique_rounds,
        "tournament_size": tournament_size,
    }


# --------------------------------------------------------------------------- #
# Intake orchestration
# --------------------------------------------------------------------------- #
def _buffer() -> list[ConversationEntry]:
    return st.session_state.setdefault(_INTAKE_THREAD_KEY, [])


def _build_intake(manager: WorkflowManager, form: Mapping[str, Any]) -> QuestionIntake:
    resolved_llm = build_hypothesis_llm(model=(form.get("model") or "").strip() or None)
    literature = (
        None
        if form.get("skip_literature")
        else LiteratureOrchestrator(pubmed=PubMedClient(), openalex=OpenAlexClient())
    )
    return QuestionIntake(
        manager=manager,
        resolved_llm=resolved_llm,
        literature=literature,
        data_source=BigQueryDatasets.from_env().describe(),
    )


def _handle_submission(manager: WorkflowManager, text: str, form: Mapping[str, Any]) -> None:
    buffer = _buffer()
    clarification = st.session_state.get(_CLARIFICATION_KEY)
    buffer.append(ConversationEntry(role="user", kind="message", text=text))

    rounds = appended_rounds(clarification, text)
    if clarification:
        original_question = clarification["original_question"]
        if clarification_exhausted(rounds):
            buffer.append(
                ConversationEntry(
                    role="assistant",
                    kind="message",
                    text=(
                        "The question is still ambiguous after several rounds. Rephrase it, "
                        'or enable "Assume the question is already causal" in the draft options.'
                    ),
                )
            )
            st.session_state.pop(_CLARIFICATION_KEY, None)
            st.rerun()
            return
        question = combine_clarifications(original_question, rounds)
    else:
        original_question = text
        question = text

    status = st.status("Working on the question...", expanded=True)
    try:
        intake = _build_intake(manager, form)
        outcome = intake.run(
            question,
            draft_options_from_form(form),
            on_stage=status.write,
            on_llm_output=lambda output: status.write(f"Received `{output.stage}` from the LLM."),
            on_gate_result=lambda result: buffer.append(
                ConversationEntry(
                    role="assistant",
                    kind="question_gate_result",
                    text=result.reasoning,
                    data={"category": result.category},
                )
            ),
        )
    except (FastOMOPError, FileNotFoundError) as exc:
        status.update(label="FastOMOP unavailable", state="error")
        buffer.append(
            ConversationEntry(
                role="assistant", kind="message", text=f"FastOMOP could not answer: {exc}"
            )
        )
        st.rerun()
        return
    except Exception as exc:  # noqa: BLE001 - surfaced in the thread, not swallowed
        status.update(label="Draft failed", state="error")
        buffer.append(
            ConversationEntry(role="assistant", kind="message", text=f"Draft failed: {exc}")
        )
        st.rerun()
        return

    _apply_outcome(manager, outcome, buffer, original_question, rounds, status)


def _apply_outcome(
    manager: WorkflowManager,
    outcome: DescriptiveAnswer | ClarificationNeeded | StudyDrafted,
    buffer: list[ConversationEntry],
    original_question: str,
    rounds: Sequence[ClarificationRound],
    status: Any,
) -> None:
    if isinstance(outcome, DescriptiveAnswer):
        status.update(label="FastOMOP answered", state="complete")
        buffer.append(
            ConversationEntry(
                role="assistant",
                kind="fastomop_answer",
                text=outcome.answer,
                data={"sql": outcome.sql, "reasoning": outcome.reasoning},
            )
        )
        st.session_state.pop(_CLARIFICATION_KEY, None)
        st.rerun()
        return

    if isinstance(outcome, ClarificationNeeded):
        status.update(label="Clarification needed", state="complete")
        buffer.append(
            ConversationEntry(
                role="assistant",
                kind="clarification_request",
                text=outcome.clarifying_question,
                data={"reasoning": outcome.reasoning},
            )
        )
        st.session_state[_CLARIFICATION_KEY] = {
            "original_question": original_question,
            "clarifying_question": outcome.clarifying_question,
            "rounds": [list(item) for item in rounds],
        }
        st.rerun()
        return

    # StudyDrafted
    state = outcome.state
    status.update(label=f"Study {state.run_id} created", state="complete")
    buffer.append(
        ConversationEntry(
            role="event",
            kind="run_created",
            text=f"Study “{state.study_id}” created.",
            phase=state.phase,
            data={
                "run_id": state.run_id,
                "study_id": state.study_id,
                "provider": outcome.provider,
                "model": outcome.model,
            },
        )
    )
    flush_intake(manager.runs_dir / state.run_id, buffer)
    st.session_state[_INTAKE_THREAD_KEY] = []
    st.session_state.pop(_CLARIFICATION_KEY, None)
    # Applied at the top of `main()` before the run selector widget is rebuilt.
    st.session_state["_pending_run_selection"] = state.run_id
    st.rerun()


def render_conversation_panel(manager: WorkflowManager, run_id: str | None) -> None:
    """Entry point for `main()`'s left column."""
    st.subheader("Conversation", anchor=False)

    if run_id is not None:
        _render_thread(ConversationStore(manager.runs_dir / run_id).read())
        st.caption(
            'Transcript of the open study. Use "New study" in the sidebar to start another.'
        )
        return

    _render_thread(_buffer())
    form = _render_draft_options()
    placeholder = (
        "Answer the question above..."
        if st.session_state.get(_CLARIFICATION_KEY)
        else "Ask a clinical question to start a study..."
    )
    submitted = st.chat_input(placeholder)
    if submitted and submitted.strip():
        _handle_submission(manager, submitted.strip(), form)
