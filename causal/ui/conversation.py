"""Left-column conversation panel for the Streamlit review app.

Two conversation-first surfaces live here:

- Phase 1: clinical-question intake and the clarification loop, as a chat
  thread instead of the old top-right popover. Buffered in
  `st.session_state` (`_intake_thread`) until `start_generated` creates the
  run, then flushed to `conversation.jsonl` (`causal.companion.conversation`).

- Phase 2: the estimand dialogue. While a run sits in the `estimand_review`
  phase the panel becomes an active conversation with the study companion
  (`causal.companion.companion` via `WorkflowManager.run_companion_turn`),
  which can end a turn by proposing `set_statistical_estimand_strategy` as a
  card the reviewer confirms.

Outside those two, the panel shows the run's persisted transcript read-only.
Nothing here decides a gate: intake opens a run with every gate pending, and
the estimand dialogue's one possible write is an explicitly confirmed
`set_statistical_estimand_strategy` -- approving the `statistical_estimand`
gate itself stays a separate manual action in the workspace.

The pure helpers (entry -> chat shape, form values -> `DraftOptions`,
clarification bookkeeping, dialogue history, pending-card lookup) are kept
free of `st.*` so `tests/test_ui_conversation.py` can exercise them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import streamlit as st

from causal.agents.results_narrative import normalize_narrative_artifact
from causal.companion.companion import CompanionMessage
from causal.companion.conversation import ConversationEntry, ConversationStore, flush_intake
from causal.companion.intake import (
    MAX_CLARIFICATION_ROUNDS,
    MAX_DESCRIPTIVE_CONTEXT,
    ClarificationNeeded,
    ClarificationRound,
    DescriptiveAnswer,
    DescriptiveExchange,
    DraftOptions,
    QuestionIntake,
    StudyDrafted,
    combine_clarifications,
)
from causal.config import BigQueryDatasets
from causal.fastomop_bridge import FastOMOPError
from causal.literature import LiteratureOrchestrator, OpenAlexClient, PubMedClient
from causal.llm import build_hypothesis_llm
from causal.orchestrator.state import RunState
from causal.orchestrator.workflow import WorkflowManager

_INTAKE_THREAD_KEY = "_intake_thread"
_CLARIFICATION_KEY = "_clarification"
_THREAD_HEIGHT = 460
# Below this many turns the thread renders inline; a fixed-height scroll
# container for one or two messages just leaves a large empty box.
_THREAD_SCROLL_AFTER = 5
_ESTIMAND_PHASE = "estimand_review"
# The results-interpretation dialogue runs across two phases; its transcript
# turns are all tagged with this one value so history filtering stays stable
# even after the run advances from `results_review` to `report`.
_RESULTS_PHASES = frozenset({"results_review", "report"})
_RESULTS_DIALOGUE_PHASE = "results_review"

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


def prior_descriptive_exchanges(
    entries: Sequence[ConversationEntry],
) -> list[DescriptiveExchange]:
    """The answered FastOMOP exchanges in the thread so far, oldest first,
    as `(question, answer)` pairs -- the session window that lets a terse
    follow-up be resolved (`combine_descriptive_followup`). The question is
    read from the answer entry's `data["question"]`, falling back to the
    most recent preceding `user` turn for entries written before that field
    existed. Capped at `MAX_DESCRIPTIVE_CONTEXT`."""
    exchanges: list[DescriptiveExchange] = []
    last_user_text = ""
    for entry in entries:
        if entry.role == "user" and entry.text.strip():
            last_user_text = entry.text.strip()
        if entry.kind != "fastomop_answer":
            continue
        question = str(entry.data.get("question") or "").strip() or last_user_text
        answer = entry.text.strip()
        if question and answer:
            exchanges.append((question, answer))
    return exchanges[-MAX_DESCRIPTIVE_CONTEXT:]


def reset_intake() -> None:
    """Clear the buffered intake thread and any pending clarification -- wired
    to the sidebar "New study" action so a fresh thread starts cleanly."""
    st.session_state.pop(_INTAKE_THREAD_KEY, None)
    st.session_state.pop(_CLARIFICATION_KEY, None)


def estimand_dialogue_history(entries: Sequence[ConversationEntry]) -> list[CompanionMessage]:
    """The `estimand_review` turns, as the companion sees them: reviewer
    (`user`) and companion (`assistant`) messages only, in order. Workflow
    `event` lines (a card confirmed / dismissed) and turns from other phases
    are dropped."""
    history: list[CompanionMessage] = []
    for entry in entries:
        if entry.phase != _ESTIMAND_PHASE or entry.role not in ("user", "assistant"):
            continue
        if entry.text.strip():
            history.append(CompanionMessage(role=entry.role, text=entry.text))
    return history


def latest_pending_action(entries: Sequence[ConversationEntry]) -> ConversationEntry | None:
    """The most recent `proposed_action` turn that has not yet been confirmed
    or dismissed. A later `proposed_action` supersedes an earlier one; any
    `action_taken` / `action_dismissed` after it clears it."""
    pending: ConversationEntry | None = None
    for entry in entries:
        if entry.phase != _ESTIMAND_PHASE:
            continue
        if entry.kind == "proposed_action":
            pending = entry
        elif entry.kind in ("action_taken", "action_dismissed"):
            pending = None
    return pending


def results_dialogue_history(entries: Sequence[ConversationEntry]) -> list[CompanionMessage]:
    """The results-dialogue turns, as the companion sees them: reviewer
    (`user`) and companion (`assistant`) messages in order. Workflow `event`
    lines (a draft saved / dismissed) and turns from other phases are
    dropped; a `drafted_narrative` turn's `reply` text is kept (it is the
    companion speaking)."""
    history: list[CompanionMessage] = []
    for entry in entries:
        if entry.phase != _RESULTS_DIALOGUE_PHASE or entry.role not in ("user", "assistant"):
            continue
        if entry.text.strip():
            history.append(CompanionMessage(role=entry.role, text=entry.text))
    return history


def latest_drafted_narrative(entries: Sequence[ConversationEntry]) -> ConversationEntry | None:
    """The most recent `drafted_narrative` turn not yet saved or dismissed. A
    later draft supersedes an earlier one; a `narrative_saved` /
    `narrative_dismissed` after it clears it."""
    pending: ConversationEntry | None = None
    for entry in entries:
        if entry.phase != _RESULTS_DIALOGUE_PHASE:
            continue
        if entry.kind == "drafted_narrative":
            pending = entry
        elif entry.kind in ("narrative_saved", "narrative_dismissed"):
            pending = None
    return pending


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
        elif entry.kind == "proposed_action":
            if entry.text:
                st.markdown(entry.text)
            st.caption(f":material/description: {entry.data.get('summary', 'Proposed action')}")
        elif entry.kind in ("action_taken", "action_dismissed"):
            icon = ":material/check_circle:" if entry.kind == "action_taken" else ":material/cancel:"
            st.markdown(f"{icon} {entry.text}")
        elif entry.kind == "drafted_narrative":
            if entry.text:
                st.markdown(entry.text)
            claim_count = len(entry.data.get("claims") or ())
            st.caption(f":material/description: Drafted interpretation · {claim_count} checkable claim(s)")
        elif entry.kind in ("narrative_saved", "narrative_dismissed"):
            icon = ":material/check_circle:" if entry.kind == "narrative_saved" else ":material/cancel:"
            st.markdown(f"{icon} {entry.text}")
        else:  # "message" and any kind a later build introduced
            st.markdown(entry.text or "_(empty)_")


_INTAKE_EMPTY_MESSAGE = (
    "Ask a clinical question to start a causal study. A descriptive question "
    "(a count, a distribution) is answered inline via FastOMOP; an ambiguous "
    "one gets a clarifying question first."
)


def _render_thread(entries: Sequence[ConversationEntry], *, empty_message: str = _INTAKE_EMPTY_MESSAGE) -> None:
    if not entries:
        st.chat_message("assistant", avatar=_ROLE_AVATARS["assistant"]).markdown(empty_message)
        return
    if len(entries) <= _THREAD_SCROLL_AFTER:
        for entry in entries:
            _render_entry(entry)
        return
    with st.container(height=_THREAD_HEIGHT, border=False):
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
            descriptive_context=prior_descriptive_exchanges(buffer),
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
                data={
                    "question": original_question,
                    "sql": outcome.sql,
                    "reasoning": outcome.reasoning,
                },
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


# --------------------------------------------------------------------------- #
# Estimand dialogue (phase estimand_review)
# --------------------------------------------------------------------------- #
_ESTIMAND_EMPTY_MESSAGE = (
    "This is where the statistical estimand is settled: the missing-data and "
    "censoring strategies, and which reviewed estimator runs. Start the dialogue "
    "below, or fill the form directly under **Current task**."
)


def _companion_error_key(run_id: str) -> str:
    return f"_companion_error_{run_id}"


def _run_companion_reply(manager: WorkflowManager, run_id: str, store: ConversationStore) -> bool:
    """Read the dialogue so far, take one companion turn, append its reply
    (and any proposed action) to the transcript. Returns True on success;
    on any provider/parse failure it records the message for the panel to
    show and returns False without writing anything."""
    history = estimand_dialogue_history(store.read())
    status = st.status("Companion is thinking...", expanded=False)
    try:
        resolved_llm = build_hypothesis_llm()
        turn = manager.run_companion_turn(run_id, resolved_llm.call, history)
    except Exception as exc:  # noqa: BLE001 - surfaced in the panel, not swallowed
        status.update(label="Companion unavailable", state="error")
        st.session_state[_companion_error_key(run_id)] = str(exc)
        return False
    status.update(label="Companion replied", state="complete")
    st.session_state.pop(_companion_error_key(run_id), None)
    if turn.proposed_action is not None:
        store.append(
            ConversationEntry(
                role="assistant",
                kind="proposed_action",
                phase=_ESTIMAND_PHASE,
                text=turn.reply,
                data={
                    "action": turn.proposed_action.action,
                    "args": turn.proposed_action.args,
                    "summary": turn.proposed_action.summary,
                },
            )
        )
    else:
        store.append(
            ConversationEntry(
                role="assistant", kind="message", phase=_ESTIMAND_PHASE, text=turn.reply
            )
        )
    return True


def _render_pending_action_card(
    manager: WorkflowManager, run_id: str, entry: ConversationEntry
) -> None:
    args = dict(entry.data.get("args") or {})
    reviewer = (st.session_state.get("reviewer_name") or "").strip()
    with st.container(border=True):
        st.markdown(":material/description: **Proposed action — not yet applied**")
        st.caption(entry.data.get("summary", ""))
        st.markdown(
            f"- Missing data: `{args.get('missing_data_strategy', '—')}`\n"
            f"- Censoring: `{args.get('censoring_strategy', '—')}`\n"
            f"- Estimator: `{args.get('estimation_procedure', '—')}`"
        )
        rationale = st.text_area(
            "Rationale (editable before you confirm)",
            value=args.get("censoring_rationale", ""),
            key=f"card_rationale_{entry.id}",
        )
        if not reviewer:
            st.caption("Enter your name in the sidebar to confirm.")
        confirm_col, dismiss_col = st.columns(2)
        confirm = confirm_col.button(
            "Confirm & save",
            type="primary",
            disabled=not (reviewer and rationale.strip()),
            key=f"card_confirm_{entry.id}",
            width="stretch",
        )
        dismiss = dismiss_col.button(
            "Dismiss", key=f"card_dismiss_{entry.id}", width="stretch"
        )
        store = ConversationStore(manager.runs_dir / run_id)
        if confirm:
            applied = {**args, "censoring_rationale": rationale.strip()}
            try:
                manager.set_statistical_estimand_strategy(
                    run_id, reviewer=reviewer, **applied
                )
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
                return
            store.append(
                ConversationEntry(
                    role="event",
                    kind="action_taken",
                    phase=_ESTIMAND_PHASE,
                    text=f"Statistical-estimand strategy saved by {reviewer}.",
                    data={"action_id": entry.id, "args": applied, "reviewer": reviewer},
                )
            )
            st.rerun()
        if dismiss:
            store.append(
                ConversationEntry(
                    role="event",
                    kind="action_dismissed",
                    phase=_ESTIMAND_PHASE,
                    text="Proposed strategy dismissed.",
                    data={"action_id": entry.id},
                )
            )
            st.rerun()


def _render_estimand_dialogue(manager: WorkflowManager, run_id: str) -> None:
    store = ConversationStore(manager.runs_dir / run_id)

    def _load() -> tuple[list[ConversationEntry], list[ConversationEntry]]:
        entries = store.read()
        return entries, [entry for entry in entries if entry.phase == _ESTIMAND_PHASE]

    entries, dialogue_entries = _load()

    # Open the dialogue automatically on first arrival. On success rerun so the
    # companion's opening turn is part of a clean render; on failure fall
    # through -- the warning below explains it and offers the form fallback.
    opened_key = f"_estimand_opened_{run_id}"
    if not dialogue_entries and not st.session_state.get(opened_key):
        st.session_state[opened_key] = True
        if _run_companion_reply(manager, run_id, store):
            st.rerun()
        entries, dialogue_entries = _load()

    st.caption(
        "Settle the statistical estimand with the study companion. Nothing is applied until you confirm."
    )
    _render_thread(dialogue_entries, empty_message=_ESTIMAND_EMPTY_MESSAGE)

    error = st.session_state.get(_companion_error_key(run_id))
    if error:
        st.warning(
            f"The companion is unavailable ({error}). Use the form under **Current task** "
            "to set the strategy directly."
        )
        if st.button("Retry the companion", icon=":material/refresh:", key=f"estimand_retry_{run_id}"):
            st.session_state.pop(_companion_error_key(run_id), None)
            _run_companion_reply(manager, run_id, store)
            st.rerun()

    pending = latest_pending_action(entries)
    if pending is not None:
        _render_pending_action_card(manager, run_id, pending)

    submitted = st.chat_input("Answer, or ask about a choice…", key=f"estimand_chat_{run_id}")
    if submitted and submitted.strip():
        store.append(
            ConversationEntry(
                role="user", kind="message", phase=_ESTIMAND_PHASE, text=submitted.strip()
            )
        )
        _run_companion_reply(manager, run_id, store)
        st.rerun()


# --------------------------------------------------------------------------- #
# Results-interpretation dialogue (phase results_review / report)
# --------------------------------------------------------------------------- #
_RESULTS_EMPTY_MESSAGE = (
    "The estimate, diagnostics, and robustness analyses are ready. Talk them "
    "through here and the companion will draft a plain-language interpretation "
    "for you to approve -- or draft it in one step under **Estimation & report**."
)


def _run_results_companion_reply(
    manager: WorkflowManager, run_id: str, store: ConversationStore
) -> bool:
    """Read the results dialogue so far, take one companion turn, append its
    reply (and any drafted narrative) to the transcript. Returns True on
    success; on any provider/parse failure it records the message for the
    panel to show and returns False without writing anything."""
    history = results_dialogue_history(store.read())
    status = st.status("Companion is thinking...", expanded=False)
    try:
        resolved_llm = build_hypothesis_llm()
        turn = manager.run_results_companion_turn(run_id, resolved_llm.call, history)
    except Exception as exc:  # noqa: BLE001 - surfaced in the panel, not swallowed
        status.update(label="Companion unavailable", state="error")
        st.session_state[_companion_error_key(run_id)] = str(exc)
        return False
    status.update(label="Companion replied", state="complete")
    st.session_state.pop(_companion_error_key(run_id), None)
    if turn.drafted_narrative is not None:
        store.append(
            ConversationEntry(
                role="assistant",
                kind="drafted_narrative",
                phase=_RESULTS_DIALOGUE_PHASE,
                text=turn.reply,
                data={
                    "narrative": turn.drafted_narrative.narrative_text,
                    "claims": [claim.to_dict() for claim in turn.drafted_narrative.claims],
                    "provider": resolved_llm.provider,
                    "model": resolved_llm.model,
                },
            )
        )
    else:
        store.append(
            ConversationEntry(
                role="assistant", kind="message", phase=_RESULTS_DIALOGUE_PHASE, text=turn.reply
            )
        )
    return True


def _render_drafted_narrative_card(
    manager: WorkflowManager, run_id: str, entry: ConversationEntry
) -> None:
    data = entry.data or {}
    narrative_text = data.get("narrative", "")
    claims = list(data.get("claims") or ())
    with st.container(border=True):
        st.markdown(":material/description: **Drafted interpretation — not yet saved**")
        st.markdown(narrative_text or "_(empty draft)_")
        if claims:
            with st.expander(f"{len(claims)} checkable claim(s)"):
                st.dataframe(
                    [
                        {
                            "Claim": claim.get("text", ""),
                            "Type": claim.get("claim_type", ""),
                            "Source fields": ", ".join(claim.get("source_fields", [])),
                        }
                        for claim in claims
                    ],
                    hide_index=True,
                    width="stretch",
                )
        st.caption(
            "Saving runs the deterministic faithfulness check and records this as the run's "
            "interpretation. It still needs your approval under **Estimation & report** before "
            "it enters the causal report."
        )
        save_col, dismiss_col = st.columns(2)
        save = save_col.button(
            "Save for review", type="primary", key=f"draft_save_{entry.id}", width="stretch"
        )
        dismiss = dismiss_col.button("Dismiss", key=f"draft_dismiss_{entry.id}", width="stretch")
        store = ConversationStore(manager.runs_dir / run_id)
        if save:
            try:
                manager.save_results_narrative_draft(
                    run_id,
                    {"narrative": narrative_text, "claims": claims},
                    results_dialogue_history(store.read()),
                    model=data.get("model", "unknown"),
                    provider=data.get("provider", "unknown"),
                )
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
                return
            store.append(
                ConversationEntry(
                    role="event",
                    kind="narrative_saved",
                    phase=_RESULTS_DIALOGUE_PHASE,
                    text="Interpretation saved — approve or exclude it under Estimation & report.",
                    data={"draft_id": entry.id},
                )
            )
            st.rerun()
        if dismiss:
            store.append(
                ConversationEntry(
                    role="event",
                    kind="narrative_dismissed",
                    phase=_RESULTS_DIALOGUE_PHASE,
                    text="Drafted interpretation dismissed.",
                    data={"draft_id": entry.id},
                )
            )
            st.rerun()


def _render_results_dialogue(manager: WorkflowManager, run_id: str, run_dir: Any) -> None:
    if not (run_dir / "estimation.json").is_file():
        _render_thread(ConversationStore(run_dir).read())
        st.info(
            "The interpretation dialogue opens once the estimator has run. Start it from "
            "**Estimation & report**.",
            icon=":material/query_stats:",
        )
        return

    store = ConversationStore(run_dir)

    def _load() -> tuple[list[ConversationEntry], list[ConversationEntry]]:
        entries = store.read()
        return entries, [e for e in entries if e.phase == _RESULTS_DIALOGUE_PHASE]

    entries, dialogue_entries = _load()

    # Open the dialogue automatically on first arrival (once per session). On
    # success rerun for a clean render; on failure fall through to the warning
    # + one-step fallback below.
    opened_key = f"_results_opened_{run_id}"
    if not dialogue_entries and not st.session_state.get(opened_key):
        st.session_state[opened_key] = True
        if _run_results_companion_reply(manager, run_id, store):
            st.rerun()
        entries, dialogue_entries = _load()

    st.caption(
        "Interpret the finished estimate with the study companion. Nothing enters the report "
        "until you save a draft and approve it."
    )
    _render_thread(dialogue_entries, empty_message=_RESULTS_EMPTY_MESSAGE)

    error = st.session_state.get(_companion_error_key(run_id))
    if error:
        st.warning(
            f"The companion is unavailable ({error}). Use **Estimation & report** to draft "
            "the interpretation in one step instead."
        )
        if st.button("Retry the companion", icon=":material/refresh:", key=f"results_retry_{run_id}"):
            st.session_state.pop(_companion_error_key(run_id), None)
            _run_results_companion_reply(manager, run_id, store)
            st.rerun()

    pending = latest_drafted_narrative(entries)
    if pending is not None:
        _render_drafted_narrative_card(manager, run_id, pending)

    submitted = st.chat_input("Ask about the estimate, or say you're ready…", key=f"results_chat_{run_id}")
    if submitted and submitted.strip():
        store.append(
            ConversationEntry(
                role="user", kind="message", phase=_RESULTS_DIALOGUE_PHASE, text=submitted.strip()
            )
        )
        _run_results_companion_reply(manager, run_id, store)
        st.rerun()


def _render_study_progress(state: RunState, run_dir: Any) -> None:
    """Compact "where is this study" view at the top of the conversation
    panel: phase progress, next action, the reviewed question and P/T/C/O,
    gate count, and available evidence. This is what the old "Study summary"
    workspace tab showed, sized for the narrow left column.

    `causal.ui.streamlit_app`'s helpers are reached through a lazy import:
    that module imports this one, so a top-level import here would be a
    cycle, but by the time this function runs both are fully loaded.
    """
    from causal.orchestrator.state import PHASES
    from causal.ui import streamlit_app as app

    protocol = app._read_json(run_dir / "protocol.json") or {}
    target_trial = protocol.get("target_trial") or {}
    estimation = app._read_json(run_dir / "estimation.json") or {}
    diagnostics = app._read_json(run_dir / "diagnostics.json") or {}
    sensitivity = app._read_json(run_dir / "sensitivity_analysis.json") or {}
    narrative_raw = app._read_json(run_dir / "results_narrative.json")

    index = PHASES.index(state.phase) if state.phase in PHASES else 0
    phase_label = app._PHASE_LABELS.get(state.phase, state.phase)
    approved = sum(gate.status == "approved" for gate in state.gates.values())
    rejected = [
        app._GATE_INFO.get(name, {}).get("title", name)
        for name, gate in state.gates.items()
        if gate.status == "rejected"
    ]

    with st.container(border=True):
        st.progress((index + 1) / len(PHASES), text=f"{phase_label} · step {index + 1} of {len(PHASES)}")
        if rejected:
            st.warning(f"Blocked: {', '.join(rejected)} need resolution.", icon=":material/warning:")
        else:
            st.caption(f"Next: {app._next_action_text(state)}")

        question = app._source_question(run_dir)
        if question:
            st.caption("Clinical question")
            st.write(question)

        if target_trial:
            st.markdown(
                f"**P** {target_trial.get('population', '—')}  \n"
                f"**T** {target_trial.get('treatment', '—')}  \n"
                f"**C** {target_trial.get('comparator', '—')}  \n"
                f"**O** {target_trial.get('outcome', '—')}"
            )

        left, right = st.columns(2)
        left.metric("Gates approved", f"{approved} / {len(state.gates)}")
        estimand = estimation.get("estimand") or {}
        if estimand:
            right.metric(
                app._humanize(estimand.get("effect_measure")),
                f"{estimand.get('point_estimate', 0):.3f}",
            )
        else:
            right.metric("Estimate", "—")

        with st.expander("Workflow & evidence", icon=":material/account_tree:"):
            for position, phase in enumerate(PHASES):
                mark = ":material/check_circle:" if position < index else (
                    ":material/play_circle:" if position == index else ":material/lock:"
                )
                st.markdown(f"{mark} {app._PHASE_LABELS.get(phase, phase)}")
            st.divider()
            flagged = sum(1 for entry in diagnostics.get("balance", ()) if entry.get("flagged"))
            narrative = normalize_narrative_artifact(narrative_raw) if narrative_raw else None
            st.caption(
                f"Diagnostics: {f'{flagged} balance flag(s)' if diagnostics else 'not run'} · "
                f"Sensitivity: {'complete' if sensitivity else 'not run'} · "
                f"Interpretation: {(narrative or {}).get('review_status', 'not generated')}"
            )


def render_conversation_panel(manager: WorkflowManager, run_id: str | None) -> None:
    """Entry point for `main()`'s left column."""
    st.subheader("Conversation", anchor=False)

    if run_id is not None:
        state: RunState = manager.load(run_id)
        run_dir = manager.runs_dir / run_id
        _render_study_progress(state, run_dir)
        st.divider()
        if state.phase == _ESTIMAND_PHASE:
            _render_estimand_dialogue(manager, run_id)
        elif state.phase in _RESULTS_PHASES:
            _render_results_dialogue(manager, run_id, run_dir)
        else:
            _render_thread(ConversationStore(run_dir).read())
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
