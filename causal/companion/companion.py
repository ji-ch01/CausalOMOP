"""Prompt-build and strict-parse for one study-companion turn.

Two conversational assists live here, both pure (no LLM call, no I/O beyond
reading the prompt template) -- like `causal.agents.question_gate` and
`causal.agents.estimation_procedure_suggestion`. The matching
`causal.orchestrator.workflow.WorkflowManager` method owns the `llm(prompt)`
call, gathers the artifacts, and persists the turn.

1. The **estimand dialogue** (`build_estimand_dialogue_prompt` /
   `parse_companion_turn`, driver `run_companion_turn`): at the
   `statistical_estimand` gate, talks the reviewer through the
   missing-data / censoring / estimator choices one at a time, and can end a
   turn by proposing the `set_statistical_estimand_strategy` write as a card
   the reviewer confirms. `proposed_action` is run through
   `causal.companion.actions.validate_proposed_action` here, so a turn that
   reaches the caller can only ever carry an allow-listed write with in-range
   arguments -- and the reviewer still has to click Confirm for it to happen.

2. The **results dialogue** (`build_results_dialogue_prompt` /
   `parse_results_companion_turn`, driver `run_results_companion_turn`): at
   `results_review` / `report`, talks the reviewer through interpreting the
   already-computed estimate, and can end a turn by carrying a
   `drafted_narrative` in exactly the `{narrative, claims}` shape the
   one-shot `causal.agents.results_narrative` step produces. Persisting that
   draft still runs the deterministic `causal.agents.results_faithfulness.
   verify_narrative` and still needs the separate human
   `review_results_narrative` approval before it can reach `causal_report.md`.

Neither turn ever approves a gate or advances a phase.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from causal.agents.results_narrative import (
    NarrativeInput,
    NarrativeResult,
    narrative_result_from_payload,
)
from causal.companion.actions import ProposedAction, validate_proposed_action
from causal.companion.decision_memory import DecisionMemory

DEFAULT_ESTIMAND_DIALOGUE_PROMPT_PATH = Path("causal/prompts/estimand_dialogue.md")
DEFAULT_RESULTS_DIALOGUE_PROMPT_PATH = Path("causal/prompts/results_dialogue.md")

_MESSAGE_ROLES = frozenset({"user", "assistant"})


@dataclass(frozen=True)
class CompanionMessage:
    """One prior turn replayed to the companion as context. `role` is the
    reviewer (`user`) or the companion (`assistant`); workflow `event` lines
    from the transcript are not passed here."""

    role: str
    text: str

    def __post_init__(self) -> None:
        if self.role not in _MESSAGE_ROLES:
            raise ValueError(f"CompanionMessage role must be one of {sorted(_MESSAGE_ROLES)}")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("CompanionMessage text must be non-empty")


@dataclass(frozen=True)
class EstimandDialogueContext:
    """JSON-safe grounding for an estimand-dialogue turn.

    Every artifact field is the raw dict as read from the run directory, or
    `{}` when that artifact is not on disk yet. `analytic_dataset.parquet` is
    never part of this -- the companion is fed the same class of artifacts as
    `causal.agents.results_narrative`.
    """

    study_specification: dict[str, Any] = field(default_factory=dict)
    current_settings: dict[str, Any] = field(default_factory=dict)
    cohort_manifest: dict[str, Any] = field(default_factory=dict)
    feasibility: dict[str, Any] = field(default_factory=dict)
    followup_audit: dict[str, Any] = field(default_factory=dict)
    positivity_audit: dict[str, Any] = field(default_factory=dict)
    covariate_missingness: dict[str, Any] = field(default_factory=dict)
    censoring_definition_approved: bool = False
    #: What the reviewer has already ruled on this run
    #: (`causal.companion.decision_memory`) -- excluded covariates and settled
    #: choices the companion must not silently re-propose.
    decision_memory: DecisionMemory = field(default_factory=DecisionMemory)


@dataclass(frozen=True)
class CompanionTurn:
    """The parsed result of one companion turn."""

    reply: str
    proposed_action: ProposedAction | None


def _counts_summary(artifact: dict[str, Any]) -> str:
    counts = artifact.get("counts") if isinstance(artifact, dict) else None
    return json.dumps(counts, indent=2) if counts else "Not yet run."


def _adjustment_set(cohort_manifest: dict[str, Any]) -> str:
    columns = cohort_manifest.get("columns") if isinstance(cohort_manifest, dict) else None
    if not isinstance(columns, list):
        return "Not available yet (build the cohort plan first)."
    names = [
        column.get("name")
        for column in columns
        if isinstance(column, dict) and column.get("role") == "baseline_covariate"
    ]
    return ", ".join(name for name in names if name) or "(no baseline covariates)"


def _primary_estimand(study_specification: dict[str, Any]) -> str:
    estimand = study_specification.get("estimand") if isinstance(study_specification, dict) else None
    if not isinstance(estimand, dict):
        return "Not available yet."
    summary = estimand.get("population_summary", "?")
    measure = estimand.get("effect_measure", "?")
    return f"{summary} on the {measure} scale"


def _format_history(history: Sequence[CompanionMessage]) -> str:
    if not history:
        return "(no messages yet -- open the dialogue)"
    speaker = {"user": "Reviewer", "assistant": "Companion"}
    return "\n\n".join(f"{speaker[message.role]}: {message.text}" for message in history)


def build_estimand_dialogue_prompt(
    template_path: Path,
    *,
    history: Sequence[CompanionMessage],
    context: EstimandDialogueContext,
) -> str:
    """Render `causal/prompts/estimand_dialogue.md`.

    Same required-placeholder substitution as
    `causal.agents.estimation_procedure_suggestion.
    build_estimation_procedure_suggestion_prompt`: a missing placeholder in
    the template raises rather than passing silently.
    """
    template = template_path.read_text(encoding="utf-8")
    required = {
        "{{CONVERSATION}}": _format_history(history),
        "{{PRIMARY_ESTIMAND}}": _primary_estimand(context.study_specification),
        "{{ADJUSTMENT_SET}}": _adjustment_set(context.cohort_manifest),
        "{{CENSORING_DEFINITION_APPROVED}}": "yes" if context.censoring_definition_approved else "no",
        "{{DECISION_MEMORY}}": context.decision_memory.format_for_prompt(),
        "{{CURRENT_SETTINGS}}": (
            json.dumps(context.current_settings, indent=2)
            if context.current_settings
            else "None recorded yet."
        ),
        "{{FEASIBILITY_SUMMARY}}": _counts_summary(context.feasibility),
        "{{FOLLOWUP_AUDIT_SUMMARY}}": _counts_summary(context.followup_audit),
        "{{POSITIVITY_AUDIT_SUMMARY}}": _counts_summary(context.positivity_audit),
        "{{COVARIATE_MISSINGNESS_SUMMARY}}": _counts_summary(context.covariate_missingness),
    }
    for placeholder, value in required.items():
        if placeholder not in template:
            raise ValueError(f"Estimand dialogue prompt is missing {placeholder}")
        template = template.replace(placeholder, value)
    return template


def parse_companion_turn(raw: str) -> CompanionTurn:
    """Structurally validate a companion turn.

    Empty, non-JSON, non-object, or missing/empty `reply` all raise
    `ValueError`. `proposed_action` (if present) is passed through
    `causal.companion.actions.validate_proposed_action`, whose
    `ActionValidationError` is itself a `ValueError` -- so a malformed or
    non-allow-listed action fails the parse rather than reaching the caller.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("LLM returned an empty companion response")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Companion response must be one valid JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("Companion response must be a JSON object")
    reply = payload.get("reply")
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError("Companion response requires a non-empty reply")
    return CompanionTurn(
        reply=reply.strip(),
        proposed_action=validate_proposed_action(payload.get("proposed_action")),
    )


# --------------------------------------------------------------------------- #
# Results-interpretation dialogue (phase results_review / report)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ResultsCompanionTurn:
    """The parsed result of one results-dialogue turn.

    `drafted_narrative` is `None` while the reviewer is still talking through
    the estimate; once they signal they are done the companion fills it with
    the same `{narrative, claims}` structure the one-shot
    `causal.agents.results_narrative` step produces, already validated by
    `causal.agents.results_narrative.narrative_result_from_payload`. It is a
    draft only -- persisting it runs `verify_narrative` and it still needs
    `WorkflowManager.review_results_narrative` approval to reach the report.
    """

    reply: str
    drafted_narrative: NarrativeResult | None


def build_results_dialogue_prompt(
    template_path: Path,
    *,
    history: Sequence[CompanionMessage],
    narrative_input: NarrativeInput,
) -> str:
    """Render `causal/prompts/results_dialogue.md`.

    The grounding is exactly `NarrativeInput` -- the same JSON-safe,
    patient-level-free bundle the one-shot narrative step is fed and that
    `verify_narrative` later checks the draft against -- serialized the same
    way `build_narrative_prompt` serializes it, so a claim the companion
    drafts can be verified against precisely what it was shown.

    Same required-placeholder rule as `build_estimand_dialogue_prompt`: a
    missing placeholder in the template raises rather than passing silently.
    """
    template = template_path.read_text(encoding="utf-8")
    required = {
        "{{CONVERSATION}}": _format_history(history),
        "{{RESULTS_JSON}}": json.dumps(narrative_input.to_dict(), indent=2, sort_keys=True),
    }
    for placeholder, value in required.items():
        if placeholder not in template:
            raise ValueError(f"Results dialogue prompt is missing {placeholder}")
        template = template.replace(placeholder, value)
    return template


def parse_results_companion_turn(raw: str) -> ResultsCompanionTurn:
    """Structurally validate one results-dialogue turn.

    Empty, non-JSON, non-object, or a missing/empty `reply` all raise
    `ValueError`. `drafted_narrative` (if present and non-empty) must be a
    `{narrative, claims}` object; it is passed through
    `causal.agents.results_narrative.narrative_result_from_payload`, whose
    `ValueError` propagates -- so a turn that reaches the caller with a
    non-null `drafted_narrative` always carries a structurally valid,
    deterministically checkable narrative.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("LLM returned an empty results-companion response")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Results-companion response must be one valid JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("Results-companion response must be a JSON object")
    reply = payload.get("reply")
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError("Results-companion response requires a non-empty reply")

    raw_draft = payload.get("drafted_narrative")
    drafted: NarrativeResult | None = None
    if raw_draft not in (None, {}):
        if not isinstance(raw_draft, dict):
            raise ValueError("drafted_narrative must be a JSON object")
        drafted = narrative_result_from_payload(raw_draft, context="drafted_narrative")

    return ResultsCompanionTurn(reply=reply.strip(), drafted_narrative=drafted)
