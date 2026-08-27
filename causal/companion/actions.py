"""Allow-list for the writes a study companion may propose.

The companion (`causal.companion.companion`) can end a turn with a
`proposed_action` -- a `WorkflowManager` write it wants a human to make. This
module is the gate between that free-text LLM output and any real call: it
accepts an action only if its name is on a fixed allow-list and every
argument is one of a known set of values. The companion can never propose
`run_id` or `reviewer` (the UI supplies those at confirm time) or any method
not listed here.

Nothing in this module calls `WorkflowManager` or an LLM. It validates a
shape; the UI renders the result as a card and, on an explicit human click,
calls the named method with the validated args plus `run_id`/`reviewer`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from causal.roadmap import CENSORING_STRATEGIES, ESTIMATION_PROCEDURES, MISSING_DATA_STRATEGIES

#: Field name -> the set of values the companion may propose for it. Every
#: enum-valued argument of every allow-listed action must appear here.
_ENUM_FIELDS: dict[str, frozenset[str]] = {
    "missing_data_strategy": frozenset(MISSING_DATA_STRATEGIES),
    "censoring_strategy": frozenset(CENSORING_STRATEGIES),
    "estimation_procedure": frozenset(ESTIMATION_PROCEDURES),
}

_MAX_RATIONALE_CHARS = 2000


@dataclass(frozen=True)
class _ActionSpec:
    enum_fields: tuple[str, ...]
    text_fields: tuple[str, ...]


#: The only writes a companion may propose. Widening what the companion can
#: do to a run means adding an entry here (and a `_summarize` branch) -- there
#: is no other path.
_ALLOWED_ACTIONS: dict[str, _ActionSpec] = {
    "set_statistical_estimand_strategy": _ActionSpec(
        enum_fields=("missing_data_strategy", "censoring_strategy", "estimation_procedure"),
        text_fields=("censoring_rationale",),
    ),
}

ALLOWED_ACTION_NAMES = frozenset(_ALLOWED_ACTIONS)


class ActionValidationError(ValueError):
    """A `proposed_action` was present but malformed or not allow-listed."""


@dataclass(frozen=True)
class ProposedAction:
    """A validated companion-proposed `WorkflowManager` write.

    `action` is the method name; `args` are the LLM-supplied keyword
    arguments, each already checked. The UI adds `run_id` and `reviewer` when
    the human confirms -- those are never in `args`. `summary` is a one-line
    description for the card.
    """

    action: str
    args: dict[str, Any]
    summary: str


def validate_proposed_action(raw: Any) -> ProposedAction | None:
    """Turn a turn's raw `proposed_action` field into a `ProposedAction`.

    Returns `None` when there is no action this turn (`raw` is `None` or an
    empty dict). Raises `ActionValidationError` when `raw` is present but is
    not a well-formed, allow-listed action with every argument in range --
    including the security-relevant cases of an unknown action name or an
    unexpected argument such as `reviewer` or `run_id`.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ActionValidationError("proposed_action must be an object")
    if not raw:
        return None

    name = raw.get("action")
    if name not in _ALLOWED_ACTIONS:
        raise ActionValidationError(
            f"action must be one of {sorted(_ALLOWED_ACTIONS)}, got {name!r}"
        )
    spec = _ALLOWED_ACTIONS[name]

    raw_args = raw.get("args")
    if not isinstance(raw_args, dict):
        raise ActionValidationError("proposed_action.args must be an object")

    allowed = set(spec.enum_fields) | set(spec.text_fields)
    unexpected = set(raw_args) - allowed
    if unexpected:
        raise ActionValidationError(
            f"unexpected arg(s) for {name}: {sorted(unexpected)} (the companion may not set these)"
        )

    args: dict[str, Any] = {}
    for field in spec.enum_fields:
        value = raw_args.get(field)
        if value not in _ENUM_FIELDS[field]:
            raise ActionValidationError(
                f"{field} must be one of {sorted(_ENUM_FIELDS[field])}, got {value!r}"
            )
        args[field] = value
    for field in spec.text_fields:
        value = raw_args.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ActionValidationError(f"{field} must be non-empty text")
        if len(value) > _MAX_RATIONALE_CHARS:
            raise ActionValidationError(
                f"{field} must be at most {_MAX_RATIONALE_CHARS} characters"
            )
        args[field] = value.strip()

    return ProposedAction(action=name, args=args, summary=_summarize(name, args))


def _summarize(name: str, args: dict[str, Any]) -> str:
    if name == "set_statistical_estimand_strategy":
        return "Save statistical-estimand strategy: " + " · ".join(
            (
                args["missing_data_strategy"],
                args["censoring_strategy"],
                args["estimation_procedure"],
            )
        )
    return name  # pragma: no cover - every allow-listed action has a branch above
