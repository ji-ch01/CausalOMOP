"""Distil a run's reviewer decisions into a compact, replayable summary.

The study companion (`causal.companion.companion`) takes one dialogue turn at
a time and is re-fed only the natural-language transcript. Without this it has
no memory of what the reviewer already *decided* -- which covariates they
excluded, which gate they rejected and why, which censoring strategy they
already settled -- and will happily re-propose a covariate the reviewer threw
out three turns ago.

`DecisionMemory` is that missing layer: a structured, JSON-safe digest of

- `hypothesis.json` / `protocol.json` `rejected_covariates` (the audited
  covariate exclusions -- this field already exists and nothing ever reads it
  back), and
- the decision-shaped entries of `RunState.history` (covariate/mapping/
  estimand/scope reviews, gate invalidations) plus any `rejected`
  `ApprovalGate` with its review comment,

merged into one timeline. It is produced by a pure, deterministic distillation
(`distill_decision_memory`) -- no LLM, in the same spirit as
`causal.agents.results_faithfulness.verify_narrative` -- and rendered into the
companion prompt by `format_for_prompt`. It never decides anything; it only
tells the companion what the human has already ruled on.

`causal.orchestrator.workflow.WorkflowManager` owns the run directory and
adapts `RunState` to the plain-data inputs here, exactly as it does for
`EstimandDialogueContext`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

#: `RunState.history` event name -> the decision category it becomes. Only
#: events that record a *reviewer* choice are here; progress markers
#: (`agent_completed`, `*_completed`, `run_started`), LLM suggestions
#: (`estimation_procedure_proposed`, `negative_controls_proposed`), and the
#: search-hint-only `concept_set_lookup_terms_set` are deliberately left out
#: so the digest stays a list of things the human actually settled.
_DECISION_EVENTS: dict[str, str] = {
    "covariate_reviewed": "covariate",
    "covariate_concepts_mapped": "mapping",
    "covariate_aggregation_reviewed": "covariate",
    "covariate_window_reviewed": "covariate",
    "covariate_domain_reviewed": "covariate",
    "concept_set_mapped": "mapping",
    "negative_control_mapped": "mapping",
    "target_trial_estimand_corrected": "estimand",
    "exposure_contrast_set": "scope",
    "washout_window_set": "scope",
    "mortality_ascertainment_set": "estimand",
    "censoring_definition_approved": "estimand",
    "statistical_estimand_strategy_set": "estimand",
    "cate_cutpoint_set": "estimand",
    "population_concept_set_added": "scope",
    "population_concept_set_removed": "scope",
    # A reviewer edit that reset an already-approved gate: the companion
    # should know that stage is back in play, not treat it as settled.
    "gate_invalidated": "gate",
}

#: Gate statuses worth replaying from the gate objects. A routine `approved`
#: does not change how the companion should behave (the approved content is
#: already in the prompt); a `rejected` gate is the reviewer's standing
#: objection and must be replayed. `provisional` is transient. Invalidations
#: come through `_DECISION_EVENTS` (`gate_invalidated`), not from here.
_REPLAYED_GATE_STATUSES = frozenset({"rejected"})

#: `covariate_reviewed` fires for both accept and reject; a reject is already
#: carried authoritatively by `rejected_covariates`, so its history line is
#: dropped to avoid listing the same exclusion twice.
_REJECT_MARKER = "=reject by "


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RejectedCovariate:
    """One audited covariate exclusion, as stored in `protocol.json`'s
    `rejected_covariates`. The companion must not re-propose these."""

    label: str
    reviewer: str
    comment: str
    decided_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "label": self.label,
            "reviewer": self.reviewer,
            "comment": self.comment,
            "decided_at": self.decided_at,
        }


@dataclass(frozen=True)
class RecordedDecision:
    """One reviewer decision distilled from history or a decided gate.

    `at` is the ISO timestamp it happened; `category` is one of `gate`,
    `covariate`, `mapping`, `estimand`, `scope`; `summary` is the
    human-readable one-liner (the history `detail`, or a gate line built
    with its review comment).
    """

    at: str
    category: str
    summary: str

    def to_dict(self) -> dict[str, str]:
        return {"at": self.at, "category": self.category, "summary": self.summary}


@dataclass(frozen=True)
class DecisionMemory:
    """A run's reviewer decisions, distilled for replay to the companion."""

    rejected_covariates: tuple[RejectedCovariate, ...] = ()
    decisions: tuple[RecordedDecision, ...] = ()
    generated_at: str = field(default_factory=_utc_now_iso)

    def is_empty(self) -> bool:
        return not self.rejected_covariates and not self.decisions

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "generated_at": self.generated_at,
            "rejected_covariates": [item.to_dict() for item in self.rejected_covariates],
            "decisions": [item.to_dict() for item in self.decisions],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DecisionMemory":
        return cls(
            rejected_covariates=tuple(
                RejectedCovariate(
                    label=str(item.get("label", "")),
                    reviewer=str(item.get("reviewer", "")),
                    comment=str(item.get("comment", "")),
                    decided_at=str(item.get("decided_at", "")),
                )
                for item in value.get("rejected_covariates", ())
            ),
            decisions=tuple(
                RecordedDecision(
                    at=str(item.get("at", "")),
                    category=str(item.get("category", "")),
                    summary=str(item.get("summary", "")),
                )
                for item in value.get("decisions", ())
            ),
            generated_at=str(value.get("generated_at", "")) or _utc_now_iso(),
        )

    def format_for_prompt(self) -> str:
        """Render the digest as the prose block a dialogue prompt embeds.

        Deterministic and side-effect-free -- the same digest always renders
        the same string, so a prompt-hash check stays meaningful.
        """
        if self.is_empty():
            return "No reviewer decisions have been recorded for this run yet."

        lines: list[str] = []
        if self.rejected_covariates:
            lines.append("Covariates the reviewer has excluded:")
            for item in self.rejected_covariates:
                reason = item.comment.strip() or "no rationale recorded"
                who = " / ".join(part for part in (item.reviewer, _date_only(item.decided_at)) if part)
                suffix = f" ({who})" if who else ""
                lines.append(f'- {item.label}: "{reason}"{suffix}')

        if self.decisions:
            if lines:
                lines.append("")
            lines.append("Decisions already recorded this run:")
            for item in self.decisions:
                stamp = _date_only(item.at)
                prefix = f"[{item.category}] " if item.category else ""
                stamp_suffix = f" ({stamp})" if stamp else ""
                lines.append(f"- {prefix}{item.summary}{stamp_suffix}")

        return "\n".join(lines)


def _date_only(iso_timestamp: str) -> str:
    """`2026-08-20T14:03:11+00:00` -> `2026-08-20`; anything unparseable is
    passed straight through so a hand-written fixture still renders."""
    text = (iso_timestamp or "").strip()
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    return text


def _gate_summary(name: str, gate: Mapping[str, Any]) -> str:
    status = str(gate.get("status", "")).strip()
    reviewer = str(gate.get("reviewer", "") or "").strip()
    comment = str(gate.get("comment", "") or "").strip()
    line = f"{name} {status}"
    if reviewer:
        line += f" by {reviewer}"
    if comment:
        line += f' — "{comment}"'
    return line


def distill_decision_memory(
    *,
    history: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Mapping[str, Any]] | None = None,
    rejected_covariates: Sequence[Mapping[str, Any]] | None = None,
    generated_at: str | None = None,
) -> DecisionMemory:
    """Build a `DecisionMemory` from a run's plain-data state.

    `history` is `RunState.history` (a list of `{at, event, detail}` dicts);
    `gates` is `{name: {status, reviewer, comment, decided_at}}` as
    `RunState.to_dict()` serialises them; `rejected_covariates` is
    `protocol.json`'s list of `{label, reviewer, comment, decided_at}`.

    Pure and deterministic: no I/O, no LLM, no clock read except the
    `generated_at` stamp (injectable for tests). Unknown history events are
    skipped, and from the gate objects only a `rejected` gate is replayed
    (`_REPLAYED_GATE_STATUSES`) -- a routine approval does not change how the
    companion should behave. Widening the digest means adding to
    `_DECISION_EVENTS`, nothing here guesses.
    """
    rejected = tuple(
        RejectedCovariate(
            label=str(item.get("label", "")).strip(),
            reviewer=str(item.get("reviewer", "")).strip(),
            comment=str(item.get("comment", "")).strip(),
            decided_at=str(item.get("decided_at", "")).strip(),
        )
        for item in (rejected_covariates or ())
        if str(item.get("label", "")).strip()
    )

    decisions: list[RecordedDecision] = []

    for entry in history:
        event = str(entry.get("event", ""))
        category = _DECISION_EVENTS.get(event)
        if category is None:
            continue
        detail = str(entry.get("detail", "")).strip()
        if not detail:
            continue
        if event == "covariate_reviewed" and _REJECT_MARKER in detail:
            continue  # already in rejected_covariates -- do not list it twice
        decisions.append(
            RecordedDecision(at=str(entry.get("at", "")), category=category, summary=detail)
        )

    for name, gate in (gates or {}).items():
        if str(gate.get("status", "")) not in _REPLAYED_GATE_STATUSES:
            continue
        decisions.append(
            RecordedDecision(
                at=str(gate.get("decided_at", "") or ""),
                category="gate",
                summary=_gate_summary(name, gate),
            )
        )

    decisions.sort(key=lambda item: (item.at == "", item.at))

    return DecisionMemory(
        rejected_covariates=rejected,
        decisions=tuple(decisions),
        generated_at=generated_at or _utc_now_iso(),
    )
