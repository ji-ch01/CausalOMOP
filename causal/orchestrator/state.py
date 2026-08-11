"""Serializable workflow state and transition rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


GateStatus = Literal["pending", "approved", "rejected", "provisional"]

PHASES = (
    # The first three phases mirror docs/causal_roadmap.md's proposed
    # governance split of the earlier combined "protocol_review" phase --
    # question/target-trial, DAG/clinical completeness, and identification/
    # estimand are three separate scientific review decisions, each with its
    # own gate (see REQUIRED_GATES in causal.orchestrator.workflow).
    "target_trial_review",
    "dag_review",
    "identification_review",
    "concept_mapping",
    "feasibility",
    "analysis_plan",
    "execution",
    "diagnostics",
    "results_review",
    "report",
)

NEXT_PHASE = dict(zip(PHASES, PHASES[1:]))


@dataclass
class ApprovalGate:
    name: str
    status: GateStatus = "pending"
    reviewer: str | None = None
    comment: str | None = None
    decided_at: str | None = None

    def decide(self, status: GateStatus, reviewer: str, comment: str | None = None) -> None:
        if status == "pending":
            raise ValueError("A gate decision cannot reset status to pending")
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        self.status = status
        self.reviewer = reviewer
        self.comment = comment
        self.decided_at = datetime.now(timezone.utc).isoformat()


@dataclass
class RunState:
    run_id: str
    study_id: str
    protocol_path: str
    phase: str = "target_trial_review"
    mode: Literal["exploratory", "final"] = "exploratory"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    gates: dict[str, ApprovalGate] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "RunState":
        value = dict(value)
        value["gates"] = {name: ApprovalGate(**gate) for name, gate in value.get("gates", {}).items()}
        return cls(**value)

    def record(self, event: str, detail: str, *, changes: dict[str, Any] | None = None) -> None:
        """Append one audit-history entry. `detail` stays the human-readable
        summary every event already had; `changes` (optional, `{"before":
        ..., "after": ...}`) lets a caller also record the prior/new value
        of whatever it mutated -- without it, `state.json` alone cannot
        reconstruct what a review actually changed, only that something did.
        Omitted (not stored as `None`) when a caller has nothing structured
        to add, so events that never carried this keep their old shape.
        """
        now = datetime.now(timezone.utc).isoformat()
        self.updated_at = now
        entry: dict[str, Any] = {"at": now, "event": event, "detail": detail}
        if changes is not None:
            entry["changes"] = changes
        self.history.append(entry)
