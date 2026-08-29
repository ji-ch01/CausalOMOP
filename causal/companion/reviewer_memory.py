"""Cross-study reviewer memory -- the fourth and outermost memory layer.

`causal.companion.decision_memory` remembers what a reviewer decided *in one
run*. This layer is opt-in and carries a reviewer's conventions *across*
studies: which covariates they have excluded before (and why), and which
statistical-estimand strategy they have settled on. It exists so the estimand
dialogue can say "you excluded `baseline_lactate` as post-treatment in
`aspirin_ami` -- the same reasoning may apply here" instead of the reviewer
re-deriving it every study.

It is advisory only, exactly like the per-run decision memory: it feeds a
prompt, never a gate, never a covariate list, and it holds only
decision-level text (labels, rationales, strategy names) -- never a
patient-level value. It is populated only when a human explicitly asks
(`WorkflowManager.remember_run_for_reviewer`), never automatically.

Stored at `artifacts/reviewers/<slug>/memory.json`, one file per reviewer
id. `merge_run_into_memory` is the pure fold of one run's decisions into an
existing `ReviewerMemory`; `WorkflowManager` owns the file I/O around it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from causal.companion.decision_memory import DecisionMemory

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify_reviewer(reviewer: str) -> str:
    """`"Dr. Chen"` -> `"dr_chen"`. Raises if nothing usable remains -- a
    reviewer id is required to key the store, and two different reviewers
    whose names slug to the same value share a file (documented, acceptable
    for a single-team tool)."""
    slug = _SLUG_RE.sub("_", (reviewer or "").strip().lower()).strip("_")
    if not slug:
        raise ValueError("A non-empty reviewer id is required for reviewer memory")
    return slug


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RememberedExclusion:
    """A covariate this reviewer excluded in some prior study."""

    label: str
    reason: str
    study_id: str
    decided_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "label": self.label,
            "reason": self.reason,
            "study_id": self.study_id,
            "decided_at": self.decided_at,
        }


@dataclass(frozen=True)
class RememberedEstimandChoice:
    """The statistical-estimand strategy this reviewer settled on for a prior
    study."""

    study_id: str
    missing_data_strategy: str
    censoring_strategy: str
    estimation_procedure: str
    decided_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "study_id": self.study_id,
            "missing_data_strategy": self.missing_data_strategy,
            "censoring_strategy": self.censoring_strategy,
            "estimation_procedure": self.estimation_procedure,
            "decided_at": self.decided_at,
        }


@dataclass(frozen=True)
class ReviewerMemory:
    """One reviewer's cross-study conventions."""

    reviewer_id: str = ""
    exclusions: tuple[RememberedExclusion, ...] = ()
    estimand_choices: tuple[RememberedEstimandChoice, ...] = ()
    updated_at: str = field(default_factory=_utc_now_iso)

    def is_empty(self) -> bool:
        return not self.exclusions and not self.estimand_choices

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "reviewer_id": self.reviewer_id,
            "updated_at": self.updated_at,
            "exclusions": [item.to_dict() for item in self.exclusions],
            "estimand_choices": [item.to_dict() for item in self.estimand_choices],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReviewerMemory":
        return cls(
            reviewer_id=str(value.get("reviewer_id", "")),
            exclusions=tuple(
                RememberedExclusion(
                    label=str(item.get("label", "")),
                    reason=str(item.get("reason", "")),
                    study_id=str(item.get("study_id", "")),
                    decided_at=str(item.get("decided_at", "")),
                )
                for item in value.get("exclusions", ())
            ),
            estimand_choices=tuple(
                RememberedEstimandChoice(
                    study_id=str(item.get("study_id", "")),
                    missing_data_strategy=str(item.get("missing_data_strategy", "")),
                    censoring_strategy=str(item.get("censoring_strategy", "")),
                    estimation_procedure=str(item.get("estimation_procedure", "")),
                    decided_at=str(item.get("decided_at", "")),
                )
                for item in value.get("estimand_choices", ())
            ),
            updated_at=str(value.get("updated_at", "")) or _utc_now_iso(),
        )

    def format_for_prompt(self) -> str:
        """The prose block the estimand dialogue prompt embeds. Deterministic
        and side-effect-free."""
        if self.is_empty():
            return "No cross-study conventions recorded for this reviewer yet."
        lines: list[str] = []
        if self.exclusions:
            lines.append("Covariates you have excluded in earlier studies:")
            for item in self.exclusions:
                reason = item.reason.strip() or "no rationale recorded"
                where = " / ".join(p for p in (item.study_id, _date_only(item.decided_at)) if p)
                lines.append(f'- {item.label}: "{reason}"' + (f" ({where})" if where else ""))
        if self.estimand_choices:
            if lines:
                lines.append("")
            lines.append("Statistical-estimand strategies you have chosen before:")
            for item in self.estimand_choices:
                where = " / ".join(p for p in (item.study_id, _date_only(item.decided_at)) if p)
                lines.append(
                    f"- {item.missing_data_strategy} · {item.censoring_strategy} · "
                    f"{item.estimation_procedure}" + (f" ({where})" if where else "")
                )
        return "\n".join(lines)


def _date_only(iso_timestamp: str) -> str:
    text = (iso_timestamp or "").strip()
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    return text


def merge_run_into_memory(
    memory: ReviewerMemory,
    *,
    study_id: str,
    decision_memory: DecisionMemory,
    estimand_settings: Mapping[str, Any] | None = None,
    now: str | None = None,
) -> ReviewerMemory:
    """Fold one run's decisions into a reviewer's memory.

    Pure. Deduplicates: an exclusion is keyed by `(study_id, label)` and a
    strategy by `study_id`, so re-running this for the same study replaces
    that study's entries rather than appending. Entries from *other* studies
    are untouched. `estimand_settings` is `protocol.json`'s
    `statistical_estimand_settings` (or `None`/`{}` if the reviewer has not
    set one yet -- then no strategy is recorded for this study).
    """
    stamp = now or _utc_now_iso()

    exclusions = [item for item in memory.exclusions if item.study_id != study_id]
    for rejected in decision_memory.rejected_covariates:
        exclusions.append(
            RememberedExclusion(
                label=rejected.label,
                reason=rejected.comment,
                study_id=study_id,
                decided_at=rejected.decided_at or stamp,
            )
        )

    choices = [item for item in memory.estimand_choices if item.study_id != study_id]
    settings = estimand_settings or {}
    if all(
        settings.get(key)
        for key in ("missing_data_strategy", "censoring_strategy", "estimation_procedure")
    ):
        choices.append(
            RememberedEstimandChoice(
                study_id=study_id,
                missing_data_strategy=str(settings["missing_data_strategy"]),
                censoring_strategy=str(settings["censoring_strategy"]),
                estimation_procedure=str(settings["estimation_procedure"]),
                decided_at=str(settings.get("decided_at") or stamp),
            )
        )

    exclusions.sort(key=lambda item: (item.decided_at == "", item.decided_at, item.label))
    choices.sort(key=lambda item: (item.decided_at == "", item.decided_at, item.study_id))

    return ReviewerMemory(
        reviewer_id=memory.reviewer_id,
        exclusions=tuple(exclusions),
        estimand_choices=tuple(choices),
        updated_at=stamp,
    )
