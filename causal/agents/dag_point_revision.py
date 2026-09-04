"""Turn a reviewer's free-text DAG correction points into structured edits.

Unlike `causal.agents.dag_critique` (the *literature-grounded* self-critique
loop, where a revised variable must still cite a retrieved PMID), this path
is driven by a human's correction points: the reviewer is the authority, so
an added or reclassified variable is expert-asserted (`pmid = ""`), exactly
as `WorkflowManager.review_covariate` already records a hand-added covariate.

The LLM here is a *formatter*, not an author: it maps each free-text point
("add baseline creatinine as a confounder", "vasopressors are a mediator, not
a confounder", "drop ICU LOS, it's a collider") onto one of three explicit
operations. It never invents a correction the reviewer did not ask for --
points it cannot map are returned in `unaddressed_points` for the reviewer to
rephrase, not silently dropped or guessed at.

No LLM call and no I/O beyond reading the prompt template happen here;
`WorkflowManager.revise_candidate_dag_from_points` owns the `llm(prompt)`
call and applies the parsed result. Every resulting `candidate_dag` still
goes through `causal.agents.hypothesis.HypothesisAgent._candidate_dag`
(role/temporal legality, cycles, window bounds) before anything is written.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causal.agents.hypothesis import (
    DAG_VARIABLE_ROLES,
    DAG_VARIABLE_TEMPORAL_CLASSES,
)
from causal.roadmap import EDGE_CERTAINTY

DEFAULT_DAG_POINT_REVISION_PROMPT_PATH = Path("causal/prompts/dag_point_revision.md")

_OPERATIONS = frozenset({"add", "reclassify", "remove"})
_MAX_OPERATIONS = 40
_MAX_RATIONALE_CHARS = 2000


@dataclass(frozen=True)
class DagPointOperation:
    op: str
    label: str
    rationale: str
    role: str | None = None
    temporal_classification: str | None = None
    observed: bool = True
    certainty: str = "established"


@dataclass(frozen=True)
class DagPointRevision:
    operations: tuple[DagPointOperation, ...] = ()
    acknowledged_errors: tuple[str, ...] = ()
    unaddressed_points: tuple[str, ...] = ()


def format_current_dag(variables: list[dict[str, Any]]) -> str:
    """The variables the reviewer is correcting, as compact bullet text for
    the prompt -- label, role, timing and a one-line rationale, nothing the
    model needs to preserve verbatim (`pmid`, concept ids, windows stay in
    `protocol.json` and are carried across untouched on a reclassify)."""
    if not variables:
        return "(the current candidate DAG has no variables yet)"
    lines = []
    for variable in variables:
        label = str(variable.get("label", "")).strip()
        if not label:
            continue
        role = variable.get("role", "?")
        timing = variable.get("temporal_classification", "?")
        rationale = str(variable.get("rationale", "")).strip()
        lines.append(f"- {label} [role={role}, timing={timing}] {rationale}")
    return "\n".join(lines) or "(the current candidate DAG has no variables yet)"


def build_dag_point_revision_prompt(
    template_path: Path,
    *,
    target: dict[str, Any],
    current_dag_text: str,
    points: list[str],
) -> str:
    """Render `causal/prompts/dag_point_revision.md`.

    Same required-placeholder rule as the other companion/agent prompts: a
    missing placeholder in the template raises rather than passing silently.
    """
    template = template_path.read_text(encoding="utf-8")
    points_text = "\n".join(f"{index}. {point}" for index, point in enumerate(points, start=1))
    required = {
        "{{POPULATION}}": str(target.get("population", "")),
        "{{TREATMENT}}": str(target.get("treatment", "")),
        "{{COMPARATOR}}": str(target.get("comparator", "")),
        "{{OUTCOME}}": str(target.get("outcome", "")),
        "{{CURRENT_DAG}}": current_dag_text,
        "{{CORRECTION_POINTS}}": points_text,
        "{{ROLES}}": ", ".join(DAG_VARIABLE_ROLES),
        "{{TEMPORAL_CLASSES}}": ", ".join(DAG_VARIABLE_TEMPORAL_CLASSES),
    }
    for placeholder, value in required.items():
        if placeholder not in template:
            raise ValueError(f"DAG point-revision prompt is missing {placeholder}")
        template = template.replace(placeholder, value)
    return template


def _clean_str_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())


def parse_dag_point_revision_response(raw: str) -> DagPointRevision:
    """Strictly validate the formatter LLM's response.

    Empty, non-JSON, non-object, or a non-list `operations` all raise
    `ValueError`. Individual operations that are malformed (bad `op`, missing
    `label`/`rationale`, or an `add`/`reclassify` without a valid `role` and
    `temporal_classification`) are dropped, not raised on -- the reviewer
    sees the survivors applied and any point the model fumbled comes back in
    `unaddressed_points`. A response whose `operations` list is entirely
    unusable raises, so the caller does not report "0 changes applied" as if
    that were a successful revision.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("LLM returned an empty DAG point-revision response")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("DAG point-revision response must be one valid JSON object") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("operations"), list):
        raise ValueError("DAG point-revision response must contain an operations list")

    operations: list[DagPointOperation] = []
    for item in payload["operations"][:_MAX_OPERATIONS]:
        if not isinstance(item, dict):
            continue
        op = item.get("op")
        label = item.get("label")
        rationale = item.get("rationale")
        if op not in _OPERATIONS:
            continue
        if not isinstance(label, str) or not label.strip():
            continue
        if not isinstance(rationale, str) or not rationale.strip():
            continue
        rationale = rationale.strip()[:_MAX_RATIONALE_CHARS]
        if op == "remove":
            operations.append(DagPointOperation(op="remove", label=label.strip(), rationale=rationale))
            continue
        role = item.get("role")
        temporal = item.get("temporal_classification")
        if role not in DAG_VARIABLE_ROLES or temporal not in DAG_VARIABLE_TEMPORAL_CLASSES:
            continue
        observed = item.get("observed", True)
        certainty = item.get("certainty", "established")
        operations.append(
            DagPointOperation(
                op=op,
                label=label.strip(),
                rationale=rationale,
                role=role,
                temporal_classification=temporal,
                observed=observed if isinstance(observed, bool) else True,
                certainty=certainty if certainty in EDGE_CERTAINTY else "established",
            )
        )

    if payload["operations"] and not operations:
        raise ValueError(
            "DAG point-revision response proposed only malformed operations; nothing to apply"
        )
    return DagPointRevision(
        operations=tuple(operations),
        acknowledged_errors=_clean_str_list(payload.get("acknowledged_errors")),
        unaddressed_points=_clean_str_list(payload.get("unaddressed_points")),
    )


_CARRIED_ON_RECLASSIFY = (
    "pmid", "kg_evidence_id", "evidence_level", "domain", "concept_ids",
    "mapping_status", "aggregation", "window_start_hours", "window_end_hours",
)


def apply_dag_point_operations(
    protocol: dict[str, Any], revision: DagPointRevision, reviewer: str, *, now: str
) -> tuple[dict[str, Any], list[str]]:
    """Apply `revision` to a copy of `protocol` and return `(new_protocol,
    notes)`.

    Pure: never mutates the argument, never touches disk, never validates the
    resulting graph (the caller runs `HypothesisAgent` for that and only
    persists if it passes). `add` for an already-present label is treated as
    `reclassify`; `reclassify`/`remove` for an unknown label is recorded in
    `notes` and skipped. A reclassify preserves any concept-mapping/window
    work already recorded for that label (`_CARRIED_ON_RECLASSIFY`).
    """
    updated = json.loads(json.dumps(protocol))
    candidate_dag: list[dict[str, Any]] = list(updated.get("candidate_dag", []) or [])
    rejected: list[dict[str, Any]] = list(updated.get("rejected_covariates", []) or [])
    by_label = {str(entry.get("label", "")).strip().lower(): entry for entry in candidate_dag}
    notes: list[str] = []

    for operation in revision.operations:
        key = operation.label.strip().lower()
        existing = by_label.get(key)
        if operation.op == "remove":
            if existing is None:
                notes.append(f"remove: '{operation.label}' is not in the candidate DAG -- skipped")
                continue
            candidate_dag = [e for e in candidate_dag if e is not existing]
            by_label.pop(key, None)
            rejected = [r for r in rejected if str(r.get("label", "")).strip().lower() != key]
            rejected.append(
                {
                    "label": operation.label.strip(),
                    "reviewer": reviewer,
                    "comment": operation.rationale,
                    "decided_at": now,
                }
            )
            continue

        # add / reclassify
        new_entry: dict[str, Any] = {
            "label": operation.label.strip(),
            "role": operation.role,
            "temporal_classification": operation.temporal_classification,
            "rationale": operation.rationale,
            "pmid": "",
            "observed": operation.observed,
            "certainty": operation.certainty,
            "evidence_level": "uncited",
        }
        if existing is not None:
            for carried in _CARRIED_ON_RECLASSIFY:
                if carried in existing:
                    new_entry[carried] = existing[carried]
            candidate_dag = [new_entry if e is existing else e for e in candidate_dag]
        else:
            candidate_dag.append(new_entry)
        by_label[key] = new_entry
        # A label being corrected back into the DAG is no longer an exclusion.
        rejected = [r for r in rejected if str(r.get("label", "")).strip().lower() != key]

    kept_labels = {str(e.get("label", "")).strip() for e in candidate_dag}
    relations = [
        relation
        for relation in updated.get("covariate_relations", []) or []
        if str(relation.get("source", "")).strip() in kept_labels
        and str(relation.get("target", "")).strip() in kept_labels
    ]

    updated["candidate_dag"] = candidate_dag
    updated["rejected_covariates"] = rejected
    updated["covariate_relations"] = relations
    return updated, notes
