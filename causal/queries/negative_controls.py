"""Negative-control exposures/outcomes: a phenotype (population/exposure/
outcome-shaped) answering a different question than the study's own --
"does this exposure known to have no effect on the outcome still show an
association here?" (negative control exposure), or "does the real exposure
still show an association with an outcome it should have no effect on?"
(negative control outcome). A non-null result on either is a warning that
the adjustment set may not be blocking all confounding.

Structurally a negative control is just a substituted `ConceptSetSpec` fed
through `causal.queries.feasibility`'s existing cohort-building machinery
(`AnalyticCohortBuilder`/`analytic_cohort_sql`) -- this module adds no new
SQL-generation logic, only the substitution helper (`substitute_spec`) and
the candidate-proposal/review contract that gets a negative control from
"an LLM's clinical suggestion" to "an approved `OMOPVariableDefinition` a
query can be generated for."

Candidate proposal mirrors `causal.agents.dag_critique`'s pattern: pure
prompt-building/parsing helpers, no LLM call and no orchestration here
(`causal.orchestrator.workflow.WorkflowManager.propose_negative_controls`
owns the `llm(prompt)` call). `parse_negative_control_response` never
returns an approved candidate, no matter what the raw response claims --
the same "no LLM auto-approves a clinical mapping" invariant
`causal.mapping.registry._UNSUPERVISED_PROVENANCE` already enforces for
every other concept mapping in this codebase, reused here rather than
reinvented.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from causal.mapping.registry import MAPPING_STATUS_APPROVED, OMOPVariableDefinition, parse_variable
from causal.queries.feasibility import ConceptSetSpec

# The two causal roles a negative control resolves to (see module
# docstring). A phenotype's `causal_role` is free text (unlike a
# covariate's, checked against `causal.agents.hypothesis.DAG_VARIABLE_ROLES`
# -- see `causal.mapping.registry.parse_variable`), so no change to that
# validation is needed to introduce these two new values.
NEGATIVE_CONTROL_ROLES = ("negative_control_exposure", "negative_control_outcome")

_MAX_NEGATIVE_CONTROL_CANDIDATES = 6

DEFAULT_NEGATIVE_CONTROL_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "negative_control_proposal.md"


def substitute_spec(
    specs: dict[str, ConceptSetSpec], control_role: str, control_spec: ConceptSetSpec,
) -> dict[str, ConceptSetSpec]:
    """Return a copy of `specs` (the real study's resolved population/
    treatment/comparator/outcome, see `causal.orchestrator.workflow.
    _load_concept_specs`) with `treatment` or `outcome` replaced by
    `control_spec` -- the entire "SQL reuse" this module needs:
    `AnalyticCohortBuilder`/`analytic_cohort_sql` take population/treatment/
    comparator/outcome as plain keyword arguments, so a negative-control
    cohort is just a normal call with one of them swapped, not a new SQL
    path.

    `control_role` must be one of `NEGATIVE_CONTROL_ROLES`:
    `"negative_control_exposure"` substitutes `treatment` (does *this*
    exposure, believed unrelated to the outcome, still show an
    association with it?); `"negative_control_outcome"` substitutes
    `outcome` (does the real exposure still show an association with
    *this* outcome, believed unrelated to it?). `population`/`comparator`
    are always left unchanged -- a reviewer proposing a negative-control
    exposure is responsible for it making sense against the study's
    existing comparator, same as they are for the real treatment.
    """
    if control_role not in NEGATIVE_CONTROL_ROLES:
        raise ValueError(f"control_role must be one of {NEGATIVE_CONTROL_ROLES}")
    missing = [role for role in ("population", "treatment", "comparator", "outcome") if role not in specs]
    if missing:
        raise ValueError(f"specs is missing required role(s): {', '.join(missing)}")
    substituted_role = "treatment" if control_role == "negative_control_exposure" else "outcome"
    return {**specs, substituted_role: control_spec}


def variable_from_negative_control_proposal(proposal: dict[str, Any]) -> OMOPVariableDefinition:
    """Adapter: one negative-control proposal/review dict (from
    `parse_negative_control_response`, or `protocol.json["negative_
    controls"][control_id]` after a human has mapped/approved it) -> one
    registry entry, via the same validating constructor (`parse_variable`)
    every other registry adapter in `causal.mapping.registry` goes through.

    A negative control is `variable_kind="phenotype"` (it is ascertained by
    presence, exactly like population/treatment/comparator/outcome -- see
    `causal.mapping.registry.variable_from_concept_set`, which this
    mirrors) with `causal_role` one of `NEGATIVE_CONTROL_ROLES`.
    """
    if not isinstance(proposal, dict):
        raise ValueError("negative control proposal must be an object")
    control_role = proposal.get("control_type")
    if control_role not in NEGATIVE_CONTROL_ROLES:
        raise ValueError(f"control_type must be one of {NEGATIVE_CONTROL_ROLES}")
    label = proposal.get("label")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("negative control proposal requires a non-empty label")
    domain = proposal.get("domain") or ""
    mapping_status = proposal.get("mapping_status", "candidate_pending_human_approval")
    concept_ids = proposal.get("concept_ids", ())
    return parse_variable({
        "variable_id": proposal.get("control_id") or _slugify_label(label),
        "dag_node_label": label,
        "variable_kind": "phenotype",
        "causal_role": control_role,
        "observed": True,
        "domain": domain,
        "root_concept_ids": concept_ids if mapping_status == MAPPING_STATUS_APPROVED else (),
        "candidate_concept_ids": concept_ids if mapping_status != MAPPING_STATUS_APPROVED else (),
        "include_descendants": bool(proposal.get("include_descendants", True)),
        "excluded_concept_ids": proposal.get("excluded_concept_ids", ()),
        "aggregation": "presence_in_window",
        "mapping_status": mapping_status,
        "reviewer": proposal.get("reviewer", ""),
        "review_comment": proposal.get("rationale", "") or proposal.get("comment", ""),
        "provenance": proposal.get("provenance", "llm_proposed"),
    })


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify_label(label: str) -> str:
    slug = _SLUG_RE.sub("_", label.strip().lower()).strip("_")
    return slug or "negative_control"


def build_negative_control_prompt(
    template_path: Path, target: dict[str, Any],
) -> str:
    """Render `causal/prompts/negative_control_proposal.md`. Same required-
    placeholder substitution discipline as `causal.agents.dag_critique.
    build_critique_prompt`: a required placeholder missing from the
    template file is a hard `ValueError`."""
    template = template_path.read_text(encoding="utf-8")
    required = {
        "{{POPULATION}}": str(target.get("population", "")),
        "{{TREATMENT}}": str(target.get("treatment", "")),
        "{{COMPARATOR}}": str(target.get("comparator", "")),
        "{{OUTCOME}}": str(target.get("outcome", "")),
    }
    for placeholder, value in required.items():
        if placeholder not in template:
            raise ValueError(f"Negative control prompt is missing {placeholder}")
        template = template.replace(placeholder, value)
    return template


def parse_negative_control_response(raw: str) -> tuple[dict[str, Any], ...]:
    """Structurally validate an LLM's negative-control-candidate response.

    Same strictness as `causal.agents.dag_critique.parse_critique_response`:
    empty, non-JSON, non-object, or a missing/malformed `candidates` list
    all raise `ValueError`. Every returned candidate has `mapping_status`
    forced to `"candidate_pending_human_approval"` and `provenance` forced
    to `"llm_proposed"` -- *regardless of what the raw response claims* --
    so there is no code path through this function that can produce an
    already-approved candidate. `concept_ids`/`domain` are never read from
    the response either: an LLM proposes a clinical label and rationale
    only, the same "propose a label, map concepts separately" split
    `causal.agents.hypothesis`'s DAG-drafting step already uses for
    `candidate_dag` covariates.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("LLM returned an empty negative control response")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Negative control response must be one valid JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("Negative control response must be a JSON object")
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("Negative control response must set candidates to a list")

    candidates: list[dict[str, Any]] = []
    for item in raw_candidates:
        if len(candidates) >= _MAX_NEGATIVE_CONTROL_CANDIDATES:
            break
        if not isinstance(item, dict):
            continue
        control_role = item.get("control_type")
        label = item.get("label")
        rationale = item.get("rationale")
        if control_role not in NEGATIVE_CONTROL_ROLES:
            continue
        if not isinstance(label, str) or not label.strip():
            continue
        if not isinstance(rationale, str) or not rationale.strip():
            continue
        candidates.append({
            "control_id": _slugify_label(label),
            "control_type": control_role,
            "label": label.strip(),
            "rationale": rationale.strip(),
            "domain": "",
            "concept_ids": [],
            "mapping_status": "candidate_pending_human_approval",
            "provenance": "llm_proposed",
        })
    if not candidates:
        raise ValueError("Negative control response produced no usable candidates")
    return tuple(candidates)
