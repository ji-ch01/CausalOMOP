"""Pure prompt-build/response-parse pair for the optional results-narrative
step -- an LLM interpretation of an already-computed statistical estimate.
Mirrors `causal.agents.question_gate`'s split: this module never calls an
LLM and never touches the analytic dataset; the caller (`causal.orchestrator.
workflow.WorkflowManager.run_results_narrative`) owns the actual `llm(prompt)`
call and reads only already-written JSON artifacts to build `NarrativeInput`.

Deterministic verification of the parsed response against those same
artifacts lives in the sibling module `causal.agents.results_faithfulness`,
not here -- this module only knows how to build the prompt and structurally
validate the JSON shape of what came back.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_NARRATIVE_PROMPT_PATH = Path("causal/prompts/results_narrative.md")

# Always appended verbatim by the caller, never trusted to the model's own
# phrasing -- matches docs/causal_roadmap.md's "Evidence and LLM boundary":
# LLM steps are advisory only, never authoritative.
NARRATIVE_BOUNDARY_STATEMENT = (
    "This is an LLM-generated interpretation of the statistical results above. "
    "It is advisory only, is not itself a statistical claim, and does not "
    "replace human review of the estimate, diagnostics, and assumptions."
)

# One claim's factual assertion type -- fixed, closed set so both the prompt
# and `causal.agents.results_faithfulness` can apply type-specific
# deterministic checks (e.g. only an `effect_direction` claim is checked
# against the point estimate's sign).
CLAIM_TYPES = (
    "effect_direction",
    "effect_magnitude",
    "uncertainty",
    "sample_size",
    "diagnostic_warning",
    "assumption_or_limitation",
)

# The top-level keys of `NarrativeInput.to_dict()` -- exactly what is
# serialized into the prompt as RESULTS_JSON. A claim's `source_fields` must
# be dotted paths rooted in one of these; anything else references an
# artifact (or a field) the model was never actually shown, which
# `causal.agents.results_faithfulness.verify_narrative` treats as a failed
# check regardless of whether the cited number happens to be correct.
ALLOWED_SOURCE_FIELD_ROOTS = (
    "study_id",
    "question",
    "estimand",
    "dataset_summary",
    "estimation",
    "diagnostics",
    "gate_history",
    "assumptions_and_limitations",
)


@dataclass(frozen=True)
class NarrativeInput:
    """Exactly the JSON-safe fields fed to the prompt -- enumerated so it is
    auditable that no patient-level artifact (`analytic_dataset.parquet`)
    ever reaches this function. `dataset_summary` carries `dataset_manifest.
    json`'s row/column counts only, never a patient value; `gate_history` is
    `{name, status, reviewer, decided_at}` tuples only, never a free-text
    history entry.
    """

    study_id: str
    question: dict[str, Any]
    estimand: dict[str, Any]
    dataset_summary: dict[str, Any]
    estimation: dict[str, Any]
    diagnostics: dict[str, Any]
    gate_history: tuple[dict[str, Any], ...]
    assumptions_and_limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_narrative_prompt(template_path: Path, narrative_input: NarrativeInput) -> str:
    template = template_path.read_text(encoding="utf-8")
    if "{{RESULTS_JSON}}" not in template:
        raise ValueError("Results-narrative prompt is missing {{RESULTS_JSON}}")
    results_json = json.dumps(narrative_input.to_dict(), indent=2, sort_keys=True)
    return template.replace("{{RESULTS_JSON}}", results_json)


@dataclass(frozen=True)
class Claim:
    """One structured, individually-verifiable factual assertion the model's
    narrative makes. `source_fields` is what lets `causal.agents.
    results_faithfulness.verify_narrative` check provenance deterministically
    instead of trusting the model's own citation."""

    text: str
    claim_type: str
    source_fields: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "claim_type": self.claim_type, "source_fields": list(self.source_fields)}


@dataclass(frozen=True)
class NarrativeResult:
    narrative_text: str
    claims: tuple[Claim, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"narrative_text": self.narrative_text, "claims": [claim.to_dict() for claim in self.claims]}


def parse_narrative_response(raw: str) -> NarrativeResult:
    """Structurally validate the narrative-step LLM response.

    Deliberately does NOT read or trust any boundary/disclaimer text the
    model itself may have included -- `NARRATIVE_BOUNDARY_STATEMENT` is
    always the fixed string the caller appends, regardless of model output,
    since a safety invariant this important should not depend on LLM
    self-disclosure compliance.

    Requires a non-empty `claims` array (each with `text`, a `claim_type`
    from `CLAIM_TYPES`, and a non-empty `source_fields` list of strings) in
    addition to the free-text `narrative` -- a model response with a
    narrative but no claims cannot be deterministically fact-checked by
    `causal.agents.results_faithfulness`, so it is rejected here rather than
    silently accepted with an empty claim list. Empty, non-JSON, non-object,
    or a missing/blank `narrative` field all raise `ValueError`, same as
    `causal.agents.question_gate.parse_gate_response`.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("LLM returned an empty results-narrative response")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Results-narrative response must be one valid JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("Results-narrative response must be a JSON object")
    return narrative_result_from_payload(payload)


def narrative_result_from_payload(
    payload: dict[str, Any], *, context: str = "Results-narrative response"
) -> NarrativeResult:
    """Validate an already-parsed `{narrative, claims}` object into a
    `NarrativeResult`.

    Shared by `parse_narrative_response` (the one-shot results-narrative
    step, which owns the JSON decode) and `causal.companion.companion.
    parse_results_companion_turn` (the drafted narrative a results-dialogue
    turn can carry as a nested field). `context` is only used to make the
    `ValueError` messages point at the right caller.

    Requires a non-empty free-text `narrative` and a non-empty `claims`
    array -- each claim needs `text`, a `claim_type` from `CLAIM_TYPES`, and
    a non-empty `source_fields` list of non-empty strings -- so a narrative
    that `causal.agents.results_faithfulness` could not deterministically
    fact-check is rejected here rather than silently accepted.
    """
    narrative = payload.get("narrative")
    if not isinstance(narrative, str) or not narrative.strip():
        raise ValueError(f"{context} must include a non-empty 'narrative' field")

    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list) or not raw_claims:
        raise ValueError(f"{context} must include a non-empty 'claims' list")

    claims: list[Claim] = []
    for index, item in enumerate(raw_claims):
        if not isinstance(item, dict):
            raise ValueError(f"{context}: claims[{index}] must be a JSON object")
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{context}: claims[{index}] must include a non-empty 'text' field")
        claim_type = item.get("claim_type")
        if claim_type not in CLAIM_TYPES:
            raise ValueError(
                f"{context}: claims[{index}].claim_type must be one of {CLAIM_TYPES}, got {claim_type!r}"
            )
        source_fields = item.get("source_fields")
        if (
            not isinstance(source_fields, list)
            or not source_fields
            or not all(isinstance(field, str) and field.strip() for field in source_fields)
        ):
            raise ValueError(
                f"{context}: claims[{index}].source_fields must be a non-empty list of non-empty strings"
            )
        claims.append(Claim(text=text.strip(), claim_type=claim_type, source_fields=tuple(source_fields)))

    return NarrativeResult(narrative_text=narrative.strip(), claims=tuple(claims))


def normalize_narrative_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    """Tolerant reader for `results_narrative.json`, old or new shape.

    The pre-review-workflow artifact (`schema_version` "1.0") only ever had
    `narrative`, `boundary_statement`, `model`, `provider`, `prompt_sha256`.
    Every key defaulted here is one introduced by the faithfulness-
    verification/human-review workflow (`schema_version` "2.0") -- callers
    (`WorkflowManager.build_causal_report`, `WorkflowManager.
    review_results_narrative`, the Streamlit UI) read through this function
    so a legacy artifact on disk from an older run is displayed as an
    unreviewed, unverified narrative instead of crashing or being silently
    treated as already approved.
    """
    normalized = dict(payload)
    normalized.setdefault("claims", [])
    normalized.setdefault("verification", {"status": "not_run", "checks": [], "checked_at_utc": None})
    normalized.setdefault("review_status", "pending")
    normalized.setdefault("reviewer", None)
    normalized.setdefault("reviewed_at", None)
    normalized.setdefault("review_note", None)
    return normalized
