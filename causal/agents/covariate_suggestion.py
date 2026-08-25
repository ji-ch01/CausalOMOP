"""Pure helpers for the covariate-classification suggestion assist.

No LLM calls and no I/O beyond reading a prompt template happen here --
`causal.ui.streamlit_app` owns the `llm(prompt)` call, matching the split
already used between `dag_critique.py` (pure prompt/response helpers) and
`user_question.py` (the only place that actually calls an LLM).

This is an assist, not a decision: every `CovariateSuggestion` this module
produces still has to be reviewed and explicitly submitted through
`WorkflowManager.review_covariate` (the same "Unresolved covariate review"
batch form that already exists) -- nothing here writes to a run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causal.agents.hypothesis import DAG_VARIABLE_DOMAINS, ROLE_ALLOWED_TEMPORAL_CLASSES

DEFAULT_COVARIATE_SUGGESTION_PROMPT_PATH = Path("causal/prompts/covariate_suggestion.md")

_MAX_SUGGESTIONS = 50


@dataclass(frozen=True)
class CovariateSuggestion:
    label: str
    decision: str  # "reject" or a causal role key from ROLE_ALLOWED_TEMPORAL_CLASSES
    temporal_classification: str  # "" when decision == "reject"
    domain: str  # "" when not proposed
    rationale: str


def build_covariate_suggestion_prompt(
    template_path: Path,
    target: dict[str, Any],
    unresolved: list[dict[str, str]],
) -> str:
    """Render `causal/prompts/covariate_suggestion.md`.

    `unresolved` is a list of `{"label": ..., "reason": ...}` -- the same
    shape already recorded in `protocol["candidate_dag_unresolved"]` by the
    DAG-drafting step, reused here as context instead of re-deriving why each
    covariate is still unclassified. Same required-placeholder substitution
    pattern as `causal.agents.dag_critique.build_critique_prompt`.
    """
    template = template_path.read_text(encoding="utf-8")
    required = {
        "{{POPULATION}}": str(target.get("population", "")),
        "{{TREATMENT}}": str(target.get("treatment", "")),
        "{{COMPARATOR}}": str(target.get("comparator", "")),
        "{{OUTCOME}}": str(target.get("outcome", "")),
        "{{UNRESOLVED_COVARIATES}}": json.dumps(unresolved, indent=2),
    }
    for placeholder, value in required.items():
        if placeholder not in template:
            raise ValueError(f"Covariate suggestion prompt is missing {placeholder}")
        template = template.replace(placeholder, value)
    return template


def parse_covariate_suggestion_response(raw: str, known_labels: set[str]) -> dict[str, CovariateSuggestion]:
    """Structurally validate the suggestion-assist LLM response.

    Empty, non-JSON, non-object, or missing/non-list `suggestions` all raise
    `ValueError` -- same top-level strictness as
    `causal.agents.dag_critique.parse_critique_response`. Individual entries
    are validated and silently dropped if malformed (unknown label, invalid
    decision, or a temporal_classification/decision combination that
    `causal.agents.hypothesis.ROLE_ALLOWED_TEMPORAL_CLASSES` would reject) --
    one bad suggestion should not discard every good one, and `known_labels`
    guards against acting on a hallucinated label the reviewer never saw.

    Returns a `{label: CovariateSuggestion}` map; a label repeated in the
    response keeps its first occurrence.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("LLM returned an empty covariate-suggestion response")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Covariate suggestion response must be one valid JSON object") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("suggestions"), list):
        raise ValueError("Covariate suggestion response must set 'suggestions' to a list")

    suggestions: dict[str, CovariateSuggestion] = {}
    for item in payload["suggestions"]:
        if len(suggestions) >= _MAX_SUGGESTIONS:
            break
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        decision = item.get("decision")
        rationale = item.get("rationale")
        if (
            not isinstance(label, str) or label not in known_labels
            or label in suggestions
            or not isinstance(rationale, str) or not rationale.strip()
        ):
            continue
        if decision == "reject":
            suggestions[label] = CovariateSuggestion(
                label=label, decision="reject", temporal_classification="",
                domain="", rationale=rationale.strip(),
            )
            continue
        if not isinstance(decision, str) or decision not in ROLE_ALLOWED_TEMPORAL_CLASSES:
            continue
        temporal = item.get("temporal_classification")
        if temporal not in ROLE_ALLOWED_TEMPORAL_CLASSES[decision]:
            continue
        domain = item.get("domain") or ""
        if domain not in DAG_VARIABLE_DOMAINS and domain != "":
            domain = ""
        suggestions[label] = CovariateSuggestion(
            label=label, decision=decision, temporal_classification=temporal,
            domain=domain, rationale=rationale.strip(),
        )
    return suggestions
