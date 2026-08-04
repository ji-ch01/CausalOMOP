"""Upstream causal-question gate.

Roadmap step 1 ("Clarify the causal question") had no guard before this:
nothing checked that a natural-language question actually asks for a causal
effect before `causal.agents.user_question.UserQuestionHypothesisService`
spent an LLM call, a literature search, and a DAG draft on it. This module is
that check -- pure prompt-build/response-parse, no LLM call and no I/O beyond
reading the prompt template, mirroring the separation already used in
`causal.agents.dag_critique`. The caller (`causal.cli`, `causal.ui.
streamlit_app`) owns the actual `llm(prompt)` call and decides what to do with
a "descriptive" or "needs_clarification" result -- this module only classifies.

See docs/causal_roadmap.md "Known gaps" for why this exists and what it does
not cover: FastOMOP is where a "descriptive" question should go, but nothing
here calls into FastOMOP -- the caller is responsible for that routing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_GATE_PROMPT_PATH = Path("causal/prompts/question_gate.md")

_GATE_CATEGORIES = {"causal", "descriptive", "needs_clarification"}


@dataclass(frozen=True)
class QuestionGateResult:
    category: str
    reasoning: str
    identified_intervention: str
    identified_outcome: str
    clarifying_question: str


def build_gate_prompt(template_path: Path, question: str) -> str:
    question = question.strip()
    if not question:
        raise ValueError("User question is required")
    template = template_path.read_text(encoding="utf-8")
    if "{{USER_QUESTION}}" not in template:
        raise ValueError("Question-gate prompt is missing {{USER_QUESTION}}")
    return template.replace("{{USER_QUESTION}}", question)


def parse_gate_response(raw: str) -> QuestionGateResult:
    """Structurally validate the gate-stage LLM response.

    Same strictness as `dag_critique.parse_critique_response`: empty,
    non-JSON, non-object, or missing/invalid `category` all raise
    `ValueError`. A `needs_clarification` category with no usable
    `clarifying_question` is also treated as malformed -- it gives the
    caller nothing to show the user, so silently falling through would just
    look like a bug.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("LLM returned an empty question-gate response")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Question-gate response must be one valid JSON object") from exc
    if not isinstance(payload, dict) or payload.get("category") not in _GATE_CATEGORIES:
        raise ValueError(f"Question-gate response must set category to one of {_GATE_CATEGORIES}")

    category = payload["category"]
    reasoning = payload.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("Question-gate response must include a non-empty reasoning")

    def _optional_text(field: str) -> str:
        value = payload.get(field, "")
        return value.strip() if isinstance(value, str) else ""

    clarifying_question = _optional_text("clarifying_question")
    if category == "needs_clarification" and not clarifying_question:
        raise ValueError(
            "Question-gate category is 'needs_clarification' but no usable "
            "clarifying_question was provided"
        )

    return QuestionGateResult(
        category=category,
        reasoning=reasoning.strip(),
        identified_intervention=_optional_text("identified_intervention"),
        identified_outcome=_optional_text("identified_outcome"),
        clarifying_question=clarifying_question,
    )
