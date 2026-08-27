"""Pure helpers for the "suggest OMOP search terms" assist.

One shared button in the Streamlit concept-mapping section asks an LLM for a
short OMOP-vocabulary search string for every concept set that still needs
one, in a single call, instead of the reviewer retyping each search box.

No LLM call and no I/O beyond reading the prompt template happen here --
`causal.ui.streamlit_app` owns the `llm(prompt)` call, the same split used by
`causal.agents.covariate_suggestion`. This is an assist, not a decision: the
only thing a suggestion does is pre-fill the per-role "Search term" box
(persisted as `concept_sets[name].lookup_term` via
`WorkflowManager.set_concept_set_lookup_terms`); the reviewer still runs the
search and approves each mapping through the unchanged forms.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_CONCEPT_LOOKUP_TERMS_PROMPT_PATH = Path("causal/prompts/concept_lookup_terms.md")

#: A search string longer than this is almost certainly a phrase the model
#: pasted from the label rather than a vocabulary search term; drop it.
_MAX_TERM_CHARS = 80


def build_concept_lookup_terms_prompt(
    template_path: Path,
    target: dict[str, Any],
    concept_sets: list[dict[str, str]],
) -> str:
    """Render `causal/prompts/concept_lookup_terms.md`.

    `concept_sets` is a list of `{"name", "role", "label"}` -- one entry per
    concept set that should get a suggested term. Same required-placeholder
    substitution pattern as
    `causal.agents.covariate_suggestion.build_covariate_suggestion_prompt`.
    """
    template = template_path.read_text(encoding="utf-8")
    required = {
        "{{POPULATION}}": str(target.get("population", "")),
        "{{TREATMENT}}": str(target.get("treatment", "")),
        "{{COMPARATOR}}": str(target.get("comparator", "")),
        "{{OUTCOME}}": str(target.get("outcome", "")),
        "{{CONCEPT_SETS}}": json.dumps(concept_sets, indent=2),
    }
    for placeholder, value in required.items():
        if placeholder not in template:
            raise ValueError(f"Concept lookup terms prompt is missing {placeholder}")
        template = template.replace(placeholder, value)
    return template


def parse_concept_lookup_terms_response(raw: str, known_names: set[str]) -> dict[str, str]:
    """Structurally validate the search-term-assist LLM response.

    Empty, non-JSON, non-object, or missing/non-list `terms` all raise
    `ValueError` -- same top-level strictness as
    `causal.agents.covariate_suggestion.parse_covariate_suggestion_response`.
    Individual entries are silently dropped when malformed: an unknown name
    (guards against a hallucinated concept set the reviewer never saw), a
    blank or over-long `search_term`, or a repeated name (first wins).

    Returns a `{name: search_term}` map.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("LLM returned an empty concept-lookup-terms response")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Concept lookup terms response must be one valid JSON object") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("terms"), list):
        raise ValueError("Concept lookup terms response must set 'terms' to a list")

    terms: dict[str, str] = {}
    for item in payload["terms"]:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        search_term = item.get("search_term")
        if (
            not isinstance(name, str)
            or name not in known_names
            or name in terms
            or not isinstance(search_term, str)
            or not search_term.strip()
            or len(search_term.strip()) > _MAX_TERM_CHARS
        ):
            continue
        terms[name] = search_term.strip()
    return terms
