"""Pure helpers for the DAG generation-critique-revision loop.

No LLM calls and no I/O beyond reading a prompt template happen here --
`causal.agents.user_question.UserQuestionHypothesisService` owns every
`llm(prompt)` call and the `llm_outputs` audit trail. This module only:
(1) runs the existing deterministic identification/clinical checks
(`causal.agents.dag_identification`) on a not-yet-approved draft DAG to
produce structured findings for the critique prompt, and (2) builds/parses
the critique-stage prompt and response. Mirrors the separation already used
between `dag_identification.py` (pure graph/clinical logic) and
`user_question.py` (orchestration).

The critic never emits DAG-variable objects itself -- only a revised
`candidate_dag`/`covariate_relations` from the drafting model (parsed by
`UserQuestionHypothesisService._parse_dag_response`, the same gate the first
draft goes through) can change what ends up in the protocol. This keeps
exactly one code path responsible for validating PMIDs/roles/temporal
classes, instead of duplicating that trust boundary here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causal.agents.dag_identification import DAGReviewResult, review_dag
from causal.agents.hypothesis import HypothesisAgent


_CRITIQUE_VERDICTS = {"approve", "revise"}
_MAX_CRITIQUE_CONCERNS = 8


@dataclass(frozen=True)
class CritiqueResult:
    verdict: str
    concerns: tuple[dict[str, str], ...]


def build_pre_check(protocol: dict[str, Any], variables: list[dict], relations: list[dict]) -> DAGReviewResult | None:
    """Run the existing backdoor-identification and clinical-completeness
    checks against the not-yet-approved draft DAG.

    Best-effort: returns None instead of raising if `protocol` is not yet
    complete enough for `HypothesisAgent` (e.g. `target_trial` missing a
    required field at this point in drafting) -- the critique prompt simply
    notes the pre-check was unavailable rather than blocking the draft.
    """
    draft = {**protocol, "candidate_dag": variables, "covariate_relations": relations}
    try:
        return review_dag(HypothesisAgent().run(draft))
    except ValueError:
        return None


def format_structural_findings(review: DAGReviewResult | None) -> str:
    """Render `build_pre_check`'s result as bullet text for the critique prompt.

    Note `blocking_issues` will typically be empty at this stage:
    `major_confounders` is a human-added field (via the covariate-review CLI
    flow) that does not exist yet when the DAG is first drafted. Most
    findings here come from `technical_notes`/`advisory_notes` and are
    informative, not "problems found by default".
    """
    if review is None:
        return "Deterministic structural/clinical pre-check was unavailable."
    lines = [f"- {note}" for note in review.identification.technical_notes]
    lines += [f"- [blocking] {issue}" for issue in review.clinical.blocking_issues]
    lines += [f"- [advisory] {note}" for note in review.clinical.advisory_notes]
    if not lines:
        return "No structural or clinical concerns were flagged by the deterministic pre-check."
    return "\n".join(lines)


def build_critique_prompt(
    template_path: Path,
    target: dict[str, Any],
    articles_text: str,
    kg_relations_text: str,
    proposed_dag_json: str,
    structural_findings_text: str,
) -> str:
    """Render `causal/prompts/dag_critique.md`.

    Same required-placeholder substitution pattern as
    `UserQuestionHypothesisService._build_dag_prompt`: a required
    placeholder missing from the template file is a hard `ValueError`,
    `{{KG_RELATIONS}}` is optional so older/custom templates still work.
    """
    template = template_path.read_text(encoding="utf-8")
    required = {
        "{{POPULATION}}": str(target.get("population", "")),
        "{{TREATMENT}}": str(target.get("treatment", "")),
        "{{COMPARATOR}}": str(target.get("comparator", "")),
        "{{OUTCOME}}": str(target.get("outcome", "")),
        "{{ARTICLES}}": articles_text,
        "{{PROPOSED_DAG}}": proposed_dag_json,
        "{{STRUCTURAL_FINDINGS}}": structural_findings_text,
    }
    for placeholder, value in required.items():
        if placeholder not in template:
            raise ValueError(f"DAG critique prompt is missing {placeholder}")
        template = template.replace(placeholder, value)
    if "{{KG_RELATIONS}}" in template:
        template = template.replace("{{KG_RELATIONS}}", kg_relations_text)
    return template


def parse_critique_response(raw: str) -> CritiqueResult:
    """Structurally validate the critique-stage LLM response.

    Same strictness as `UserQuestionHypothesisService._parse_dag_response`:
    empty, non-JSON, non-object, or missing/invalid `verdict` all raise
    `ValueError` so the caller degrades to the pre-critique DAG rather than
    acting on a malformed response. A `"revise"` verdict with no usable
    `concerns` is also treated as malformed -- it gives the drafting model
    nothing actionable, so revising on it would just invite ungrounded,
    unprompted changes.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("LLM returned an empty critique response")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Critique response must be one valid JSON object") from exc
    if not isinstance(payload, dict) or payload.get("verdict") not in _CRITIQUE_VERDICTS:
        raise ValueError(f"Critique response must set verdict to one of {_CRITIQUE_VERDICTS}")

    concerns: list[dict[str, str]] = []
    raw_concerns = payload.get("concerns", [])
    if isinstance(raw_concerns, list):
        for item in raw_concerns:
            if len(concerns) >= _MAX_CRITIQUE_CONCERNS:
                break
            if (
                isinstance(item, dict)
                and isinstance(item.get("target"), str) and item["target"].strip()
                and isinstance(item.get("issue"), str) and item["issue"].strip()
            ):
                concerns.append({"target": item["target"].strip(), "issue": item["issue"].strip()})

    verdict = payload["verdict"]
    if verdict == "revise" and not concerns:
        raise ValueError("Critique verdict is 'revise' but no usable concerns were provided")
    return CritiqueResult(verdict=verdict, concerns=tuple(concerns))


def build_revision_prompt(base_dag_prompt: str, proposed_dag_json: str, critique: CritiqueResult) -> str:
    """Append a "prior proposal and critique" section to the already-built
    `dag.md` prompt, so the revision response has the identical required
    JSON shape `_parse_dag_response` already validates -- no separate
    revision template to keep in sync with `dag.md`.
    """
    concerns_text = "\n".join(f"- [{concern['target']}] {concern['issue']}" for concern in critique.concerns)
    return (
        f"{base_dag_prompt}\n\n"
        "## Prior proposal and reviewer critique\n\n"
        "An independent reviewer already examined your proposal below and raised the "
        "concerns listed. Produce a revised, complete response in the same required JSON "
        "shape described above (candidate_dag/covariate_relations/unresolved) that "
        "addresses these concerns using only the evidence already provided above -- do "
        "not cite any pmid or knowledge-graph id not listed there.\n\n"
        f"PRIOR PROPOSAL:\n{proposed_dag_json}\n\n"
        f"REVIEWER CONCERNS:\n{concerns_text}\n"
    )
