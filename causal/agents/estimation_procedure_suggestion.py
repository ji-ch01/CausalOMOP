"""Pure helpers for the estimation-procedure suggestion assist at the
`statistical_estimand` gate.

No LLM call and no I/O beyond reading a prompt template happen here --
`causal.orchestrator.WorkflowManager.propose_estimation_procedure` owns the
`llm(prompt)` call, matching the split already used by `causal.agents.
covariate_suggestion` and `causal.queries.negative_controls`.

When `population_summary == "CATE"`, `causal.roadmap.StatisticalEstimand`'s
own `__post_init__` cross-check already forces `estimation_procedure ==
"causal_forest_dml"` -- there is nothing to recommend, so
`deterministic_cate_suggestion` below answers that directly and the caller
must never invoke `llm()` in that case. The LLM is only ever asked to choose
between `plug_in_standardization` and `targeted_minimum_loss_based_
estimation` (`population_summary` in `{"ATE", "ATT"}`).

This is an assist, not a decision: every `EstimationProcedureSuggestion` this
module produces still has to be reviewed and explicitly submitted through
`WorkflowManager.set_statistical_estimand_strategy` (the same manual form
that already exists) -- nothing here writes to a run's approved settings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causal.roadmap import ESTIMATION_PROCEDURES

DEFAULT_ESTIMATION_PROCEDURE_SUGGESTION_PROMPT_PATH = Path(
    "causal/prompts/estimation_procedure_suggestion.md"
)


@dataclass(frozen=True)
class EstimationProcedureSuggestion:
    estimation_procedure: str  # one of causal.roadmap.ESTIMATION_PROCEDURES
    rationale: str
    provenance: str  # "llm_proposed" or "deterministic_cate_rule"


def deterministic_cate_suggestion(ready_effect_modifier_count: int) -> EstimationProcedureSuggestion:
    """No LLM call: `population_summary == "CATE"` already forces
    `estimation_procedure == "causal_forest_dml"` (`causal.roadmap.
    StatisticalEstimand.__post_init__`) -- this just restates that existing,
    enforced rule as a suggestion so the gate has something to show and
    pre-fill, not a judgment call."""
    return EstimationProcedureSuggestion(
        estimation_procedure="causal_forest_dml",
        rationale=(
            "population_summary is 'CATE'; StatisticalEstimand requires "
            "estimation_procedure='causal_forest_dml' for CATE studies -- "
            "plug_in_standardization/targeted_minimum_loss_based_estimation only "
            "report the ATE. This run's analysis_plan.json currently has "
            f"{ready_effect_modifier_count} reviewed, ready effect_modifier "
            "covariate(s)."
        ),
        provenance="deterministic_cate_rule",
    )


def build_estimation_procedure_suggestion_prompt(
    template_path: Path,
    *,
    population_summary: str,
    effect_measure: str,
    feasibility: dict[str, Any],
    followup_audit: dict[str, Any],
    positivity_audit: dict[str, Any],
    covariate_missingness: dict[str, Any],
) -> str:
    """Render `causal/prompts/estimation_procedure_suggestion.md`. Only ever
    called for `population_summary` in `{"ATE", "ATT"}` (see module
    docstring). Each diagnostic argument is the raw JSON artifact dict as
    read from a run directory, or `{}` if that diagnostic hasn't been run
    yet -- summarized here as `"Not yet run."` rather than omitted, so the
    LLM's rationale can explicitly note when a signal is unavailable instead
    of silently reasoning as if it were absent for a reason. Same required-
    placeholder substitution pattern as `causal.agents.covariate_suggestion.
    build_covariate_suggestion_prompt`.
    """
    template = template_path.read_text(encoding="utf-8")

    def _summarize(artifact: dict[str, Any]) -> str:
        counts = artifact.get("counts") if isinstance(artifact, dict) else None
        return json.dumps(counts, indent=2) if counts else "Not yet run."

    required = {
        "{{POPULATION_SUMMARY}}": population_summary,
        "{{EFFECT_MEASURE}}": effect_measure,
        "{{FEASIBILITY_SUMMARY}}": _summarize(feasibility),
        "{{FOLLOWUP_AUDIT_SUMMARY}}": _summarize(followup_audit),
        "{{POSITIVITY_AUDIT_SUMMARY}}": _summarize(positivity_audit),
        "{{COVARIATE_MISSINGNESS_SUMMARY}}": _summarize(covariate_missingness),
    }
    for placeholder, value in required.items():
        if placeholder not in template:
            raise ValueError(f"Estimation procedure suggestion prompt is missing {placeholder}")
        template = template.replace(placeholder, value)
    return template


def parse_estimation_procedure_suggestion_response(raw: str) -> EstimationProcedureSuggestion:
    """Structurally validate the suggestion-assist LLM response.

    Empty, non-JSON, non-object, missing/invalid `estimation_procedure`, or
    missing/empty `rationale` all raise `ValueError` -- same top-level
    strictness as `causal.queries.negative_controls.
    parse_negative_control_response`. There is exactly one recommendation
    object here, not a list, so there is no lenient per-item path to fall
    back to.

    Validates against the full `ESTIMATION_PROCEDURES` set (not just the two
    ATE-capable ones); the caller (`WorkflowManager.
    propose_estimation_procedure`) additionally rejects a returned
    `"causal_forest_dml"` when it only asked for an ATE/ATT choice, since
    that would violate `StatisticalEstimand`'s own `population_summary`
    cross-check.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("LLM returned an empty estimation-procedure response")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Estimation procedure response must be one valid JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("Estimation procedure response must be a JSON object")
    procedure = payload.get("estimation_procedure")
    if procedure not in ESTIMATION_PROCEDURES:
        raise ValueError(f"estimation_procedure must be one of {sorted(ESTIMATION_PROCEDURES)}")
    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("Estimation procedure response requires a non-empty rationale")
    return EstimationProcedureSuggestion(
        estimation_procedure=procedure, rationale=rationale.strip(), provenance="llm_proposed",
    )
