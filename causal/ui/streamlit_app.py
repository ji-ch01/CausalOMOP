"""Read-focused Streamlit dashboard for validating and demoing a causal workflow run.

Run with: `streamlit run causal/ui/streamlit_app.py` from the repository root.

This app mainly calls the existing `WorkflowManager` (load/decide_gate/advance/
build_demo_report) and reads the JSON artifacts it already writes under
`artifacts/runs/<run_id>/`. A few actions do reach further, the same way the
`causalomop` CLI does: drafting a hypothesis calls the configured LLM directly
(see `_render_new_question_form`); mapping a `concept_sets`/`query_concept_sets`
entry (`_render_concept_mapping_section`, mirrors `workflow concept-set-map`/
`query-concept-set-map`) or approving a covariate's concept mapping
(`_render_analysis_plan_tab`, mirrors `workflow covariate-map-concepts
--with-mcp`) both call OMCP directly (`OMCPClient`) to search/verify OMOP
concepts, unconditionally here rather than opt-in, since this app is where a
reviewer with real BigQuery/OMOP access does the actual clinical mapping
review, not a fast offline dev loop; and regenerating the analysis plan
(`_render_regenerate_plan_section`, mirrors `workflow plan [--with-mcp]`)
optionally calls OMCP too, opt-in there since refreshing OMOP concept
*candidates* is a slower discovery aid, not a correctness gate.
"""

from __future__ import annotations

import csv
import html
import io
import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st

# `streamlit run` puts this file's own directory (causal/ui/) on sys.path,
# not the repository root, so the `causal` package is not importable without
# this regardless of the shell's current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from causal.agents.covariate_suggestion import (
    DEFAULT_COVARIATE_SUGGESTION_PROMPT_PATH,
    build_covariate_suggestion_prompt,
    parse_covariate_suggestion_response,
)
from causal.agents.dag_identification import (
    DEFAULT_MAX_ADJUSTMENT_CANDIDATES,
    RECOMMENDED_MAX_CONFOUNDERS,
    review_dag,
)
from causal.agents.dag_metrics import (
    max_structural_hamming_distance,
    max_structural_intervention_distance,
    normalized_agreement,
)
from causal.agents.dag_tournament import MAX_TOURNAMENT_SIZE
from causal.agents.data_science import LIFETIME_LOOKBACK_HOURS
from causal.agents.hypothesis import (
    DAG_VARIABLE_DOMAINS,
    DAG_VARIABLE_ROLES,
    DAG_VARIABLE_TEMPORAL_CLASSES,
    CovariateRelation,
    DAGVariable,
    HypothesisArtifact,
    RejectedCovariate,
)
from causal.agents.question_gate import DEFAULT_GATE_PROMPT_PATH, build_gate_prompt, parse_gate_response
from causal.agents.results_narrative import normalize_narrative_artifact
from causal.agents.user_question import LLMRawOutput, UserQuestionHypothesisService
from causal.config import BigQueryDatasets
from causal.literature import LiteratureOrchestrator, OpenAlexClient, PubMedClient
from causal.llm import build_hypothesis_llm
from causal.mcp_client import OMCPClient
from causal.orchestrator.state import NEXT_PHASE, PHASES, RunState
from causal.orchestrator.workflow import WorkflowManager
from causal.queries import COVARIATE_AGGREGATIONS_BY_DOMAIN, EXPOSURE_CONTRASTS, OUTCOME_DOMAINS
from causal.roadmap import CENSORING_STRATEGIES, MISSING_DATA_STRATEGIES, CausalDAG


RUNS_DIR = _PROJECT_ROOT / "artifacts" / "runs"

_GATE_COLORS = {
    "approved": "#2E7D32",
    "rejected": "#C62828",
    "provisional": "#EF6C00",
    "pending": "#757575",
}

_GATE_INFO: dict[str, dict[str, str]] = {
    "target_trial": {
        "title": "Question & target trial",
        "phase": "target_trial_review",
        "description": (
            "Approves the normalized causal question (population / treatment / comparator / "
            "outcome) and the target-trial protocol around it — eligibility criteria, "
            "assignment procedure, censoring strategy, analysis strategy, and the counterfactual "
            "estimand (effect measure, treatment/comparator contrast values)."
        ),
    },
    "dag_review": {
        "title": "Causal DAG & clinical completeness",
        "phase": "dag_review",
        "description": (
            "Approves the causal DAG proposed for this question — the assumed confounders, "
            "mediators, and colliders. Must be approved (every major confounder resolved) "
            "before proceeding."
        ),
    },
    "identification_estimand": {
        "title": "Identification & estimand",
        "phase": "identification_review",
        "description": (
            "Approves that the declared DAG is identified by backdoor adjustment and that the "
            "full causal study specification (question, target trial, estimand, DAG) validates "
            "as one internally consistent contract. Writes `study_specification.json` on approval."
        ),
    },
    "clinical_mapping": {
        "title": "Clinical concept mapping",
        "phase": "concept_mapping",
        "description": (
            "Approves the OMOP concept sets (ICD/RxNorm/LOINC codes, etc.) used to operationalize "
            "population / treatment / comparator / outcome for querying. In **exploratory** mode, "
            "a `provisional` decision is enough to proceed with an unvalidated first-pass "
            "phenotype; **final** mode requires full `approved`."
        ),
    },
    "analysis_plan": {
        "title": "Analysis plan",
        "phase": "analysis_plan",
        "description": (
            "Approves the statistical design (data source, phenotypes, planned feasibility "
            "checks) before the pipeline is allowed to build the analytic cohort and execute."
        ),
    },
    "statistical_estimand": {
        "title": "Statistical estimand",
        "phase": "estimand_review",
        "description": (
            "Approves the observed-data statistical estimand: the outcome-model specification, "
            "the adjustment set as actually built into `cohort.sql`, and an explicit missing-"
            "data/censoring strategy. Writes `statistical_estimand.json` on approval; distinct "
            "from the counterfactual estimand approved by identification_estimand."
        ),
    },
    "results": {
        "title": "Results review",
        "phase": "results_review",
        "description": (
            "Approves the final causal effect estimate and diagnostics before the demo report "
            "is finalized."
        ),
    },
}

_NODE_COLORS = {
    "exposure": "#4C78A8",
    "outcome": "#E45756",
    "confounder": "#54A24B",
    "mediator": "#F2A900",
    "collider": "#9D7BBF",
    "prognostic": "#72727A",
    "effect_modifier": "#17A2B8",
    "selection": "#EE7C2E",
    "instrument": "#8C8C8C",
    "other": "#B0B0B0",
}

# Shown above any table that exposes a raw "Vocabulary"/"Domain" column, so a
# non-expert reviewer doesn't need outside knowledge to read it.
_VOCABULARY_CAPTION = (
    "Vocabulary is the coding system a concept comes from -- e.g. RxNorm for medications, "
    "ICD-10-CM/SNOMED for diagnoses, LOINC for lab tests. Domain is the OMOP table it lives in."
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _dag_agreement_score(distance: int, variable_count: int | None, max_distance_fn) -> float | None:
    """`distance` (a raw SHD/SID count from `dag_tournament.py`) rescaled to
    a 0.0-1.0 agreement score for display (see `normalized_agreement`).

    `variable_count` is `None` for runs recorded before this field was
    added -- there is no way to recover it after the fact, so this returns
    `None` (rendered as a blank cell) rather than a guess.
    """
    if variable_count is None:
        return None
    return normalized_agreement(distance, max_distance_fn(variable_count))


def _prettify_metric_key(key: str) -> str:
    """`snake_case` key -> sentence-cased label, e.g. `outcome_events_treatment`
    -> "Outcome events treatment". These keys come straight from SQL column
    aliases (`causal/queries/feasibility.py`, `AnalyticCohortBuilder.profile_sql`)
    with no fixed, enumerable set this app can hold its own display names for
    -- this is a generic fallback, not a per-key mapping."""
    label = key.replace("_", " ").strip()
    return label[:1].upper() + label[1:] if label else label


def _render_count_metrics(counts: dict[str, Any]) -> None:
    """KPI-tile view of a flat {key: number} dict (feasibility/cohort-profile/
    follow-up counts) instead of a raw JSON blob. The exact machine-readable
    values are always still one click away via the caller's "raw data"
    expander -- this is a friendlier default view of the same numbers, not a
    replacement for them.
    """
    if not counts:
        st.caption("No counts returned.")
        return
    with st.container(horizontal=True):
        for key, value in counts.items():
            display_value = f"{value:.2f}" if isinstance(value, float) else ("—" if value is None else value)
            st.metric(_prettify_metric_key(key), display_value, border=True)


def _badge(status: str) -> str:
    color = _GATE_COLORS.get(status, "#757575")
    return (
        f'<span style="background-color:{color};color:white;padding:2px 10px;'
        f'border-radius:10px;font-size:0.85em;">{html.escape(status)}</span>'
    )


def _source_question(run_dir: Path) -> str | None:
    for filename in ("protocol.json", "hypothesis.json"):
        data = _read_json(run_dir / filename)
        question = (data or {}).get("source_question")
        if isinstance(question, str) and question.strip():
            return question.strip()
    return None


def _list_runs() -> list[str]:
    if not RUNS_DIR.is_dir():
        return []
    return sorted(
        (p.name for p in RUNS_DIR.iterdir() if p.is_dir() and (p / "state.json").is_file()),
        reverse=True,
    )


def _reviewer_default() -> str:
    """Sidebar-entered reviewer name, reused so a reviewer stepping through
    several gates/covariates in one sitting doesn't retype it each time."""
    return st.session_state.get("reviewer_name", "")


_REVIEWER_FIELD_KEY_PREFIXES = (
    "reviewer_",
    "covariate_reviewer_",
    "concept_set_reviewer_",
    "exposure_contrast_reviewer_",
    "covariate_domain_reviewer_",
    "covariate_window_reviewer_",
    "covariate_aggregation_reviewer_",
    "covariate_map_reviewer_",
    "add_pop_reviewer_",
    "remove_pop_reviewer_",
)


def _sync_reviewer_fields() -> None:
    """Push the sidebar name into every per-gate/per-covariate Reviewer
    field already on screen. Needed because `st.text_input(value=...)` only
    seeds a widget's session_state the first time its key is created --
    once rendered, later reruns ignore `value=` and keep the stale text."""
    name = st.session_state.get("reviewer_name", "")
    for key in list(st.session_state.keys()):
        if key.startswith(_REVIEWER_FIELD_KEY_PREFIXES):
            st.session_state[key] = name


def _run_option_label(run_id: str) -> str:
    question = _source_question(RUNS_DIR / run_id)
    if not question:
        return run_id
    preview = question if len(question) <= 70 else f"{question[:67]}..."
    return f"{run_id}  —  {preview}"


def _hypothesis_artifact_from_dict(data: dict[str, Any]) -> HypothesisArtifact:
    dag_variables = tuple(
        DAGVariable(
            label=v["label"],
            role=v["role"],
            rationale=v["rationale"],
            # Older runs predate temporal_classification; fall back to a
            # display-only guess so historical artifacts still render. This
            # bypasses HypothesisAgent's role/temporal consistency check
            # (direct dataclass construction, not `_candidate_dag`), which is
            # fine here since the UI never re-validates or re-executes this
            # artifact.
            temporal_classification=v.get("temporal_classification", "pre_treatment"),
            pmid=v.get("pmid", ""),
            observed=v.get("observed", True),
            certainty=v.get("certainty", "established"),
            kg_evidence_id=v.get("kg_evidence_id", ""),
            evidence_level=v.get("evidence_level", "unclassified"),
        )
        for v in data.get("candidate_dag", ())
    )
    relations = tuple(
        CovariateRelation(
            source=r["source"],
            target=r["target"],
            rationale=r["rationale"],
            pmid=r.get("pmid", ""),
            certainty=r.get("certainty", "established"),
            kg_evidence_id=r.get("kg_evidence_id", ""),
            evidence_level=r.get("evidence_level", "unclassified"),
        )
        for r in data.get("covariate_relations", ())
    )
    rejected_covariates = tuple(
        RejectedCovariate(
            label=r["label"],
            reviewer=r["reviewer"],
            comment=r.get("comment", ""),
            decided_at=r.get("decided_at", ""),
        )
        for r in data.get("rejected_covariates", ())
    )
    return HypothesisArtifact(
        metadata=data.get("metadata", {}),
        schema_version=data.get("schema_version", ""),
        study_id=data["study_id"],
        question_type=data.get("question_type", "causal"),
        source_question=data.get("source_question", ""),
        population=data["population"],
        treatment=data["treatment"],
        comparator=data["comparator"],
        outcome=data["outcome"],
        time_zero=data["time_zero"],
        grace_period_hours=data["grace_period_hours"],
        follow_up_days=data["follow_up_days"],
        primary_estimand=data["primary_estimand"],
        secondary_estimands=tuple(data.get("secondary_estimands", ())),
        candidate_confounders=tuple(data.get("candidate_confounders", ())),
        candidate_effect_modifiers=tuple(data.get("candidate_effect_modifiers", ())),
        candidate_dag=dag_variables,
        covariate_relations=relations,
        rejected_covariates=rejected_covariates,
        major_confounders=tuple(data.get("major_confounders", ())),
        unresolved_covariates=tuple(data.get("unresolved_covariates", ())),
        unresolved_major_confounders=tuple(data.get("unresolved_major_confounders", ())),
        design_constraints=tuple(data.get("design_constraints", ())),
        unresolved_issues=tuple(data.get("unresolved_issues", ())),
        executable=data.get("executable", False),
    )


def _dag_to_dot(dag: CausalDAG) -> str:
    lines = [
        "digraph G {",
        "rankdir=LR;",
        'graph [pad="0.25", nodesep="0.45", ranksep="0.9", bgcolor="transparent"];',
        'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, margin="0.16,0.11"];',
        'edge [fontname="Helvetica", fontsize=10, color="#667085", arrowsize=0.8];',
    ]
    for node in dag.nodes:
        color = _NODE_COLORS.get(node.role, "#B0B0B0")
        role_label = node.role.replace("_", " ")
        fontcolor = "#1F2937" if node.role in {"mediator", "other"} else "white"
        emphasis = ", penwidth=3" if node.node_id in {dag.exposure_id, dag.outcome_id} else ""
        label = node.label.replace('"', "'")
        lines.append(
            f'"{node.node_id}" [label="{label}\\n{role_label}", fillcolor="{color}", '
            f'fontcolor="{fontcolor}"{emphasis}];'
        )
    for edge in dag.edges:
        style = "solid" if edge.status in {"evidence_supported", "expert_approved"} else "dashed"
        lines.append(f'"{edge.source}" -> "{edge.target}" [style={style}];')
    lines.append("}")
    return "\n".join(lines)


_PHASE_TO_GATE = {info["phase"]: gate_name for gate_name, info in _GATE_INFO.items()}

_PHASE_LABELS: dict[str, str] = {
    "target_trial_review": "question & target trial review",
    "dag_review": "causal DAG review",
    "identification_review": "identification & estimand review",
    "concept_mapping": "clinical concept mapping",
    "feasibility": "feasibility check",
    "analysis_plan": "analysis plan",
    "estimand_review": "statistical estimand review",
    "execution": "cohort execution",
    "diagnostics": "diagnostics",
    "results_review": "results review",
    "report": "demo report",
}

# Phases with no approval gate (`_PHASE_TO_GATE` has no entry) have no
# `advance()`-blocking decision to point to, so the hint instead names the
# concrete action that produces this phase's artifacts.
_NO_GATE_NEXT_ACTION: dict[str, str] = {
    "feasibility": "run the feasibility check (Feasibility & cohort)",
    "execution": "build the cohort plan and run it (Feasibility & cohort)",
    "diagnostics": "review the cohort diagnostics (Feasibility & cohort)",
    "report": "generate the demo report (Report & audit)",
}


def _next_action_text(state: RunState) -> str:
    """One sentence naming the next concrete thing a reviewer should do,
    kept consistent with the same rule `WorkflowManager.advance` enforces
    (`REQUIRED_GATES`/`NEXT_PHASE`) so this can never suggest an action the
    Advance button would then reject.
    """
    gate_name = _PHASE_TO_GATE.get(state.phase)
    if gate_name is None:
        return _NO_GATE_NEXT_ACTION.get(state.phase, "advance to the next phase") + "."
    gate = state.gates[gate_name]
    title = _GATE_INFO[gate_name]["title"]
    if gate.status == "rejected":
        return f"resolve the blocking issues, then re-decide the {title} gate (Current task)."
    accepted = gate.status == "approved" or (
        gate.status == "provisional" and gate_name == "clinical_mapping" and state.mode == "exploratory"
    )
    if accepted:
        next_phase = NEXT_PHASE[state.phase]
        return f"advance to {_PHASE_LABELS.get(next_phase, next_phase)} (Current task)."
    return f"decide the {title} gate (Current task)."


def _render_phase_tracker(state: RunState) -> None:
    current_phase = state.phase
    current_index = PHASES.index(current_phase)
    current_gate = _PHASE_TO_GATE.get(current_phase)
    gate_status = state.gates[current_gate].status if current_gate else "action required"

    phase_col, progress_col, gate_col = st.columns(3)
    phase_col.metric("Current phase", _PHASE_LABELS.get(current_phase, current_phase), border=True)
    progress_col.metric("Workflow progress", f"{current_index + 1} of {len(PHASES)}", border=True)
    gate_col.metric(
        "Current checkpoint",
        gate_status.replace("_", " "),
        border=True,
    )
    st.info(f"Next action: {_next_action_text(state)}", icon=":material/arrow_forward:")

    with st.expander("View the complete workflow", icon=":material/account_tree:"):
        for start in range(0, len(PHASES), 3):
            columns = st.columns(3)
            for index, phase in enumerate(PHASES[start : start + 3], start=start):
                phase_label = _PHASE_LABELS.get(phase, phase)
                if index < current_index:
                    status, color, icon = "Completed", "green", ":material/check_circle:"
                elif index == current_index:
                    status, color, icon = "Current", "blue", ":material/play_circle:"
                else:
                    status, color, icon = "Upcoming", "gray", ":material/lock:"
                with columns[index - start]:
                    st.markdown(f"**{phase_label}**")
                    st.badge(status, color=color, icon=icon)
        st.caption("A locked step may require an approval decision before the workflow can continue.")


def _render_target_trial_summary(run_dir: Path) -> None:
    """Read-only view of `protocol["target_trial"]` -- what the `target_trial`
    gate actually approves (see `WorkflowManager._block_if_target_trial_
    incomplete`). Deliberately read-only: these fields are drafted upfront
    (by the LLM prompt or a manually authored protocol.json), reviewed and
    approved/rejected here, not iteratively built through the workflow the
    way concept mappings are -- so there is no `set_*` review action to pair
    it with, unlike `_render_exposure_contrast_selector`.
    """
    protocol = _read_json(run_dir / "protocol.json")
    target = (protocol or {}).get("target_trial")
    if not isinstance(target, dict):
        st.info("No target_trial object in protocol.json yet.")
        return
    with st.expander("Target trial & estimand (target_trial gate content)", expanded=False):
        col1, col2 = st.columns(2)
        col1.markdown(f"**Population**  \n{target.get('population', '—')}")
        col1.markdown(f"**Treatment**  \n{target.get('treatment', '—')}")
        col1.markdown(f"**Comparator**  \n{target.get('comparator', '—')}")
        col1.markdown(f"**Outcome**  \n{target.get('outcome', '—')}")
        col1.markdown(f"**Time zero**  \n{target.get('time_zero', '—')}")
        col2.markdown(f"**Assignment procedure**  \n{target.get('assignment_procedure', '—')}")
        col2.markdown(f"**Censoring strategy**  \n{target.get('censoring_strategy', '—')}")
        col2.markdown(f"**Analysis strategy**  \n{target.get('analysis_strategy', '—')}")
        col2.markdown(
            f"**Estimand**  \n{target.get('effect_measure', '—')}: "
            f"{target.get('treatment_value', '—')} vs {target.get('comparator_value', '—')}"
        )
        criteria = target.get("eligibility_criteria") or []
        st.markdown("**Eligibility criteria**")
        if criteria:
            st.markdown("\n".join(f"- {item}" for item in criteria))
        else:
            st.caption("None recorded.")


_MISSING_DATA_STRATEGY_LABELS = {
    "complete_case_analysis": "Complete-case analysis (drop rows with any missing covariate)",
}

_CENSORING_STRATEGY_LABELS = {
    "exclude_censored_before_analysis": "Exclude censored rows (drop censored=1 before fitting)",
    "ignore_censoring_administrative_waiver": "Ignore censoring (explicit waiver, not an oversight)",
    "ipcw_administrative_censoring": "IPCW (reweight by inverse probability of remaining uncensored)",
}


def _render_statistical_estimand_review(manager: WorkflowManager, state: RunState, run_dir: Path) -> None:
    """Lets a reviewer record the missing-data/censoring strategy
    (`WorkflowManager.set_statistical_estimand_strategy`) that
    `_write_statistical_estimand` needs before the `statistical_estimand`
    gate can be approved. Unlike `target_trial`'s read-only summary, this
    gate's content has no other tab that already owns setting it, so the
    form to actually choose it lives here.
    """
    run_id = state.run_id
    protocol = _read_json(run_dir / "protocol.json") or {}
    cohort_manifest = _read_json(run_dir / "cohort_manifest.json")
    settings = protocol.get("query_settings") or {}
    censoring_definition_approved = bool(settings.get("censoring_definition_approved", False))
    saved = protocol.get("statistical_estimand_settings")
    saved = saved if isinstance(saved, dict) else {}

    with st.expander("Statistical estimand (statistical_estimand gate content)", expanded=True):
        if cohort_manifest is None:
            st.warning(
                "No cohort_manifest.json yet -- build the cohort plan first (Feasibility & cohort "
                "tab) so the adjustment set actually built into cohort.sql is known."
            )
        else:
            adjustment_set = [
                column["name"] for column in cohort_manifest["columns"]
                if column["role"] == "baseline_covariate"
            ]
            st.markdown(
                f"**Adjustment set (from cohort_manifest.json)**  \n{', '.join(adjustment_set) or '—'}"
            )
        st.markdown(
            "**Per-row `censored` column approved?**  \n"
            + ("Yes" if censoring_definition_approved else "No")
        )
        if not censoring_definition_approved:
            st.caption(
                "Choosing 'Exclude censored rows' or 'IPCW' below requires approving the per-row "
                "censoring definition first (Feasibility & cohort tab)."
            )
        if saved:
            st.caption(
                f"Currently saved: missing_data_strategy={saved.get('missing_data_strategy')}, "
                f"censoring_strategy={saved.get('censoring_strategy')}, "
                f"rationale=\"{saved.get('censoring_rationale')}\""
            )

        missing_data_options = sorted(MISSING_DATA_STRATEGIES)
        censoring_options = sorted(CENSORING_STRATEGIES)
        with st.form(key=f"statistical_estimand_form_{run_id}"):
            missing_data_strategy = st.selectbox(
                "Missing-data strategy",
                missing_data_options,
                index=(
                    missing_data_options.index(saved["missing_data_strategy"])
                    if saved.get("missing_data_strategy") in missing_data_options else 0
                ),
                format_func=lambda v: _MISSING_DATA_STRATEGY_LABELS.get(v, v),
                key=f"missing_data_strategy_{run_id}",
            )
            censoring_strategy = st.selectbox(
                "Censoring strategy",
                censoring_options,
                index=(
                    censoring_options.index(saved["censoring_strategy"])
                    if saved.get("censoring_strategy") in censoring_options else 0
                ),
                format_func=lambda v: _CENSORING_STRATEGY_LABELS.get(v, v),
                key=f"censoring_strategy_{run_id}",
            )
            rationale = st.text_area(
                "Rationale (required)",
                value=saved.get("censoring_rationale", ""),
                key=f"censoring_rationale_{run_id}",
            )
            reviewer = st.text_input(
                "Reviewer", value=_reviewer_default(), key=f"statistical_estimand_reviewer_{run_id}"
            )
            if st.form_submit_button("Save missing-data/censoring strategy"):
                try:
                    manager.set_statistical_estimand_strategy(
                        run_id,
                        missing_data_strategy=missing_data_strategy,
                        censoring_strategy=censoring_strategy,
                        censoring_rationale=rationale,
                        reviewer=reviewer,
                    )
                    st.success("Statistical estimand strategy saved")
                    st.rerun()
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))


def _render_gate_card(manager: WorkflowManager, state: RunState, gate_name: str) -> None:
    gate = state.gates[gate_name]
    info = _GATE_INFO.get(gate_name, {"title": gate_name, "phase": "—", "description": ""})
    with st.container(border=True):
        header_col, badge_col = st.columns([3, 1])
        header_col.markdown(f"**{info['title']}**")
        badge_col.markdown(_badge(gate.status), unsafe_allow_html=True)
        phase_label = _PHASE_LABELS.get(info["phase"], info["phase"])
        st.caption(f"Gates leaving the {phase_label} phase. {info['description']}")
        if gate.reviewer:
            st.caption(
                f"Last decision by {html.escape(gate.reviewer)} at {gate.decided_at}"
                + (f" — {html.escape(gate.comment)}" if gate.comment else "")
            )
        with st.form(key=f"gate_form_{state.run_id}_{gate_name}"):
            reviewer = st.text_input(
                "Reviewer", value=_reviewer_default(), key=f"reviewer_{state.run_id}_{gate_name}"
            )
            comment = st.text_area("Comment (optional)", key=f"comment_{state.run_id}_{gate_name}")
            status = st.selectbox(
                "Decision",
                ("approved", "rejected", "provisional"),
                key=f"status_{state.run_id}_{gate_name}",
            )
            if st.form_submit_button("Save decision", type="primary"):
                try:
                    manager.decide_gate(state.run_id, gate_name, status, reviewer, comment or None)
                    st.success(f"{info['title']} set to {status}")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))


def _advance_phase_button(manager: WorkflowManager, state: RunState) -> None:
    if st.button("Advance to next phase", type="primary", icon=":material/arrow_forward:"):
        try:
            manager.advance(state.run_id)
            st.success("Phase advanced")
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))


def _render_gates_tab(manager: WorkflowManager, state: RunState, run_dir: Path) -> None:
    st.subheader("Current task")
    st.caption(
        "Complete the current review decision before advancing. Other gates are shown below in "
        "workflow order for reference."
    )
    _render_target_trial_summary(run_dir)

    current_gate_name = _PHASE_TO_GATE.get(state.phase)
    if current_gate_name is None:
        st.info(f"Current action: {_next_action_text(state)}", icon=":material/arrow_forward:")
        _advance_phase_button(manager, state)
    else:
        if current_gate_name == "statistical_estimand":
            _render_statistical_estimand_review(manager, state, run_dir)
        _render_gate_card(manager, state, current_gate_name)
        current_gate = state.gates[current_gate_name]
        may_advance = current_gate.status == "approved" or (
            current_gate_name == "clinical_mapping"
            and state.mode == "exploratory"
            and current_gate.status == "provisional"
        )
        if may_advance:
            st.space("small")
            _advance_phase_button(manager, state)
        else:
            st.caption("Save an approved decision to unlock the next phase.")

    hypothesis = _read_json(run_dir / "hypothesis.json")
    if (
        current_gate_name == "identification_estimand"
        and hypothesis is not None
        and state.gates["identification_estimand"].status != "approved"
    ):
        try:
            artifact = _hypothesis_artifact_from_dict(hypothesis)
            identification = review_dag(artifact).identification
            if identification.identified_by_backdoor_adjustment:
                st.success(
                    "Causal DAG is identified by backdoor adjustment — the identification_estimand "
                    "gate is clear to approve on that front. See Study design for details."
                )
            else:
                st.error(
                    "Causal DAG is **not** identified by backdoor adjustment — approving the "
                    "identification_estimand gate will be rejected until this is resolved. See Study "
                    "design for the blocking notes."
                )
        except (KeyError, ValueError):
            pass

    ordered_gate_names = [
        _PHASE_TO_GATE[phase]
        for phase in PHASES
        if phase in _PHASE_TO_GATE and _PHASE_TO_GATE[phase] in state.gates
    ]
    remaining_gate_names = [name for name in state.gates if name not in ordered_gate_names]
    with st.expander("All approval gates", icon=":material/fact_check:"):
        for gate_name in ordered_gate_names + remaining_gate_names:
            if gate_name != current_gate_name:
                _render_gate_card(manager, state, gate_name)


def _render_hypothesis_tab(manager: WorkflowManager, run_id: str, run_dir: Path) -> None:
    hypothesis = _read_json(run_dir / "hypothesis.json")
    if hypothesis is None:
        st.info("No hypothesis.json for this run yet.")
        return

    st.subheader("Normalized causal question")
    col1, col2 = st.columns(2)
    col1.markdown(f"**Population**  \n{hypothesis.get('population', '—')}")
    col1.markdown(f"**Treatment**  \n{hypothesis.get('treatment', '—')}")
    col1.markdown(f"**Comparator**  \n{hypothesis.get('comparator', '—')}")
    col1.markdown(f"**Outcome**  \n{hypothesis.get('outcome', '—')}")
    col2.markdown(f"**Time zero**  \n{hypothesis.get('time_zero', '—')}")
    col2.markdown(f"**Grace period**  \n{hypothesis.get('grace_period_hours', '—')} hours")
    col2.markdown(f"**Follow-up**  \n{hypothesis.get('follow_up_days', '—')} days")
    col2.markdown(f"**Primary estimand**  \n{hypothesis.get('primary_estimand', '—')}")

    unresolved = hypothesis.get("unresolved_issues") or []
    if unresolved:
        st.warning("Unresolved hypothesis issues:\n" + "\n".join(f"- `{issue}`" for issue in unresolved))

    st.divider()
    st.subheader("Causal DAG & identification")
    try:
        artifact = _hypothesis_artifact_from_dict(hypothesis)
        review = review_dag(artifact)
    except (KeyError, ValueError) as exc:
        st.warning(
            "Could not build the causal DAG from this run's hypothesis.json "
            f"(likely an older artifact schema): {exc}"
        )
        return
    identification = review.identification
    clinical = review.clinical
    evidence = review.evidence
    adjustment = review.adjustment

    st.warning(review.caveat)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Variables", len(identification.dag.nodes), border=True)
    metric_cols[1].metric("Confounders", identification.confounder_count, border=True)
    metric_cols[2].metric(
        "Backdoor identified",
        "Yes" if identification.identified_by_backdoor_adjustment else "No",
        border=True,
    )
    metric_cols[3].metric("Adjustment sets", len(identification.minimum_adjustment_sets), border=True)

    adjustment_candidate_count = len(identification.dag.adjustment_candidate_ids())
    if identification.confounder_count > RECOMMENDED_MAX_CONFOUNDERS:
        st.warning(
            f"{identification.confounder_count} confounders (recommended limit "
            f"{RECOMMENDED_MAX_CONFOUNDERS}) — unlike other roles, confounders can never be "
            "excluded from the backdoor search, and it gets combinatorially slower per "
            "confounder. Past this point, checking identification can take well over a minute. "
            "Consider whether every confounder here is actually load-bearing."
        )
    if adjustment_candidate_count >= DEFAULT_MAX_ADJUSTMENT_CANDIDATES:
        st.error(
            f"{adjustment_candidate_count}/{DEFAULT_MAX_ADJUSTMENT_CANDIDATES} adjustment "
            "candidates — at the hard cap, the identification_estimand gate will refuse to "
            "approve until a candidate is reclassified or rejected."
        )

    with st.container(border=True):
        st.markdown("**Candidate causal DAG**")
        st.caption(
            "Read left to right: arrows represent the assumed causal structure. "
            "Treatment and outcome nodes have a thicker border."
        )
        st.graphviz_chart(_dag_to_dot(identification.dag), width="stretch", height=520)
        st.caption(
            ":blue-badge[Treatment] :red-badge[Outcome] :green-badge[Confounder] "
            ":orange-badge[Mediator] :violet-badge[Collider] :gray-badge[Other role] · "
            "Solid arrows are evidence-supported or expert-approved; dashed arrows remain proposed."
        )

    adjustment_col, review_col = st.columns([3, 2], vertical_alignment="top")
    with adjustment_col:
        st.subheader("Adjustment strategy")
        if identification.minimum_adjustment_sets:
            st.caption("Adjust for one complete set below; do not combine partial sets.")
            for index, adjustment_set in enumerate(identification.minimum_adjustment_sets, start=1):
                labels = [identification.dag.node_map[node_id].label for node_id in adjustment_set]
                with st.container(border=True):
                    st.markdown(f"**Set {index}**")
                    st.write(", ".join(labels) if labels else "No covariate adjustment required by this DAG.")
            if adjustment.core_variable_labels or adjustment.alternative_variable_labels:
                st.caption(
                    "Core variables: "
                    f"{', '.join(adjustment.core_variable_labels) or '—'}. "
                    "Alternatives: "
                    f"{', '.join(adjustment.alternative_variable_labels) or '—'}."
                )
        else:
            st.info("No minimum adjustment set is available for this DAG.", icon=":material/info:")
        if adjustment.unused_confounder_labels:
            st.warning(
                "Review these declared confounders; they do not appear in a minimum adjustment set: "
                + ", ".join(adjustment.unused_confounder_labels),
                icon=":material/warning:",
            )

    with review_col:
        st.subheader("Review status")
        if clinical.approved_for_dag_gate:
            st.success("Clinical review is ready for the DAG gate.", icon=":material/check_circle:")
        else:
            st.error("Clinical review is blocked.", icon=":material/error:")
        st.caption(f"Comparator represented in the DAG: {artifact.comparator or '—'}")
        if clinical.blocking_issues:
            st.error("\n".join(f"- {issue}" for issue in clinical.blocking_issues))
        if clinical.advisory_notes:
            st.warning("\n".join(f"- {note}" for note in clinical.advisory_notes))
        if identification.technical_notes:
            with st.expander("Identification notes", icon=":material/functions:"):
                st.caption("Graph-theoretic checks only; not a clinical judgment.")
                st.markdown("\n".join(f"- {note}" for note in identification.technical_notes))

    if evidence.total_items:
        with st.expander("Evidence coverage", icon=":material/menu_book:"):
            evidence_metric_cols = st.columns(4)
            evidence_metric_cols[0].metric("Literature only", evidence.literature_only, border=True)
            evidence_metric_cols[1].metric("Knowledge graph only", evidence.kg_only, border=True)
            evidence_metric_cols[2].metric("Corroborated", evidence.corroborated, border=True)
            evidence_metric_cols[3].metric("Uncited", evidence.uncited, border=True)
            st.caption(
                f"{evidence.cited_fraction:.0%} of DAG items ({evidence.total_items} total) carry at least "
                f"one citation; {evidence.corroborated_fraction:.0%} are corroborated by both sources."
            )

    evidence_rows = [
        {
            "Item": variable.label,
            "Role": variable.role.replace("_", " "),
            "Evidence id": variable.pmid or "—",
            "Evidence level": variable.evidence_level.replace("_", " "),
            "Knowledge-graph corroborated": "✓" if variable.kg_evidence_id else "",
        }
        for variable in artifact.candidate_dag
    ] + [
        {
            "Item": f"{relation.source} → {relation.target}",
            "Role": "covariate relation",
            "Evidence id": relation.pmid or "—",
            "Evidence level": relation.evidence_level.replace("_", " "),
            "Knowledge-graph corroborated": "✓" if relation.kg_evidence_id else "",
        }
        for relation in artifact.covariate_relations
    ]
    if evidence_rows:
        st.markdown("**Evidence sources**")
        st.caption(
            "Evidence id is a PubMed/OpenAlex citation or a knowledge-graph relation id. "
            "Evidence level is the citation's tier (systematic review > RCT > observational). "
            "Knowledge-graph corroborated marks items where the knowledge graph independently "
            "confirmed a literature-cited relation."
        )
        st.dataframe(evidence_rows, hide_index=True, width="stretch")

    dag_unresolved = (_read_json(run_dir / "protocol.json") or {}).get("candidate_dag_unresolved") or []
    if dag_unresolved:
        st.markdown("**Flagged unresolved by the drafting model**")
        st.caption(
            "The DAG-drafting step itself could not confidently classify these — distinct "
            "from the human covariate review below."
        )
        st.json(dag_unresolved)

    tournament = (_read_json(run_dir / "protocol.json") or {}).get("dag_tournament")
    if isinstance(tournament, dict):
        st.markdown("**Draft DAG comparison**")
        st.caption(
            f"{tournament.get('candidate_count')} draft DAG(s) were generated independently. "
            f"{tournament.get('dropped_duplicate_count')} were dropped as near-identical to "
            f"another draft. The remaining {tournament.get('survivor_count')} were combined into "
            "the DAG shown above."
        )
        conflicts = tournament.get("conflicts") or []
        if conflicts:
            st.warning(
                "These disagreements between drafts need human review (never resolved "
                "automatically):\n"
                + "\n".join(f"- [{conflict['kind']}] {conflict['description']}" for conflict in conflicts)
            )

        pairwise_distances = tournament.get("pairwise_distances") or []
        merge_fidelity = tournament.get("merge_fidelity") or []
        if pairwise_distances or merge_fidelity:
            with st.expander("Agreement between independently drafted DAGs", icon=":material/compare_arrows:"):
                agreement_column = st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0)
                if pairwise_distances:
                    comparisons = []
                    all_structural_scores = []
                    all_causal_scores = []
                    expert_rows = []
                    for entry in pairwise_distances:
                        structural_score = _dag_agreement_score(
                            entry["shd"], entry.get("variable_count"), max_structural_hamming_distance
                        )
                        causal_a_to_b = _dag_agreement_score(
                            entry["sid_a_given_b"], entry.get("variable_count"), max_structural_intervention_distance
                        )
                        causal_b_to_a = _dag_agreement_score(
                            entry["sid_b_given_a"], entry.get("variable_count"), max_structural_intervention_distance
                        )
                        if structural_score is not None:
                            all_structural_scores.append(structural_score)
                        all_causal_scores.extend(
                            score for score in (causal_a_to_b, causal_b_to_a) if score is not None
                        )
                        candidate_a = entry["candidate_a"]
                        candidate_b = entry["candidate_b"]
                        causal_text = (
                            f"{causal_a_to_b:.0%} (draft {candidate_a} reference) · "
                            f"{causal_b_to_a:.0%} (draft {candidate_b} reference)"
                            if causal_a_to_b is not None and causal_b_to_a is not None
                            else "Not available for this historical run"
                        )
                        comparisons.append(
                            {
                                "Compared drafts": f"Draft {candidate_a} vs draft {candidate_b}",
                                "Relationship similarity (SHD)": structural_score,
                                "Causal implication similarity (SID)": causal_text,
                            }
                        )
                        expert_rows.append(
                            {
                                "Compared drafts": f"Draft {candidate_a} vs draft {candidate_b}",
                                "Variables compared": entry.get("variable_count"),
                                "SHD (raw disagreements)": entry["shd"],
                                "Relationship similarity (SHD)": structural_score,
                                f"SID with draft {candidate_a} as reference": entry["sid_a_given_b"],
                                f"Causal similarity, draft {candidate_a} reference": causal_a_to_b,
                                f"SID with draft {candidate_b} as reference": entry["sid_b_given_a"],
                                f"Causal similarity, draft {candidate_b} reference": causal_b_to_a,
                            }
                        )

                    summary_cols = st.columns(3)
                    summary_cols[0].metric("Draft pairs compared", len(comparisons), border=True)
                    summary_cols[1].metric(
                        "Lowest relationship similarity",
                        f"{min(all_structural_scores):.0%}" if all_structural_scores else "Not available",
                        border=True,
                        help="The most structurally different pair. 100% means identical arrows and directions.",
                    )
                    summary_cols[2].metric(
                        "Lowest causal implication similarity",
                        f"{min(all_causal_scores):.0%}" if all_causal_scores else "Not available",
                        border=True,
                        help="The most different directional comparison for predicted intervention conclusions.",
                    )
                    st.info(
                        "This is a consistency check across independently generated drafts — it does **not** "
                        "establish that any DAG is clinically correct.",
                        icon=":material/info:",
                    )
                    st.markdown("**Plain-language comparison**")
                    st.caption(
                        "Relationship similarity asks: *do the drafts draw the same arrows in the same "
                        "direction?* Causal implication similarity asks: *could the differences alter an "
                        "interventional conclusion?* Both are shown because a small drawing difference may "
                        "or may not matter for causal inference."
                    )
                    st.dataframe(
                        comparisons,
                        column_config={"Relationship similarity (SHD)": agreement_column},
                        hide_index=True,
                        width="stretch",
                    )
                if merge_fidelity:
                    st.markdown("**Does the final merged DAG preserve each draft's causal conclusions?**")
                    st.caption(
                        "100% means the merged DAG retains the causal implications of that draft. "
                        "A lower value indicates that a flagged merge conflict excluded some variables "
                        "or relationships and requires review."
                    )
                    st.dataframe(
                        [
                            {
                                "Candidate": entry["candidate_index"],
                                "Causal conclusions preserved": _dag_agreement_score(
                                    entry["sid_to_merged"],
                                    entry.get("variable_count"),
                                    max_structural_intervention_distance,
                                ),
                            }
                            for entry in merge_fidelity
                        ],
                        column_config={"Causal conclusions preserved": agreement_column},
                        hide_index=True,
                        width="stretch",
                    )
                if pairwise_distances:
                    with st.expander("Expert detail: SHD, SID, direction and raw values", icon=":material/school:"):
                        st.markdown("**How the percentages are calculated**")
                        st.markdown(
                            "- **Relationship similarity (SHD)** compares every unordered pair of variable "
                            "labels: no arrow, arrow A → B, or arrow B → A. The displayed percentage is "
                            "`1 − SHD / [n(n−1)/2]`, where *n* is the number of unique labels across the pair.\n"
                            "- **Causal implication similarity (SID)** checks every ordered intervention/outcome "
                            "pair. The percentage is `1 − SID / [n(n−1)]`. SID is directional: \"draft A as "
                            "reference\" asks whether draft B preserves causal conclusions implied by draft A.\n"
                            "- **Important limitation:** variables are matched by their displayed labels. Different "
                            "wording for the same clinical concept (for example, `age` versus `patient age`) "
                            "is counted as a difference."
                        )
                        st.markdown("**Pairwise values**")
                        st.dataframe(
                            expert_rows,
                            column_config={
                                "Relationship similarity (SHD)": agreement_column,
                                **{
                                    column: agreement_column
                                    for column in expert_rows[0]
                                    if column.startswith("Causal similarity")
                                },
                            },
                            hide_index=True,
                            width="stretch",
                        )
                        st.caption(
                            "Raw SHD/SID counts and the number of compared variables are also retained in "
                            "protocol.json for reproducibility."
                        )
                with st.expander("Further technical notes", icon=":material/menu_book:"):
                    st.caption(
                        "Structural agreement is based on Structural Hamming Distance (SHD): the "
                        "count of relationship disagreements between two drafts, rescaled to a "
                        "0-100% score. Causal agreement is based on Structural Intervention "
                        "Distance (SID; Peters & Bühlmann, 2015), which checks whether a "
                        "disagreement would actually change a computed causal effect -- not "
                        "symmetric, so both directions are shown separately (see "
                        "docs/dag_generation_audit.md §9). \"Preserved in final DAG\" applies the "
                        "same SID-based rescaling to the distance from each draft's own DAG to the "
                        "merged DAG (Proposition 7 implies it should always read 100% unless a "
                        "merge conflict excluded something). Runs from before this scoring was "
                        "added show blank scores here. Raw SHD/SID counts for every pair are always "
                        "saved in this run's protocol.json (see Artifacts below)."
                    )

    if artifact.unresolved_covariates:
        st.divider()
        st.subheader("Unresolved covariate review")
        st.caption(
            "These first-pass covariates have no assigned causal role yet, so they are "
            "excluded from the DAG above pending review. Major confounders (marked) block "
            "DAG approval until classified or explicitly rejected. Rows left on \"Skip for "
            "now\" are not submitted, so you can decide a subset now and the rest later -- "
            "one submit reviews every row you did set, in a single page reload."
        )
        st.caption(
            f"Already accepted: {identification.confounder_count} confounder(s). Recommended "
            f"limit is {RECOMMENDED_MAX_CONFOUNDERS} — classifying more of the rows below as "
            "\"confounder\" past that point will make identification checks noticeably slower "
            "(see the warning above)."
        )
        major_set = set(artifact.major_confounders)

        def _decision_label(value: str) -> str:
            if value == "skip":
                return "Skip for now"
            if value == "reject":
                return "Reject"
            return value.replace("_", " ")

        suggestions_key = f"_covariate_suggestions_{run_id}"
        if st.button(
            "Suggest classifications",
            icon=":material/auto_awesome:",
            key=f"covariate_suggest_button_{run_id}",
            help=(
                "Asks the configured LLM for a starting role/timing/domain guess for every row "
                "below, pre-filling the fields so reviewing is confirm-or-correct instead of "
                "starting blank. Nothing is saved until you submit -- edit or leave any row on "
                "\"Skip for now\" as usual."
            ),
        ):
            reason_by_label = {
                item.get("label"): item.get("reason")
                for item in dag_unresolved
                if isinstance(item, dict)
            }
            unresolved_payload = [
                {
                    "label": label,
                    "reason": reason_by_label.get(label) or "No classification reason recorded.",
                }
                for label in artifact.unresolved_covariates
            ]
            try:
                resolved_llm = build_hypothesis_llm()
                with st.spinner(f"Asking `{resolved_llm.provider}:{resolved_llm.model}` for suggestions..."):
                    prompt = build_covariate_suggestion_prompt(
                        DEFAULT_COVARIATE_SUGGESTION_PROMPT_PATH, hypothesis, unresolved_payload,
                    )
                    suggestions = parse_covariate_suggestion_response(
                        resolved_llm.call(prompt), set(artifact.unresolved_covariates),
                    )
                st.session_state[suggestions_key] = suggestions
                # Set each row's widget state directly (not via `index=` on the
                # selectbox below) so it takes effect on the very next render --
                # once a keyed widget has rendered once, `index=`/`value=` is
                # only consulted the *first* time that key appears in
                # `st.session_state`, session_state wins on every render after.
                for label, suggestion in suggestions.items():
                    st.session_state[f"covariate_decision_{run_id}_{label}"] = suggestion.decision
                    if suggestion.temporal_classification:
                        st.session_state[f"covariate_temporal_{run_id}_{label}"] = (
                            suggestion.temporal_classification
                        )
                    if suggestion.domain:
                        st.session_state[f"covariate_domain_{run_id}_{label}"] = suggestion.domain
                if not suggestions:
                    st.warning("No suggestion was confident enough to propose for any row.")
                st.rerun()
            except (ValueError, RuntimeError, OSError) as exc:
                st.error(str(exc))

        cached_suggestions: dict[str, Any] = st.session_state.get(suggestions_key, {})

        with st.form(key=f"covariate_batch_form_{run_id}"):
            reviewer = st.text_input(
                "Reviewer", value=_reviewer_default(), key=f"covariate_reviewer_{run_id}"
            )
            rows = []
            for label in artifact.unresolved_covariates:
                is_major = label in major_set
                suggestion = cached_suggestions.get(label)
                # Collapsed by default (16-25+ of these on a real DAG was the
                # single biggest contributor to this tab's page length, even
                # after compacting each row's fields into columns below) --
                # major confounders stay expanded since they block DAG
                # approval and need attention now; same
                # expand-what's-actionable philosophy as
                # `_render_covariate_card`'s `_covariate_needs_review`.
                header = label + (" 🔴 major confounder" if is_major else "")
                if suggestion is not None:
                    header += f" — 🤖 suggested: {_decision_label(suggestion.decision)}"
                with st.expander(header, expanded=is_major):
                    if suggestion is not None:
                        st.caption(f"🤖 Suggested: {_decision_label(suggestion.decision)} — {suggestion.rationale}")
                    decision_col, temporal_col, domain_col = st.columns(3)
                    decision_choice = decision_col.selectbox(
                        "Decision",
                        ("skip", "reject") + DAG_VARIABLE_ROLES,
                        key=f"covariate_decision_{run_id}_{label}",
                        format_func=_decision_label,
                    )
                    temporal_choice = temporal_col.selectbox(
                        "Temporal classification",
                        DAG_VARIABLE_TEMPORAL_CLASSES,
                        key=f"covariate_temporal_{run_id}_{label}",
                        format_func=_humanize,
                        help="Relative to treatment.",
                    )
                    domain_choice = domain_col.selectbox(
                        "OMOP domain (optional)",
                        ("",) + DAG_VARIABLE_DOMAINS,
                        key=f"covariate_domain_{run_id}_{label}",
                        format_func=lambda value: "Not yet assessed" if value == "" else value,
                        help="Enables a concept lookup later.",
                    )
                    comment = st.text_area(
                        "Rationale" if decision_choice not in ("skip", "reject") else "Comment (optional)",
                        key=f"covariate_comment_{run_id}_{label}",
                    )
                    rows.append((label, decision_choice, temporal_choice, domain_choice, comment))
            submitted = st.form_submit_button("Submit reviewed decisions")

        if submitted:
            reviewed = [row for row in rows if row[1] != "skip"]
            if not reviewed:
                st.warning('No covariate decisions to submit -- every row is still on "Skip for now".')
            else:
                succeeded, errors = [], []
                for label, decision_choice, temporal_choice, domain_choice, comment in reviewed:
                    try:
                        if decision_choice == "reject":
                            manager.review_covariate(run_id, label, "reject", reviewer, comment or None)
                        else:
                            manager.review_covariate(
                                run_id, label, "accept", reviewer, comment,
                                role=decision_choice, temporal_classification=temporal_choice,
                                domain=domain_choice or None,
                            )
                        st.success(f"'{label}' reviewed")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

    if artifact.rejected_covariates:
        st.divider()
        with st.expander(f"Reviewed exclusions ({len(artifact.rejected_covariates)})"):
            for rejection in artifact.rejected_covariates:
                detail = f" — {rejection.comment}" if rejection.comment else ""
                when = f" ({rejection.decided_at})" if rejection.decided_at else ""
                st.markdown(f"- **{rejection.label}** rejected by {rejection.reviewer}{detail}{when}")

    if artifact.design_constraints:
        st.divider()
        st.subheader("Generation notes")
        st.caption(
            "Auditable notes recorded while this draft was generated, including whether an "
            "independent critique step reviewed and revised the DAG."
        )
        for note in artifact.design_constraints:
            st.markdown(f"- {note}")

    llm_outputs = _read_json(run_dir / "llm_outputs.json")
    if llm_outputs and llm_outputs.get("outputs"):
        st.divider()
        with st.expander(f"Generation audit trail ({len(llm_outputs['outputs'])} LLM call(s))"):
            st.caption(
                f"Model `{llm_outputs.get('model', 'unknown')}`, prompt version "
                f"`{llm_outputs.get('prompt_version', '—')}`. Raw, unedited LLM responses."
            )
            for index, output in enumerate(llm_outputs["outputs"], start=1):
                st.markdown(f"**{index}. `{output.get('stage', 'unknown')}`**")
                st.code(output.get("raw_text", ""), language="json")


def _omcp_lookup(client: OMCPClient, domain: str, term: str) -> Any:
    """Dispatch to the right `OMCPClient` search method for `domain`, mirroring
    `causal.agents.data_science.DataScienceAgent._lookup_candidates`'s own
    condition/drug/(everything else) split -- the same rule, kept in one
    place, now shared by the analysis-plan phenotype/covariate lookups
    (via `DataScienceAgent`) and this app's live concept-mapping search."""
    if domain == "condition":
        return client.lookup_condition(term)
    if domain == "drug":
        return client.lookup_drug(term)
    return client.lookup_concept(term, domain)


def _concept_option_label(candidate: dict[str, Any]) -> str:
    """Multiselect option label for one search result: name and OMOP concept
    id. Options are always pre-sorted best-match-first (`_sorted_by_match_score`),
    so relative match quality is already conveyed by list order without also
    spelling out the raw edit-distance score here -- see the "Technical
    detail" table alongside this multiselect for the exact number."""
    return f"{candidate['concept_name']} — OMOP concept {candidate['concept_id']}"


def _sorted_by_match_score(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Closest OMCP match first (`match_score` is an edit distance, lower is
    closer); candidates missing a score (older OMCP responses) sort last
    rather than first, so an unscored row never looks like the best match."""
    return sorted(
        candidates,
        key=lambda c: c.get("match_score") if c.get("match_score") is not None else float("inf"),
    )


def _format_concept_ids(concept_ids: Any, candidates: Any) -> str:
    """Render approved OMOP concept ids as "name (id)" for a clinician
    reading the summary table, resolving names from `candidates` (this run's
    already-loaded `concept_candidates`, see `DataScienceAgent._attach_concept_candidates`)
    -- no extra OMCP lookup needed. Falls back to the bare id when a concept
    was approved from a search whose results this run's artifact no longer
    carries (e.g. approved before `concept_candidates` was persisted)."""
    if not concept_ids:
        return "—"
    name_by_id = {c.get("concept_id"): c.get("concept_name") for c in candidates or ()}
    return ", ".join(
        f"{name_by_id[cid]} ({cid})" if name_by_id.get(cid) else str(cid) for cid in concept_ids
    )


def _parse_concept_rows(text: str, expected_domain: str) -> list[dict[str, Any]]:
    """Parse an OMCP `Lookup_*`/`Get_Concepts_By_Id` CSV response into
    reviewable rows, dropping anything that doesn't parse as a concept_id or
    whose `domain_id` doesn't match `expected_domain` -- the same
    defense-in-depth filter `DataScienceAgent._parse_concept_candidates`
    applies, since the server already filters by domain and this is a
    belt-and-suspenders check, not the primary guarantee. `match_score`
    (OMCP's `name_edit_distance` -- lower is a closer match) is `None` when
    the response predates that column, e.g. `Get_Concepts_By_Id`."""
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            concept_id = int(row.get("concept_id", ""))
        except (TypeError, ValueError):
            continue
        domain_id = str(row.get("domain_id", "")).strip()
        if domain_id.lower() != expected_domain:
            continue
        try:
            match_score = float(row["name_edit_distance"])
        except (KeyError, TypeError, ValueError):
            match_score = None
        rows.append({
            "concept_id": concept_id,
            "concept_name": str(row.get("concept_name", "")).strip(),
            "vocabulary_id": str(row.get("vocabulary_id", "")).strip(),
            "domain_id": domain_id,
            "match_score": match_score,
        })
    return rows


def _render_omcp_concept_search(*, state_key: str, domain: str, default_term: str) -> list[dict[str, Any]]:
    """Search box + button driving one live OMCP concept lookup, with results
    cached in `st.session_state[state_key]` so the multiselect/save step a
    caller renders right after this (a separate `st.form`, since a search
    button can't itself live inside the save form) sees the same result set
    across the rerun that submitting it triggers. `state_key` should embed
    the currently selected domain so switching domains hides stale results
    from the previous one instead of showing a mismatched list.
    """
    term = st.text_input("Search term", value=default_term, key=f"{state_key}_term")
    if st.button("Search", key=f"{state_key}_btn"):
        try:
            datasets = BigQueryDatasets.from_env()
            client = OMCPClient(datasets)
            with st.spinner(f"Searching for '{term}'..."):
                response = _omcp_lookup(client, domain, term)
            st.session_state[state_key] = _parse_concept_rows(response.text, domain)
        except (ValueError, RuntimeError, OSError) as exc:
            st.session_state[state_key] = []
            st.error(str(exc))
    return st.session_state.get(state_key, [])


# role (as free-text as `concept_sets[name].role` allows) -> a *starting
# guess* for which OMOP domain a phenotype in that role usually lives in.
# Only ever picks the selectbox's initial position -- a reviewer sees and
# can change it before anything is written -- but it beats always defaulting
# to `options[0]` ("condition") purely by accident of tuple order regardless
# of role, which is silently wrong for e.g. a drug-based treatment/comparator.
_ROLE_DOMAIN_HINTS: dict[str, str] = {
    "population": "condition",
    "outcome": "condition",
    "treatment": "drug",
    "comparator": "drug",
    "treatment_and_comparator": "drug",
}


def _default_domain_index(role: str, current_domain: str, options: tuple[str, ...]) -> int:
    if current_domain in options:
        return options.index(current_domain)
    guess = _ROLE_DOMAIN_HINTS.get(role, options[0])
    return options.index(guess) if guess in options else 0


def _render_concept_set_mapping_forms(
    manager: WorkflowManager, run_id: str, concept_sets: dict, exposure_contrast: str
) -> None:
    """One search-and-approve form per `concept_sets` entry.

    This is now the only concept-mapping step for population/treatment/
    comparator/outcome: `_load_concept_specs` (feasibility/cohort SQL
    generation) resolves its `ConceptSetSpec`s directly from `concept_sets`
    by role, so there is no separate `query_concept_sets` form to keep in
    sync with this one -- previously a reviewer had to map the same clinical
    concept twice, with nothing checking the two agreed.
    """
    for name, definition in concept_sets.items():
        if not isinstance(definition, dict):
            continue
        role = str(definition.get("role", ""))
        current_domain = definition.get("domain") or ""
        # `death` (the fixed OMOP `death` table, no concept search needed)
        # is only offered for an outcome-role entry -- mirrors
        # `causal.orchestrator.workflow.map_concept_set`'s own role-gated
        # domain validation.
        domain_options = OUTCOME_DOMAINS if role == "outcome" else DAG_VARIABLE_DOMAINS
        with st.expander(
            f"Map '{name}' (role: {role or '—'})",
            expanded=definition.get("status") != "approved",
        ):
            if role == "comparator" and exposure_contrast == "vs_unexposed":
                st.caption(
                    "Comparator design is set to \"unexposed comparator\" above -- this study's "
                    "comparator arm is simply everyone who wasn't treated, so no OMOP concept is "
                    "needed here. Leave unmapped, or set status to \"No standard concept\" below."
                )
            domain_choice = st.selectbox(
                "OMOP domain", domain_options,
                index=_default_domain_index(role, current_domain, domain_options),
                key=f"concept_set_domain_{run_id}_{name}",
            )
            if not current_domain:
                st.caption(
                    "Domain pre-selected as a starting guess based on this concept set's role "
                    "-- confirm it's actually correct before searching/approving."
                )
            existing_ids = [int(c) for c in definition.get("concept_ids", ())]
            if domain_choice == "death":
                st.caption("`death` uses the fixed OMOP `death` table -- no concept search needed.")
                name_by_id: dict[int, str] = {}
                option_ids: list[int] = []
            else:
                candidates = _sorted_by_match_score(_render_omcp_concept_search(
                    state_key=f"concept_set_candidates_{run_id}_{name}_{domain_choice}",
                    domain=domain_choice,
                    default_term=(
                        definition.get("lookup_term") or definition.get("label") or name.replace("_", " ")
                    ),
                ))
                name_by_id = {c["concept_id"]: _concept_option_label(c) for c in candidates}
                option_ids = list(dict.fromkeys([c["concept_id"] for c in candidates] + existing_ids))
            with st.form(key=f"concept_set_map_form_{run_id}_{name}"):
                if domain_choice == "death":
                    chosen_ids: list[int] = []
                else:
                    chosen_ids = st.multiselect(
                        "Approve OMOP concept(s)", options=option_ids, default=existing_ids,
                        format_func=lambda cid: name_by_id.get(cid, f"{cid} — (previously approved, not in current search)"),
                        key=f"concept_set_map_ids_{run_id}_{name}",
                    )
                status_choice = st.selectbox(
                    "Status", ("approved", "no_standard_concept", "pending_omop_mapping"),
                    key=f"concept_set_status_{run_id}_{name}",
                    format_func=_humanize,
                )
                reviewer = st.text_input(
                    "Reviewer", value=_reviewer_default(), key=f"concept_set_reviewer_{run_id}_{name}"
                )
                submitted = st.form_submit_button("Save mapping")
                if submitted:
                    try:
                        with st.spinner("Saving..."):
                            manager.map_concept_set(
                                run_id, name, domain_choice, chosen_ids, reviewer, status=status_choice,
                            )
                        st.success(f"Mapping saved for '{name}'")
                        st.rerun()
                    except (ValueError, RuntimeError, OSError) as exc:
                        st.error(str(exc))


# exposure_contrast value -> (clinician-facing label, plain-language
# description). See `causal.queries.feasibility.EXPOSURE_CONTRASTS`'s own
# comment for the full modeling rationale behind each design.
_EXPOSURE_CONTRAST_INFO: dict[str, tuple[str, str]] = {
    "add_on": (
        "Add-on (comparator is a required backbone therapy)",
        "Everyone gets the comparator; the treatment arm additionally gets the treatment on top "
        'of it. E.g. "crystalloids plus albumin vs. crystalloids alone".',
    ),
    "mutually_exclusive": (
        "Active comparator (drug vs. drug)",
        "Keeps only people who received exactly one of treatment/comparator, each mapped to "
        'its own OMOP concept. E.g. "aspirin vs. clopidogrel".',
    ),
    "vs_unexposed": (
        "Unexposed comparator (drug vs. no drug)",
        "The comparator arm is simply everyone who did not receive the treatment -- no OMOP "
        'concept is mapped (or needed) for the comparator. E.g. "aspirin vs. no aspirin".',
    ),
}


def _render_exposure_contrast_selector(manager: WorkflowManager, run_id: str, protocol: dict) -> str:
    """Lets a reviewer choose how `treatment`/`comparator` combine into the
    cohort inclusion filter and the binary arm indicator (see
    `causal.queries.feasibility.EXPOSURE_CONTRASTS`) -- a real modeling
    decision, not a query-engine detail: getting it wrong silently changes
    who ends up in the cohort. Previously only settable by hand-editing
    protocol.json, with every run silently defaulting to "add_on".

    Returns the currently-saved value (before any unsubmitted form edit), so
    callers can adapt other widgets (e.g. the comparator mapping form) to it
    without re-reading protocol.json themselves.
    """
    settings = protocol.get("query_settings")
    current = (settings or {}).get("exposure_contrast", "add_on")
    st.markdown("**Comparator design**")
    st.caption(
        "How the comparator relates to the treatment. Getting this wrong silently changes who "
        "ends up in the cohort, so pick the one that actually matches this study's question:"
    )
    for value in EXPOSURE_CONTRASTS:
        label, description = _EXPOSURE_CONTRAST_INFO[value]
        current_marker = " — **current**" if value == current else ""
        st.caption(f"- **{label}**{current_marker}: {description}")
    with st.form(key=f"exposure_contrast_form_{run_id}"):
        choice = st.selectbox(
            "Comparator design",
            EXPOSURE_CONTRASTS,
            index=EXPOSURE_CONTRASTS.index(current) if current in EXPOSURE_CONTRASTS else 0,
            format_func=lambda value: _EXPOSURE_CONTRAST_INFO[value][0],
            key=f"exposure_contrast_choice_{run_id}",
        )
        reviewer = st.text_input(
            "Reviewer", value=_reviewer_default(), key=f"exposure_contrast_reviewer_{run_id}"
        )
        submitted = st.form_submit_button("Save comparator design")
        if submitted:
            try:
                manager.set_exposure_contrast(run_id, choice, reviewer)
                st.success("Comparator design saved")
                st.rerun()
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
    return current


def _render_concept_mapping_section(manager: WorkflowManager, run_id: str, run_dir: Path) -> None:
    """Concept mapping (`concept_sets`) lives in `protocol.json`, not
    `analysis_plan.json` -- so it's shown even before the Data Science Agent
    has run. Each entry below is editable: `Search OMCP` runs a live lookup
    (see `_render_omcp_concept_search`), and `Save mapping` calls
    `WorkflowManager.map_concept_set` -- mirroring `causalomop workflow
    concept-set-map` on the CLI, which stays available for scripted/offline
    use.

    This is the single concept-mapping step for population/treatment/
    comparator/outcome: it also feeds cohort/feasibility SQL generation
    (`causal.orchestrator.workflow._load_concept_specs` resolves query
    `ConceptSetSpec`s directly from `concept_sets`, by role). There used to
    be a second, independent `query_concept_sets` mapping step for the same
    role here, with nothing checking the two agreed -- removed.
    """
    protocol = _read_json(run_dir / "protocol.json")
    concept_sets = (protocol or {}).get("concept_sets") or {}
    if not concept_sets:
        return

    st.subheader("Concept mapping")
    st.caption(
        "OMOP concept sets used to operationalize this question's population/treatment/"
        "comparator/outcome (and any other declared phenotype). Feeds both the analysis plan "
        "below and cohort/feasibility query generation -- one mapping per role."
    )

    exposure_contrast = _render_exposure_contrast_selector(manager, run_id, protocol or {})
    st.divider()

    st.caption(
        "OMOP concept(s) are shown as numeric codes here because their matched clinical names "
        "aren't saved once approved -- expand a row below to search and see the name before "
        "approving, or re-run the search there to look one up again."
    )
    st.dataframe(
        [
            {
                "Name": name,
                "Role": _humanize(definition.get("role")),
                "OMOP domain": definition.get("domain") or "—",
                "Status": _humanize(definition.get("status")),
                "OMOP concept(s)": ", ".join(str(c) for c in definition.get("concept_ids", ())) or "none yet",
                "Description": definition.get("label", ""),
            }
            for name, definition in concept_sets.items()
            if isinstance(definition, dict)
        ],
        hide_index=True,
        width="stretch",
    )
    _render_concept_set_mapping_forms(manager, run_id, concept_sets, exposure_contrast)
    _render_population_concept_set_manager(manager, run_id, protocol or {})


def _render_population_concept_set_manager(manager: WorkflowManager, run_id: str, protocol: dict[str, Any]) -> None:
    """A population defined by more than one independently-required
    diagnosis (e.g. "adults with type 2 diabetes AND COPD") needs more than
    the single `concept_sets["population"]` entry the table/forms above
    manage -- see `WorkflowManager.add_population_concept_set`'s docstring
    for why a second required diagnosis can't just be mapped into that one
    slot. This adds/removes entries; the new entry then gets mapped/approved
    through the ordinary, unmodified forms above like any other concept set.
    """
    additional_names = tuple(protocol.get("additional_population_concept_sets", ()) or ())
    if additional_names:
        st.caption(
            "Required, ANDed with population: " + ", ".join(additional_names)
        )
        for name in additional_names:
            with st.form(key=f"remove_population_concept_set_{run_id}_{name}"):
                st.caption(f"Remove '{name}' as a required population diagnosis")
                reviewer = st.text_input(
                    "Reviewer", value=_reviewer_default(), key=f"remove_pop_reviewer_{run_id}_{name}"
                )
                comment = st.text_area(
                    "Reason (optional)", key=f"remove_pop_comment_{run_id}_{name}"
                )
                if st.form_submit_button(f"Remove '{name}'"):
                    try:
                        manager.remove_population_concept_set(run_id, name, reviewer, comment or None)
                        st.success(f"'{name}' removed")
                        st.rerun()
                    except (ValueError, RuntimeError) as exc:
                        st.error(str(exc))

    with st.expander("+ Add a required population diagnosis", expanded=False):
        st.caption(
            "Adds a second (or further) diagnosis that must ALSO be present -- ANDed with "
            "population, not an alternative to it. Only creates the entry; map/approve it via "
            "the table above afterward."
        )
        with st.form(key=f"add_population_concept_set_{run_id}"):
            name = st.text_input(
                "Short key (e.g. 'copd')", key=f"add_pop_name_{run_id}",
                help="Becomes the concept_sets dict key -- lowercase, no spaces.",
            )
            label = st.text_input(
                "Diagnosis/condition (e.g. 'Chronic obstructive pulmonary disease')",
                key=f"add_pop_label_{run_id}",
            )
            lookup_term = st.text_input(
                "OMOP search term (optional, e.g. 'COPD')", key=f"add_pop_lookup_{run_id}"
            )
            reviewer = st.text_input(
                "Reviewer", value=_reviewer_default(), key=f"add_pop_reviewer_{run_id}"
            )
            comment = st.text_area("Comment (optional)", key=f"add_pop_comment_{run_id}")
            if st.form_submit_button("Add required diagnosis"):
                try:
                    manager.add_population_concept_set(
                        run_id,
                        name=name,
                        label=label,
                        lookup_term=lookup_term,
                        reviewer=reviewer,
                        comment=comment or None,
                    )
                    st.success(f"'{name.strip()}' added -- map it to OMOP concepts above")
                    st.rerun()
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))


def _render_regenerate_plan_section(manager: WorkflowManager, run_id: str) -> None:
    """Run the Data Science Agent from the app (mirrors `workflow plan
    [--with-mcp]` on the CLI) so a reviewer can pick up covariate-review
    changes (role, domain, concept mapping, aggregation, window) into a fresh
    `analysis_plan.json` without leaving the app. Regenerating itself never
    needs OMCP -- every field it derives comes from `hypothesis.json`/
    `protocol.json` -- so refreshing OMOP concept *candidates* (a slower,
    one-lookup-per-still-unmapped-phenotype-or-covariate discovery aid, not a
    correctness gate like the concept-mapping domain check above) stays an
    explicit opt-in here, unlike that check.
    """
    with st.form(key="regenerate_analysis_plan_form"):
        refresh_candidates = st.checkbox(
            "Also refresh OMOP concept suggestions (slower -- one lookup per phenotype/"
            "covariate still pending a mapping)",
            key="regenerate_plan_with_mcp",
        )
        regenerate = st.form_submit_button("Regenerate analysis plan")
    if not regenerate:
        return
    try:
        lookup = None
        if refresh_candidates:
            datasets = BigQueryDatasets.from_env()
            lookup = OMCPClient(datasets)
        with st.spinner("Regenerating the analysis plan..."):
            manager.run_data_science_agent(run_id, lookup)
        st.success("Analysis plan regenerated.")
        st.rerun()
    except (ValueError, RuntimeError, OSError) as exc:
        st.error(str(exc))


_COVARIATE_STATUS_BADGE: dict[str, tuple[str, str]] = {
    "ready": ("Ready", "green"),
    "definition_pending": ("Needs review", "orange"),
    "unobserved": ("Not observed", "gray"),
}

_COVARIATE_MAPPING_BADGE: dict[str, tuple[str, str]] = {
    "approved": ("Mapped", "green"),
    "no_standard_concept": ("No OMOP concept", "gray"),
    "unmapped": ("Not mapped", "orange"),
}

_COVARIATE_LEAKAGE_BADGE: dict[str, tuple[str, str]] = {
    "low": ("Low leakage risk", "green"),
    "review_required": ("Review required", "orange"),
}


def _covariate_needs_review(covariate: dict[str, Any]) -> bool:
    return covariate.get("status") != "ready" or covariate.get("leakage_risk") == "review_required"


def _covariate_review_reasons(covariate: dict[str, Any]) -> list[str]:
    """Why this covariate isn't extraction-ready yet, mirroring the exact fields
    `DataScienceAgent._covariate_from_variable` checks to compute
    `extraction_ready` (mapping_status/domain/aggregation) -- surfaced here,
    not re-derived, so this can never disagree with the actual gating logic.
    """
    if covariate.get("status") == "unobserved":
        return ["Not observed for this study -- no covariate decision needed."]
    reasons = []
    if not covariate.get("domain"):
        reasons.append("OMOP domain not yet confirmed.")
    if covariate.get("mapping_status", "unmapped") != "approved":
        reasons.append("Concept mapping not yet approved.")
    if covariate.get("aggregation", "definition_pending") in ("", "definition_pending"):
        reasons.append("Aggregation method not yet chosen.")
    if covariate.get("leakage_risk") == "review_required":
        reasons.append("Ascertainment window needs clinical review (measured at/after treatment).")
    return reasons


def _humanize(token: Any) -> str:
    """Turn an internal snake_case enum/status token (e.g. "no_standard_concept")
    into space-separated words for clinician-facing display. Only ever
    reformats a code's punctuation, never its meaning -- so a still-unhandled
    status shows up as readable words instead of a raw underscored code,
    rather than silently disappearing."""
    if token in (None, ""):
        return "—"
    return str(token).replace("_", " ")


def _format_blocker(blocker: str) -> str:
    """Render one `execution_readiness.json` blocker code (see
    `causal.agents.execution.ExecutionAgent.run`) as a clinician-readable
    sentence instead of the raw `prefix:label:status`-shaped string it's
    stored as. Recognizes the fixed set of prefixes that agent emits;
    anything else (a future blocker kind, or an opaque hypothesis-level
    `unresolved_issues` entry) falls back to a generic humanized rendering
    so it stays readable without needing this function updated in lockstep.
    """
    parts = blocker.split(":")
    head = parts[0]
    if head == "analysis_plan_gate" and len(parts) == 2:
        return f"Analysis plan: {_humanize(parts[1])}."
    if head == "clinical_mapping_gate" and len(parts) == 2:
        return f"Clinical concept mapping: {_humanize(parts[1])}."
    if head == "covariate_not_ready" and len(parts) == 2:
        return f"Covariate '{parts[1]}' is not ready yet (mapping/aggregation still pending)."
    if head == "phenotype_not_approved" and len(parts) == 2:
        return f"Phenotype '{parts[1]}' is not yet approved."
    if head == "unresolved" and len(parts) >= 2:
        rest = parts[1:]
        if rest[0] == "concept_set" and len(rest) == 3:
            return f"Concept set '{rest[1]}': {_humanize(rest[2])}."
        if rest[0] == "phenotype" and len(rest) == 3:
            return f"Phenotype '{rest[1]}': {_humanize(rest[2])}."
        if rest[0] == "covariate" and len(rest) == 3:
            return f"Covariate '{rest[1]}': {_humanize(rest[2])}."
        return _humanize(":".join(rest))
    return _humanize(blocker)


def _format_covariate_window(window_start_hours: Any, window_end_hours: Any) -> str:
    """Render a covariate's ascertainment window in plain language, mapping
    `LIFETIME_LOOKBACK_HOURS` back to "any time before treatment" instead of
    its raw (and clinician-meaningless) hour count. Shared by the covariate
    table and the per-covariate detail card so the two never drift apart."""
    start_display = (
        "any time before treatment" if window_start_hours == LIFETIME_LOOKBACK_HOURS
        else f"{window_start_hours}h"
    )
    return f"{start_display} to {window_end_hours}h"


def _render_covariate_card(manager: WorkflowManager, run_id: str, c: dict[str, Any]) -> None:
    """One collapsible card per covariate: current state (badges), why a
    decision is required (if any), and the four review actions as tabs --
    domain, window, aggregation, and concept mapping. Replaces what used to
    be four stacked `st.form`s per covariate so a run with several covariates
    doesn't turn the tab into one long scroll of near-identical forms.

    Each tab's save action also calls `manager.run_data_science_agent(run_id)`
    (no `concept_lookup`, so it never hits OMCP -- same fast, filesystem-only
    path as `_render_regenerate_plan_section` with its checkbox unchecked)
    before `st.rerun()`. Without this, `analysis_plan.json` was a stale
    snapshot until a reviewer separately clicked "Regenerate analysis plan",
    which read as "my save didn't take" even though the underlying
    `candidate_dag` entry in protocol.json/hypothesis.json was updated fine.
    """
    name = c.get("name")
    domain = c.get("domain")
    status = c.get("status", "definition_pending")
    mapping_status = c.get("mapping_status", "unmapped")
    leakage_risk = c.get("leakage_risk", "review_required")

    with st.expander(f"{name} — {_humanize(c.get('causal_role'))}", expanded=_covariate_needs_review(c)):
        with st.container(horizontal=True):
            status_label, status_color = _COVARIATE_STATUS_BADGE.get(status, (_humanize(status), "gray"))
            st.badge(status_label, color=status_color)
            mapping_label, mapping_color = _COVARIATE_MAPPING_BADGE.get(
                mapping_status, (_humanize(mapping_status), "gray")
            )
            st.badge(mapping_label, color=mapping_color)
            leakage_label, leakage_color = _COVARIATE_LEAKAGE_BADGE.get(
                leakage_risk, (_humanize(leakage_risk), "gray")
            )
            st.badge(leakage_label, color=leakage_color)

        reasons = _covariate_review_reasons(c)
        if reasons:
            st.caption("Why a decision is required: " + " ".join(reasons))

        domain_tab, window_tab, aggregation_tab, mapping_tab = st.tabs(
            ["Domain", "Window", "Aggregation", "Mapping"]
        )

        with domain_tab, st.form(key=f"covariate_domain_form_{run_id}_{name}"):
            st.caption(
                f"Currently {domain or 'not yet assessed'}. The DAG-drafting step proposes one "
                "from the literature it read, but it is only a starting guess; confirm or "
                "correct it before concept search/aggregation depend on it. Changing an "
                "already-set domain resets any concept mapping/aggregation already saved "
                "for this covariate."
            )
            domain_choice = st.selectbox(
                "OMOP domain", DAG_VARIABLE_DOMAINS,
                index=DAG_VARIABLE_DOMAINS.index(domain) if domain in DAG_VARIABLE_DOMAINS else 0,
                key=f"covariate_domain_choice_{run_id}_{name}",
            )
            reviewer = st.text_input(
                "Reviewer", value=_reviewer_default(), key=f"covariate_domain_reviewer_{run_id}_{name}",
            )
            submitted = st.form_submit_button("Save domain")
            if submitted:
                try:
                    manager.review_covariate_domain(run_id, name, domain_choice, reviewer)
                    manager.run_data_science_agent(run_id)
                    st.success(f"Domain saved for '{name}'")
                    st.rerun()
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))

        with window_tab, st.form(key=f"covariate_window_form_{run_id}_{name}"):
            window_start_hours = c.get("window_start_hours")
            window_end_hours = c.get("window_end_hours")
            st.caption(
                f"Currently {_format_covariate_window(window_start_hours, window_end_hours)} "
                "relative to index (auto-derived unless overridden below). Choose a long, "
                "chronic lookback for stable history or comorbidities; choose a short, "
                "transient lookback for values or events that can change close to treatment. "
                "The window must end at or before the index time so it cannot include "
                "post-treatment information."
            )
            window_col1, window_col2 = st.columns(2)
            window_start = window_col1.number_input(
                "Window start (hours, ≤ 0)", value=int(window_start_hours or 0),
                max_value=0, step=1, key=f"covariate_window_start_{run_id}_{name}",
                help=f"{LIFETIME_LOOKBACK_HOURS} means \"any time before treatment\" -- the "
                "auto-derived default for chronic, low-risk covariates, not a typo. Use a "
                "shorter interval for transient clinical state close to treatment.",
            )
            window_end = window_col2.number_input(
                "Window end (hours, ≤ 0)", value=int(window_end_hours or 0),
                max_value=0, step=1, key=f"covariate_window_end_{run_id}_{name}",
            )
            reviewer = st.text_input(
                "Reviewer", value=_reviewer_default(), key=f"covariate_window_reviewer_{run_id}_{name}",
            )
            submitted = st.form_submit_button("Save window")
            if submitted:
                try:
                    manager.review_covariate_window(
                        run_id, name, int(window_start), int(window_end), reviewer,
                    )
                    manager.run_data_science_agent(run_id)
                    st.success(f"Window saved for '{name}'")
                    st.rerun()
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))

        with aggregation_tab:
            allowed_aggregations = COVARIATE_AGGREGATIONS_BY_DOMAIN.get(domain) if domain else None
            if not allowed_aggregations:
                st.caption("Set a domain first — aggregation choices depend on it.")
            else:
                with st.form(key=f"covariate_aggregation_form_{run_id}_{name}"):
                    st.caption(
                        "Choose what one value per person should mean in this window. For an "
                        "event domain, use presence when the question is whether it occurred "
                        "(or count for the number of occurrences); no event is recorded as 0. "
                        "For a numeric measurement, use the last pre-index value for the most "
                        "recent baseline state, or max/min/mean when the clinical question calls "
                        "for the peak, nadir, or average. A missing numeric measurement remains "
                        "NULL: it is unknown, not zero."
                    )
                    aggregation_choice = st.selectbox(
                        f"Aggregation (domain: {domain})",
                        allowed_aggregations,
                        key=f"covariate_aggregation_choice_{run_id}_{name}",
                        format_func=_humanize,
                        help="Presence/count encodes absent events as 0. Last, maximum, minimum, "
                        "and mean encode numeric values; no observed value remains NULL rather than 0.",
                    )
                    reviewer = st.text_input(
                        "Reviewer", value=_reviewer_default(),
                        key=f"covariate_aggregation_reviewer_{run_id}_{name}",
                    )
                    submitted = st.form_submit_button("Save aggregation")
                    if submitted:
                        try:
                            manager.review_covariate_aggregation(
                                run_id, name, aggregation_choice, reviewer,
                            )
                            manager.run_data_science_agent(run_id)
                            st.success(f"Aggregation saved for '{name}'")
                            st.rerun()
                        except (ValueError, RuntimeError) as exc:
                            st.error(str(exc))

        with mapping_tab:
            candidates = c.get("concept_candidates") or []
            if not candidates:
                st.caption(
                    "No OMOP concept candidates yet. First set and save a domain, then click "
                    "Regenerate analysis plan and select ‘Also refresh OMOP concept suggestions’ "
                    "to refresh candidates with OMCP."
                )
            else:
                ranked_candidates = _sorted_by_match_score(candidates)
                st.caption("Best matches first. Pick the correct OMOP concept(s) below.")
                with st.expander("Search results (technical detail)", icon=":material/data_object:"):
                    st.caption(_VOCABULARY_CAPTION)
                    st.dataframe(
                        [
                            {
                                "OMOP concept ID": candidate.get("concept_id"),
                                "Name": candidate.get("concept_name"),
                                "Vocabulary": candidate.get("vocabulary_id"),
                                "Domain": candidate.get("domain_id"),
                                "Match score (lower = closer)": candidate.get("match_score"),
                            }
                            for candidate in ranked_candidates
                        ],
                        hide_index=True,
                        width="stretch",
                    )
                name_by_id = {
                    candidate.get("concept_id"): _concept_option_label(candidate)
                    for candidate in ranked_candidates
                }
                with st.form(key=f"covariate_map_form_{run_id}_{name}"):
                    chosen_ids = st.multiselect(
                        "Approve OMOP concept(s) for this covariate",
                        options=[candidate.get("concept_id") for candidate in ranked_candidates],
                        format_func=lambda cid: name_by_id.get(cid, f"{cid} — (previously approved, not in current search)"),
                        key=f"covariate_map_ids_{run_id}_{name}",
                    )
                    mapping_status_choice = st.selectbox(
                        "Mapping status",
                        ("approved", "no_standard_concept"),
                        key=f"covariate_map_status_{run_id}_{name}",
                        format_func=_humanize,
                    )
                    reviewer = st.text_input(
                        "Reviewer", value=_reviewer_default(), key=f"covariate_map_reviewer_{run_id}_{name}"
                    )
                    submitted = st.form_submit_button("Save concept mapping")
                    if submitted:
                        try:
                            # Only "approved" mappings need OMCP's domain check
                            # (see `WorkflowManager.map_covariate_concepts`) --
                            # skip building a client for "no_standard_concept"
                            # so a submodule/credentials issue can't block that
                            # (already OMCP-free) reviewer decision.
                            concept_lookup = None
                            if mapping_status_choice == "approved":
                                datasets = BigQueryDatasets.from_env()
                                concept_lookup = OMCPClient(datasets)
                            with st.spinner("Verifying concept domain..."):
                                manager.map_covariate_concepts(
                                    run_id, name, chosen_ids, reviewer,
                                    mapping_status=mapping_status_choice,
                                    concept_lookup=concept_lookup,
                                )
                            manager.run_data_science_agent(run_id)
                            st.success(f"Concept mapping saved for '{name}'")
                            st.rerun()
                        except (ValueError, RuntimeError, OSError) as exc:
                            st.error(str(exc))


def _render_analysis_plan_tab(manager: WorkflowManager, run_id: str, run_dir: Path) -> None:
    _render_concept_mapping_section(manager, run_id, run_dir)

    st.divider()
    _render_regenerate_plan_section(manager, run_id)

    plan = _read_json(run_dir / "analysis_plan.json")
    if plan is None:
        st.info("No analysis_plan.json for this run yet (use the button above to generate it).")
        return

    st.divider()
    st.subheader("Analysis plan")
    col1, col2 = st.columns(2)
    col1.markdown(f"**Data source**  \n`{plan.get('data_source', '—')}`")
    col1.markdown(f"**Design**  \n{_humanize(plan.get('design'))}")
    col2.markdown(
        f"**OK to run in exploratory mode**  \n{'Yes' if plan.get('exploratory_execution_allowed') else 'No'}"
    )
    col2.markdown(f"**OK to run in final mode**  \n{'Yes' if plan.get('final_execution_allowed') else 'No'}")

    phenotypes = plan.get("phenotypes") or []
    if phenotypes:
        st.markdown("**Phenotypes**")
        st.dataframe(
            [
                {
                    "Name": p.get("name"),
                    "Role": _humanize(p.get("role")),
                    "OMOP domain": p.get("domain"),
                    "Mode": _humanize(p.get("mode")),
                    "Status": _humanize(p.get("status")),
                    "OMOP concept(s)": _format_concept_ids(p.get("concept_ids", ()), p.get("concept_candidates")),
                    "Matching rule": p.get("source_value_rule") or "—",
                    "OMOP tables": ", ".join(p.get("omop_tables", ())),
                }
                for p in phenotypes
            ],
            hide_index=True,
            width="stretch",
        )
        for p in phenotypes:
            candidates = p.get("concept_candidates") or []
            if candidates:
                with st.expander(f"Candidate OMOP concepts for '{p.get('name')}' ({len(candidates)})"):
                    st.caption(_VOCABULARY_CAPTION)
                    st.dataframe(
                        [
                            {
                                "OMOP concept ID": c.get("concept_id"),
                                "Name": c.get("concept_name"),
                                "Vocabulary": c.get("vocabulary_id"),
                                "Domain": c.get("domain_id"),
                                "Match score (lower = closer)": c.get("match_score"),
                            }
                            for c in _sorted_by_match_score(candidates)
                        ],
                        hide_index=True,
                        width="stretch",
                    )

    covariates = plan.get("covariates") or []
    if covariates:
        st.markdown("**Covariates** (from the reviewed causal DAG)")
        st.dataframe(
            [
                {
                    "Name": c.get("name"),
                    "Causal role": _humanize(c.get("causal_role")),
                    "Adjustment role": _humanize(c.get("adjustment_role")),
                    "Window": _format_covariate_window(
                        c.get("window_start_hours"), c.get("window_end_hours")
                    ),
                    "Leakage risk": _humanize(c.get("leakage_risk")),
                    "Status": _humanize(c.get("status")),
                    "OMOP domain": c.get("domain") or "—",
                    "Mapping status": _humanize(c.get("mapping_status", "unmapped")),
                    "OMOP concept(s)": _format_concept_ids(c.get("concept_ids", ()), c.get("concept_candidates")),
                    "Aggregation": _humanize(c.get("aggregation", "definition_pending")),
                }
                for c in covariates
            ],
            hide_index=True,
            width="stretch",
        )
        for c in covariates:
            _render_covariate_card(manager, run_id, c)

    excluded_from_adjustment = plan.get("excluded_from_adjustment") or []
    if excluded_from_adjustment:
        with st.expander(f"Excluded from adjustment ({len(excluded_from_adjustment)})"):
            for entry in excluded_from_adjustment:
                st.markdown(f"- {entry}")

    checks = plan.get("feasibility_checks") or []
    if checks:
        st.markdown("**Planned feasibility checks**")
        for check in checks:
            st.markdown(f"- {check}")


def _render_feasibility_actions(manager: WorkflowManager, state: RunState, run_id: str) -> None:
    """Trigger the steps that produce this tab's artifacts -- mirrors `workflow
    feasibility`/`cohort-plan`/`cohort-profile`/`followup-audit` on the CLI.
    All go through OMCP (the CLI's default `--transport mcp`), consistent with
    every other OMOP-querying action in this app; the CLI's direct-BigQuery
    transport is dev/test-only and stays CLI-only.
    """
    datasets = BigQueryDatasets.from_env()

    def _run(title: str, label: str, ready: bool, requirement: str, spinner: str, action, icon: str) -> None:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.badge("Ready" if ready else "Not ready", color="green" if ready else "gray", icon=icon)
            if st.button(
                label,
                disabled=not ready,
                type="primary" if ready else "secondary",
                icon=icon,
                key=f"feasibility_action_{label}",
                width="stretch",
            ):
                try:
                    with st.spinner(spinner):
                        action()
                    st.success(f"{label} completed.")
                    st.rerun()
                except (ValueError, RuntimeError, OSError) as exc:
                    st.error(str(exc))
            st.caption("Available now." if ready else requirement)

    col1, col2 = st.columns(2)
    with col1:
        _run(
            "1. Check feasibility",
            "Run feasibility check",
            state.phase == "feasibility"
            and state.mode == "exploratory"
            and state.gates["clinical_mapping"].status == "provisional",
            "Requires the feasibility phase, exploratory mode, and a provisional clinical-mapping decision.",
            "Running feasibility query...",
            lambda: manager.run_feasibility_mcp(run_id, OMCPClient(datasets), datasets),
            ":material/query_stats:",
        )
        _run(
            "3. Profile the cohort",
            "Run cohort profile",
            state.phase in {"feasibility", "analysis_plan"},
            "Requires the feasibility or analysis-plan phase and an existing analysis plan.",
            "Profiling the draft cohort...",
            lambda: manager.run_cohort_profile_mcp(run_id, OMCPClient(datasets), datasets),
            ":material/groups:",
        )
    with col2:
        _run(
            "2. Create the cohort plan",
            "Build cohort plan",
            state.phase in {"analysis_plan", "estimand_review", "execution"},
            "Requires the analysis-plan, estimand-review, or cohort-execution phase and an "
            "existing analysis plan.",
            "Building governed cohort SQL...",
            lambda: manager.build_cohort_plan(run_id, datasets),
            ":material/description:",
        )
        _run(
            "4. Audit follow-up",
            "Run follow-up audit",
            state.phase in {"feasibility", "analysis_plan"},
            "Requires the feasibility or analysis-plan phase and an existing analysis plan.",
            "Checking follow-up completeness...",
            lambda: manager.run_followup_audit_mcp(run_id, OMCPClient(datasets), datasets),
            ":material/fact_check:",
        )


def _render_feasibility_tab(manager: WorkflowManager, state: RunState, run_dir: Path) -> None:
    feasibility = _read_json(run_dir / "feasibility.json")
    cohort_manifest = _read_json(run_dir / "cohort_manifest.json")
    cohort_profile = _read_json(run_dir / "cohort_profile.json")
    followup_audit = _read_json(run_dir / "followup_audit.json")
    readiness = _read_json(run_dir / "execution_readiness.json")

    st.subheader("Feasibility and cohort readiness")
    st.caption(
        "Use aggregate diagnostics to assess whether the reviewed study design can progress. "
        "These checks do not estimate a treatment effect."
    )

    any_rendered = False

    with st.container(border=True):
        st.markdown("**Can this study proceed?**")
        if readiness:
            any_rendered = True
            cols = st.columns(3)
            feasibility_allowed = readiness.get("aggregate_feasibility_allowed", False)
            cohort_allowed = readiness.get("analytic_dataset_build_allowed", False)
            estimation_allowed = readiness.get("causal_estimation_allowed", False)
            cols[0].metric("Feasibility counts", "Allowed" if feasibility_allowed else "Blocked", border=True)
            cols[1].metric("Cohort build", "Allowed" if cohort_allowed else "Blocked", border=True)
            cols[2].metric("Causal effect estimation", "Allowed" if estimation_allowed else "Not implemented", border=True)
            if readiness.get("blockers"):
                st.warning("What still needs attention:", icon=":material/warning:")
                st.markdown("\n".join(f"- {_format_blocker(blocker)}" for blocker in readiness["blockers"]))
            elif feasibility_allowed or cohort_allowed:
                st.success("No execution blocker is recorded in the current readiness assessment.", icon=":material/check_circle:")
        else:
            st.info(
                "No readiness assessment has been generated yet. Complete the available steps below "
                "to create one.",
                icon=":material/pending:",
            )

    st.subheader("Available actions")
    st.caption("Actions are enabled only when the workflow phase and required review decisions permit them.")
    _render_feasibility_actions(manager, state, state.run_id)

    st.subheader("Aggregate results")

    if feasibility:
        any_rendered = True
        st.subheader("Provisional feasibility counts")
        st.caption(
            "Uses human-approved concept sets for this question's population/treatment/"
            "comparator/outcome — not a causal estimate."
        )
        _render_count_metrics(feasibility.get("counts", {}))
        for warning in feasibility.get("warnings", ()):
            st.warning(warning)
        with st.expander("Raw data (JSON)", icon=":material/data_object:"):
            st.json(feasibility.get("counts", {}))

    if cohort_manifest:
        any_rendered = True
        st.subheader("Draft analytic cohort contract")
        cols = st.columns(3)
        cols[0].metric("Status", _humanize(cohort_manifest.get("status", "unknown")))
        cols[1].metric("Ready to run", "Yes" if cohort_manifest.get("execution_allowed") else "No")
        cols[2].metric(
            "Results may be shared with the AI assistant",
            "Yes" if cohort_manifest.get("delivery_to_llm_allowed") else "No",
        )
        for issue in cohort_manifest.get("unresolved_issues", ()):
            st.markdown(f"- {issue}")

    query_validation = _read_json(run_dir / "cohort_query_validation.json")
    if query_validation:
        any_rendered = True
        st.subheader("Cohort query safety check")
        st.caption(
            "Automated checks that the underlying database query is safe and only touches the "
            "expected clinical data tables. Does not judge clinical correctness of the query."
        )
        status = query_validation.get("status", "unknown")
        (st.success if status == "passed" else st.error)(f"Status: {status}")
        for violation in query_validation.get("violations", ()):
            st.error(f"- {violation}")
        with st.expander("Technical detail", icon=":material/terminal:"):
            st.caption(
                "Checks performed: no unbound parameters, no `SELECT *`, only expected OMOP "
                "tables."
            )
            estimated_bytes = query_validation.get("estimated_bytes_processed")
            st.caption(
                "Estimated data scanned: "
                + (f"{estimated_bytes:,} bytes" if isinstance(estimated_bytes, int) else "not available (not a dry run)")
            )

    if cohort_profile:
        any_rendered = True
        st.subheader("Aggregate cohort profile")
        st.caption("Validates cohort-building mechanics only; contains no patient-level rows.")
        _render_count_metrics(cohort_profile.get("profile", {}))
        for warning in cohort_profile.get("warnings", ()):
            st.warning(warning)
        with st.expander("Raw data (JSON)", icon=":material/data_object:"):
            st.json(cohort_profile.get("profile", {}))

    if followup_audit:
        any_rendered = True
        st.subheader("Follow-up completeness check")
        censoring_approved = bool(
            ((_read_json(run_dir / "protocol.json") or {}).get("query_settings") or {})
            .get("censoring_definition_approved", False)
        )
        st.caption(
            "Aggregate observability diagnostic. Per-row `censored` column: "
            + ("approved." if censoring_approved else "not yet clinically approved.")
        )
        _render_count_metrics(followup_audit.get("counts", {}))
        for warning in followup_audit.get("warnings", ()):
            st.warning(warning)
        with st.expander("Raw data (JSON)", icon=":material/data_object:"):
            st.json(followup_audit.get("counts", {}))
        if not censoring_approved:
            with st.form(key=f"approve_censoring_definition_form_{state.run_id}"):
                st.caption(
                    "Approving turns on the real per-row `censored` column (see counts above) "
                    "instead of leaving it `NULL` for every row -- needed only if the statistical "
                    "estimand gate will choose 'Exclude censored rows'."
                )
                reviewer = st.text_input(
                    "Reviewer", value=_reviewer_default(), key=f"censoring_definition_reviewer_{state.run_id}"
                )
                if st.form_submit_button("Approve per-row censoring definition"):
                    try:
                        manager.approve_censoring_definition(state.run_id, reviewer)
                        st.success("Per-row censoring definition approved")
                        st.rerun()
                    except (ValueError, RuntimeError) as exc:
                        st.error(str(exc))

    if not any_rendered:
        st.info("No aggregate feasibility, cohort, or execution-readiness result is available yet.")


# `RunState.record`'s first argument (see `causal/orchestrator/state.py` and
# every `state.record(...)` call site in `causal/orchestrator/workflow.py`)
# -- a fixed, human-curated label per event kind. An event not in this dict
# (future addition, or an older run) falls back to `_humanize` so it still
# reads as words instead of a raw snake_case code.
_EVENT_LABELS: dict[str, str] = {
    "run_started": "Run started",
    "gate_decided": "Approval gate decided",
    "gate_invalidated": "Approval gate reset (upstream change)",
    "phase_advanced": "Phase advanced",
    "agent_completed": "Automated step completed",
    "concept_set_mapped": "Concept mapping saved",
    "covariate_reviewed": "Covariate reviewed",
    "covariate_domain_reviewed": "Covariate OMOP domain saved",
    "covariate_concepts_mapped": "Covariate concept mapping saved",
    "covariate_aggregation_reviewed": "Covariate aggregation saved",
    "covariate_window_reviewed": "Covariate window saved",
    "registry_built": "Concept mapping registry built",
    "cohort_plan_generated": "Cohort plan generated",
    "cohort_profile_completed": "Cohort profile completed",
    "followup_audit_completed": "Follow-up audit completed",
    "mapping_audit_completed": "Concept mapping audit completed",
    "report_generated": "Demo report generated",
    "exposure_contrast_set": "Comparator design saved",
    "feasibility_completed": "Feasibility check completed",
}


def _format_history_event(event: str) -> str:
    return _EVENT_LABELS.get(event, _humanize(event))


def _render_report_tab(manager: WorkflowManager, state: RunState, run_dir: Path) -> None:
    st.subheader("Demo report")
    if st.button("Regenerate demo report"):
        manager.build_demo_report(state.run_id)
        st.rerun()

    report_path = run_dir / "demo_report.md"
    if report_path.is_file():
        report_text = report_path.read_text(encoding="utf-8")
        st.download_button("Download demo_report.md", report_text, file_name="demo_report.md")
        st.markdown(report_text)
    else:
        st.info("No demo report generated yet.")

    st.divider()
    st.subheader("History")
    if state.history:
        st.dataframe(
            [
                {"When": h["at"], "Event": _format_history_event(h["event"]), "Detail": h["detail"]}
                for h in reversed(state.history)
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No history recorded yet.")

    st.divider()
    st.subheader("Artifacts")
    for artifact_name in state.artifacts:
        artifact_path = run_dir / artifact_name
        if artifact_path.is_file():
            st.download_button(
                f"Download {artifact_name}",
                artifact_path.read_bytes(),
                file_name=artifact_name,
                key=f"download_{artifact_name}",
            )


def _render_estimation_actions(manager: WorkflowManager, state: RunState, run_id: str, run_dir: Path) -> None:
    """Trigger the steps that produce this tab's artifacts -- mirrors
    `workflow materialize-dataset`/`estimate`/`balance-diagnostics`/
    `causal-report`/`narrative` on the CLI. Materialization always goes
    through OMCP's signed path (see `WorkflowManager.materialize_analytic_
    dataset`'s own docstring) -- there is no direct-database bypass for it,
    unlike the CLI's dev-only direct-BigQuery feasibility transport."""
    datasets = BigQueryDatasets.from_env()
    readiness = _read_json(run_dir / "execution_readiness.json") or {}
    dataset_ready = (run_dir / "analytic_dataset.parquet").is_file()

    def _run(title: str, label: str, ready: bool, requirement: str, spinner: str, action, icon: str) -> None:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.badge("Ready" if ready else "Not ready", color="green" if ready else "gray", icon=icon)
            if st.button(
                label,
                disabled=not ready,
                type="primary" if ready else "secondary",
                icon=icon,
                key=f"estimation_action_{label}",
                width="stretch",
            ):
                try:
                    with st.spinner(spinner):
                        action()
                    st.success(f"{label} completed.")
                    st.rerun()
                except (ValueError, RuntimeError, OSError) as exc:
                    st.error(str(exc))
            st.caption("Available now." if ready else requirement)

    st.subheader("Prepare deterministic results")
    st.caption(
        "These governed steps create the estimate and diagnostics that the interpretation workflow can read."
    )
    materialize_col, diagnostics_col = st.columns(2)
    with materialize_col:
        _run(
            "1. Materialize the analytic dataset",
            "Materialize dataset",
            state.phase == "execution" and bool(readiness.get("analytic_dataset_build_allowed")),
            "Requires the execution phase and an allowed analytic-dataset build.",
            "Executing the governed cohort query through OMCP...",
            lambda: manager.materialize_analytic_dataset(run_id, OMCPClient(datasets)),
            ":material/table_rows:",
        )
    with diagnostics_col:
        _run(
            "2. Compute covariate balance",
            "Run balance diagnostics",
            state.phase in {"execution", "diagnostics"} and dataset_ready,
            "Requires a materialized dataset.",
            "Computing standardized mean differences...",
            lambda: manager.run_diagnostics(run_id),
            ":material/balance:",
        )

    statistical_estimand = _read_json(run_dir / "statistical_estimand.json") or {}
    procedure = statistical_estimand.get("estimation_procedure", "plug_in_standardization")
    is_tmle = procedure == "targeted_minimum_loss_based_estimation"
    estimator_label = "Run TMLE estimate" if is_tmle else "Run g-computation estimate"
    estimator_spinner = "Fitting the TMLE estimator..." if is_tmle else "Fitting the g-computation baseline..."
    estimator_description = (
        "TMLE uses doubly robust nuisance models and an efficient-influence-curve confidence interval."
        if is_tmle
        else "G-computation estimates the prespecified marginal treatment effect with bootstrap confidence intervals."
    )

    with st.container(border=True):
        st.markdown("**3. Estimate the treatment effect**")
        estimate_ready = (
            state.phase in {"execution", "diagnostics"}
            and dataset_ready and bool(readiness.get("causal_estimation_allowed"))
        )
        st.badge(
            "Ready" if estimate_ready else "Not ready",
            color="green" if estimate_ready else "gray",
            icon=":material/functions:",
        )
        n_bootstrap = st.number_input(
            "Bootstrap resamples",
            min_value=10, max_value=5000, value=1000, step=10,
            key=f"estimate_n_bootstrap_{run_id}",
            help="Nonparametric bootstrap resamples used for the 95% confidence interval.",
        )
        if st.button(
            estimator_label,
            disabled=not estimate_ready,
            type="primary" if estimate_ready else "secondary",
            icon=":material/functions:",
            key=f"estimation_action_{procedure}_{run_id}",
            width="stretch",
        ):
            try:
                with st.spinner(estimator_spinner):
                    manager.run_estimation(run_id, n_bootstrap=int(n_bootstrap))
                st.success(f"{estimator_label} completed.")
                st.rerun()
            except (ValueError, RuntimeError, OSError) as exc:
                st.error(str(exc))
        st.caption(estimator_description)
        if not estimate_ready:
            st.caption("Requires a materialized dataset and an allowed causal estimation.")


def _render_advanced_analyses(manager: WorkflowManager, state: RunState, run_dir: Path) -> None:
    """Show optional robustness and heterogeneity analyses next to the primary
    estimate. These are diagnostic extensions: they do not alter a reviewed
    estimand or permit bypassing any workflow gate.
    """
    estimation = _read_json(run_dir / "estimation.json")
    sensitivity = _read_json(run_dir / "sensitivity_analysis.json")
    cate = _read_json(run_dir / "cate.json")
    protocol = _read_json(run_dir / "protocol.json") or {}
    statistical_estimand = _read_json(run_dir / "statistical_estimand.json") or {}
    negative_controls = protocol.get("negative_controls", {})
    results_reviewable = state.phase in {"execution", "diagnostics", "results_review", "report"}

    st.subheader("Robustness & heterogeneity")
    st.caption(
        "Optional analyses make assumptions and subgroup findings easier to review. "
        "They supplement the primary estimate; they do not replace it."
    )

    sensitivity_col, cate_col = st.columns(2)
    with sensitivity_col:
        with st.container(border=True):
            st.markdown("**Unmeasured-confounding sensitivity**")
            st.badge(
                "Complete" if sensitivity else "Available" if estimation else "Waiting for estimate",
                color="green" if sensitivity else "blue" if estimation else "gray",
                icon=":material/shield:",
            )
            if sensitivity:
                metric_col, ci_col = st.columns(2)
                metric_col.metric("E-value (point estimate)", f"{sensitivity.get('e_value_point', 0):.2f}")
                ci_col.metric("E-value (CI limit)", f"{sensitivity.get('e_value_ci', 0):.2f}")
                st.caption(sensitivity.get("interpretation", ""))
            else:
                st.caption("Quantifies the minimum confounding strength needed to explain away the observed association.")
            if st.button(
                "Run E-value analysis",
                disabled=not (estimation and results_reviewable),
                icon=":material/shield:",
                key=f"run_sensitivity_{state.run_id}",
                width="stretch",
            ):
                try:
                    with st.spinner("Computing the E-value sensitivity analysis..."):
                        manager.run_sensitivity_analysis(state.run_id)
                    st.success("Sensitivity analysis completed.")
                    st.rerun()
                except (ValueError, RuntimeError, OSError) as exc:
                    st.error(str(exc))

    with cate_col:
        with st.container(border=True):
            population_summary = statistical_estimand.get("population_summary")
            cate_declared = population_summary == "CATE"
            st.markdown("**Prespecified subgroup effects (CATE)**")
            st.badge(
                "Complete" if cate else "Available" if cate_declared else "Not declared",
                color="green" if cate else "blue" if cate_declared else "gray",
                icon=":material/account_tree:",
            )
            if cate:
                cate_estimand = cate.get("estimand", {})
                st.metric("Overall forest ATE", f"{cate_estimand.get('overall_ate_from_forest', 0):.4f}")
                st.caption(
                    f"{cate_estimand.get('n_prespecified_subgroups_tested', 0)} prespecified subgroup(s), "
                    f"adjusted with {cate_estimand.get('multiple_testing_method', 'the recorded method')}."
                )
            elif cate_declared:
                st.caption("Runs the reviewed causal forest across the prespecified effect-modifier family.")
            else:
                st.caption("Available only when the approved statistical estimand declares CATE as the population summary.")
            if st.button(
                "Run CATE analysis",
                disabled=not (cate_declared and results_reviewable),
                icon=":material/account_tree:",
                key=f"run_cate_{state.run_id}",
                width="stretch",
            ):
                try:
                    with st.spinner("Fitting the causal forest for prespecified subgroups..."):
                        manager.run_cate(state.run_id)
                    st.success("CATE analysis completed.")
                    st.rerun()
                except (ValueError, RuntimeError, OSError) as exc:
                    st.error(str(exc))

    if negative_controls:
        with st.expander("Negative-control checks", icon=":material/compare_arrows:"):
            rows = []
            for control_id, control in negative_controls.items():
                result = _read_json(run_dir / f"negative_control_estimation_{control_id}.json")
                rows.append(
                    {
                        "Control": control.get("label", control_id),
                        "Role": control.get("control_role", "").replace("_", " "),
                        "Mapping": control.get("mapping_status", "pending").replace("_", " "),
                        "Estimate": "Complete" if result else "Not run",
                    }
                )
            st.dataframe(rows, hide_index=True, width="stretch")
            st.caption("Negative controls are diagnostic checks and require separate clinical mapping and review.")


def _render_balance_chart(balance: list[dict[str, Any]]) -> None:
    """Compact visual summary of absolute standardized mean differences.

    The table remains available below for precise values; this chart makes
    potentially imbalanced covariates immediately scannable during review.
    """
    chart_rows = []
    for entry in balance:
        try:
            absolute_smd = abs(float(entry["smd"]))
        except (KeyError, TypeError, ValueError):
            continue
        chart_rows.append(
            {
                "Covariate": entry.get("column_name", "Unknown covariate"),
                "Absolute SMD": absolute_smd,
                "Balance status": "Flagged" if entry.get("flagged") else "Within threshold",
            }
        )
    if not chart_rows:
        return

    chart_rows.sort(key=lambda row: row["Absolute SMD"], reverse=True)
    chart_height = min(480, max(180, len(chart_rows) * 32))
    st.vega_lite_chart(
        chart_rows,
        {
            "layer": [
                {
                    "mark": {"type": "bar", "cornerRadiusEnd": 3},
                    "encoding": {
                        "y": {
                            "field": "Covariate",
                            "type": "nominal",
                            "sort": "-x",
                            "title": None,
                        },
                        "x": {
                            "field": "Absolute SMD",
                            "type": "quantitative",
                            "title": "Absolute standardized mean difference",
                        },
                        "color": {
                            "field": "Balance status",
                            "type": "nominal",
                            "scale": {
                                "domain": ["Within threshold", "Flagged"],
                                "range": ["#4C78A8", "#E45756"],
                            },
                            "title": None,
                        },
                        "tooltip": [
                            {"field": "Covariate", "type": "nominal"},
                            {"field": "Absolute SMD", "type": "quantitative", "format": ".3f"},
                            {"field": "Balance status", "type": "nominal"},
                        ],
                    },
                },
                {
                    "mark": {"type": "rule", "color": "#F59E0B", "strokeDash": [6, 4], "strokeWidth": 2},
                    "encoding": {"x": {"datum": 0.1, "type": "quantitative"}},
                },
            ]
        },
        height=chart_height,
        width="stretch",
    )
    st.caption("Bars show absolute SMD. The dashed line marks the 0.10 review threshold.")


def _narrative_blocks_report(narrative: dict[str, Any] | None) -> bool:
    """Whether an existing `results_narrative.json` should block "Build
    causal report" -- true for anything short of a clean approval or an
    explicit exclusion. Pure/no `st.*` so it is unit-testable on its own
    (see `tests/test_streamlit_app.py`); the caller passes an already
    `normalize_narrative_artifact`-normalized dict, or `None` when no
    narrative was ever generated (never blocks -- report building without
    a narrative is always allowed)."""
    if not narrative:
        return False
    if narrative.get("review_status") == "excluded":
        return False
    if narrative.get("review_status") == "approved":
        verification = narrative.get("verification") or {}
        no_failed_checks = not any(
            check.get("status") == "failed" for check in verification.get("checks", ())
        )
        if verification.get("status") in {"passed", "warning"} and no_failed_checks:
            return False
    return True


def _render_results_narrative_section(manager: WorkflowManager, state: RunState, run_dir: Path) -> dict[str, Any] | None:
    """Steps 2-4 of the estimation/report flow: generate the optional LLM
    interpretation, show its deterministic faithfulness verification, and
    let a human approve or exclude it. Returns the normalized artifact (or
    `None` if none exists yet) so the caller can gate "Build causal report"
    on it via `_narrative_blocks_report`."""
    run_id = state.run_id
    st.subheader("LLM-assisted interpretation")
    st.caption(
        "Advisory only -- reads the already-computed estimate, diagnostics, and gate history, "
        "never the patient-level dataset. It cannot approve the results gate or change the estimate."
    )
    narrative_raw = _read_json(run_dir / "results_narrative.json")
    narrative = normalize_narrative_artifact(narrative_raw) if narrative_raw else None

    narrative_ready = state.phase in {"results_review", "report"} and (run_dir / "estimation.json").is_file()
    label = "Regenerate interpretation" if narrative else "Generate interpretation"
    if st.button(
        label,
        disabled=not narrative_ready,
        type="primary" if narrative_ready else "secondary",
        icon=":material/smart_toy:",
        key=f"generate_narrative_{run_id}",
        width="stretch",
    ):
        try:
            with st.spinner("Requesting an interpretive narrative..."):
                resolved_llm = build_hypothesis_llm()
                manager.run_results_narrative(
                    run_id, resolved_llm.call,
                    model=resolved_llm.model, provider=resolved_llm.provider,
                )
            st.success("Interpretation generated.")
            st.rerun()
        except Exception as exc:
            # Any provider/network/parsing failure here must degrade this one
            # optional action, not the rest of the tab.
            st.error(f"Interpretation generation failed (the causal report is unaffected): {exc}")
    if not narrative_ready:
        st.caption("Requires the results-review or report phase and an existing estimate.")

    if not narrative:
        st.info("No LLM interpretation has been generated for this run yet.")
        return None

    with st.container(border=True):
        st.markdown(narrative.get("narrative", ""))
    st.caption(narrative.get("boundary_statement", ""))
    st.caption(f"Model: {narrative.get('provider', 'unknown')}:{narrative.get('model', 'unknown')}")

    claims = narrative.get("claims", [])
    st.markdown("**Structured claims**")
    if claims:
        st.dataframe(
            [
                {
                    "Claim": claim.get("text", ""),
                    "Type": claim.get("claim_type", ""),
                    "Source fields": ", ".join(claim.get("source_fields", [])),
                }
                for claim in claims
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption("No structured claims (this artifact predates faithfulness verification).")

    st.markdown("**Faithfulness verification**")
    st.caption("Deterministic checks only -- no second LLM is used to grade this narrative.")
    verification = narrative.get("verification") or {"status": "not_run", "checks": []}
    verification_status = verification.get("status", "not_run")
    checks = verification.get("checks", [])
    if verification_status == "failed":
        st.error("At least one faithfulness check failed. This interpretation cannot be auto-approved.")
    elif verification_status == "warning":
        st.warning("Faithfulness verification has warning(s) to review before approval.")
    elif verification_status == "passed":
        st.success("All deterministic faithfulness checks passed.")
    else:
        st.warning(
            "This interpretation predates faithfulness verification -- regenerate it to enable review."
        )
    if checks:
        st.dataframe(
            [
                {
                    "Subject": check.get("subject", ""),
                    "Claim type": check.get("claim_type", ""),
                    "Status": check.get("status", ""),
                    "Reason": check.get("reason", ""),
                    "Evidence": check.get("evidence", ""),
                }
                for check in checks
            ],
            hide_index=True,
            width="stretch",
        )

    st.markdown("**Human review**")
    review_status = narrative.get("review_status", "pending")
    st.badge(
        review_status.replace("_", " ").capitalize(),
        color={"approved": "green", "excluded": "orange"}.get(review_status, "gray"),
    )
    if narrative.get("reviewer"):
        note_suffix = f" -- {narrative['review_note']}" if narrative.get("review_note") else ""
        st.caption(f"Last decision by {narrative['reviewer']} on {narrative.get('reviewed_at', '-')}{note_suffix}")

    can_approve = verification_status in {"passed", "warning"} and not any(
        check.get("status") == "failed" for check in checks
    )
    review_phase_ok = state.phase in {"results_review", "report"}
    with st.form(key=f"narrative_review_form_{run_id}", border=False):
        reviewer = st.text_input("Reviewer", value=_reviewer_default(), key=f"narrative_reviewer_{run_id}")
        note = st.text_area("Review note (optional)", key=f"narrative_review_note_{run_id}")
        with st.container(horizontal=True):
            approve_clicked = st.form_submit_button(
                "Approve", type="primary", disabled=not (can_approve and review_phase_ok),
            )
            exclude_clicked = st.form_submit_button("Exclude from report", disabled=not review_phase_ok)
        if approve_clicked or exclude_clicked:
            decision = "approved" if approve_clicked else "excluded"
            try:
                manager.review_results_narrative(run_id, decision, reviewer, note or None)
                st.success(f"Interpretation {decision}.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    if not can_approve:
        st.caption("Approval is disabled while a faithfulness check has failed or has not run yet.")

    return narrative


def _render_estimation_report_tab(manager: WorkflowManager, state: RunState, run_dir: Path) -> None:
    st.subheader("Estimation & causal report")
    st.caption(
        "Run the prespecified estimator, review diagnostics and robustness analyses, then "
        "assemble an auditable causal report. No patient-level value ever leaves the governed "
        "run directory."
    )

    estimation_preview = _read_json(run_dir / "estimation.json")
    narrative_raw = _read_json(run_dir / "results_narrative.json")
    narrative_preview = normalize_narrative_artifact(narrative_raw) if narrative_raw else None
    report_exists = (run_dir / "causal_report.md").is_file()
    with st.container(border=True):
        st.markdown("**Final interpretation workflow**")
        interpretation_col, results_col, report_col = st.columns(3)
        with interpretation_col:
            st.caption("1. Interpretation and review")
            if narrative_preview:
                review_status = narrative_preview.get("review_status", "pending")
                status_label = review_status.replace("_", " ").capitalize()
                status_color = {"approved": "green", "excluded": "orange"}.get(review_status, "gray")
            else:
                status_label, status_color = "Not requested", "gray"
            st.badge(status_label, color=status_color, icon=":material/rate_review:")
        with results_col:
            st.caption("2. Results and diagnostics")
            st.badge(
                "Available" if estimation_preview else "Pending",
                color="green" if estimation_preview else "gray",
                icon=":material/query_stats:",
            )
        with report_col:
            st.caption("3. Final causal report")
            st.badge(
                "Built" if report_exists else "Pending",
                color="green" if report_exists else "gray",
                icon=":material/description:",
            )

    if estimation_preview:
        narrative = _render_results_narrative_section(manager, state, run_dir)
    else:
        st.subheader("Prepare results")
        _render_estimation_actions(manager, state, state.run_id, run_dir)
        with st.container(border=True):
            st.markdown("**Interpretation assisted by LLM**")
            st.caption(
                "An interpretation becomes available after the deterministic treatment-effect estimate is complete."
            )
        narrative = None

    dataset_manifest = _read_json(run_dir / "dataset_manifest.json")
    estimation = _read_json(run_dir / "estimation.json")
    diagnostics = _read_json(run_dir / "diagnostics.json")

    st.subheader("Results and diagnostics")

    if dataset_manifest:
        st.subheader("Analytic dataset")
        cols = st.columns(2)
        cols[0].metric("Rows", dataset_manifest.get("row_count", "—"))
        cols[1].metric("Columns", dataset_manifest.get("column_count", "—"))

    if estimation:
        estimand = estimation.get("estimand", {})
        st.subheader("Causal effect estimate")
        st.caption(f"{estimand.get('method', '—')} — {estimand.get('citation', '')}")
        cols = st.columns(3)
        cols[0].metric(_humanize(estimand.get("effect_measure")), f"{estimand.get('point_estimate', 0):.4f}")
        cols[1].metric(
            f"{int(estimand.get('ci_level', 0.95) * 100)}% CI",
            f"[{estimand.get('ci_lower', 0):.4f}, {estimand.get('ci_upper', 0):.4f}]",
        )
        cols[2].metric("N (total)", estimand.get("n_total", "—"))
        for warning in estimand.get("warnings", ()):
            st.warning(warning)
        with st.expander("Raw data (JSON)", icon=":material/data_object:"):
            st.json(estimand)
    else:
        st.info("No causal effect has been estimated yet.")

    if diagnostics:
        st.subheader("Covariate balance (SMD)")
        st.caption(diagnostics.get("citation", ""))
        _render_balance_chart(list(diagnostics.get("balance", ())))
        st.dataframe(
            [
                {
                    "Covariate": entry["column_name"],
                    "SMD": round(entry["smd"], 3),
                    "Treated mean": round(entry["treated_mean"], 3),
                    "Comparator mean": round(entry["comparator_mean"], 3),
                    "Flagged": "Yes" if entry["flagged"] else "No",
                }
                for entry in diagnostics.get("balance", ())
            ],
            hide_index=True,
            width="stretch",
        )
        for warning in diagnostics.get("warnings", ()):
            st.warning(warning)

    _render_advanced_analyses(manager, state, run_dir)

    if estimation_preview:
        with st.expander("Refresh analysis", icon=":material/refresh:"):
            st.caption("Re-run materialization, diagnostics, or the primary estimator when the reviewed study inputs change.")
            _render_estimation_actions(manager, state, state.run_id, run_dir)

    st.subheader("3. Causal report")
    report_phase_ok = state.phase in {"results_review", "report"}
    narrative_blocks = _narrative_blocks_report(narrative)
    build_disabled = not report_phase_ok or narrative_blocks
    if st.button(
        "Build final causal report",
        disabled=build_disabled,
        type="primary" if not build_disabled else "secondary",
        icon=":material/description:",
        key=f"build_causal_report_{state.run_id}",
        width="stretch",
    ):
        try:
            with st.spinner("Assembling the auditable causal report..."):
                manager.build_causal_report(state.run_id)
            st.success("Causal report built.")
            st.rerun()
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(str(exc))
    if narrative_blocks:
        st.caption(
            "An LLM interpretation is pending review. Approve it or exclude it from the report to continue."
        )
    elif not report_phase_ok:
        st.caption("Requires the results-review or report phase.")

    report_path = run_dir / "causal_report.md"
    if report_path.is_file():
        report_text = report_path.read_text(encoding="utf-8")
        st.download_button("Download causal_report.md", report_text, file_name="causal_report.md")
        with st.expander("Preview", icon=":material/description:"):
            st.markdown(report_text)
    else:
        st.info("No causal report generated yet.")


def _render_new_question_section(manager: WorkflowManager, *, expanded: bool = False) -> None:
    with st.expander("New causal study", icon=":material/add_circle:", expanded=expanded):
        st.caption(
            "Calls the configured LLM provider (and PubMed/OpenAlex unless skipped) "
            "and may incur API costs. Can take up to a couple of minutes — longer with "
            "more review rounds or more than one candidate DAG."
        )
        with st.form(key="new_question_form"):
            question = st.text_area("Clinical question", key="new_question_text")
            mode = st.selectbox("Mode", ("exploratory", "final"), key="new_question_mode")
            model_override = st.text_input(
                "Model override (optional)", key="new_question_model"
            )
            skip_question_gate = st.checkbox(
                "Skip causal-question gate (assume the question is already causal)",
                key="new_question_skip_gate",
            )
            skip_literature = st.checkbox(
                "Skip literature grounding (PubMed/OpenAlex)", key="new_question_skip_lit"
            )
            skip_critique = st.checkbox(
                "Skip automated DAG review", key="new_question_skip_critique"
            )
            critique_col, tournament_col = st.columns(2)
            max_critique_rounds = critique_col.number_input(
                "Review rounds",
                min_value=1,
                max_value=5,
                value=2,
                disabled=skip_critique,
                key="new_question_max_critique_rounds",
                help="How many rounds of automated review and revision the draft DAG goes through.",
            )
            tournament_size = tournament_col.number_input(
                "Candidate DAGs to draft",
                min_value=1,
                max_value=MAX_TOURNAMENT_SIZE,
                value=1,
                key="new_question_tournament_size",
                help=(
                    "Draft this many DAGs independently, then automatically combine them into "
                    "one (1 = today's default, a single draft)."
                ),
            )
            submitted = st.form_submit_button("Generate hypothesis draft")

        if not submitted:
            return
        if not question.strip():
            st.error("Clinical question is required")
            return

        status_log = st.status("Generating hypothesis draft...", expanded=True)

        def _on_llm_output(output: LLMRawOutput) -> None:
            status_log.write(f"Received `{output.stage}` from the LLM.")

        try:
            resolved_llm = build_hypothesis_llm(model=model_override.strip() or None)
            status_log.write(f"Using provider `{resolved_llm.provider}:{resolved_llm.model}`.")
            if not skip_question_gate:
                gate_result = parse_gate_response(
                    resolved_llm.call(build_gate_prompt(DEFAULT_GATE_PROMPT_PATH, question))
                )
                if gate_result.category == "descriptive":
                    status_log.update(label="Not a causal question", state="error")
                    st.warning(
                        "This looks like a descriptive/retrieval question, not a causal one -- "
                        "FastOMOP (agno_fastomop) handles this, not the causal target-trial pipeline.\n\n"
                        f"Reasoning: {gate_result.reasoning}"
                    )
                    return
                if gate_result.category == "needs_clarification":
                    status_log.update(label="Needs clarification", state="error")
                    st.warning(
                        "This question needs to be sharpened before drafting a causal study.\n\n"
                        f"Reasoning: {gate_result.reasoning}\n\n"
                        f"Clarifying question: {gate_result.clarifying_question}"
                    )
                    return
                status_log.write(f"Causal-question gate: {gate_result.category} -- {gate_result.reasoning}")
            literature = (
                None
                if skip_literature
                else LiteratureOrchestrator(pubmed=PubMedClient(), openalex=OpenAlexClient())
            )
            generated = UserQuestionHypothesisService().generate(
                question,
                resolved_llm.call,
                data_source=BigQueryDatasets.from_env().describe(),
                literature=literature,
                on_llm_output=_on_llm_output,
                enable_critique=not skip_critique,
                max_critique_rounds=int(max_critique_rounds),
                tournament_size=int(tournament_size),
            )
            state = manager.start_generated(
                generated, mode, f"{resolved_llm.provider}:{resolved_llm.model}"
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            status_log.update(label="Failed", state="error")
            st.error(str(exc))
            return

        status_log.update(label=f"Created run {state.run_id}", state="complete")
        # The `run_selector` widget is already instantiated earlier in this same
        # script run, so it can't be reassigned directly here (Streamlit raises
        # StreamlitAPIException). Stash the target run id and apply it at the
        # top of the next run, before the widget is (re-)created.
        st.session_state["_pending_run_selection"] = state.run_id
        st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="CausalOMOP",
        page_icon=":material/account_tree:",
        layout="wide",
    )

    manager = WorkflowManager(runs_dir=RUNS_DIR)

    pending_selection = st.session_state.pop("_pending_run_selection", None)
    if pending_selection is not None:
        st.session_state["run_selector"] = pending_selection

    with st.sidebar:
        st.title("CausalOMOP")
        st.caption("Causal study design and review")
        st.text_input(
            "Reviewer name",
            key="reviewer_name",
            on_change=_sync_reviewer_fields,
            help="Pre-fills the reviewer field on every gate and covariate decision below.",
        )
        st.divider()
        st.header("Study runs")
        run_ids = _list_runs()
        run_id = (
            st.selectbox(
                "Open a study",
                run_ids,
                key="run_selector",
                format_func=_run_option_label,
            )
            if run_ids
            else None
        )

    title_col, action_col = st.columns([4, 1], vertical_alignment="center")
    with title_col:
        st.title("Causal workflow review")
        st.caption("Draft, review, and audit a causal study. Each important decision remains human-reviewed.")
    with action_col:
        with st.popover("New causal study", type="primary", icon=":material/add:", width="stretch"):
            _render_new_question_section(manager, expanded=True)

    if run_id is None:
        with st.container(border=True):
            st.subheader("Start your first causal study")
            st.write(
                "Create a study from a clinical question, then review the target trial, causal DAG, "
                "and OMOP concept mappings step by step."
            )
            st.info("Use **New causal study** in the top-right corner to begin.", icon=":material/arrow_upward:")
        return

    state = manager.load(run_id)
    run_dir = RUNS_DIR / run_id

    st.caption(f"Study `{state.run_id}` · {state.mode} mode · updated {state.updated_at}")
    source_question = _source_question(run_dir)
    if source_question:
        with st.container(border=True):
            st.caption("Clinical question")
            st.write(source_question)
    _render_phase_tracker(state)
    workspace = st.radio(
        "Study workspace",
        (
            "Current task",
            "Study design",
            "Mapping & plan",
            "Feasibility & cohort",
            "Estimation & report",
            "Report & audit",
        ),
        key=f"workspace_{run_id}",
        horizontal=True,
        label_visibility="collapsed",
        width="stretch",
    )

    if workspace == "Current task":
        _render_gates_tab(manager, state, run_dir)
    elif workspace == "Study design":
        _render_hypothesis_tab(manager, run_id, run_dir)
    elif workspace == "Mapping & plan":
        _render_analysis_plan_tab(manager, run_id, run_dir)
    elif workspace == "Feasibility & cohort":
        _render_feasibility_tab(manager, state, run_dir)
    elif workspace == "Estimation & report":
        _render_estimation_report_tab(manager, state, run_dir)
    else:
        _render_report_tab(manager, state, run_dir)


if __name__ == "__main__":
    main()
