"""Read-focused Streamlit dashboard for validating and demoing a causal workflow run.

Run with: `streamlit run causal/ui/streamlit_app.py` from the repository root.

This app only calls the existing `WorkflowManager` (load/decide_gate/advance/
build_demo_report) and reads the JSON artifacts it already writes under
`artifacts/runs/<run_id>/`. It never calls an LLM or BigQuery/OMCP directly —
those actions stay on the `causalomop` CLI.
"""

from __future__ import annotations

import html
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

from causal.agents.dag_identification import review_dag
from causal.agents.hypothesis import (
    DAG_VARIABLE_ROLES,
    DAG_VARIABLE_TEMPORAL_CLASSES,
    CovariateRelation,
    DAGVariable,
    HypothesisArtifact,
    RejectedCovariate,
)
from causal.agents.user_question import LLMRawOutput, UserQuestionHypothesisService
from causal.config import BigQueryDatasets
from causal.literature import LiteratureOrchestrator, OpenAlexClient, PubMedClient
from causal.llm import build_hypothesis_llm
from causal.orchestrator.state import PHASES, RunState
from causal.orchestrator.workflow import WorkflowManager
from causal.roadmap import CausalDAG


RUNS_DIR = _PROJECT_ROOT / "artifacts" / "runs"

_GATE_COLORS = {
    "approved": "#2E7D32",
    "rejected": "#C62828",
    "provisional": "#EF6C00",
    "pending": "#757575",
}

_GATE_INFO: dict[str, dict[str, str]] = {
    "protocol": {
        "title": "Protocol & causal DAG",
        "phase": "protocol_review",
        "description": (
            "Approves the normalized causal question (population / treatment / comparator / "
            "outcome) and the causal DAG proposed for it — the assumed confounders, mediators, "
            "and colliders. Must be approved (major confounders resolved, DAG identified by "
            "backdoor adjustment) before proceeding."
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


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


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
        'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];',
        'edge [fontname="Helvetica", fontsize=9];',
    ]
    for node in dag.nodes:
        color = _NODE_COLORS.get(node.role, "#B0B0B0")
        label = node.label.replace('"', "'")
        lines.append(
            f'"{node.node_id}" [label="{label}\\n({node.role})", fillcolor="{color}", fontcolor="white"];'
        )
    for edge in dag.edges:
        style = "solid" if edge.status in {"evidence_supported", "expert_approved"} else "dashed"
        lines.append(f'"{edge.source}" -> "{edge.target}" [style={style}];')
    lines.append("}")
    return "\n".join(lines)


_PHASE_TO_GATE = {info["phase"]: gate_name for gate_name, info in _GATE_INFO.items()}


def _render_phase_tracker(current_phase: str) -> None:
    current_index = PHASES.index(current_phase)
    columns = st.columns(len(PHASES))
    for index, (column, phase) in enumerate(zip(columns, PHASES)):
        if index < current_index:
            marker = "✅"
        elif index == current_index:
            marker = "▶️"
        else:
            marker = "⬜"
        label = f"{phase} 🔒" if phase in _PHASE_TO_GATE else phase
        column.markdown(f"{marker}  \n**{label}**" if index == current_index else f"{marker}  \n{label}")
    st.caption("🔒 = leaving this phase requires an approval gate decision (see the Gates & Validation tab).")


def _render_gates_tab(manager: WorkflowManager, state: RunState, run_dir: Path) -> None:
    st.subheader("Approval gates")
    st.caption(
        "The pipeline has 4 human sign-off points, each tied to one phase of the run (🔒 in the "
        "tracker above). A phase can't advance to the next until its gate is `approved` — or, for "
        "clinical mapping in exploratory mode, `provisional`."
    )

    hypothesis = _read_json(run_dir / "hypothesis.json")
    if hypothesis is not None and state.gates["protocol"].status != "approved":
        try:
            artifact = _hypothesis_artifact_from_dict(hypothesis)
            identification = review_dag(artifact).identification
            if identification.identified_by_backdoor_adjustment:
                st.success(
                    "Causal DAG is identified by backdoor adjustment — the protocol gate "
                    "is clear to approve on that front. See the Hypothesis & DAG tab for details."
                )
            else:
                st.error(
                    "Causal DAG is **not** identified by backdoor adjustment — approving the "
                    "protocol gate will be rejected until this is resolved. See the Hypothesis "
                    "& DAG tab for the blocking notes."
                )
        except (KeyError, ValueError):
            pass

    for gate_name, gate in state.gates.items():
        info = _GATE_INFO.get(gate_name, {"title": gate_name, "phase": "—", "description": ""})
        with st.container(border=True):
            header_col, badge_col = st.columns([3, 1])
            header_col.markdown(f"**{info['title']}** &nbsp; `{gate_name}`", unsafe_allow_html=True)
            badge_col.markdown(_badge(gate.status), unsafe_allow_html=True)
            st.caption(f"Gates leaving phase `{info['phase']}`. {info['description']}")
            if gate.reviewer:
                st.caption(
                    f"Last decision by {html.escape(gate.reviewer)} at {gate.decided_at}"
                    + (f" — {html.escape(gate.comment)}" if gate.comment else "")
                )
            with st.form(key=f"gate_form_{gate_name}"):
                reviewer = st.text_input(
                    "Reviewer", value=_reviewer_default(), key=f"reviewer_{gate_name}"
                )
                comment = st.text_area("Comment (optional)", key=f"comment_{gate_name}")
                status = st.selectbox(
                    "Decision",
                    ("approved", "rejected", "provisional"),
                    key=f"status_{gate_name}",
                )
                submitted = st.form_submit_button("Decide")
                if submitted:
                    try:
                        manager.decide_gate(
                            state.run_id, gate_name, status, reviewer, comment or None
                        )
                        st.success(f"{gate_name} set to {status}")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

    st.divider()
    st.subheader("Advance phase")
    st.caption(f"Current phase: `{state.phase}`")
    if st.button("Advance to next phase"):
        try:
            manager.advance(state.run_id)
            st.success("Phase advanced")
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))


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

    st.caption(f"Active comparator represented on the DAG: **{artifact.comparator or '—'}**")

    metric_cols = st.columns(3)
    metric_cols[0].metric("Confounders", identification.confounder_count)
    metric_cols[1].metric(
        "Identified by backdoor adjustment", "Yes" if identification.identified_by_backdoor_adjustment else "No"
    )
    metric_cols[2].metric("Candidate adjustment sets", len(identification.minimum_adjustment_sets))

    if identification.minimum_adjustment_sets:
        st.markdown("**Minimum adjustment sets**")
        for adjustment_set in identification.minimum_adjustment_sets:
            labels = [identification.dag.node_map[node_id].label for node_id in adjustment_set]
            st.markdown(f"- {', '.join(labels) if labels else '(empty set)'}")
        if adjustment.core_variable_labels or adjustment.alternative_variable_labels:
            st.caption(
                "**Core** (in every minimum set — required, not optional): "
                f"{', '.join(adjustment.core_variable_labels) or '—'}. "
                "**Alternative** (in some but not all — interchangeable for backdoor-blocking, "
                "may still differ clinically): "
                f"{', '.join(adjustment.alternative_variable_labels) or '—'}."
            )
        if adjustment.unused_confounder_labels:
            st.warning(
                "Declared confounder(s) that never appear in any minimum adjustment set — "
                "either redundant given the rest of the DAG, or worth double-checking the role "
                "assignment: " + ", ".join(adjustment.unused_confounder_labels)
            )

    if evidence.total_items:
        evidence_metric_cols = st.columns(4)
        evidence_metric_cols[0].metric("Evidence: literature only", evidence.literature_only)
        evidence_metric_cols[1].metric("Evidence: knowledge graph only", evidence.kg_only)
        evidence_metric_cols[2].metric("Evidence: corroborated (both)", evidence.corroborated)
        evidence_metric_cols[3].metric("Evidence: uncited", evidence.uncited)
        st.caption(
            f"{evidence.cited_fraction:.0%} of DAG items ({evidence.total_items} total) carry at least "
            f"one citation; {evidence.corroborated_fraction:.0%} are corroborated by two independent "
            "sources (literature and knowledge graph)."
        )

    if identification.technical_notes:
        st.info(
            "Graph-theoretic notes (identification only — not a clinical judgment):\n"
            + "\n".join(f"- {note}" for note in identification.technical_notes)
        )

    st.markdown(
        f"**Clinical validation: {'approved for DAG gate' if clinical.approved_for_dag_gate else 'blocked'}**"
    )
    if clinical.blocking_issues:
        st.error("\n".join(f"- {issue}" for issue in clinical.blocking_issues))
    if clinical.advisory_notes:
        st.warning("\n".join(f"- {note}" for note in clinical.advisory_notes))

    st.graphviz_chart(_dag_to_dot(identification.dag))

    evidence_rows = [
        {
            "Item": variable.label,
            "Role": variable.role.replace("_", " "),
            "Evidence id": variable.pmid or "—",
            "KG-corroborated": "✓" if variable.kg_evidence_id else "",
        }
        for variable in artifact.candidate_dag
    ] + [
        {
            "Item": f"{relation.source} → {relation.target}",
            "Role": "covariate relation",
            "Evidence id": relation.pmid or "—",
            "KG-corroborated": "✓" if relation.kg_evidence_id else "",
        }
        for relation in artifact.covariate_relations
    ]
    if evidence_rows:
        st.markdown("**Evidence sources**")
        st.caption(
            "Evidence id is a PubMed/OpenAlex citation or a knowledge-graph relation id. "
            "KG-corroborated marks items where the knowledge graph independently confirmed "
            "a literature-cited relation."
        )
        st.dataframe(evidence_rows, width="stretch")

    dag_unresolved = (_read_json(run_dir / "protocol.json") or {}).get("candidate_dag_unresolved") or []
    if dag_unresolved:
        st.markdown("**Flagged unresolved by the drafting model**")
        st.caption(
            "The DAG-drafting step itself could not confidently classify these — distinct "
            "from the human covariate review below."
        )
        st.json(dag_unresolved)

    if artifact.unresolved_covariates:
        st.divider()
        st.subheader("Unresolved covariate review")
        st.caption(
            "These first-pass covariates have no assigned causal role yet, so they are "
            "excluded from the DAG above pending review. Major confounders (marked) block "
            "DAG approval until classified or explicitly rejected."
        )
        major_set = set(artifact.major_confounders)
        for label in artifact.unresolved_covariates:
            is_major = label in major_set
            with st.container(border=True), st.form(key=f"covariate_form_{label}"):
                st.markdown(f"**{label}**" + (" 🔴 major confounder" if is_major else ""))
                reviewer = st.text_input(
                    "Reviewer", value=_reviewer_default(), key=f"covariate_reviewer_{label}"
                )
                decision_choice = st.selectbox(
                    "Decision",
                    ("reject",) + DAG_VARIABLE_ROLES,
                    key=f"covariate_decision_{label}",
                    format_func=lambda value: "Reject" if value == "reject" else value.replace("_", " "),
                )
                temporal_choice = st.selectbox(
                    "Temporal classification (relative to treatment)",
                    DAG_VARIABLE_TEMPORAL_CLASSES,
                    key=f"covariate_temporal_{label}",
                )
                comment = st.text_area(
                    "Rationale" if decision_choice != "reject" else "Comment (optional)",
                    key=f"covariate_comment_{label}",
                )
                submitted = st.form_submit_button("Submit decision")
                if submitted:
                    try:
                        if decision_choice == "reject":
                            manager.review_covariate(run_id, label, "reject", reviewer, comment or None)
                        else:
                            manager.review_covariate(
                                run_id, label, "accept", reviewer, comment,
                                role=decision_choice, temporal_classification=temporal_choice,
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


def _render_concept_mapping_section(run_dir: Path) -> None:
    """Concept mapping (`concept_sets` / `query_concept_sets`) lives in
    `protocol.json`, not `analysis_plan.json` — so it's shown even before the
    Data Science Agent has run. Today this data is only ever populated via
    the `causalomop concepts <term>` CLI search followed by manual
    protocol.json edits; there is no in-app concept search yet.
    """
    protocol = _read_json(run_dir / "protocol.json")
    concept_sets = (protocol or {}).get("concept_sets") or {}
    query_concept_sets = (protocol or {}).get("query_concept_sets") or {}
    if not concept_sets and not query_concept_sets:
        return

    st.subheader("Concept mapping")
    st.caption(
        "OMOP concept sets used to operationalize this question. Mapped outside this app today "
        "(`causalomop concepts <term>` to search the vocabulary, then a manual protocol.json "
        "edit) — this is a read-only view of what has been mapped so far."
    )

    if concept_sets:
        st.markdown("**Phenotype review concept sets** (`concept_sets`, reviewed via the Data Science Agent)")
        st.dataframe(
            [
                {
                    "name": name,
                    "role": definition.get("role", "—"),
                    "domain": definition.get("domain") or "—",
                    "status": definition.get("status", "—"),
                    "concept_ids": ", ".join(str(c) for c in definition.get("concept_ids", ())) or "none yet",
                    "label": definition.get("label", ""),
                }
                for name, definition in concept_sets.items()
                if isinstance(definition, dict)
            ],
            width="stretch",
        )

    if query_concept_sets:
        st.markdown("**Query-generation concept sets** (`query_concept_sets`, used to build cohort/feasibility SQL)")
        st.dataframe(
            [
                {
                    "role": role,
                    "domain": definition.get("domain") or "—",
                    "concept_ids": ", ".join(str(c) for c in definition.get("concept_ids", ())) or "none yet",
                }
                for role, definition in query_concept_sets.items()
                if isinstance(definition, dict)
            ],
            width="stretch",
        )
        if any(not d.get("concept_ids") for d in query_concept_sets.values() if isinstance(d, dict)):
            st.warning(
                "One or more roles have no concept_ids yet — cohort/feasibility SQL generation "
                "will fail until these are filled in."
            )


def _render_analysis_plan_tab(run_dir: Path) -> None:
    _render_concept_mapping_section(run_dir)

    plan = _read_json(run_dir / "analysis_plan.json")
    if plan is None:
        st.info("No analysis_plan.json for this run yet (run the Data Science Agent to generate it).")
        return

    st.divider()
    st.subheader("Analysis plan")
    col1, col2 = st.columns(2)
    col1.markdown(f"**Data source**  \n`{plan.get('data_source', '—')}`")
    col1.markdown(f"**Design**  \n`{plan.get('design', '—')}`")
    col2.markdown(f"**Exploratory execution allowed**  \n`{plan.get('exploratory_execution_allowed', '—')}`")
    col2.markdown(f"**Final execution allowed**  \n`{plan.get('final_execution_allowed', '—')}`")

    phenotypes = plan.get("phenotypes") or []
    if phenotypes:
        st.markdown("**Phenotypes**")
        st.dataframe(
            [
                {
                    "name": p.get("name"),
                    "role": p.get("role"),
                    "domain": p.get("domain"),
                    "mode": p.get("mode"),
                    "status": p.get("status"),
                    "concept_ids": ", ".join(str(c) for c in p.get("concept_ids", ())) or "—",
                    "source_value_rule": p.get("source_value_rule") or "—",
                    "omop_tables": ", ".join(p.get("omop_tables", ())),
                }
                for p in phenotypes
            ],
            width="stretch",
        )
        for p in phenotypes:
            candidates = p.get("concept_candidates") or []
            if candidates:
                with st.expander(f"Candidate OMOP concepts for '{p.get('name')}' ({len(candidates)})"):
                    st.dataframe(
                        [
                            {
                                "concept_id": c.get("concept_id"),
                                "concept_name": c.get("concept_name"),
                                "vocabulary_id": c.get("vocabulary_id"),
                                "domain_id": c.get("domain_id"),
                            }
                            for c in candidates
                        ],
                        width="stretch",
                    )

    checks = plan.get("feasibility_checks") or []
    if checks:
        st.markdown("**Planned feasibility checks**")
        for check in checks:
            st.markdown(f"- {check}")


def _render_feasibility_tab(run_dir: Path) -> None:
    feasibility = _read_json(run_dir / "feasibility.json")
    cohort_manifest = _read_json(run_dir / "cohort_manifest.json")
    cohort_profile = _read_json(run_dir / "cohort_profile.json")
    followup_audit = _read_json(run_dir / "followup_audit.json")
    readiness = _read_json(run_dir / "execution_readiness.json")

    any_rendered = False

    if feasibility:
        any_rendered = True
        st.subheader("Provisional feasibility counts")
        st.caption(
            "Uses human-approved concept sets for this question's population/treatment/"
            "comparator/outcome — not a causal estimate."
        )
        st.json(feasibility.get("counts", {}))
        for warning in feasibility.get("warnings", ()):
            st.warning(warning)

    if cohort_manifest:
        any_rendered = True
        st.subheader("Draft analytic cohort contract")
        cols = st.columns(3)
        cols[0].metric("Status", cohort_manifest.get("status", "unknown"))
        cols[1].metric("Execution allowed", str(cohort_manifest.get("execution_allowed", False)))
        cols[2].metric("Delivery to LLM allowed", str(cohort_manifest.get("delivery_to_llm_allowed", False)))
        for issue in cohort_manifest.get("unresolved_issues", ()):
            st.markdown(f"- `{issue}`")

    query_validation = _read_json(run_dir / "cohort_query_validation.json")
    if query_validation:
        any_rendered = True
        st.subheader("Cohort query governance check")
        st.caption(
            "Deterministic checks only — no unbound parameters, no `SELECT *`, only expected "
            "OMOP tables. Does not judge clinical correctness of the query."
        )
        cols = st.columns(2)
        cols[0].metric("Status", query_validation.get("status", "unknown"))
        estimated_bytes = query_validation.get("estimated_bytes_processed")
        cols[1].metric(
            "Estimated bytes scanned",
            f"{estimated_bytes:,}" if isinstance(estimated_bytes, int) else "not dry-run",
        )
        for violation in query_validation.get("violations", ()):
            st.error(f"`{violation}`")

    if cohort_profile:
        any_rendered = True
        st.subheader("Aggregate cohort profile")
        st.caption("Validates cohort-building mechanics only; contains no patient-level rows.")
        st.json(cohort_profile.get("profile", {}))
        for warning in cohort_profile.get("warnings", ()):
            st.warning(warning)

    if followup_audit:
        any_rendered = True
        st.subheader("Nominal follow-up audit")
        st.caption("Aggregate observability diagnostic; censoring definition is not clinically approved.")
        st.json(followup_audit.get("counts", {}))
        for warning in followup_audit.get("warnings", ()):
            st.warning(warning)

    if readiness:
        any_rendered = True
        st.subheader("Execution Agent decision")
        cols = st.columns(3)
        cols[0].metric("Aggregate feasibility allowed", str(readiness.get("aggregate_feasibility_allowed", False)))
        cols[1].metric("Analytic dataset build allowed", str(readiness.get("analytic_dataset_build_allowed", False)))
        cols[2].metric("Causal estimation allowed", str(readiness.get("causal_estimation_allowed", False)))
        for blocker in readiness.get("blockers", ()):
            st.markdown(f"- `{blocker}`")

    if not any_rendered:
        st.info("No feasibility, cohort, or execution-readiness artifacts for this run yet.")


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
            [{"at": h["at"], "event": h["event"], "detail": h["detail"]} for h in reversed(state.history)],
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


def _render_new_question_section(manager: WorkflowManager) -> None:
    with st.expander("➕ New run from a clinical question"):
        st.caption(
            "Calls the configured LLM provider (and PubMed/OpenAlex unless skipped) "
            "and may incur API costs. Can take up to a couple of minutes."
        )
        with st.form(key="new_question_form"):
            question = st.text_area("Clinical question", key="new_question_text")
            mode = st.selectbox("Mode", ("exploratory", "final"), key="new_question_mode")
            model_override = st.text_input(
                "Model override (optional)", key="new_question_model"
            )
            skip_literature = st.checkbox(
                "Skip literature grounding (PubMed/OpenAlex)", key="new_question_skip_lit"
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
            literature = (
                None
                if skip_literature
                else LiteratureOrchestrator(pubmed=PubMedClient(), openalex=OpenAlexClient())
            )
            generated = UserQuestionHypothesisService().generate(
                question,
                resolved_llm.call,
                data_source=BigQueryDatasets.from_env().fq_dataset("omop"),
                literature=literature,
                on_llm_output=_on_llm_output,
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
    st.set_page_config(page_title="CausalOMOP — workflow review", layout="wide")
    st.title("CausalOMOP — workflow review")

    manager = WorkflowManager(runs_dir=RUNS_DIR)

    pending_selection = st.session_state.pop("_pending_run_selection", None)
    if pending_selection is not None:
        st.session_state["run_selector"] = pending_selection

    with st.sidebar:
        st.text_input(
            "Your name",
            key="reviewer_name",
            help="Pre-fills the reviewer field on every gate and covariate decision below.",
        )
        st.divider()
        st.header("Runs")
        run_ids = _list_runs()
        run_id = (
            st.selectbox(
                "Select a run",
                run_ids,
                key="run_selector",
                format_func=_run_option_label,
            )
            if run_ids
            else None
        )
        st.divider()
        _render_new_question_section(manager)

    if run_id is None:
        st.warning(f"No runs found under `{RUNS_DIR}`. Create one from the sidebar.")
        return

    state = manager.load(run_id)
    run_dir = RUNS_DIR / run_id

    st.caption(
        f"`{state.run_id}` · mode `{state.mode}` · created {state.created_at} · updated {state.updated_at}"
    )
    source_question = _source_question(run_dir)
    if source_question:
        st.markdown(f"> {source_question}")
    _render_phase_tracker(state.phase)
    st.divider()

    tabs = st.tabs(
        ["Gates & Validation", "Hypothesis & DAG", "Analysis Plan", "Feasibility & Cohort", "Demo Report & History"]
    )
    with tabs[0]:
        _render_gates_tab(manager, state, run_dir)
    with tabs[1]:
        _render_hypothesis_tab(manager, run_id, run_dir)
    with tabs[2]:
        _render_analysis_plan_tab(run_dir)
    with tabs[3]:
        _render_feasibility_tab(run_dir)
    with tabs[4]:
        _render_report_tab(manager, state, run_dir)


if __name__ == "__main__":
    main()
