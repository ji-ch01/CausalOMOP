"""Filesystem-backed workflow manager with explicit human approval gates."""

from __future__ import annotations

import json
import csv
import io
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from causal.orchestrator.state import ApprovalGate, NEXT_PHASE, RunState
from causal.agents import DataScienceAgent, ExecutionAgent, HypothesisAgent
from causal.agents.hypothesis import DAG_VARIABLE_ROLES, DAG_VARIABLE_TEMPORAL_CLASSES
from causal.agents.user_question import GeneratedHypothesis
from causal.bigquery import BigQueryOMOPGateway
from causal.config import BigQueryDatasets
from causal.data_builder import AnalyticCohortBuilder
from causal.mcp_client import OMCPClient
from causal.queries import albumin_sepsis_feasibility_sql


REQUIRED_GATES = {
    "protocol_review": "protocol",
    "concept_mapping": "clinical_mapping",
    "analysis_plan": "analysis_plan",
    "results_review": "results",
}


def _make_run_id(study_id: str) -> str:
    """Build a chronologically sortable, question-identifiable run id.

    Format: {UTC timestamp}_{study_id}-{short random hash}, e.g.
    20260723-143012_albumin_sepsis_28d_mimic_omop-f00fefd1
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}_{study_id}-{uuid.uuid4().hex[:8]}"


class WorkflowManager:
    def __init__(self, runs_dir: str | Path = "artifacts/runs") -> None:
        self.runs_dir = Path(runs_dir)

    def start(self, protocol_path: str | Path, mode: str = "exploratory") -> RunState:
        if mode not in {"exploratory", "final"}:
            raise ValueError("mode must be exploratory or final")
        source = Path(protocol_path)
        protocol = json.loads(source.read_text(encoding="utf-8"))
        study_id = protocol.get("study_id")
        if not isinstance(study_id, str) or not study_id.strip():
            raise ValueError("Protocol requires a non-empty study_id")

        run_id = _make_run_id(study_id)
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        snapshot = run_dir / "protocol.json"
        shutil.copyfile(source, snapshot)
        state = RunState(
            run_id=run_id,
            study_id=study_id,
            protocol_path=str(snapshot),
            mode=mode,
            gates={
                "protocol": ApprovalGate("protocol"),
                "clinical_mapping": ApprovalGate("clinical_mapping"),
                "analysis_plan": ApprovalGate("analysis_plan"),
                "results": ApprovalGate("results"),
            },
            artifacts=["protocol.json"],
        )
        state.record("run_started", f"mode={mode}")
        self.save(state)
        return state

    def start_generated(
        self,
        generated: GeneratedHypothesis,
        mode: str = "exploratory",
        model: str = "unknown",
    ) -> RunState:
        """Create a governed run from an LLM draft without bypassing gates."""
        if mode not in {"exploratory", "final"}:
            raise ValueError("mode must be exploratory or final")
        study_id = generated.protocol.get("study_id")
        if not isinstance(study_id, str) or not study_id.strip():
            raise ValueError("Generated protocol requires a non-empty study_id")
        run_id = _make_run_id(study_id)
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "protocol.json").write_text(
            json.dumps(generated.protocol, indent=2), encoding="utf-8"
        )
        (run_dir / "hypothesis.json").write_text(
            json.dumps(generated.hypothesis.to_dict(), indent=2), encoding="utf-8"
        )
        (run_dir / "llm_outputs.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "model": model,
                    "prompt_version": generated.prompt_version,
                    "contains_raw_llm_text": True,
                    "outputs": [item.to_dict() for item in generated.llm_outputs],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        state = RunState(
            run_id=run_id,
            study_id=study_id,
            protocol_path=str(run_dir / "protocol.json"),
            mode=mode,
            gates={
                "protocol": ApprovalGate("protocol"),
                "clinical_mapping": ApprovalGate("clinical_mapping"),
                "analysis_plan": ApprovalGate("analysis_plan"),
                "results": ApprovalGate("results"),
            },
            artifacts=["protocol.json", "hypothesis.json", "llm_outputs.json"],
        )
        state.record("run_started", f"mode={mode}; source=llm_draft")
        state.record(
            "agent_completed",
            f"gpt_hypothesis_draft; model={model}; raw_outputs={len(generated.llm_outputs)}",
        )
        self.save(state)
        return state

    def load(self, run_id: str) -> RunState:
        path = self._state_path(run_id)
        if not path.is_file():
            raise FileNotFoundError(f"Unknown run_id: {run_id}")
        return RunState.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, state: RunState) -> None:
        path = self._state_path(state.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(path)

    def decide_gate(self, run_id: str, gate: str, status: str, reviewer: str, comment: str | None = None) -> RunState:
        state = self.load(run_id)
        if gate not in state.gates:
            raise ValueError(f"Unknown gate: {gate}")
        if status not in {"approved", "rejected", "provisional"}:
            raise ValueError("status must be approved, rejected, or provisional")
        if gate == "protocol" and status == "approved":
            self._block_if_major_confounders_unresolved(run_id)
        state.gates[gate].decide(status, reviewer, comment)  # type: ignore[arg-type]
        state.record("gate_decided", f"{gate}={status} by {reviewer}")
        self.save(state)
        return state

    def advance(self, run_id: str) -> RunState:
        state = self.load(run_id)
        gate_name = REQUIRED_GATES.get(state.phase)
        if gate_name:
            gate = state.gates[gate_name]
            accepted = gate.status == "approved" or (
                state.mode == "exploratory" and gate.status == "provisional" and gate_name == "clinical_mapping"
            )
            if not accepted:
                raise RuntimeError(f"Gate {gate_name} must be accepted before leaving {state.phase}")
        next_phase = NEXT_PHASE.get(state.phase)
        if next_phase is None:
            raise RuntimeError("Workflow is already at its final phase")
        previous = state.phase
        state.phase = next_phase
        state.record("phase_advanced", f"{previous}->{next_phase}")
        self.save(state)
        return state

    def run_hypothesis_agent(self, run_id: str) -> RunState:
        state = self.load(run_id)
        if state.phase not in {"protocol_review", "concept_mapping"}:
            raise RuntimeError("Hypothesis Agent can only run during protocol_review or concept_mapping")
        protocol_path = self.runs_dir / run_id / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        self._write_hypothesis(run_id, protocol)
        if "hypothesis.json" not in state.artifacts:
            state.artifacts.append("hypothesis.json")
        state.record("agent_completed", "hypothesis_agent->hypothesis.json")
        self.save(state)
        return state

    def review_covariate(
        self,
        run_id: str,
        label: str,
        decision: str,
        reviewer: str,
        comment: str | None = None,
        *,
        role: str | None = None,
        temporal_classification: str | None = None,
        pmid: str = "",
        observed: bool = True,
        certainty: str = "established",
    ) -> RunState:
        """Record a human accept/reject decision for a pending first-pass covariate.

        "Accept" adds a role-classified `candidate_dag` entry (same shape the
        LLM/literature stage produces, minus a pmid); "reject" records an
        audited exclusion instead of leaving the covariate silently absent.
        Either way, `hypothesis.json` is regenerated so the decision is
        immediately reflected in `assess_identification`.
        """
        if decision not in {"accept", "reject"}:
            raise ValueError("decision must be accept or reject")
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        label = label.strip()
        if not label:
            raise ValueError("label is required")

        state = self.load(run_id)
        run_dir = self.runs_dir / run_id
        protocol_path = run_dir / "protocol.json"
        if not (run_dir / "hypothesis.json").is_file():
            raise RuntimeError("Run the Hypothesis Agent before reviewing covariates")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

        candidate_dag = [item for item in protocol.get("candidate_dag", ()) if item.get("label") != label]
        rejected_covariates = [
            item for item in protocol.get("rejected_covariates", ()) if item.get("label") != label
        ]

        if decision == "accept":
            if role not in DAG_VARIABLE_ROLES:
                raise ValueError(f"role must be one of {DAG_VARIABLE_ROLES}")
            if temporal_classification not in DAG_VARIABLE_TEMPORAL_CLASSES:
                raise ValueError(
                    f"temporal_classification must be one of {DAG_VARIABLE_TEMPORAL_CLASSES}"
                )
            if not comment or not comment.strip():
                raise ValueError("comment is required as the rationale when accepting a covariate")
            candidate_dag.append({
                "label": label,
                "role": role,
                "temporal_classification": temporal_classification,
                "rationale": comment.strip(),
                "pmid": pmid,
                "observed": observed,
                "certainty": certainty,
            })
        else:
            rejected_covariates.append({
                "label": label,
                "reviewer": reviewer.strip(),
                "comment": (comment or "").strip(),
                "decided_at": datetime.now(timezone.utc).isoformat(),
            })

        protocol["candidate_dag"] = candidate_dag
        protocol["rejected_covariates"] = rejected_covariates
        temporary = protocol_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        temporary.replace(protocol_path)

        self._write_hypothesis(run_id, protocol)
        state.record("covariate_reviewed", f"{label}={decision} by {reviewer}")
        self.save(state)
        return state

    def _block_if_major_confounders_unresolved(self, run_id: str) -> None:
        """The DAG-approval gate ("protocol") cannot be approved while a major
        confounder is still unresolved: the run may never have called the
        Hypothesis Agent yet, so absence of hypothesis.json is not itself a
        blocker here (protocol.json-only workflows never touch this gate's
        clinical content). Once hypothesis.json exists, its
        `unresolved_major_confounders` is authoritative.
        """
        hypothesis_path = self.runs_dir / run_id / "hypothesis.json"
        if not hypothesis_path.is_file():
            return
        hypothesis = json.loads(hypothesis_path.read_text(encoding="utf-8"))
        unresolved_major = hypothesis.get("unresolved_major_confounders") or []
        if unresolved_major:
            raise ValueError(
                "Cannot approve the protocol/DAG gate while major confounders remain "
                f"unresolved: {', '.join(unresolved_major)}. Classify each one with a role and "
                "rationale, or explicitly reject it with a reviewer and rationale."
            )

    def _write_hypothesis(self, run_id: str, protocol: dict) -> None:
        artifact = HypothesisAgent().run(protocol).to_dict()
        artifact_path = self.runs_dir / run_id / "hypothesis.json"
        temporary = artifact_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        temporary.replace(artifact_path)

    def run_data_science_agent(
        self,
        run_id: str,
        concept_lookup: OMCPClient | None = None,
    ) -> RunState:
        state = self.load(run_id)
        if state.phase not in {"concept_mapping", "feasibility", "analysis_plan"}:
            raise RuntimeError("Data Science Agent requires concept_mapping, feasibility, or analysis_plan phase")
        run_dir = self.runs_dir / run_id
        hypothesis_path = run_dir / "hypothesis.json"
        if not hypothesis_path.is_file():
            raise RuntimeError("Run the Hypothesis Agent first")
        hypothesis = json.loads(hypothesis_path.read_text(encoding="utf-8"))
        protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
        artifact = DataScienceAgent().run(
            hypothesis, protocol, concept_lookup=concept_lookup
        ).to_dict()
        artifact_path = run_dir / "analysis_plan.json"
        temporary = artifact_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        temporary.replace(artifact_path)
        if "analysis_plan.json" not in state.artifacts:
            state.artifacts.append("analysis_plan.json")
        source = "omcp_candidates" if concept_lookup is not None else "protocol_only"
        state.record(
            "agent_completed",
            f"data_science_agent->analysis_plan.json; mapping_source={source}",
        )
        self.save(state)
        return state

    def run_execution_readiness(self, run_id: str) -> RunState:
        """Create the execution policy artifact without running an estimator."""
        state = self.load(run_id)
        if state.phase not in {"feasibility", "analysis_plan", "execution"}:
            raise RuntimeError(
                "Execution readiness requires feasibility, analysis_plan, or execution phase"
            )
        run_dir = self.runs_dir / run_id
        plan_path = run_dir / "analysis_plan.json"
        if not plan_path.is_file():
            raise RuntimeError("Run the Data Science Agent first")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        artifact = ExecutionAgent().run(
            plan,
            mode=state.mode,
            clinical_mapping_status=state.gates["clinical_mapping"].status,
            analysis_plan_status=state.gates["analysis_plan"].status,
        ).to_dict()
        artifact_path = run_dir / "execution_readiness.json"
        temporary = artifact_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        temporary.replace(artifact_path)
        mcp_context = {
            "run_id": state.run_id,
            "study_id": state.study_id,
            "mode": state.mode,
            "clinical_mapping_status": state.gates["clinical_mapping"].status,
            "analysis_plan_status": state.gates["analysis_plan"].status,
            "aggregate_feasibility_allowed": artifact["aggregate_feasibility_allowed"],
            "analytic_dataset_build_allowed": artifact["analytic_dataset_build_allowed"],
            "causal_estimation_allowed": artifact["causal_estimation_allowed"],
            "readiness_input_sha256": artifact["metadata"]["input_sha256"],
        }
        context_path = run_dir / "mcp_context.json"
        context_temporary = context_path.with_suffix(".tmp")
        context_temporary.write_text(
            json.dumps(mcp_context, indent=2), encoding="utf-8"
        )
        context_temporary.replace(context_path)
        if "execution_readiness.json" not in state.artifacts:
            state.artifacts.append("execution_readiness.json")
        if "mcp_context.json" not in state.artifacts:
            state.artifacts.append("mcp_context.json")
        state.record(
            "agent_completed",
            "execution_agent->execution_readiness.json+mcp_context.json; no causal effect estimated",
        )
        self.save(state)
        return state

    def build_cohort_plan(
        self,
        run_id: str,
        datasets: BigQueryDatasets | None = None,
    ) -> RunState:
        """Write cohort SQL and its governance manifest without executing it."""
        state = self.load(run_id)
        if state.phase not in {"analysis_plan", "execution"}:
            raise RuntimeError("Cohort planning requires analysis_plan or execution phase")
        run_dir = self.runs_dir / run_id
        self.run_execution_readiness(run_id)
        state = self.load(run_id)
        readiness = json.loads(
            (run_dir / "execution_readiness.json").read_text(encoding="utf-8")
        )
        sql, manifest = AnalyticCohortBuilder().build(
            state.study_id,
            datasets or BigQueryDatasets.from_env(),
            execution_allowed=bool(readiness["analytic_dataset_build_allowed"]),
        )
        (run_dir / "cohort.sql").write_text(sql, encoding="utf-8")
        (run_dir / "cohort_manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
        )
        for artifact in ("cohort.sql", "cohort_manifest.json"):
            if artifact not in state.artifacts:
                state.artifacts.append(artifact)
        state.record(
            "cohort_plan_generated",
            f"cohort.sql; execution_allowed={manifest.execution_allowed}",
        )
        self.save(state)
        return state

    def run_cohort_profile_mcp(
        self,
        run_id: str,
        client: OMCPClient,
        datasets: BigQueryDatasets,
    ) -> RunState:
        """Validate the draft cohort using aggregate results only."""
        state = self.load(run_id)
        if state.phase not in {"feasibility", "analysis_plan"}:
            raise RuntimeError("Cohort profiling requires feasibility or analysis_plan phase")
        self.run_execution_readiness(run_id)
        state = self.load(run_id)
        run_dir = self.runs_dir / run_id
        context = json.loads((run_dir / "mcp_context.json").read_text(encoding="utf-8"))
        response = client.causal_select(
            AnalyticCohortBuilder().profile_sql(state.study_id, datasets),
            "aggregate_feasibility",
            context,
        )
        rows = list(csv.DictReader(io.StringIO(response.text)))
        if len(rows) != 1:
            raise RuntimeError("OMCP cohort profile must return one aggregate row")
        profile: dict[str, Any] = {}
        for key, value in rows[0].items():
            if key is None or value in (None, ""):
                profile[key] = None
            elif "." in value:
                profile[key] = float(value)
            else:
                profile[key] = int(value)
        artifact = {
            "status": "aggregate_validation_only",
            "execution_transport": "omcp",
            "causal_effect_estimated": False,
            "patient_level_rows_returned": False,
            "profile": profile,
            "mcp_metadata": response.metadata,
            "warnings": [
                "The cohort still uses provisional phenotypes.",
                "Censoring is unresolved for every row in the draft contract.",
                "This profile validates mechanics and cannot support a causal conclusion.",
            ],
        }
        (run_dir / "cohort_profile.json").write_text(
            json.dumps(artifact, indent=2), encoding="utf-8"
        )
        if "cohort_profile.json" not in state.artifacts:
            state.artifacts.append("cohort_profile.json")
        state.record("cohort_profile_completed", "aggregate profile via OMCP")
        self.save(state)
        return state

    def run_followup_audit_mcp(
        self,
        run_id: str,
        client: OMCPClient,
        datasets: BigQueryDatasets,
    ) -> RunState:
        """Audit nominal follow-up completeness without returning patient rows."""
        state = self.load(run_id)
        if state.phase not in {"feasibility", "analysis_plan"}:
            raise RuntimeError("Follow-up audit requires feasibility or analysis_plan phase")
        self.run_execution_readiness(run_id)
        state = self.load(run_id)
        run_dir = self.runs_dir / run_id
        context = json.loads((run_dir / "mcp_context.json").read_text(encoding="utf-8"))
        response = client.causal_select(
            AnalyticCohortBuilder().followup_audit_sql(state.study_id, datasets),
            "aggregate_feasibility",
            context,
        )
        rows = list(csv.DictReader(io.StringIO(response.text)))
        if len(rows) != 1:
            raise RuntimeError("OMCP follow-up audit must return one aggregate row")
        counts = {
            key: int(value) for key, value in rows[0].items()
            if key and value not in (None, "")
        }
        artifact = {
            "status": "aggregate_observability_audit_only",
            "execution_transport": "omcp",
            "causal_effect_estimated": False,
            "patient_level_rows_returned": False,
            "censoring_definition_approved": False,
            "counts": counts,
            "mcp_metadata": response.metadata,
            "warnings": [
                "OMOP observation periods may encode recorded encounters rather than complete mortality ascertainment.",
                "Nominal completeness is a diagnostic and is not an approved censoring definition.",
                "Clinical and data-owner validation is required before using this field analytically.",
            ],
        }
        (run_dir / "followup_audit.json").write_text(
            json.dumps(artifact, indent=2), encoding="utf-8"
        )
        if "followup_audit.json" not in state.artifacts:
            state.artifacts.append("followup_audit.json")
        state.record("followup_audit_completed", "aggregate audit via OMCP")
        self.save(state)
        return state

    def build_demo_report(self, run_id: str) -> Path:
        state = self.load(run_id)
        run_dir = self.runs_dir / run_id
        hypothesis = self._read_optional_json(run_dir / "hypothesis.json")
        plan = self._read_optional_json(run_dir / "analysis_plan.json")
        feasibility = self._read_optional_json(run_dir / "feasibility.json")
        cohort_manifest = self._read_optional_json(run_dir / "cohort_manifest.json")
        cohort_profile = self._read_optional_json(run_dir / "cohort_profile.json")
        followup_audit = self._read_optional_json(run_dir / "followup_audit.json")
        readiness = self._read_optional_json(run_dir / "execution_readiness.json")
        lines = [
            f"# CausalOMOP demonstration — {state.study_id}",
            "",
            "> Architecture and feasibility demonstration only. No causal effect has been estimated.",
            "",
            "## Workflow status",
            "",
            f"- Run: `{state.run_id}`",
            f"- Mode: `{state.mode}`",
            f"- Phase: `{state.phase}`",
            f"- Created: `{state.created_at}`",
            "",
            "## Approval gates",
            "",
            "| Gate | Status | Reviewer |",
            "|---|---|---|",
        ]
        for gate in state.gates.values():
            lines.append(
                f"| {gate.name} | {gate.status} | {gate.reviewer or '—'} |"
            )
        if hypothesis:
            lines.extend([
                "", "## Normalized causal question", "",
                f"- Population: {hypothesis['population']}",
                f"- Treatment: {hypothesis['treatment']}",
                f"- Comparator: {hypothesis['comparator']}",
                f"- Outcome: {hypothesis['outcome']}",
                f"- Time zero: {hypothesis['time_zero']}",
                f"- Grace period: {hypothesis['grace_period_hours']} hours",
                f"- Follow-up: {hypothesis['follow_up_days']} days",
                f"- Primary estimand: {hypothesis['primary_estimand']}",
                "", "### Unresolved hypothesis issues", "",
            ])
            lines.extend(f"- `{issue}`" for issue in hypothesis.get("unresolved_issues", ()))
        if plan:
            lines.extend([
                "", "## Analysis plan", "",
                f"- Data source: `{plan['data_source']}`",
                f"- Design: `{plan['design']}`",
                f"- Exploratory execution allowed: `{plan['exploratory_execution_allowed']}`",
                f"- Final execution allowed: `{plan['final_execution_allowed']}`",
                "", "### Phenotypes", "",
                "| Name | Role | Mode | Status |",
                "|---|---|---|---|",
            ])
            for phenotype in plan.get("phenotypes", ()):
                lines.append(
                    f"| {phenotype['name']} | {phenotype['role']} | {phenotype['mode']} | {phenotype['status']} |"
                )
            lines.extend(["", "### Planned feasibility checks", ""])
            lines.extend(f"- {check}" for check in plan.get("feasibility_checks", ()))
        if feasibility:
            counts = feasibility.get("counts", {})
            lines.extend([
                "", "## Provisional feasibility counts", "",
                "> These counts use a diagnosis-based sepsis phenotype and unvalidated source values. They are not a causal estimate.",
                "",
                f"- Eligible persons: **{counts.get('eligible_persons', '—')}**",
                f"- Albumin plus crystalloids: **{counts.get('albumin_plus_crystalloid', '—')}**",
                f"- Crystalloids only: **{counts.get('crystalloid_only', '—')}**",
                f"- No recognized exposure: **{counts.get('neither_exposure', '—')}**",
                f"- Deaths within 28 days, all groups: **{counts.get('deaths_28d', '—')}**",
                "", "### Feasibility warnings", "",
            ])
            lines.extend(f"- {warning}" for warning in feasibility.get("warnings", ()))
        if cohort_manifest:
            lines.extend([
                "", "## Draft analytic cohort contract", "",
                f"- Status: `{cohort_manifest.get('status', 'unknown')}`",
                f"- Person-level data: `{cohort_manifest.get('contains_person_level_data', True)}`",
                f"- Execution allowed: `{cohort_manifest.get('execution_allowed', False)}`",
                f"- Delivery to an LLM allowed: `{cohort_manifest.get('delivery_to_llm_allowed', False)}`",
                "", "### Unresolved cohort issues", "",
            ])
            lines.extend(
                f"- `{issue}`" for issue in cohort_manifest.get("unresolved_issues", ())
            )
        if cohort_profile:
            profile = cohort_profile.get("profile", {})
            lines.extend([
                "", "## Aggregate cohort profile", "",
                "> This profile validates cohort-building mechanics only. It contains no patient-level rows and does not estimate a causal effect.",
                "",
                f"- Cohort persons: **{profile.get('cohort_persons', '—')}**",
                f"- Treated persons: **{profile.get('treated_persons', '—')}**",
                f"- Comparator persons: **{profile.get('comparator_persons', '—')}**",
                f"- Observed outcome events: **{profile.get('outcome_events', '—')}**",
                f"- Missing age: **{profile.get('missing_age', '—')}**",
                f"- Missing gender: **{profile.get('missing_gender', '—')}**",
                f"- Missing race: **{profile.get('missing_race', '—')}**",
                f"- Unresolved censoring: **{profile.get('unresolved_censoring', '—')}**",
                "", "### Cohort profile warnings", "",
            ])
            lines.extend(
                f"- {warning}" for warning in cohort_profile.get("warnings", ())
            )
        if followup_audit:
            counts = followup_audit.get("counts", {})
            lines.extend([
                "", "## Nominal 28-day follow-up audit", "",
                "> This is an aggregate observability diagnostic. The censoring definition is not clinically approved.",
                "",
                f"- Observation period covers 28 days: **{counts.get('observation_period_covers_28d', '—')}**",
                f"- Known death within 28 days: **{counts.get('known_death_within_28d', '—')}**",
                f"- Nominal follow-up complete: **{counts.get('nominal_followup_complete', '—')}**",
                f"- Nominal follow-up incomplete: **{counts.get('nominal_followup_incomplete', '—')}**",
                f"- No observation period at index: **{counts.get('no_observation_period_at_index', '—')}**",
                "", "### Follow-up audit warnings", "",
            ])
            lines.extend(f"- {warning}" for warning in followup_audit.get("warnings", ()))
        if readiness:
            lines.extend([
                "", "## Execution Agent decision", "",
                f"- Status: `{readiness.get('status', 'unknown')}`",
                f"- Aggregate feasibility allowed: `{readiness.get('aggregate_feasibility_allowed', False)}`",
                f"- Analytic dataset build allowed: `{readiness.get('analytic_dataset_build_allowed', False)}`",
                f"- Causal estimation allowed: `{readiness.get('causal_estimation_allowed', False)}`",
                "", "### Current execution blockers", "",
            ])
            lines.extend(f"- `{blocker}`" for blocker in readiness.get("blockers", ()))
        lines.extend([
            "", "## Interpretation", "",
            "The workflow contracts and audit trail are operational. Cohort feasibility refinement and causal estimates remain pending. "
            "Source-value phenotypes are restricted to development; final conclusions require clinical mapping approval.",
            "",
        ])
        report_path = run_dir / "demo_report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        if "demo_report.md" not in state.artifacts:
            state.artifacts.append("demo_report.md")
        state.record("report_generated", "demo_report.md")
        self.save(state)
        return report_path

    def run_feasibility(self, run_id: str, gateway: BigQueryOMOPGateway) -> RunState:
        state = self.load(run_id)
        if state.phase != "feasibility":
            raise RuntimeError("Feasibility execution requires the feasibility phase")
        if state.mode != "exploratory":
            raise RuntimeError("The provisional phenotype is restricted to exploratory runs")
        if state.gates["clinical_mapping"].status != "provisional":
            raise RuntimeError("clinical_mapping must be explicitly provisional")
        result = gateway.albumin_sepsis_feasibility()
        run_dir = self.runs_dir / run_id
        artifact_path = run_dir / "feasibility.json"
        temporary = artifact_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary.replace(artifact_path)
        if "feasibility.json" not in state.artifacts:
            state.artifacts.append("feasibility.json")
        state.record("feasibility_completed", "feasibility.json; no causal effect estimated")
        self.save(state)
        return state

    def run_feasibility_mcp(
        self,
        run_id: str,
        client: OMCPClient,
        datasets: BigQueryDatasets,
    ) -> RunState:
        """Run provisional aggregate feasibility through the shared OMCP server."""
        state = self.load(run_id)
        if state.phase != "feasibility":
            raise RuntimeError("Feasibility execution requires the feasibility phase")
        if state.mode != "exploratory":
            raise RuntimeError("The provisional phenotype is restricted to exploratory runs")
        if state.gates["clinical_mapping"].status != "provisional":
            raise RuntimeError("clinical_mapping must be explicitly provisional")

        self.run_execution_readiness(run_id)
        state = self.load(run_id)
        run_dir = self.runs_dir / run_id
        context = json.loads((run_dir / "mcp_context.json").read_text(encoding="utf-8"))
        response = client.causal_select(
            albumin_sepsis_feasibility_sql(datasets),
            "aggregate_feasibility",
            context,
        )
        rows = list(csv.DictReader(io.StringIO(response.text)))
        if len(rows) != 1:
            detail = response.text.replace("\n", " ")[:500]
            raise RuntimeError(
                "OMCP feasibility query must return exactly one aggregate row; "
                f"response={detail!r}"
            )
        counts = {
            key: int(value) for key, value in rows[0].items()
            if key and value not in (None, "")
        }
        result = {
            "status": "provisional_source_value_phenotype",
            "execution_transport": "omcp",
            "causal_effect_estimated": False,
            "grace_period_hours": 24,
            "follow_up_days": 28,
            "counts": counts,
            "mcp_metadata": response.metadata,
            "warnings": [
                "Sepsis uses a diagnosis concept set, not the published Sepsis-3 phenotype.",
                "Albumin and crystalloid exposure use unvalidated source-value regular expressions.",
                "Sodium-chloride products may include non-resuscitation administrations.",
                "Counts are for pipeline feasibility and must not be interpreted causally.",
            ],
        }
        artifact_path = run_dir / "feasibility.json"
        temporary = artifact_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary.replace(artifact_path)
        if "feasibility.json" not in state.artifacts:
            state.artifacts.append("feasibility.json")
        state.record(
            "feasibility_completed",
            "feasibility.json via OMCP; no causal effect estimated",
        )
        self.save(state)
        return state

    @staticmethod
    def _read_optional_json(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def _state_path(self, run_id: str) -> Path:
        if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in run_id):
            raise ValueError("Invalid run_id")
        return self.runs_dir / run_id / "state.json"
