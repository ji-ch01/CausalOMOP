"""Governed execution-readiness agent for causal OMOP studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from causal.provenance import artifact_metadata


@dataclass(frozen=True)
class ExecutionReadinessArtifact:
    metadata: dict[str, str]
    schema_version: str
    study_id: str
    status: str
    mode: str
    aggregate_feasibility_allowed: bool
    analytic_dataset_build_allowed: bool
    causal_estimation_allowed: bool
    permitted_actions: tuple[str, ...]
    prohibited_actions: tuple[str, ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analysis_plan_blockers(
    analysis_plan: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """The three reasons an analysis plan is not yet execution-ready:
    `(unresolved_issues, covariates_not_ready, phenotypes_not_approved)`.
    Shared by `ExecutionAgent.run` (the automated gate, evaluated every time
    readiness is computed) and `WorkflowManager._block_if_analysis_plan_
    incomplete` (the human-approval gate, evaluated once when a reviewer
    approves the `analysis_plan` gate) so the two can never disagree about
    what "complete" means.

    A malformed entry (not a dict, or missing the expected field) counts as
    incomplete for both covariates and phenotypes -- treating it as
    "somehow already fine" would let a hand-edited or buggily-generated
    plan slip past review.
    """
    unresolved = tuple(str(value) for value in analysis_plan.get("unresolved_issues", ()))
    incomplete_covariates = tuple(
        str(item.get("name", "unnamed")) if isinstance(item, dict) else "unnamed"
        for item in analysis_plan.get("covariates", ())
        if not isinstance(item, dict) or item.get("status") != "ready"
    )
    incomplete_phenotypes = tuple(
        str(item.get("name", "unnamed")) if isinstance(item, dict) else "unnamed"
        for item in analysis_plan.get("phenotypes", ())
        if not isinstance(item, dict)
        or (item.get("approval_required", True) and item.get("status") != "approved")
    )
    return unresolved, incomplete_covariates, incomplete_phenotypes


class ExecutionAgent:
    """Translate a reviewed plan into an explicit execution decision.

    Version 0.1 is deliberately a readiness controller. It never estimates an
    effect itself and therefore cannot turn a provisional phenotype into a
    causal result simply because a query happens to run.
    """

    def run(
        self,
        analysis_plan: dict[str, Any],
        *,
        mode: str,
        clinical_mapping_status: str,
        analysis_plan_status: str,
    ) -> ExecutionReadinessArtifact:
        if analysis_plan.get("schema_version") != "1.0":
            raise ValueError("Unsupported analysis plan schema_version")
        if mode not in {"exploratory", "final"}:
            raise ValueError("mode must be exploratory or final")
        study_id = self._required_text(analysis_plan, "study_id")

        unresolved, incomplete_covariates, incomplete_phenotypes = analysis_plan_blockers(analysis_plan)

        # Exploratory mode may probe aggregate feasibility as soon as
        # clinical mapping is at least provisional; final mode is held to
        # the stricter bar of a fully approved mapping (mirrors
        # `omcp.governance.authorize_causal_operation`'s own
        # `mode == "final"` check for this same operation -- that check is
        # unreachable unless this agent can actually grant the allowance it
        # gates on).
        aggregate_allowed = (
            clinical_mapping_status in {"provisional", "approved"}
            if mode == "exploratory"
            else clinical_mapping_status == "approved"
        )
        dataset_flag = (
            analysis_plan.get("exploratory_execution_allowed")
            if mode == "exploratory"
            else analysis_plan.get("final_execution_allowed")
        )
        dataset_allowed = bool(dataset_flag) and analysis_plan_status == "approved"
        causal_allowed = (
            dataset_allowed
            and analysis_plan_status == "approved"
            and clinical_mapping_status == "approved"
            and not unresolved
            and not incomplete_covariates
            and not incomplete_phenotypes
        )

        blockers = []
        if analysis_plan_status != "approved":
            blockers.append("analysis_plan_gate:not_approved")
        if clinical_mapping_status != "approved":
            blockers.append(f"clinical_mapping_gate:{clinical_mapping_status}")
        blockers.extend(f"unresolved:{value}" for value in unresolved)
        blockers.extend(f"covariate_not_ready:{value}" for value in incomplete_covariates)
        blockers.extend(f"phenotype_not_approved:{value}" for value in incomplete_phenotypes)

        permitted = []
        if aggregate_allowed:
            permitted.extend((
                "read_only_aggregate_feasibility",
                "cohort_attrition_counts",
                "mapping_and_missingness_audits",
            ))
        if dataset_allowed:
            permitted.append("build_deidentified_analytic_dataset")
        if causal_allowed:
            permitted.extend(("fit_nuisance_models", "estimate_prespecified_causal_effects"))

        prohibited = []
        if not dataset_allowed:
            prohibited.append("build_analysis_ready_patient_level_dataset")
        if not causal_allowed:
            prohibited.extend((
                "estimate_or_publish_causal_effect",
                "label_crude_group_differences_as_causal",
                "produce_final_clinical_conclusion",
            ))

        return ExecutionReadinessArtifact(
            metadata=artifact_metadata(
                "execution_agent",
                "0.1.0",
                {
                    "analysis_plan": analysis_plan,
                    "mode": mode,
                    "clinical_mapping_status": clinical_mapping_status,
                    "analysis_plan_status": analysis_plan_status,
                },
            ),
            schema_version="1.0",
            study_id=study_id,
            status="ready_for_causal_execution" if causal_allowed else "blocked_before_causal_execution",
            mode=mode,
            aggregate_feasibility_allowed=aggregate_allowed,
            analytic_dataset_build_allowed=dataset_allowed,
            causal_estimation_allowed=causal_allowed,
            permitted_actions=tuple(permitted),
            prohibited_actions=tuple(prohibited),
            blockers=tuple(dict.fromkeys(blockers)),
            warnings=(
                "Readiness is a technical policy decision, not clinical validation.",
                "Patient-level data must remain inside the governed analysis environment.",
            ),
        )

    @staticmethod
    def _required_text(value: dict[str, Any], field: str) -> str:
        result = value.get(field)
        if not isinstance(result, str) or not result.strip():
            raise ValueError(f"Missing {field}")
        return result.strip()
