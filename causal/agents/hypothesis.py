"""Deterministic hypothesis agent producing a normalized causal protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from causal.provenance import artifact_metadata


REQUIRED_TARGET_TRIAL_FIELDS = (
    "population",
    "treatment",
    "comparator",
    "time_zero",
    "grace_period_hours",
    "outcome",
    "follow_up_days",
    "primary_estimand",
)

DAG_RELATIONS = ("confounder_to_treatment", "confounder_to_outcome")


@dataclass(frozen=True)
class DAGEdge:
    """One literature-grounded candidate DAG edge. Always provisional."""

    confounder: str
    relation: str
    pmid: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HypothesisArtifact:
    metadata: dict[str, str]
    schema_version: str
    study_id: str
    question_type: str
    source_question: str
    population: str
    treatment: str
    comparator: str
    outcome: str
    time_zero: str
    grace_period_hours: int
    follow_up_days: int
    primary_estimand: str
    secondary_estimands: tuple[str, ...]
    candidate_confounders: tuple[str, ...]
    candidate_effect_modifiers: tuple[str, ...]
    candidate_dag: tuple[DAGEdge, ...]
    design_constraints: tuple[str, ...]
    unresolved_issues: tuple[str, ...]
    executable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HypothesisAgent:
    """Normalize an approved research idea without inventing clinical facts."""

    def run(self, protocol: dict[str, Any]) -> HypothesisArtifact:
        study_id = self._required_text(protocol, "study_id")
        source_question = self._required_text(protocol, "source_question")
        target = protocol.get("target_trial")
        if not isinstance(target, dict):
            raise ValueError("Protocol requires a target_trial object")
        missing = [name for name in REQUIRED_TARGET_TRIAL_FIELDS if target.get(name) in (None, "")]
        if missing:
            raise ValueError(f"Missing target trial fields: {', '.join(missing)}")

        grace = self._positive_int(target, "grace_period_hours", allow_zero=True)
        follow_up = self._positive_int(target, "follow_up_days")
        modifiers = tuple(str(value) for value in target.get("secondary_estimands", ()))
        covariates = tuple(str(value) for value in protocol.get("first_pass_covariates", ()))
        constraints = tuple(str(value) for value in protocol.get("design_constraints", ()))
        unresolved = self._unresolved_concept_sets(protocol.get("concept_sets", {}))

        return HypothesisArtifact(
            metadata=artifact_metadata("hypothesis_agent", "0.1.0", protocol),
            schema_version="1.0",
            study_id=study_id,
            question_type="causal",
            source_question=source_question,
            population=str(target["population"]),
            treatment=str(target["treatment"]),
            comparator=str(target["comparator"]),
            outcome=str(target["outcome"]),
            time_zero=str(target["time_zero"]),
            grace_period_hours=grace,
            follow_up_days=follow_up,
            primary_estimand=str(target["primary_estimand"]),
            secondary_estimands=modifiers,
            candidate_confounders=covariates,
            candidate_effect_modifiers=self._effect_modifiers(modifiers),
            candidate_dag=self._candidate_dag(protocol.get("candidate_dag", ())),
            design_constraints=constraints,
            unresolved_issues=unresolved,
            executable=not unresolved,
        )

    @staticmethod
    def _required_text(value: dict[str, Any], field: str) -> str:
        result = value.get(field)
        if not isinstance(result, str) or not result.strip():
            raise ValueError(f"Protocol requires {field}")
        return result.strip()

    @staticmethod
    def _positive_int(value: dict[str, Any], field: str, allow_zero: bool = False) -> int:
        result = value.get(field)
        minimum = 0 if allow_zero else 1
        if not isinstance(result, int) or isinstance(result, bool) or result < minimum:
            raise ValueError(f"{field} must be an integer >= {minimum}")
        return result

    @staticmethod
    def _effect_modifiers(estimands: tuple[str, ...]) -> tuple[str, ...]:
        modifiers = []
        for estimand in estimands:
            if estimand.lower().startswith("cate by "):
                modifiers.append(estimand[8:].strip())
        return tuple(modifiers)

    @staticmethod
    def _candidate_dag(value: Any) -> tuple[DAGEdge, ...]:
        """Structurally validate DAG edges regardless of who produced protocol.json.

        This does not re-verify that a pmid is real or was actually retrieved
        from PubMed — that grounding check happens where the retrieval result
        is known, in the LLM bridge. This is a safety net for any protocol
        (LLM-drafted or hand-authored) reaching the agent.
        """
        if not isinstance(value, (list, tuple)):
            return ()
        edges = []
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("Each candidate_dag entry must be an object")
            confounder = item.get("confounder")
            relation = item.get("relation")
            pmid = item.get("pmid")
            rationale = item.get("rationale")
            if not isinstance(confounder, str) or not confounder.strip():
                raise ValueError("candidate_dag entry requires a non-empty confounder")
            if relation not in DAG_RELATIONS:
                raise ValueError(f"candidate_dag relation must be one of {DAG_RELATIONS}")
            if not isinstance(pmid, str) or not pmid.strip():
                raise ValueError("candidate_dag entry requires a non-empty pmid")
            if not isinstance(rationale, str) or not rationale.strip():
                raise ValueError("candidate_dag entry requires a non-empty rationale")
            edges.append(DAGEdge(confounder.strip(), relation, pmid.strip(), rationale.strip()))
        return tuple(edges)

    @staticmethod
    def _unresolved_concept_sets(concept_sets: Any) -> tuple[str, ...]:
        if not isinstance(concept_sets, dict):
            return ("concept_sets: missing",)
        issues = []
        for name, definition in concept_sets.items():
            status = definition.get("status") if isinstance(definition, dict) else "invalid"
            if status != "approved":
                issues.append(f"concept_set:{name}:{status}")
        return tuple(issues)
