"""Deterministic causal-roadmap contracts and DAG identification checks.

This module is deliberately independent from OMOP and from any LLM.  A causal
question must first have a coherent causal model and an identified estimand;
only then should variables be operationalized against an observational CDM.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any


NODE_ROLES = {
    "exposure",
    "outcome",
    "confounder",
    "mediator",
    "collider",
    "prognostic",
    "effect_modifier",
    "selection",
    "other",
}
NODE_TIMINGS = ("baseline", "treatment", "post_treatment", "outcome")
_TIMING_ORDER = {name: index for index, name in enumerate(NODE_TIMINGS)}


def _require_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class CausalQuestion:
    population: str
    treatment: str
    comparator: str
    outcome: str

    def __post_init__(self) -> None:
        for field in ("population", "treatment", "comparator", "outcome"):
            object.__setattr__(self, field, _require_text(getattr(self, field), field))


@dataclass(frozen=True)
class TargetTrialProtocol:
    eligibility_criteria: tuple[str, ...]
    treatment_strategy: str
    comparator_strategy: str
    assignment_procedure: str
    time_zero: str
    grace_period_hours: int
    follow_up_days: int
    outcome_definition: str
    censoring_strategy: str
    analysis_strategy: str

    def __post_init__(self) -> None:
        if not self.eligibility_criteria or any(not item.strip() for item in self.eligibility_criteria):
            raise ValueError("target trial requires explicit eligibility criteria")
        for field in (
            "treatment_strategy", "comparator_strategy", "assignment_procedure",
            "time_zero", "outcome_definition", "censoring_strategy", "analysis_strategy",
        ):
            object.__setattr__(self, field, _require_text(getattr(self, field), field))
        if (
            not isinstance(self.grace_period_hours, int)
            or isinstance(self.grace_period_hours, bool)
            or self.grace_period_hours < 0
        ):
            raise ValueError("grace_period_hours must be an integer >= 0")
        if (
            not isinstance(self.follow_up_days, int)
            or isinstance(self.follow_up_days, bool)
            or self.follow_up_days < 1
        ):
            raise ValueError("follow_up_days must be an integer >= 1")


@dataclass(frozen=True)
class CausalEstimand:
    population_summary: str
    effect_measure: str
    treatment_value: str
    comparator_value: str
    outcome: str
    horizon_days: int

    def __post_init__(self) -> None:
        if self.population_summary not in {"ATE", "ATT", "CATE"}:
            raise ValueError("population_summary must be ATE, ATT, or CATE")
        if self.effect_measure not in {
            "risk_difference", "risk_ratio", "odds_ratio", "mean_difference", "survival_difference"
        }:
            raise ValueError("Unsupported causal effect measure")
        for field in ("treatment_value", "comparator_value", "outcome"):
            object.__setattr__(self, field, _require_text(getattr(self, field), field))
        if (
            not isinstance(self.horizon_days, int)
            or isinstance(self.horizon_days, bool)
            or self.horizon_days < 1
        ):
            raise ValueError("horizon_days must be an integer >= 1")

    @property
    def counterfactual_contrast(self) -> str:
        return (
            f"{self.effect_measure}: Y({self.treatment_value}) vs "
            f"Y({self.comparator_value}) at day {self.horizon_days}"
        )


@dataclass(frozen=True)
class CausalNode:
    node_id: str
    label: str
    role: str
    timing: str
    observed: bool

    def __post_init__(self) -> None:
        if not self.node_id or not self.node_id.replace("_", "").isalnum():
            raise ValueError("node_id must be a non-empty alphanumeric snake_case identifier")
        if not self.label.strip():
            raise ValueError(f"Causal node {self.node_id} requires a label")
        if self.role not in NODE_ROLES:
            raise ValueError(f"Unknown causal node role: {self.role}")
        if self.timing not in NODE_TIMINGS:
            raise ValueError(f"Unknown causal node timing: {self.timing}")


@dataclass(frozen=True)
class CausalEdge:
    source: str
    target: str
    rationale: str
    evidence_ids: tuple[str, ...] = ()
    status: str = "proposed"

    def __post_init__(self) -> None:
        if self.source == self.target:
            raise ValueError("A causal edge cannot be a self-loop")
        if not self.rationale.strip():
            raise ValueError("A causal edge requires a rationale")
        if self.status not in {"proposed", "expert_approved", "evidence_supported"}:
            raise ValueError(f"Unknown causal edge status: {self.status}")


@dataclass(frozen=True)
class AdjustmentAssessment:
    adjustment_set: tuple[str, ...]
    backdoor_satisfied: bool
    valid: bool
    blockers: tuple[str, ...]
    assumptions_requiring_review: tuple[str, ...] = (
        "consistency_and_well_defined_interventions",
        "conditional_exchangeability_given_adjustment_set",
        "positivity_in_the_target_population",
        "no_interference_between_individuals",
        "correct_temporal_order_and_measurement",
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CausalDAG:
    nodes: tuple[CausalNode, ...]
    edges: tuple[CausalEdge, ...]
    exposure_id: str
    outcome_id: str

    def __post_init__(self) -> None:
        node_map = self.node_map
        if len(node_map) != len(self.nodes):
            raise ValueError("Causal DAG node_id values must be unique")
        if self.exposure_id not in node_map or node_map[self.exposure_id].role != "exposure":
            raise ValueError("exposure_id must identify an exposure node")
        if self.outcome_id not in node_map or node_map[self.outcome_id].role != "outcome":
            raise ValueError("outcome_id must identify an outcome node")
        if node_map[self.exposure_id].timing != "treatment":
            raise ValueError("The exposure node must have treatment timing")
        if node_map[self.outcome_id].timing != "outcome":
            raise ValueError("The outcome node must have outcome timing")
        for edge in self.edges:
            if edge.source not in node_map or edge.target not in node_map:
                raise ValueError(f"Causal edge references an unknown node: {edge.source}->{edge.target}")
            source_timing = _TIMING_ORDER[node_map[edge.source].timing]
            target_timing = _TIMING_ORDER[node_map[edge.target].timing]
            if source_timing > target_timing:
                raise ValueError(
                    f"Causal edge violates declared temporal order: {edge.source}->{edge.target}"
                )
        self._topological_order()

    @property
    def node_map(self) -> dict[str, CausalNode]:
        return {node.node_id: node for node in self.nodes}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def descendants(self, node_id: str) -> set[str]:
        if node_id not in self.node_map:
            raise ValueError(f"Unknown causal node: {node_id}")
        children = self._children(self.edges)
        found: set[str] = set()
        pending = list(children[node_id])
        while pending:
            current = pending.pop()
            if current not in found:
                found.add(current)
                pending.extend(children[current])
        return found

    def assess_adjustment_set(self, adjustment_set: tuple[str, ...]) -> AdjustmentAssessment:
        selected = tuple(dict.fromkeys(adjustment_set))
        node_map = self.node_map
        blockers = []
        for node_id in selected:
            node = node_map.get(node_id)
            if node is None:
                blockers.append(f"unknown_node:{node_id}")
                continue
            if node_id in {self.exposure_id, self.outcome_id}:
                blockers.append(f"cannot_adjust_for_endpoint:{node_id}")
            if not node.observed:
                blockers.append(f"unobserved_node:{node_id}")
            if node.timing != "baseline":
                blockers.append(f"not_baseline:{node_id}")
            if node.role in {"mediator", "collider", "selection"}:
                blockers.append(f"forbidden_role:{node_id}:{node.role}")
        for node_id in set(selected) & self.descendants(self.exposure_id):
            blockers.append(f"descendant_of_exposure:{node_id}")

        backdoor = False if blockers else self._is_backdoor_blocked(set(selected))
        if not backdoor:
            blockers.append("open_backdoor_path")
        return AdjustmentAssessment(
            adjustment_set=selected,
            backdoor_satisfied=backdoor,
            valid=not blockers,
            blockers=tuple(dict.fromkeys(blockers)),
        )

    def minimum_adjustment_sets(self, max_candidates: int = 16) -> tuple[tuple[str, ...], ...]:
        """Return minimum-cardinality valid baseline adjustment sets.

        The bounded exhaustive search is intended for reviewable study DAGs,
        not high-dimensional variable-selection graphs.
        """
        candidates = tuple(
            node.node_id
            for node in self.nodes
            if node.observed
            and node.timing == "baseline"
            and node.role in {"confounder", "prognostic", "effect_modifier", "other"}
        )
        if len(candidates) > max_candidates:
            raise ValueError(
                f"DAG has {len(candidates)} adjustment candidates; maximum is {max_candidates}"
            )
        for size in range(len(candidates) + 1):
            valid = tuple(
                selected
                for selected in combinations(candidates, size)
                if self.assess_adjustment_set(selected).valid
            )
            if valid:
                return valid
        return ()

    def _is_backdoor_blocked(self, conditioned: set[str]) -> bool:
        # Backdoor graph: remove arrows emanating from the exposure.
        edges = tuple(edge for edge in self.edges if edge.source != self.exposure_id)
        ancestors = self._ancestors({self.exposure_id, self.outcome_id} | conditioned, edges)

        # Moralize the relevant ancestral graph.
        neighbors = {node_id: set() for node_id in ancestors}
        parents = self._parents(edges)
        for edge in edges:
            if edge.source in ancestors and edge.target in ancestors:
                neighbors[edge.source].add(edge.target)
                neighbors[edge.target].add(edge.source)
        for child in ancestors:
            relevant_parents = tuple(parents[child] & ancestors)
            for left, right in combinations(relevant_parents, 2):
                neighbors[left].add(right)
                neighbors[right].add(left)

        pending = [self.exposure_id]
        visited = set(conditioned)
        while pending:
            current = pending.pop()
            if current == self.outcome_id:
                return False
            if current in visited:
                continue
            visited.add(current)
            pending.extend(neighbors[current] - visited)
        return True

    def _topological_order(self) -> tuple[str, ...]:
        parents = self._parents(self.edges)
        children = self._children(self.edges)
        ready = [node_id for node_id, values in parents.items() if not values]
        ordered = []
        while ready:
            current = ready.pop()
            ordered.append(current)
            for child in children[current]:
                parents[child].remove(current)
                if not parents[child]:
                    ready.append(child)
        if len(ordered) != len(self.nodes):
            raise ValueError("Causal graph must be acyclic")
        return tuple(ordered)

    def _ancestors(self, node_ids: set[str], edges: tuple[CausalEdge, ...]) -> set[str]:
        parents = self._parents(edges)
        found = set(node_ids)
        pending = list(node_ids)
        while pending:
            current = pending.pop()
            for parent in parents[current]:
                if parent not in found:
                    found.add(parent)
                    pending.append(parent)
        return found

    def _parents(self, edges: tuple[CausalEdge, ...]) -> dict[str, set[str]]:
        result = {node.node_id: set() for node in self.nodes}
        for edge in edges:
            result[edge.target].add(edge.source)
        return result

    def _children(self, edges: tuple[CausalEdge, ...]) -> dict[str, set[str]]:
        result = {node.node_id: set() for node in self.nodes}
        for edge in edges:
            result[edge.source].add(edge.target)
        return result


@dataclass(frozen=True)
class CausalStudySpecification:
    """Pre-OMOP causal specification produced by the first Roadmap stages."""

    schema_version: str
    study_id: str
    source_question: str
    question: CausalQuestion
    target_trial: TargetTrialProtocol
    estimand: CausalEstimand
    dag: CausalDAG

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("Causal study specification schema_version must be 1.0")
        if not self.study_id or not self.study_id.replace("_", "").isalnum():
            raise ValueError("study_id must be an alphanumeric snake_case identifier")
        _require_text(self.source_question, "source_question")
        if self.estimand.horizon_days != self.target_trial.follow_up_days:
            raise ValueError("Estimand horizon must match target-trial follow-up")

    @property
    def identified_by_backdoor_adjustment(self) -> bool:
        return bool(self.dag.minimum_adjustment_sets())

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["identification"] = {
            "identified_by_backdoor_adjustment": self.identified_by_backdoor_adjustment,
            "minimum_adjustment_sets": self.dag.minimum_adjustment_sets(),
            "counterfactual_contrast": self.estimand.counterfactual_contrast,
        }
        return result
