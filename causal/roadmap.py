"""Deterministic causal-roadmap contracts and DAG identification checks.

This module is deliberately independent from OMOP and from any LLM.  A causal
question must first have a coherent causal model and an identified estimand;
only then should variables be operationalized against an observational CDM.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
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
    "instrument",
    "other",
}
NODE_TIMINGS = ("baseline", "treatment", "post_treatment", "outcome")
_TIMING_ORDER = {name: index for index, name in enumerate(NODE_TIMINGS)}

#: Single source of truth for `CausalDAG.minimum_adjustment_sets`'s search
#: cap. `causal.agents.dag_identification` re-exports this rather than
#: defining its own copy, and `CausalStudySpecification` below relies on
#: `minimum_adjustment_sets`'s *default* picking this up implicitly -- two
#: different literal defaults here and there previously let the
#: identification_estimand gate reject a DAG that the read-only "Causal DAG
#: & identification" tab (which does pass this constant explicitly, via
#: `assess_identification`) reported as already identified.
DEFAULT_MAX_ADJUSTMENT_CANDIDATES = 24

# Evidence-provenance status (CausalEdge.status) answers "how was this edge
# authorized?". EDGE_CERTAINTY answers a different question: "how settled is
# the causal claim itself?". An edge can be evidence_supported (a citation
# was given) while the direction it asserts is still contested in that same
# literature -- these two axes are kept independent on purpose.
EDGE_CERTAINTY = {
    "established",       # consistent, direct evidence for this direction
    "plausible",          # reasonable clinical hypothesis, not strongly evidenced
    "contested",           # the literature disagrees on this relationship or its direction
    "unknown_direction",  # an association is documented but causal direction is unclear
}


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


# Roadmap stage 8 (docs/causal_roadmap.md): three estimators are
# implemented today, all targeting the identical statistical functional
# (the g-formula) via different estimation procedures -- `causal.agents.
# estimation.EstimationAgent` (parametric g-computation, an unpenalized
# logistic outcome model), `causal.agents.tmle.TMLEAgent` (targeted maximum
# likelihood estimation, a SuperLearner-ensembled outcome/treatment model),
# and `causal.agents.cate.CATEAgent` (a causal forest, for the conditional
# rather than population-average functional -- see `StatisticalEstimand.
# population_summary`). These sets are deliberately narrow, not because no
# other functional/model could ever be legitimate, but because declaring
# one that nothing can fit would make `StatisticalEstimand` a promise the
# pipeline cannot keep -- widen them only alongside a real estimator for
# the new value.
IDENTIFYING_FUNCTIONALS = {"g_formula_standardization"}
OUTCOME_MODEL_SPECS = {"logistic_regression_unpenalized", "super_learner_ensemble"}

# Which estimation procedure computes the identifying functional above.
# `plug_in_standardization` is `EstimationAgent`'s parametric g-computation;
# `targeted_minimum_loss_based_estimation` is `TMLEAgent`'s TMLE (see that
# module's docstring for the citation and scope); `causal_forest_dml` is
# `CATEAgent`'s causal forest (Wager & Athey 2018), the only procedure that
# supports `population_summary="CATE"` (see `StatisticalEstimand`'s own
# cross-check). All are valid, reviewed choices for the same
# `identifying_functional`, not a ranked "better" option -- see
# `StatisticalEstimand.estimation_procedure`.
ESTIMATION_PROCEDURES = {
    "plug_in_standardization", "targeted_minimum_loss_based_estimation", "causal_forest_dml",
}

# Only complete-case analysis (listwise deletion of rows with any missing
# adjustment covariate) is implemented; see the same rationale above.
MISSING_DATA_STRATEGIES = {"complete_case_analysis"}

# Unlike the two sets above, all three of these are real, implemented
# choices for the analytic dataset's per-row `censored` column (`causal.
# data_builder.analytic.AnalyticCohortBuilder`): `exclude_censored_before_
# analysis` drops censored rows before fitting and `ignore_censoring_
# administrative_waiver` is an explicit reviewed waiver not to use the
# column at all -- neither corrects for *informative* censoring.
# `ipcw_administrative_censoring` does: a stabilized inverse-probability-of-
# censoring-weighted g-formula (see `causal.agents.estimation.
# EstimationAgent` for the citation and implementation). Multiple imputation
# for missing covariates remains unimplemented (`MISSING_DATA_STRATEGIES`
# above).
CENSORING_STRATEGIES = {
    "exclude_censored_before_analysis",
    "ignore_censoring_administrative_waiver",
    "ipcw_administrative_censoring",
}


@dataclass(frozen=True)
class StatisticalEstimand:
    """The observed-data statistical estimand (`docs/causal_roadmap.md`
    stage 8): the identifying functional and estimation contract that turns
    `CausalEstimand`'s counterfactual contrast into something computable
    from `analytic_dataset.parquet`.

    Deliberately a separate object from `CausalEstimand`, not a few extra
    fields bolted onto it: `CausalEstimand` fixes *what* causal contrast is
    being asked for (population summary, treatment/comparator values,
    horizon) and is meaningful even before any OMOP data exists.
    `StatisticalEstimand` fixes *how* that contrast will actually be
    estimated from the observed, possibly incomplete and administratively
    censored data this pipeline can extract -- the outcome-model
    specification, the exact adjustment set that was reviewed and built into
    `cohort.sql`, and -- unlike before this object existed -- an explicit,
    reviewed missing-data and censoring strategy, rather than a choice left
    implicit in `EstimationAgent`'s code with the analytic cohort's
    `censored` column silently unused by anything.

    `estimation_procedure` (added in schema_version 1.1) answers "how will
    this be estimated" -- `plug_in_standardization` (`EstimationAgent`) or
    `targeted_minimum_loss_based_estimation` (`TMLEAgent`) -- as a distinct
    question from `identifying_functional`, which both procedures currently
    share (`g_formula_standardization`): the functional says what
    statistical quantity identifies the causal contrast, the procedure says
    which reviewed estimator computes it.

    `population_summary` (added in schema_version 1.2) must match
    `study_specification.json`'s `estimand.population_summary` -- closes a
    real gap where a study declaring `"CATE"` at the causal-question stage
    could still silently run plain ATE g-computation, since nothing
    previously re-checked that declaration this late in the pipeline (see
    `causal.orchestrator.workflow._write_statistical_estimand`).
    `"CATE"` additionally requires `estimation_procedure=
    "causal_forest_dml"` (`CATEAgent`, the only estimator that computes
    it) -- `plug_in_standardization`/`targeted_minimum_loss_based_
    estimation` only ever report the ATE.
    """

    schema_version: str
    study_id: str
    identifying_functional: str
    outcome_model: str
    adjustment_set: tuple[str, ...]
    effect_measure: str
    missing_data_strategy: str
    censoring_strategy: str
    censoring_rationale: str
    estimation_procedure: str = "plug_in_standardization"
    population_summary: str = "ATE"

    def __post_init__(self) -> None:
        if self.schema_version not in {"1.0", "1.1", "1.2"}:
            raise ValueError("Statistical estimand schema_version must be 1.0, 1.1, or 1.2")
        if not self.study_id or not self.study_id.replace("_", "").isalnum():
            raise ValueError("study_id must be an alphanumeric snake_case identifier")
        if self.identifying_functional not in IDENTIFYING_FUNCTIONALS:
            raise ValueError(f"Unsupported identifying functional: {self.identifying_functional!r}")
        if self.outcome_model not in OUTCOME_MODEL_SPECS:
            raise ValueError(f"Unsupported outcome model specification: {self.outcome_model!r}")
        if self.estimation_procedure not in ESTIMATION_PROCEDURES:
            raise ValueError(f"Unsupported estimation procedure: {self.estimation_procedure!r}")
        if self.population_summary not in {"ATE", "ATT", "CATE"}:
            raise ValueError("population_summary must be ATE, ATT, or CATE")
        if self.population_summary == "CATE" and self.estimation_procedure != "causal_forest_dml":
            raise ValueError(
                "population_summary='CATE' requires estimation_procedure='causal_forest_dml' -- "
                "plug_in_standardization/targeted_minimum_loss_based_estimation only report the ATE"
            )
        if self.effect_measure not in {
            "risk_difference", "risk_ratio", "odds_ratio", "mean_difference", "survival_difference"
        }:
            raise ValueError("Unsupported causal effect measure")
        if len(set(self.adjustment_set)) != len(self.adjustment_set):
            raise ValueError("adjustment_set must not contain duplicate columns")
        if self.missing_data_strategy not in MISSING_DATA_STRATEGIES:
            raise ValueError(f"Unsupported missing-data strategy: {self.missing_data_strategy!r}")
        if self.censoring_strategy not in CENSORING_STRATEGIES:
            raise ValueError(f"Unsupported censoring strategy: {self.censoring_strategy!r}")
        object.__setattr__(
            self, "censoring_rationale", _require_text(self.censoring_rationale, "censoring_rationale")
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    certainty: str = "established"

    def __post_init__(self) -> None:
        if self.source == self.target:
            raise ValueError("A causal edge cannot be a self-loop")
        if not self.rationale.strip():
            raise ValueError("A causal edge requires a rationale")
        if self.status not in {"proposed", "expert_approved", "evidence_supported"}:
            raise ValueError(f"Unknown causal edge status: {self.status}")
        if self.certainty not in EDGE_CERTAINTY:
            raise ValueError(f"Unknown causal edge certainty: {self.certainty}")


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
    # The active comparator (e.g. "piperacillin-tazobactam" when exposure is
    # "ceftriaxone"): identification and adjustment are computed over the
    # single binary exposure node, but the comparator must still be visible
    # on the DAG itself rather than living only in prose, so a reviewer can
    # see which contrast the graph actually represents. Optional at this
    # generic graph layer (default "") since not every CausalDAG consumer is
    # a target-trial emulation; the clinical bridge that builds a DAG from a
    # HypothesisArtifact always supplies it.
    comparator: str = ""

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
            if node.role in {"mediator", "collider", "selection", "instrument"}:
                # Instruments are excluded alongside mediators/colliders: conditioning on a
                # pure instrument does not block a backdoor path and can amplify bias from
                # any remaining unmeasured confounding (Bhattacharya & Vogt, 2007).
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

    def adjustment_candidate_ids(self) -> tuple[str, ...]:
        """Adjustment-role node ids actually worth searching over: observed,
        baseline-timing, adjustment-eligible roles, minus any pruned as
        provably absent from every backdoor path (see
        `_isolated_non_confounder_candidates`). Shared by
        `minimum_adjustment_sets` and by UI/reporting code that wants to
        show or compare against this count without duplicating the pruning
        logic or paying for the combinatorial search below.
        """
        candidates = tuple(
            node.node_id
            for node in self.nodes
            if node.observed
            and node.timing == "baseline"
            and node.role in {"confounder", "prognostic", "effect_modifier", "other"}
        )
        isolated = self._isolated_non_confounder_candidates(candidates)
        return tuple(node_id for node_id in candidates if node_id not in isolated)

    # B019 warns that caching on `self` can leak instances; here `CausalDAG` is
    # a small frozen value object and maxsize=32 bounds what the class-level
    # cache retains, so the leak is negligible -- see the docstring for why the
    # method is memoized at all.
    @lru_cache(maxsize=32)  # noqa: B019
    def minimum_adjustment_sets(
        self, max_candidates: int = DEFAULT_MAX_ADJUSTMENT_CANDIDATES
    ) -> tuple[tuple[str, ...], ...]:
        """Return minimum-cardinality valid baseline adjustment sets.

        The bounded exhaustive search is intended for reviewable study DAGs,
        not high-dimensional variable-selection graphs. Memoized (`self` is
        hashable: `CausalDAG` and its `nodes`/`edges` are frozen dataclasses
        of hashable fields) because the identification_estimand gate builds
        two value-equal `CausalDAG`s moments apart -- one in `_block_if_
        not_identified`, one inside `CausalStudySpecification.to_dict()` via
        `_write_study_specification_if_hypothesis_exists` -- and both used to
        pay for this same combinatorial search in full while holding
        `WorkflowManager`'s per-run file lock, widening the window for a
        concurrent action on the same run to hit the lock's 10s timeout.
        """
        candidates = self.adjustment_candidate_ids()
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

    def _isolated_non_confounder_candidates(self, candidate_ids: tuple[str, ...]) -> set[str]:
        """Candidates that are provably absent from every backdoor path, and
        so are safe to drop from the combinatorial search below regardless
        of `max_candidates` or how many literature-grounded covariates a DAG
        draft happens to propose.

        A `confounder` always has an edge to *both* exposure and outcome by
        construction (see `causal.agents.dag_identification._ROLE_MAP`), so
        it always sits on at least one backdoor path and is never pruned
        here. Every other adjustment-eligible role (`prognostic`,
        `effect_modifier`, `other`/treatment_cause_only) gets only a single
        fixed edge to one endpoint. Unless a `covariate_relations` edge
        connects it to something else, that single edge makes it a
        degree-one node in the whole graph -- and a degree-one node can
        never be an *interior* vertex of a path connecting two other nodes
        (exposure and outcome), so conditioning on it can never open or
        close a backdoor path. This is checked by total edge degree, not
        role alone, so a node this same role that *does* carry a
        `covariate_relations` edge into the rest of the graph is correctly
        left in the search (its relevance is no longer provable by this
        argument).
        """
        candidate_set = set(candidate_ids)
        degree = {node_id: 0 for node_id in candidate_set}
        for edge in self.edges:
            if edge.source in degree:
                degree[edge.source] += 1
            if edge.target in degree:
                degree[edge.target] += 1
        node_map = self.node_map
        return {
            node_id for node_id in candidate_set
            if node_map[node_id].role != "confounder" and degree[node_id] <= 1
        }

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
        # Computed once and reused for both fields below -- this is the
        # same bounded-but-still-combinatorial search `assess_identification`
        # runs (see `DEFAULT_MAX_ADJUSTMENT_CANDIDATES`), so paying for it
        # twice on every identification_estimand gate approval was a real,
        # avoidable cost, not just a style nit.
        adjustment_sets = self.dag.minimum_adjustment_sets()
        result = asdict(self)
        result["identification"] = {
            "identified_by_backdoor_adjustment": bool(adjustment_sets),
            "minimum_adjustment_sets": adjustment_sets,
            "counterfactual_contrast": self.estimand.counterfactual_contrast,
        }
        return result


def to_dagitty(dag: CausalDAG, adjustment_set: tuple[str, ...] = ()) -> str:
    """Serialize a CausalDAG to dagitty text syntax.

    Paste the output into https://dagitty.net (Model > Import model from
    text). No plotting library or system binary is required for this: it is
    plain text generation, and dagitty.net renders and lays it out in the
    browser. `adjustment_set` (node_ids) is optional and only changes how the
    diagram is annotated; it never mutates `dag`.
    """
    labels = [node.label for node in dag.nodes]
    if len(labels) != len(set(labels)):
        raise ValueError("Cannot export to dagitty: duplicate node labels would collide")
    adjusted = set(adjustment_set)

    by_timing: dict[str, list[CausalNode]] = {timing: [] for timing in NODE_TIMINGS}
    for node in dag.nodes:
        by_timing[node.timing].append(node)

    lines = ["dag {"]
    if dag.comparator:
        exposure_label = dag.node_map[dag.exposure_id].label
        lines.append(f'// active comparator: "{exposure_label}" vs "{dag.comparator}"')
    for timing_index, timing in enumerate(NODE_TIMINGS):
        nodes_at_timing = by_timing[timing]
        count = len(nodes_at_timing)
        for slot, node in enumerate(nodes_at_timing):
            x = timing_index * 2
            y = (slot - (count - 1) / 2) * 1.5
            attrs = []
            if node.node_id == dag.exposure_id:
                attrs.append("exposure")
            if node.node_id == dag.outcome_id:
                attrs.append("outcome")
            if node.node_id in adjusted:
                attrs.append("adjusted")
            if not node.observed:
                attrs.append("latent")
            attrs.append(f'pos="{x},{y:.2f}"')
            lines.append(f"{_dagitty_id(node.label)} [{','.join(attrs)}]")
    for edge in dag.edges:
        source_label = dag.node_map[edge.source].label
        target_label = dag.node_map[edge.target].label
        tag = "evidence" if edge.status == "evidence_supported" else edge.status
        if edge.certainty != "established":
            tag += f"|{edge.certainty}"
        comment = f"  // [{tag}] {edge.rationale}"
        if edge.evidence_ids:
            comment += f" (PMID: {', '.join(edge.evidence_ids)})"
        lines.append(f"{_dagitty_id(source_label)} -> {_dagitty_id(target_label)}{comment}")
    lines.append("}")
    return "\n".join(lines)


def _dagitty_id(label: str) -> str:
    return '"' + label.replace('"', '\\"') + '"'
