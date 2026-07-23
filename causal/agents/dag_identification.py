"""Bridge from the Hypothesis Agent's literature-grounded DAG to roadmap.py.

`causal.roadmap` is deliberately independent from OMOP and from any LLM: it
only answers a graph-theoretic question (does this DAG identify the effect by
backdoor adjustment?). This module builds a `roadmap.CausalDAG` purely from
clinical-language nodes — the PubMed-grounded `candidate_dag` produced by the
Hypothesis Agent — so identification can be checked *before* any OMOP concept
mapping is attempted, not after.

Every variable that enters the graph carries an explicit causal role
(confounder / treatment_cause_only / outcome_cause_only / mediator /
collider / effect_modifier / selection / instrument) assigned by the
Hypothesis Agent's DAG stage — never inferred here.
A variable is never wired to both exposure and outcome just because it was
proposed as a "covariate"; that was the previous behavior and it silently
mislabeled mediators and outcome-only prognostic factors as confounders,
which can open or close the wrong paths. `candidate_confounders` (the
ungrounded first-pass list) is descriptive only: any label in it that never
received a role via `candidate_dag` is exposed as `HypothesisArtifact.
unresolved_covariates` (an "unresolved" status, not a silent exclusion) and
never guessed into the graph. If that label was also flagged
`major_confounders`, leaving it unresolved is a blocking issue surfaced by
`assess_clinical_completeness`, not merely an advisory note.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from causal.agents.hypothesis import DAGVariable, HypothesisArtifact
from causal.roadmap import CausalDAG, CausalEdge, CausalNode


_SLUG_INVALID = re.compile(r"[^a-z0-9]+")
_EXPOSURE_ID = "exposure"
_OUTCOME_ID = "outcome"

# role -> (roadmap node role, node timing, edge directions relative to the variable)
# "to_exposure" / "to_outcome" mean an edge FROM the variable; "from_exposure" /
# "from_outcome" mean an edge INTO the variable.
_ROLE_MAP: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "confounder": ("confounder", "baseline", ("to_exposure", "to_outcome")),
    "treatment_cause_only": ("other", "baseline", ("to_exposure",)),
    "outcome_cause_only": ("prognostic", "baseline", ("to_outcome",)),
    "mediator": ("mediator", "post_treatment", ("from_exposure", "to_outcome")),
    # timing="outcome" (not "post_treatment"): roadmap.CausalDAG requires the
    # outcome node to have the terminal timing and forbids any edge whose
    # source outranks its target, so an edge *from* the outcome node is only
    # representable into another node that shares that same terminal timing.
    "collider": ("collider", "outcome", ("from_exposure", "from_outcome")),
    "effect_modifier": ("effect_modifier", "baseline", ("to_outcome",)),
    # Selection variables are collider-like (jointly caused by treatment and
    # outcome, e.g. study inclusion or being alive/observed at follow-up) but
    # kept as a distinct role so warnings and exports can name the mechanism.
    "selection": ("selection", "outcome", ("from_exposure", "from_outcome")),
    # An instrument affects treatment only, with no direct or backdoor path to
    # the outcome other than through treatment. It is deliberately excluded
    # from roadmap.CausalDAG.minimum_adjustment_sets candidates: conditioning
    # on an instrument does not close a backdoor path and can amplify bias
    # from any remaining unmeasured confounding.
    "instrument": ("instrument", "baseline", ("to_exposure",)),
}


@dataclass(frozen=True)
class DAGIdentificationResult:
    """Purely graph-theoretic: does the declared DAG identify the effect by backdoor adjustment?

    This never asks whether the DAG is clinically *right* — only whether,
    taken at face value, its structure satisfies the backdoor criterion. A
    DAG missing every real confounder is trivially "identified" (there is no
    backdoor path left to block); `technical_notes` flags that specific trap,
    but distinguishing "identified" from "clinically complete" is exactly why
    this is kept separate from `ClinicalValidationResult`.
    """

    dag: CausalDAG
    confounder_count: int
    identified_by_backdoor_adjustment: bool
    minimum_adjustment_sets: tuple[tuple[str, ...], ...]
    technical_notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dag": self.dag.to_dict(),
            "confounder_count": self.confounder_count,
            "identified_by_backdoor_adjustment": self.identified_by_backdoor_adjustment,
            "minimum_adjustment_sets": self.minimum_adjustment_sets,
            "technical_notes": self.technical_notes,
        }


@dataclass(frozen=True)
class ClinicalValidationResult:
    """Clinical judgment concerns: is this DAG complete and trustworthy enough to approve?

    Deliberately independent of graph theory: a DAG can be perfectly
    identified by backdoor adjustment (`DAGIdentificationResult`) while still
    failing clinical validation because a known major confounder was never
    resolved, or a role assignment rests on contested evidence.
    """

    blocking_issues: tuple[str, ...]
    advisory_notes: tuple[str, ...]

    @property
    def approved_for_dag_gate(self) -> bool:
        return not self.blocking_issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocking_issues": self.blocking_issues,
            "advisory_notes": self.advisory_notes,
            "approved_for_dag_gate": self.approved_for_dag_gate,
        }


@dataclass(frozen=True)
class DAGReviewResult:
    """Bundles the two independent assessments a human reviews before approving a DAG."""

    identification: DAGIdentificationResult
    clinical: ClinicalValidationResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "identification": self.identification.to_dict(),
            "clinical": self.clinical.to_dict(),
        }


def build_causal_dag(hypothesis: HypothesisArtifact) -> CausalDAG:
    """Build a clinical-language DAG from the role-classified variables the
    Hypothesis Agent proposed.

    No OMOP concept ever enters this graph; node labels are the free-text
    clinical terms already validated by `HypothesisAgent._candidate_dag`.
    Each `candidate_dag` entry carries an explicit causal role (see
    `causal.agents.hypothesis.DAG_VARIABLE_ROLES`) that determines its
    roadmap role, timing, and which edges it gets — nothing here re-derives
    or guesses that role.
    """
    nodes = [
        CausalNode(_EXPOSURE_ID, hypothesis.treatment, "exposure", "treatment", observed=True),
        CausalNode(_OUTCOME_ID, hypothesis.outcome, "outcome", "outcome", observed=True),
    ]
    label_to_id: dict[str, str] = {}
    edges = [
        CausalEdge(
            source=_EXPOSURE_ID,
            target=_OUTCOME_ID,
            rationale=f"The causal hypothesis under study: does {hypothesis.treatment!r} "
            f"affect {hypothesis.outcome!r}?",
            status="proposed",
        )
    ]

    for variable in hypothesis.candidate_dag:
        node_role, timing, directions = _ROLE_MAP[variable.role]
        node_id = _variable_node_id(variable.label, label_to_id)
        nodes.append(CausalNode(node_id, variable.label, node_role, timing, observed=variable.observed))
        status = "evidence_supported" if variable.pmid else "proposed"
        evidence_ids = (variable.pmid,) if variable.pmid else ()
        for direction in directions:
            source, target = _direction_to_edge(direction, node_id)
            edges.append(
                CausalEdge(
                    source=source,
                    target=target,
                    rationale=variable.rationale,
                    evidence_ids=evidence_ids,
                    status=status,
                    certainty=variable.certainty,
                )
            )

    post_treatment_labels = {
        variable.label
        for variable in hypothesis.candidate_dag
        if variable.role in ("mediator", "collider", "selection")
    }
    for relation in hypothesis.covariate_relations:
        if relation.source in post_treatment_labels or relation.target in post_treatment_labels:
            raise ValueError(
                f"covariate_relations cannot reference a mediator/collider/selection label: "
                f"{relation.source!r} -> {relation.target!r}"
            )
        status = "evidence_supported" if relation.pmid else "proposed"
        evidence_ids = (relation.pmid,) if relation.pmid else ()
        edges.append(
            CausalEdge(
                source=label_to_id[relation.source],
                target=label_to_id[relation.target],
                rationale=relation.rationale,
                evidence_ids=evidence_ids,
                status=status,
                certainty=relation.certainty,
            )
        )

    return CausalDAG(tuple(nodes), tuple(edges), _EXPOSURE_ID, _OUTCOME_ID, comparator=hypothesis.comparator)


def assess_identification(hypothesis: HypothesisArtifact, max_candidates: int = 24) -> DAGIdentificationResult:
    """Answer only the graph-theoretic question: does this DAG, as declared, identify the effect?

    Contains no clinical judgment: it does not know or care whether a first-
    pass covariate was never classified, whether a role rests on contested
    evidence, or whether a known major confounder is missing. Those are
    `assess_clinical_completeness`'s job — an empty `candidate_dag` is
    trivially "identified" here (there is no backdoor path left to block),
    and `technical_notes` exists specifically so that fact is never read as
    "no confounding was found."
    """
    dag = build_causal_dag(hypothesis)
    confounder_count = sum(1 for node in dag.nodes if node.role == "confounder")
    notes = []
    try:
        minimum_sets = dag.minimum_adjustment_sets(max_candidates=max_candidates)
    except ValueError as exc:
        minimum_sets = ()
        notes.append(
            f"Backdoor identification could not be verified (not merely 'not identified'): {exc}"
        )
    if not confounder_count:
        notes.append(
            "No confounders were proposed at all; backdoor identification "
            "is trivially satisfied and must not be read as confirmation of no confounding."
        )
    return DAGIdentificationResult(
        dag=dag,
        confounder_count=confounder_count,
        identified_by_backdoor_adjustment=bool(minimum_sets),
        minimum_adjustment_sets=minimum_sets,
        technical_notes=tuple(notes),
    )


def assess_clinical_completeness(hypothesis: HypothesisArtifact) -> ClinicalValidationResult:
    """Answer the clinical question: is this DAG complete and trustworthy enough to approve?

    Independent of `assess_identification` on purpose (requirement: graph-
    theoretic identification and clinical validation are separate checks).
    `blocking_issues` is what a DAG-approval gate must consult — a DAG can be
    backdoor-identified and still be rejected here because a major
    confounder (age, illness severity, septic shock, ...) was never resolved.
    """
    blocking: list[str] = []
    advisory: list[str] = []

    classified_labels = {variable.label for variable in hypothesis.candidate_dag}
    rejected_by_label = {rejection.label: rejection for rejection in hypothesis.rejected_covariates}
    major = set(hypothesis.major_confounders)
    for label in hypothesis.unresolved_covariates:
        rejection = rejected_by_label.get(label)
        # unresolved_covariates already excludes rejected labels (see
        # HypothesisAgent.run); this branch only exists as a defensive check
        # in case a caller hand-builds a HypothesisArtifact inconsistently.
        if rejection is not None:  # pragma: no cover - defensive
            advisory.append(
                f"first-pass covariate '{label}' was reviewed and excluded by "
                f"{rejection.reviewer}" + (f": {rejection.comment}" if rejection.comment else "")
            )
            continue
        message = (
            f"first-pass covariate '{label}' has no assigned causal role and is "
            "unresolved: it is excluded from the DAG until it is classified or "
            "explicitly rejected with a reviewer and rationale"
        )
        if label in major:
            blocking.append(
                f"major confounder '{label}' is unresolved and must be classified or explicitly "
                "rejected (with reviewer and rationale) before this DAG can be approved"
            )
        else:
            advisory.append(message)

    for rejection in hypothesis.rejected_covariates:
        advisory.append(
            f"first-pass covariate '{rejection.label}' was reviewed and excluded by "
            f"{rejection.reviewer}" + (f": {rejection.comment}" if rejection.comment else "")
        )

    dag = build_causal_dag(hypothesis)
    disputed_edges = [edge for edge in dag.edges if edge.certainty in {"contested", "unknown_direction"}]
    if disputed_edges:
        advisory.append(
            "Identification results assume a fixed direction for "
            + ", ".join(
                f"{dag.node_map[edge.source].label} -> {dag.node_map[edge.target].label} "
                f"({edge.certainty})"
                for edge in disputed_edges
            )
            + "; this direction is not well-established and should be reviewed before "
            "trusting the adjustment set it produces."
        )

    return ClinicalValidationResult(blocking_issues=tuple(blocking), advisory_notes=tuple(advisory))


def review_dag(hypothesis: HypothesisArtifact, max_candidates: int = 24) -> DAGReviewResult:
    """Convenience entry point bundling the graph-theoretic and clinical checks for review/UI use."""
    return DAGReviewResult(
        identification=assess_identification(hypothesis, max_candidates=max_candidates),
        clinical=assess_clinical_completeness(hypothesis),
    )


def _direction_to_edge(direction: str, node_id: str) -> tuple[str, str]:
    if direction == "to_exposure":
        return node_id, _EXPOSURE_ID
    if direction == "to_outcome":
        return node_id, _OUTCOME_ID
    if direction == "from_exposure":
        return _EXPOSURE_ID, node_id
    if direction == "from_outcome":
        return _OUTCOME_ID, node_id
    raise ValueError(f"Unknown edge direction: {direction}")  # pragma: no cover - internal invariant


def _variable_node_id(label: str, seen: dict[str, str]) -> str:
    if label in seen:
        return seen[label]
    slug = _SLUG_INVALID.sub("_", label.strip().lower()).strip("_") or "variable"
    candidate = f"variable_{slug}"
    existing = set(seen.values())
    suffix = 1
    while candidate in existing:
        suffix += 1
        candidate = f"variable_{slug}_{suffix}"
    seen[label] = candidate
    return candidate
