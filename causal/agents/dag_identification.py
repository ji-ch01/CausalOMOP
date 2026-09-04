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

from causal.agents.hypothesis import HypothesisArtifact
from causal.roadmap import (
    DEFAULT_MAX_ADJUSTMENT_CANDIDATES,
    CausalDAG,
    CausalEdge,
    CausalEstimand,
    CausalNode,
    CausalQuestion,
    CausalStudySpecification,
    TargetTrialProtocol,
)


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
class EvidenceSourceSummary:
    """How candidate_dag variables and covariate_relations were grounded.

    Every accepted item falls into exactly one bucket: `uncited` (role
    asserted on rationale alone, `pmid == ""`), `literature_only` (a PubMed/
    OpenAlex citation, no independent KG confirmation), `kg_only` (an
    OptimusKG edge id was the *only* evidence -- no retrieved abstract
    discussed it), or `corroborated` (a literature citation *and* an
    independent `kg_evidence_id` -- two unrelated sources agree, the
    strongest evidence tier this pipeline can produce). This is a coverage
    metric, not a quality one: it says nothing about whether the cited
    evidence is itself strong (see `causal.literature.orchestrator`'s
    evidence-tier grading for that), only where each edge's grounding came
    from.
    """

    total_items: int
    literature_only: int
    kg_only: int
    corroborated: int
    uncited: int

    @property
    def cited_fraction(self) -> float:
        """Share of items grounded by at least one source (any of the three cited buckets)."""
        return (self.total_items - self.uncited) / self.total_items if self.total_items else 0.0

    @property
    def corroborated_fraction(self) -> float:
        return self.corroborated / self.total_items if self.total_items else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_items": self.total_items,
            "literature_only": self.literature_only,
            "kg_only": self.kg_only,
            "corroborated": self.corroborated,
            "uncited": self.uncited,
            "cited_fraction": self.cited_fraction,
            "corroborated_fraction": self.corroborated_fraction,
        }


def evidence_summary(hypothesis: HypothesisArtifact) -> EvidenceSourceSummary:
    """Summarize evidence-source coverage across `candidate_dag` and `covariate_relations`.

    Classification relies only on `pmid`/`kg_evidence_id`, following the same
    convention used everywhere else in this module: a knowledge-graph-only
    citation lives in `pmid` itself (prefixed `optimuskg:`, see
    `causal.knowledge_graph.KGEdge.citation_id`), while `kg_evidence_id`
    marks a *second*, independent KG confirmation of an already
    literature-cited item (see `DAGVariable`'s docstring). An item can never
    be `kg_only` and `corroborated` at once because a KG-only variable never
    populates `kg_evidence_id` (`causal.agents.user_question._kg_reinforcement`
    short-circuits on an `optimuskg:`-prefixed `pmid`).
    """
    pairs: list[tuple[str, str]] = [
        (variable.pmid, variable.kg_evidence_id) for variable in hypothesis.candidate_dag
    ] + [
        (relation.pmid, relation.kg_evidence_id) for relation in hypothesis.covariate_relations
    ]
    literature_only = kg_only = corroborated = uncited = 0
    for pmid, kg_evidence_id in pairs:
        if not pmid:
            uncited += 1
        elif kg_evidence_id:
            corroborated += 1
        elif pmid.startswith("optimuskg:"):
            kg_only += 1
        else:
            literature_only += 1
    return EvidenceSourceSummary(
        total_items=len(pairs),
        literature_only=literature_only,
        kg_only=kg_only,
        corroborated=corroborated,
        uncited=uncited,
    )


@dataclass(frozen=True)
class AdjustmentSetSummary:
    """Cross-set structure of `DAGIdentificationResult.minimum_adjustment_sets`.

    A single set size/count is easy to read off `DAGIdentificationResult`
    directly, but a reviewer choosing which variables actually matter needs
    to know how the *sets* relate to each other, not just how many there
    are: `core_variable_labels` are in every minimum set (dropping one
    breaks every option this search found -- there is no way to identify
    the effect without it), `alternative_variable_labels` appear in some but
    not all (interchangeable with each other for backdoor-blocking purposes
    alone -- a reviewer may still prefer one on clinical or measurement
    grounds), and `unused_confounder_labels` are declared confounders that
    never appear in *any* minimum set. That last one is not necessarily a
    bug -- a confounder whose only backdoor path is already blocked by
    another selected variable is genuinely redundant for identification --
    but it is exactly the kind of thing `DAGIdentificationResult.
    technical_notes` cannot flag (it only tracks failures/edge cases in
    computing the sets, not this kind of within-DAG redundancy), so it is
    worth a reviewer's attention: it can also mean a role was misassigned.
    """

    set_count: int
    set_size: int
    core_variable_labels: tuple[str, ...]
    alternative_variable_labels: tuple[str, ...]
    unused_confounder_labels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "set_count": self.set_count,
            "set_size": self.set_size,
            "core_variable_labels": self.core_variable_labels,
            "alternative_variable_labels": self.alternative_variable_labels,
            "unused_confounder_labels": self.unused_confounder_labels,
        }


def adjustment_set_summary(identification: DAGIdentificationResult) -> AdjustmentSetSummary:
    """Derive core/alternative/unused variable structure from `identification.minimum_adjustment_sets`.

    All label lookups go through `identification.dag.node_map`, so this
    reads only the already-computed `DAGIdentificationResult` -- it never
    recomputes adjustment sets itself and is safe to call unconditionally,
    including when the DAG is not identified (`set_count=0`).
    """
    node_map = identification.dag.node_map
    confounder_ids = {node.node_id for node in identification.dag.nodes if node.role == "confounder"}
    sets = identification.minimum_adjustment_sets
    if not sets:
        return AdjustmentSetSummary(
            set_count=0,
            set_size=0,
            core_variable_labels=(),
            alternative_variable_labels=(),
            unused_confounder_labels=tuple(sorted(node_map[node_id].label for node_id in confounder_ids)),
        )
    node_id_sets = [set(adjustment_set) for adjustment_set in sets]
    core = set.intersection(*node_id_sets)
    union = set.union(*node_id_sets)
    alternative = union - core
    unused_confounders = confounder_ids - union
    return AdjustmentSetSummary(
        set_count=len(sets),
        set_size=len(sets[0]),
        core_variable_labels=tuple(sorted(node_map[node_id].label for node_id in core)),
        alternative_variable_labels=tuple(sorted(node_map[node_id].label for node_id in alternative)),
        unused_confounder_labels=tuple(sorted(node_map[node_id].label for node_id in unused_confounders)),
    )


def grounding_caveat(evidence: EvidenceSourceSummary) -> str:
    """A fixed disclaimer meant to be shown next to `identification`/`clinical`
    every time they are, so "identified"/"approved for DAG gate" is never
    misread as "clinically validated".

    Nothing in this pipeline checks a `candidate_dag` against a published or
    expert-drawn reference DAG for the question actually being asked. The
    artifacts that look like that, the `reference_dag.json` files under
    `studies/`, are hand-built one-off comparison targets for specific
    validation studies (see `docs/dag_generation_audit.md`) -- not a
    mechanism this pipeline runs automatically for an arbitrary user
    question. `identified_by_
    backdoor_adjustment` and `approved_for_dag_gate` mean "no known
    structural or clinical blocking issue was found", never "a domain
    expert or a published causal diagram confirmed this structure" --
    conflating the two is the exact failure mode this caveat exists to head
    off. `EvidenceSourceSummary`'s citation coverage is the closest thing
    this pipeline has to a grounding signal, and even that is explicitly a
    *coverage* metric, not a *correctness* one (see its own docstring):
    it says where a claim came from, never whether the claim is right for
    this population.
    """
    if not evidence.total_items:
        return (
            "No published or expert reference DAG was checked against this graph. "
            "'Identified' / 'approved for DAG gate' mean no known blocking issue was found -- "
            "not that this DAG has been clinically validated. Human covariate review remains "
            "required before treating it as more than a first-pass draft."
        )
    return (
        "No published or expert reference DAG was checked against this graph. "
        "'Identified' / 'approved for DAG gate' mean no known blocking issue was found -- not "
        "that a domain expert or a published causal diagram has validated this structure. "
        f"{evidence.cited_fraction:.0%} of items carry at least one citation "
        f"({evidence.corroborated_fraction:.0%} corroborated by two independent sources) -- "
        "treat that as evidence *coverage*, not a correctness guarantee. Human covariate review "
        "remains required before treating this DAG as more than a first-pass draft."
    )


@dataclass(frozen=True)
class DAGReviewResult:
    """Bundles the assessments a human reviews before approving a DAG.

    `identification`/`clinical` are the two graph-theoretic/clinical
    verdicts documented on their own classes above; `evidence`/`adjustment`
    are descriptive metrics computed from those results and never feed back
    into either verdict -- they exist to make the review surface (CLI/UI)
    legible at a glance, not to gate approval themselves. `caveat` is a
    fixed disclaimer derived from `evidence` (see `grounding_caveat`) meant
    to be surfaced alongside `identification`/`clinical` everywhere they are,
    so a passing verdict is never mistaken for clinical validation.
    """

    identification: DAGIdentificationResult
    clinical: ClinicalValidationResult
    evidence: EvidenceSourceSummary
    adjustment: AdjustmentSetSummary
    caveat: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "identification": self.identification.to_dict(),
            "clinical": self.clinical.to_dict(),
            "evidence": self.evidence.to_dict(),
            "adjustment": self.adjustment.to_dict(),
            "caveat": self.caveat,
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
        evidence_ids = tuple(
            evidence_id for evidence_id in (variable.pmid, variable.kg_evidence_id) if evidence_id
        )
        status = "evidence_supported" if evidence_ids else "proposed"
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
        evidence_ids = tuple(
            evidence_id for evidence_id in (relation.pmid, relation.kg_evidence_id) if evidence_id
        )
        status = "evidence_supported" if evidence_ids else "proposed"
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


def build_study_specification(protocol: dict[str, Any], hypothesis: HypothesisArtifact) -> CausalStudySpecification:
    """Bridge `protocol.json` + the Hypothesis Agent's output into the
    typed, validated `roadmap.CausalStudySpecification` contract -- the
    counterfactual-estimand roadmap stage (see `docs/causal_roadmap.md`
    stage 3), wired for real instead of merely defined.

    `CausalQuestion`/most of `TargetTrialProtocol` are already fully
    determined by `hypothesis` (itself already validated by
    `HypothesisAgent`). The remaining `TargetTrialProtocol` fields
    (`eligibility_criteria`, `assignment_procedure`, `censoring_strategy`,
    `analysis_strategy`) and all of `CausalEstimand` except
    `population_summary`/`outcome`/`horizon_days` (`effect_measure`,
    `treatment_value`, `comparator_value`) are read directly from
    `protocol["target_trial"]`, which `HypothesisArtifact` never carries.

    Deliberately does no validation of its own: every field is handed to the
    dataclasses exactly as found (missing text defaults to `""`, missing
    `eligibility_criteria` to `()`), so `CausalQuestion.__post_init__`/
    `TargetTrialProtocol.__post_init__`/`CausalEstimand.__post_init__`/
    `CausalStudySpecification.__post_init__` raise the one, already-correct
    `ValueError` for whatever is missing or invalid -- including the
    horizon/follow-up consistency check. This function only decides *where*
    each field comes from, never *whether* it is acceptable.
    """
    target = protocol.get("target_trial")
    target = target if isinstance(target, dict) else {}

    question = CausalQuestion(
        population=hypothesis.population,
        treatment=hypothesis.treatment,
        comparator=hypothesis.comparator,
        outcome=hypothesis.outcome,
    )
    target_trial = TargetTrialProtocol(
        eligibility_criteria=tuple(str(item) for item in target.get("eligibility_criteria") or ()),
        treatment_strategy=hypothesis.treatment,
        comparator_strategy=hypothesis.comparator,
        assignment_procedure=str(target.get("assignment_procedure") or ""),
        time_zero=hypothesis.time_zero,
        grace_period_hours=hypothesis.grace_period_hours,
        follow_up_days=hypothesis.follow_up_days,
        outcome_definition=hypothesis.outcome,
        censoring_strategy=str(target.get("censoring_strategy") or ""),
        analysis_strategy=str(target.get("analysis_strategy") or ""),
    )
    estimand = CausalEstimand(
        population_summary=hypothesis.primary_estimand,
        effect_measure=str(target.get("effect_measure") or ""),
        treatment_value=str(target.get("treatment_value") or ""),
        comparator_value=str(target.get("comparator_value") or ""),
        outcome=hypothesis.outcome,
        horizon_days=hypothesis.follow_up_days,
    )
    dag = build_causal_dag(hypothesis)
    return CausalStudySpecification(
        schema_version="1.0",
        study_id=hypothesis.study_id,
        source_question=hypothesis.source_question,
        question=question,
        target_trial=target_trial,
        estimand=estimand,
        dag=dag,
    )


# `DEFAULT_MAX_ADJUSTMENT_CANDIDATES` above is imported, not redefined here:
# it is `CausalDAG.minimum_adjustment_sets`'s own default in `causal.roadmap`
# (the actual source of truth), so every caller that omits `max_candidates`
# -- including `CausalStudySpecification`, used when the
# identification_estimand gate writes `study_specification.json` -- agrees
# with this module's `assess_identification`/`review_dag` on what "too many
# candidates" means. Two independent literal defaults here and there
# previously let the gate reject a DAG that `assess_identification` reported
# as already identified.

#: Soft, UI-only guardrail on `confounder`-role variables specifically, not
#: enforced by any gate. Unlike `prognostic`/`effect_modifier`/`other`
#: candidates, a `confounder` always has edges to *both* exposure and
#: outcome (see `_ROLE_MAP`) and so can never be pruned by
#: `CausalDAG._isolated_non_confounder_candidates` -- confounder count is
#: therefore what actually drives the combinatorial search's runtime.
#: Empirically, 18 confounders (all individually necessary, i.e. the
#: minimum valid adjustment set was the full set) took ~90s to verify;
#: kept well below `DEFAULT_MAX_ADJUSTMENT_CANDIDATES` for headroom.
RECOMMENDED_MAX_CONFOUNDERS = 15


def assess_identification(
    hypothesis: HypothesisArtifact, max_candidates: int = DEFAULT_MAX_ADJUSTMENT_CANDIDATES
) -> DAGIdentificationResult:
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
    confounder_count = sum(1 for node in dag.nodes if node.role == "confounder")
    if confounder_count == 0 and hypothesis.candidate_confounders and hypothesis.unresolved_covariates:
        # Deliberately advisory, not blocking: `major_confounders` above is the
        # only sanctioned way to force a block, so a reviewer is never
        # ambushed by an automatically-inferred one (see
        # `causal.agents.hypothesis._major_confounders`). But an empty
        # confounder set trivially satisfies backdoor identification (no path
        # left to block), which silently reads as "no confounding" unless a
        # reviewer notices -- this note exists so that reading is never made
        # by omission, without deciding for the reviewer that it must block.
        advisory.append(
            "no candidate_dag variable has been classified as a confounder, and "
            f"{len(hypothesis.unresolved_covariates)} first-pass covariate(s) remain unresolved "
            "(" + ", ".join(f"'{label}'" for label in hypothesis.unresolved_covariates) + "); "
            "backdoor identification is only trivially satisfied here because no confounder was "
            "ever declared, not because none exists. Review whether any of these should be "
            "classified as a confounder before approving -- add it to major_confounders if it "
            "must block approval until resolved."
        )
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


def review_dag(
    hypothesis: HypothesisArtifact, max_candidates: int = DEFAULT_MAX_ADJUSTMENT_CANDIDATES
) -> DAGReviewResult:
    """Convenience entry point bundling the graph-theoretic, clinical, and descriptive-metric checks for review/UI use."""
    identification = assess_identification(hypothesis, max_candidates=max_candidates)
    evidence = evidence_summary(hypothesis)
    return DAGReviewResult(
        identification=identification,
        clinical=assess_clinical_completeness(hypothesis),
        evidence=evidence,
        adjustment=adjustment_set_summary(identification),
        caveat=grounding_caveat(evidence),
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
