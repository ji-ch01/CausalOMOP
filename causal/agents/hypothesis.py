"""Deterministic hypothesis agent producing a normalized causal protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from causal.provenance import artifact_metadata
from causal.roadmap import EDGE_CERTAINTY


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

DAG_VARIABLE_ROLES = (
    "confounder",            # common cause of treatment and outcome
    "treatment_cause_only",  # influences treatment choice, not the outcome directly
    "outcome_cause_only",    # prognostic factor, does not influence treatment choice
    "mediator",               # consequence of treatment, on the pathway to outcome
    "collider",                # caused by both treatment and outcome (or their proxies)
    "effect_modifier",       # baseline factor that changes the size/direction of the treatment effect
    "selection",              # caused by treatment and outcome (or their proxies) via study inclusion/follow-up
    "instrument",             # affects treatment only, with no plausible direct path to the outcome
)

DAG_VARIABLE_TEMPORAL_CLASSES = (
    "pre_treatment",   # determined before the treatment decision; can plausibly cause treatment choice
    "at_treatment",     # determined at/around treatment initiation itself (e.g. the choice between comparators)
    "post_treatment",  # only arises, or is only measured, after treatment has started
)

# A causal role makes an implicit temporal claim (e.g. a confounder must precede
# treatment to cause it). `temporal_classification` is asserted independently by
# the classifier so this claim can be checked, not merely assumed: a variable
# that is clinically a post-treatment consequence (e.g. a hospital-acquired
# infection) cannot be smuggled in as a `confounder` just by picking that role.
ROLE_ALLOWED_TEMPORAL_CLASSES: dict[str, tuple[str, ...]] = {
    "confounder": ("pre_treatment", "at_treatment"),
    "treatment_cause_only": ("pre_treatment", "at_treatment"),
    "outcome_cause_only": ("pre_treatment", "at_treatment"),
    "effect_modifier": ("pre_treatment", "at_treatment"),
    "instrument": ("pre_treatment", "at_treatment"),
    "mediator": ("post_treatment",),
    "collider": ("post_treatment",),
    "selection": ("post_treatment",),
}


@dataclass(frozen=True)
class DAGVariable:
    """One candidate DAG variable with an explicit causal role. Always provisional.

    `pmid` is optional: "" means the role was asserted without a literature
    citation (still requires a `rationale`), a non-empty PMID/OpenAlex id
    means the role is literature-grounded. `observed` marks whether the
    variable is expected to be measurable at all (False = latent/unmeasured).
    `certainty` (see `causal.roadmap.EDGE_CERTAINTY`) is orthogonal to `pmid`:
    a cited role can still be `contested` or `unknown_direction` if the
    literature itself disagrees on the relationship.

    `temporal_classification` (see `DAG_VARIABLE_TEMPORAL_CLASSES`) is a
    second, independent assertion about when the variable is determined
    relative to treatment. It is required, not derived from `role`, so a
    contradiction between the two (e.g. `role="confounder"` on a variable
    the classifier also calls `post_treatment`) is a structural error caught
    at validation time rather than a silent adjustment-set leak.
    """

    label: str
    role: str
    rationale: str
    temporal_classification: str
    pmid: str = ""
    observed: bool = True
    certainty: str = "established"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CovariateRelation:
    """One causal edge between two already-declared covariates (not touching exposure/outcome)."""

    source: str
    target: str
    rationale: str
    pmid: str = ""
    certainty: str = "established"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RejectedCovariate:
    """A first-pass covariate a human reviewer explicitly excluded from the DAG.

    Distinct from simply never assigning a role: this is an audited decision
    (reviewer + rationale + timestamp), so it stops being reported as
    "pending classification" and instead as a reviewed exclusion.
    """

    label: str
    reviewer: str
    comment: str = ""
    decided_at: str = ""

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
    candidate_dag: tuple[DAGVariable, ...]
    covariate_relations: tuple[CovariateRelation, ...]
    rejected_covariates: tuple[RejectedCovariate, ...]
    major_confounders: tuple[str, ...]
    unresolved_covariates: tuple[str, ...]
    unresolved_major_confounders: tuple[str, ...]
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
        dag_variables = self._candidate_dag(protocol.get("candidate_dag", ()))
        known_labels = {variable.label for variable in dag_variables}
        covariate_relations = self._covariate_relations(protocol.get("covariate_relations", ()), known_labels)
        rejected_covariates = self._rejected_covariates(protocol.get("rejected_covariates", ()))
        major_confounders = self._major_confounders(protocol.get("major_confounders", ()), set(covariates))

        classified_labels = known_labels
        rejected_labels = {rejection.label for rejection in rejected_covariates}
        unresolved_covariates = tuple(
            label for label in covariates if label not in classified_labels and label not in rejected_labels
        )
        unresolved_major_confounders = tuple(
            label for label in unresolved_covariates if label in major_confounders
        )

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
            candidate_dag=dag_variables,
            covariate_relations=covariate_relations,
            rejected_covariates=rejected_covariates,
            major_confounders=major_confounders,
            unresolved_covariates=unresolved_covariates,
            unresolved_major_confounders=unresolved_major_confounders,
            design_constraints=constraints,
            unresolved_issues=unresolved,
            # A major confounder left unresolved (neither classified nor an
            # audited rejection) blocks execution the same way an unmapped
            # concept set does: it is a known threat to validity, not merely
            # an open item.
            executable=not unresolved and not unresolved_major_confounders,
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
    def _candidate_dag(value: Any) -> tuple[DAGVariable, ...]:
        """Structurally validate DAG variables regardless of who produced protocol.json.

        Every variable must carry an explicit causal `role` (see
        DAG_VARIABLE_ROLES) and a `rationale`: a variable never enters the
        graph just by being named. `pmid` is optional — "" marks an
        expert-asserted, ungrounded role; a non-empty id marks a
        literature-grounded one. This does not re-verify that a pmid is real
        or was actually retrieved — that grounding check happens where the
        retrieval result is known, in the LLM bridge. This is a safety net
        for any protocol (LLM-drafted or hand-authored) reaching the agent.
        """
        if not isinstance(value, (list, tuple)):
            return ()
        variables = []
        seen_labels: set[str] = set()
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("Each candidate_dag entry must be an object")
            label = item.get("label")
            role = item.get("role")
            rationale = item.get("rationale")
            temporal_classification = item.get("temporal_classification")
            pmid = item.get("pmid", "")
            observed = item.get("observed", True)
            certainty = item.get("certainty", "established")
            if not isinstance(label, str) or not label.strip():
                raise ValueError("candidate_dag entry requires a non-empty label")
            if role not in DAG_VARIABLE_ROLES:
                raise ValueError(f"candidate_dag role must be one of {DAG_VARIABLE_ROLES}")
            if not isinstance(rationale, str) or not rationale.strip():
                raise ValueError("candidate_dag entry requires a non-empty rationale")
            if temporal_classification not in DAG_VARIABLE_TEMPORAL_CLASSES:
                raise ValueError(
                    f"candidate_dag entry '{label}' requires temporal_classification to be one of "
                    f"{DAG_VARIABLE_TEMPORAL_CLASSES}"
                )
            allowed_temporal = ROLE_ALLOWED_TEMPORAL_CLASSES[role]
            if temporal_classification not in allowed_temporal:
                raise ValueError(
                    f"candidate_dag entry '{label}' has role '{role}', which requires "
                    f"temporal_classification to be one of {allowed_temporal}, not "
                    f"'{temporal_classification}'. A post-treatment variable cannot be asserted "
                    "as a role that implies it precedes treatment, and vice versa."
                )
            if not isinstance(pmid, str):
                raise ValueError("candidate_dag pmid must be a string")
            if not isinstance(observed, bool):
                raise ValueError("candidate_dag observed must be a boolean")
            if certainty not in EDGE_CERTAINTY:
                raise ValueError(f"candidate_dag certainty must be one of {EDGE_CERTAINTY}")
            label = label.strip()
            if label in seen_labels:
                raise ValueError(f"candidate_dag defines '{label}' more than once")
            seen_labels.add(label)
            variables.append(
                DAGVariable(
                    label, role, rationale.strip(), temporal_classification, pmid.strip(), observed, certainty
                )
            )
        return tuple(variables)

    @staticmethod
    def _covariate_relations(value: Any, known_labels: set[str]) -> tuple[CovariateRelation, ...]:
        """Structurally validate covariate-to-covariate edges.

        Both endpoints must already be declared in candidate_dag: a relation
        cannot silently introduce a variable that was never assigned a role.
        """
        if not isinstance(value, (list, tuple)):
            return ()
        relations = []
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("Each covariate_relations entry must be an object")
            source = item.get("source")
            target = item.get("target")
            rationale = item.get("rationale")
            pmid = item.get("pmid", "")
            certainty = item.get("certainty", "established")
            if not isinstance(source, str) or not source.strip():
                raise ValueError("covariate_relations entry requires a non-empty source")
            if not isinstance(target, str) or not target.strip():
                raise ValueError("covariate_relations entry requires a non-empty target")
            if not isinstance(rationale, str) or not rationale.strip():
                raise ValueError("covariate_relations entry requires a non-empty rationale")
            if not isinstance(pmid, str):
                raise ValueError("covariate_relations pmid must be a string")
            if certainty not in EDGE_CERTAINTY:
                raise ValueError(f"covariate_relations certainty must be one of {EDGE_CERTAINTY}")
            source, target = source.strip(), target.strip()
            if source == target:
                raise ValueError("covariate_relations entry cannot relate a variable to itself")
            if source not in known_labels:
                raise ValueError(f"covariate_relations source '{source}' is not a declared candidate_dag label")
            if target not in known_labels:
                raise ValueError(f"covariate_relations target '{target}' is not a declared candidate_dag label")
            relations.append(CovariateRelation(source, target, rationale.strip(), pmid.strip(), certainty))
        return tuple(relations)

    @staticmethod
    def _rejected_covariates(value: Any) -> tuple[RejectedCovariate, ...]:
        """Structurally validate human decisions to exclude a first-pass covariate.

        Requires a reviewer identity (audit trail), same expectation as
        ApprovalGate.decide. `comment`/`decided_at` are optional free text.
        """
        if not isinstance(value, (list, tuple)):
            return ()
        rejections = []
        seen_labels: set[str] = set()
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("Each rejected_covariates entry must be an object")
            label = item.get("label")
            reviewer = item.get("reviewer")
            comment = item.get("comment", "")
            decided_at = item.get("decided_at", "")
            if not isinstance(label, str) or not label.strip():
                raise ValueError("rejected_covariates entry requires a non-empty label")
            if not isinstance(reviewer, str) or not reviewer.strip():
                raise ValueError("rejected_covariates entry requires a non-empty reviewer")
            if not isinstance(comment, str):
                raise ValueError("rejected_covariates comment must be a string")
            if not isinstance(decided_at, str):
                raise ValueError("rejected_covariates decided_at must be a string")
            label = label.strip()
            if label in seen_labels:
                raise ValueError(f"rejected_covariates defines '{label}' more than once")
            seen_labels.add(label)
            rejections.append(RejectedCovariate(label, reviewer.strip(), comment.strip(), decided_at.strip()))
        return tuple(rejections)

    @staticmethod
    def _major_confounders(value: Any, known_covariates: set[str]) -> tuple[str, ...]:
        """Structurally validate the protocol author's explicit "must not be dropped silently" flag.

        This is never inferred from a label (e.g. by keyword-matching "age" or
        "severity"): the codebase's convention is that clinical judgment is
        asserted explicitly and checked, not guessed. Each entry must already
        be one of the first-pass covariates, so this cannot be used to sneak
        an unreviewed label into the pipeline.
        """
        if not isinstance(value, (list, tuple)):
            return ()
        majors = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("major_confounders entries must be non-empty strings")
            label = item.strip()
            if label not in known_covariates:
                raise ValueError(
                    f"major_confounders entry '{label}' is not a declared first_pass_covariates label"
                )
            if label not in seen:
                seen.add(label)
                majors.append(label)
        return tuple(majors)

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
