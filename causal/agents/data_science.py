"""Deterministic analysis planner for an OMOP target-trial emulation."""

from __future__ import annotations

import csv
import io
from dataclasses import asdict, dataclass, replace
from typing import Any, Protocol

from causal.provenance import artifact_metadata
from causal.queries import EXPOSURE_CONTRASTS


class ConceptLookup(Protocol):
    def lookup_drug(self, term: str) -> Any: ...
    def lookup_condition(self, term: str) -> Any: ...
    def lookup_concept(self, term: str, domain: str) -> Any: ...


def _parse_optional_float(value: Any) -> float | None:
    """Best-effort float parse for an optional CSV column (e.g. OMCP's
    `name_edit_distance` ranking column) -- `None`/empty/non-numeric all
    mean "not available", not an error, since older OMCP servers won't
    return this column at all."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ConceptCandidate:
    concept_id: int
    concept_name: str
    vocabulary_id: str
    domain_id: str
    match_score: float | None = None


# domain -> OMOP tables a phenotype in that domain is built from. Mirrors
# `causal.queries.feasibility.DOMAIN_TABLES` but at the coarser granularity
# this review-only plan needs (table names, not columns).
_PHENOTYPE_DOMAIN_TABLES: dict[str, tuple[str, ...]] = {
    "condition": ("condition_occurrence", "concept_ancestor"),
    "drug": ("drug_exposure",),
    "procedure": ("procedure_occurrence",),
    "measurement": ("measurement",),
    "observation": ("observation",),
}


@dataclass(frozen=True)
class PhenotypePlan:
    name: str
    role: str
    status: str
    mode: str
    domain: str
    omop_tables: tuple[str, ...]
    concept_ids: tuple[int, ...]
    source_value_rule: str | None = None
    lookup_term: str | None = None
    approval_required: bool = True
    concept_candidates: tuple[ConceptCandidate, ...] = ()


@dataclass(frozen=True)
class CovariatePlan:
    name: str
    causal_role: str
    adjustment_role: str
    temporal_role: str
    window_start_hours: int
    window_end_hours: int
    aggregation: str
    leakage_risk: str
    status: str = "definition_pending"
    concept_candidates: tuple[ConceptCandidate, ...] = ()
    concept_ids: tuple[int, ...] = ()
    mapping_status: str = "unmapped"
    domain: str = ""


@dataclass(frozen=True)
class AnalysisPlanArtifact:
    metadata: dict[str, str]
    schema_version: str
    study_id: str
    status: str
    data_source: str
    design: str
    time_zero: str
    grace_period_hours: int
    follow_up_days: int
    estimands: tuple[str, ...]
    phenotypes: tuple[PhenotypePlan, ...]
    covariates: tuple[CovariatePlan, ...]
    excluded_from_adjustment: tuple[str, ...]
    required_tables: tuple[str, ...]
    feasibility_checks: tuple[str, ...]
    diagnostics: tuple[str, ...]
    unresolved_issues: tuple[str, ...]
    exploratory_execution_allowed: bool
    final_execution_allowed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_REQUIRED_PHENOTYPE_ROLES = (
    "population", "treatment", "comparator", "treatment_and_comparator", "time_zero", "outcome",
)

# Every DAG_VARIABLE_ROLE (causal.agents.hypothesis.DAG_VARIABLE_ROLES) falls
# into exactly one of these two sets -- a candidate_dag entry always becomes
# either a CovariatePlan or an audited `excluded_from_adjustment` entry, never
# silently neither. Adjustment-eligible roles are baseline factors a backdoor
# adjustment set (or a CATE model, for effect_modifier) can legitimately use;
# excluded roles are post-treatment/instrument roles where conditioning would
# bias the estimate rather than help it (see causal.roadmap.CausalDAG.
# assess_adjustment_set's own forbidden-role check, which this mirrors at the
# OMOP-planning layer). Public (not module-private) because
# `causal.mapping.registry` reuses this exact partition to block a
# mediator/collider/selection/instrument from ever being converted into a
# baseline-adjustment `CovariateSpec` -- one shared definition of "which DAG
# roles may never be adjusted for", not a second copy of the same rule.
ADJUSTMENT_ELIGIBLE_ROLES = {"confounder", "effect_modifier", "outcome_cause_only", "treatment_cause_only"}
EXCLUDED_FROM_ADJUSTMENT_ROLES = {"mediator", "collider", "selection", "instrument"}

_DEMOGRAPHIC_COVARIATES = {"age", "sex", "race"}

#: Stand-in for "no real limit, can be measured any time before treatment"
#: (100 years back from index, in hours) -- window bounds are plain
#: hour-integers with no explicit unbounded sentinel, so a low-risk/
#: demographic-like covariate gets this concrete-but-effectively-unbounded
#: value instead. Public (not module-private) so callers displaying a
#: reviewed window (see `causal.ui.streamlit_app`) can recognize this exact
#: value and render it in plain language instead of the raw hour count.
LIFETIME_LOOKBACK_HOURS = -876000


class DataScienceAgent:
    """Turn a normalized hypothesis into a reviewable, non-executing plan.

    Phenotypes are derived generically from `protocol["concept_sets"]`: each
    entry declares its own `role` and `domain` (see `_phenotype_from_concept_set`)
    instead of the agent hardcoding clinical names. This is what lets the same
    agent plan a sepsis/albumin study and an unrelated one without touching
    Python code -- only `concept_sets` changes.

    Covariates are derived the same way, from `hypothesis["candidate_dag"]`
    (the role- and temporal-classified DAG variables the Hypothesis Agent /
    DAG-identification stage already produced) instead of the raw, unreviewed
    `candidate_confounders` list -- see `_covariate` and the module-level role
    partition above. This is what makes a rejected covariate, an unclassified
    covariate, and a mediator/collider/selection/instrument behave
    differently here instead of all becoming an indistinguishable generic
    "definition_pending" plan entry.
    """

    def run(
        self,
        hypothesis: dict[str, Any],
        protocol: dict[str, Any],
        concept_lookup: ConceptLookup | None = None,
        adjustment: dict[str, Any] | None = None,
    ) -> AnalysisPlanArtifact:
        if hypothesis.get("schema_version") != "1.0":
            raise ValueError("Unsupported hypothesis schema_version")
        study_id = self._required_text(hypothesis, "study_id")
        concept_sets = protocol.get("concept_sets")
        if not isinstance(concept_sets, dict):
            raise ValueError("Protocol requires concept_sets")
        exposure_contrast = self._exposure_contrast(protocol)

        phenotypes = tuple(
            self._phenotype_from_concept_set(name, definition, exposure_contrast)
            for name, definition in concept_sets.items()
        )
        declared_roles = {phenotype.role for phenotype in phenotypes}
        # Fallback scaffolding for the two OMOP-standard building blocks that
        # do not need a concept set of their own (all-cause death lives in
        # `death`; a study's time-zero location is a visit-level definition).
        # A protocol that instead defines a non-mortality outcome or a
        # non-visit time-zero via its own concept_sets entry (role="outcome"
        # / role="time_zero") overrides these defaults rather than getting
        # both.
        if "outcome" not in declared_roles:
            phenotypes += (PhenotypePlan(
                name="all_cause_death", role="outcome", status="standard_table",
                mode="standard", domain="standard", omop_tables=("death",), concept_ids=(),
                approval_required=False,
            ),)
        if "time_zero" not in declared_roles:
            phenotypes += (PhenotypePlan(
                name="index_event", role="time_zero", status="standard_table",
                mode="standard", domain="standard",
                omop_tables=("visit_detail", "visit_occurrence"), concept_ids=(),
                approval_required=False,
            ),)

        if concept_lookup is not None:
            phenotypes = self._attach_concept_candidates(phenotypes, concept_lookup)

        dag_variables = tuple(
            variable for variable in hypothesis.get("candidate_dag", ()) if isinstance(variable, dict)
        )
        covariates = tuple(
            self._covariate(variable, adjustment)
            for variable in dag_variables
            if variable.get("role") in ADJUSTMENT_ELIGIBLE_ROLES
        )
        covariates_missing_domain: tuple[str, ...] = ()
        if concept_lookup is not None:
            covariates, covariates_missing_domain = self._attach_covariate_concept_candidates(
                covariates, dag_variables, concept_lookup
            )
        excluded_from_adjustment = self._excluded_from_adjustment(dag_variables, hypothesis)
        unclassified_covariates = self._unclassified_covariates(hypothesis, dag_variables)
        # A confounder that never reaches `status == "ready"` (unmapped,
        # no aggregation reviewed, or altogether unobserved) must block
        # `final_execution_allowed` the same way a missing-domain or
        # unclassified covariate already does -- otherwise a study can run
        # in final mode without ever adjusting for it, silently biasing the
        # estimate instead of surfacing a blocking warning.
        covariates_not_ready = tuple(
            covariate.name for covariate in covariates
            if covariate.causal_role == "confounder" and covariate.status != "ready"
        )
        unresolved = (
            tuple(hypothesis.get("unresolved_issues", ()))
            + tuple(
                f"phenotype:{item.name}:{item.status}"
                for item in phenotypes if item.approval_required and item.status != "approved"
            )
            + tuple(f"covariate:{label}:unclassified_by_dag" for label in unclassified_covariates)
            + tuple(f"covariate:{label}:missing_domain" for label in covariates_missing_domain)
            + tuple(f"covariate:{label}:not_ready" for label in covariates_not_ready)
        )
        required_tables = tuple(sorted({table for phenotype in phenotypes for table in phenotype.omop_tables} | {
            "person", "measurement", "procedure_occurrence", "drug_exposure", "condition_occurrence"
        }))
        source_fallback_available = all(
            item.mode in {"standard", "source_value_provisional"} for item in phenotypes
        )
        # `additional_population_concept_sets` (see `WorkflowManager.
        # add_population_concept_set`) names further, independently-required
        # population concept sets that ANDed onto `population` -- their role
        # ("population_secondary") is not itself in `_REQUIRED_PHENOTYPE_ROLES`
        # since it's not always required (an unlisted, removed entry must not
        # block this), so active ones are checked here explicitly instead.
        active_population_secondary = set(protocol.get("additional_population_concept_sets", ()) or ())
        core_definitions_ready = all(
            item.status not in {"missing", "definition_pending"}
            for item in phenotypes
            if item.role in _REQUIRED_PHENOTYPE_ROLES or item.name in active_population_secondary
        )

        treatment_name = self._phenotype_name_for_role(phenotypes, "treatment", "treatment_and_comparator")
        comparator_name = self._phenotype_name_for_role(phenotypes, "comparator", "treatment_and_comparator")
        follow_up_days = self._required_int(hypothesis, "follow_up_days", 1)

        return AnalysisPlanArtifact(
            metadata=artifact_metadata(
                "data_science_agent", "0.1.0", {"hypothesis": hypothesis, "protocol": protocol}
            ),
            schema_version="1.0",
            study_id=study_id,
            status="draft_pending_review",
            data_source=self._required_text(protocol, "data_source"),
            design=(
                "add_on_target_trial_emulation" if exposure_contrast == "add_on"
                else "active_comparator_target_trial_emulation"
            ),
            time_zero=self._required_text(hypothesis, "time_zero"),
            grace_period_hours=self._required_int(hypothesis, "grace_period_hours", 0),
            follow_up_days=follow_up_days,
            estimands=(self._required_text(hypothesis, "primary_estimand"), *tuple(hypothesis.get("secondary_estimands", ()))),
            phenotypes=phenotypes,
            covariates=covariates,
            excluded_from_adjustment=excluded_from_adjustment,
            required_tables=required_tables,
            feasibility_checks=(
                "eligible population and unique persons",
                f"{treatment_name} exposure within grace period",
                f"{comparator_name} exposure within grace period",
                "mutually exclusive treatment assignment" if exposure_contrast == "mutually_exclusive"
                else "add-on treatment assignment relative to the comparator backbone",
                f"{follow_up_days}-day outcome events by treatment arm",
                "missingness for every baseline covariate",
                "published cohort-size comparison",
            ),
            diagnostics=(
                "cohort attrition", "treatment prevalence", "event rate by arm",
                "propensity overlap", "covariate balance", "extreme weights",
                "immortal-time alignment", "post-treatment leakage",
            ),
            unresolved_issues=tuple(dict.fromkeys(unresolved)),
            exploratory_execution_allowed=source_fallback_available and core_definitions_ready,
            final_execution_allowed=not unresolved,
        )

    @staticmethod
    def _exposure_contrast(protocol: dict[str, Any]) -> str:
        settings = protocol.get("query_settings")
        value = settings.get("exposure_contrast", "add_on") if isinstance(settings, dict) else "add_on"
        if value not in EXPOSURE_CONTRASTS:
            raise ValueError(f"query_settings.exposure_contrast must be one of {EXPOSURE_CONTRASTS}")
        return value

    @staticmethod
    def _phenotype_name_for_role(phenotypes: tuple[PhenotypePlan, ...], *roles: str) -> str:
        for phenotype in phenotypes:
            if phenotype.role in roles:
                return phenotype.name
        return "/".join(roles)

    @classmethod
    def _attach_concept_candidates(
        cls,
        phenotypes: tuple[PhenotypePlan, ...],
        lookup: ConceptLookup,
    ) -> tuple[PhenotypePlan, ...]:
        enriched = []
        for phenotype in phenotypes:
            if phenotype.domain not in _PHENOTYPE_DOMAIN_TABLES or phenotype.status == "approved":
                enriched.append(phenotype)
                continue
            term = phenotype.lookup_term or phenotype.name.replace("_", " ")
            enriched.append(replace(
                phenotype,
                concept_candidates=cls._lookup_candidates(term, phenotype.domain, lookup),
            ))
        return tuple(enriched)

    @classmethod
    def _attach_covariate_concept_candidates(
        cls,
        covariates: tuple[CovariatePlan, ...],
        dag_variables: tuple[dict[str, Any], ...],
        lookup: ConceptLookup,
    ) -> tuple[tuple[CovariatePlan, ...], tuple[str, ...]]:
        """Attach reviewable `concept_candidates` to covariates whose
        `candidate_dag` entry declares a `domain` -- the same review-only
        contract phenotypes get via `_attach_concept_candidates`. A covariate
        whose DAG entry has no domain yet (`DAGVariable.domain`'s default,
        meaning "not yet assessed") is left untouched rather than guessed --
        but its name is returned alongside the enriched covariates so `run()`
        can surface it in `unresolved_issues` (`covariate:{label}:missing_domain`)
        instead of the covariate silently sitting at `definition_pending`
        forever with nothing telling a reviewer it needs a domain assigned.
        """
        domain_by_label = {variable["label"]: variable.get("domain", "") for variable in dag_variables}
        enriched = []
        missing_domain = []
        for covariate in covariates:
            domain = domain_by_label.get(covariate.name, "")
            if not domain:
                enriched.append(covariate)
                missing_domain.append(covariate.name)
                continue
            term = covariate.name.replace("_", " ")
            enriched.append(replace(
                covariate,
                concept_candidates=cls._lookup_candidates(term, domain, lookup),
            ))
        return tuple(enriched), tuple(missing_domain)

    @classmethod
    def _lookup_candidates(cls, term: str, domain: str, lookup: ConceptLookup) -> tuple["ConceptCandidate", ...]:
        """Dispatch a concept-candidate search to the right `ConceptLookup`
        method for `domain`. `condition`/`drug` reuse the two long-standing,
        OMCP-backed methods (`Lookup_Condition`/`Lookup_Drug`, a plain search
        over the OMOP `concept` table already loaded in the CDM); the other
        three domains a phenotype or covariate can declare go through the
        newer, generalized `lookup_concept`. Shared by phenotypes and
        covariates so this dispatch rule is defined exactly once.
        """
        if domain == "condition":
            response = lookup.lookup_condition(term)
        elif domain == "drug":
            response = lookup.lookup_drug(term)
        else:
            response = lookup.lookup_concept(term, domain)
        return cls._parse_concept_candidates(response.text, domain)

    @staticmethod
    def _parse_concept_candidates(text: str, expected_domain: str) -> tuple[ConceptCandidate, ...]:
        candidates = []
        for row in csv.DictReader(io.StringIO(text)):
            try:
                concept_id = int(row.get("concept_id", ""))
            except (TypeError, ValueError):
                continue
            domain_id = str(row.get("domain_id", "")).strip()
            if domain_id.lower() != expected_domain:
                continue
            candidates.append(ConceptCandidate(
                concept_id=concept_id,
                concept_name=str(row.get("concept_name", "")).strip(),
                vocabulary_id=str(row.get("vocabulary_id", "")).strip(),
                domain_id=domain_id,
                match_score=_parse_optional_float(row.get("name_edit_distance")),
            ))
        return tuple(candidates)

    @staticmethod
    def _phenotype_from_concept_set(name: str, definition: Any, exposure_contrast: str) -> PhenotypePlan:
        """Build one reviewable phenotype from a protocol-declared concept set.

        `role` and `domain` are asserted by the protocol author (human or
        LLM draft, same "clinical judgment is explicit and checked" contract
        as `causal.agents.hypothesis`), never guessed from the set's name.
        `source_value_rule` is optional: it exists only for domains (like
        `drug`) where a standard-concept mapping can plausibly be missing and
        a versioned, human-authored regex fallback is meaningful; supplying
        one is what turns `mode` into "source_value_provisional" while the
        set is not yet `approved`. `lookup_term` optionally overrides the
        term sent to `concept_lookup` (see `_attach_concept_candidates`) when
        the set's own name is not a good vocabulary search term (e.g. a
        `crystalloids` class name searched as "sodium chloride" instead).

        `approval_required` is `False` only for a `role="comparator"` entry
        under `exposure_contrast="vs_unexposed"` (e.g. "aspirin vs no
        aspirin", see `causal.queries.feasibility.EXPOSURE_CONTRASTS`) --
        there is no OMOP concept for "absence of a drug", so this role never
        needs approval under that design, regardless of its recorded
        `status`. Every other role/design combination keeps requiring one
        (the dataclass default).
        """
        if not isinstance(definition, dict):
            raise ValueError(f"concept_sets['{name}'] must be an object")
        role = definition.get("role")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"concept_sets['{name}'] requires a role")
        role = role.strip()
        domain = definition.get("domain")
        if domain == "death":
            # Mirrors `causal.orchestrator.workflow._load_concept_specs`'s
            # own `domain='death'` handling on the query-generation side of
            # this same `concept_sets` entry: all-cause mortality lives in
            # the standard OMOP `death` table, not a concept domain, and is
            # only a valid choice for the outcome role.
            if role != "outcome":
                raise ValueError(f"concept_sets['{name}']: domain 'death' is only valid for role='outcome'")
            omop_tables = ("death",)
        elif domain in _PHENOTYPE_DOMAIN_TABLES:
            omop_tables = _PHENOTYPE_DOMAIN_TABLES[domain]
        else:
            raise ValueError(
                f"concept_sets['{name}'] requires domain to be one of "
                f"{tuple(_PHENOTYPE_DOMAIN_TABLES) + ('death',)}"
            )
        source_value_rule = definition.get("source_value_rule")
        if source_value_rule is not None and not isinstance(source_value_rule, str):
            raise ValueError(f"concept_sets['{name}'].source_value_rule must be a string")
        lookup_term = definition.get("lookup_term")
        if lookup_term is not None and not isinstance(lookup_term, str):
            raise ValueError(f"concept_sets['{name}'].lookup_term must be a string")
        ids = tuple(int(value) for value in definition.get("concept_ids", ()))
        status = str(definition.get("status", "missing"))
        mode = "source_value_provisional" if status != "approved" and source_value_rule else "standard"
        approval_required = not (role == "comparator" and exposure_contrast == "vs_unexposed")
        return PhenotypePlan(
            name=name, role=role, status=status, mode=mode, domain=domain,
            omop_tables=omop_tables, concept_ids=ids,
            source_value_rule=source_value_rule if mode == "source_value_provisional" else None,
            lookup_term=lookup_term, approval_required=approval_required,
        )

    @staticmethod
    def _covariate(variable: dict[str, Any], adjustment: dict[str, Any] | None) -> CovariatePlan:
        """Build one reviewable covariate from a role-classified `candidate_dag` entry.

        `role`/`temporal_classification` are asserted by the Hypothesis Agent's
        DAG stage (see `causal.agents.hypothesis.DAGVariable`), never guessed
        here -- same "clinical judgment is explicit and checked" contract as
        `_phenotype_from_concept_set`. A demographic (age/sex/race) keeps its
        long-standing special case (always a `person`-table field, safely
        `"ready"` and low-risk regardless of its DAG timing); every other
        variable's window/leakage-risk now come from `temporal_classification`
        instead of a name-keyword guess: `pre_treatment` variables can be
        measured any time before treatment (wide window, low leakage risk,
        like a demographic), `at_treatment` variables are only defined at the
        treatment-decision instant itself (narrow window, review required --
        the classic immortal-time/leakage zone). `adjustment` (see
        `causal.agents.dag_identification.AdjustmentSetSummary.to_dict()`) is
        optional and only annotates `confounder`-role variables with whether
        they are core/alternative/unused for the DAG's minimum backdoor
        adjustment set; it never changes whether a variable becomes a
        covariate at all, since even a redundant confounder can still be
        worth mapping (see `AdjustmentSetSummary`'s own docstring on why
        "unused" is not the same as "wrong").
        """
        label = str(variable["label"]).strip()
        role = str(variable["role"])
        observed = bool(variable.get("observed", True))
        demographic = label.lower() in _DEMOGRAPHIC_COVARIATES
        low_risk_window = demographic or variable.get("temporal_classification") == "pre_treatment"
        adjustment_role = ""
        if role == "confounder" and adjustment:
            if label in adjustment.get("core_variable_labels", ()):
                adjustment_role = "core"
            elif label in adjustment.get("alternative_variable_labels", ()):
                adjustment_role = "alternative"
            elif label in adjustment.get("unused_confounder_labels", ()):
                adjustment_role = "unused"
        domain = str(variable.get("domain", ""))
        mapping_status = str(variable.get("mapping_status", "unmapped"))
        aggregation = (
            "value_at_index" if demographic else str(variable.get("aggregation", "definition_pending"))
        )
        # A non-demographic covariate is only extraction-ready once its OMOP
        # concept(s) are approved *and* a human has reviewed how to combine
        # repeated occurrences (`causal.agents.hypothesis.DAGVariable.
        # aggregation`) -- knowing "which rows" is not the same as knowing
        # "how to combine them". `domain` must also be a known extraction
        # domain, since `causal.data_builder.analytic.AnalyticCohortBuilder`
        # needs one to generate SQL at all.
        extraction_ready = (
            mapping_status == "approved"
            and domain in _PHENOTYPE_DOMAIN_TABLES
            and aggregation not in ("", "definition_pending")
        )
        status = "unobserved" if not observed else (
            "ready" if (demographic or extraction_ready) else "definition_pending"
        )
        # A reviewer-set window (`causal.agents.hypothesis.DAGVariable.
        # window_start_hours`/`window_end_hours`, see `review_covariate_window`)
        # overrides the auto-derived one; `None` (the default) means no
        # override was ever recorded.
        window_start_hours = variable.get("window_start_hours")
        window_end_hours = variable.get("window_end_hours")
        if window_start_hours is None or window_end_hours is None:
            window_start_hours = LIFETIME_LOOKBACK_HOURS if low_risk_window else -24
            window_end_hours = 0
        return CovariatePlan(
            name=label,
            causal_role=role,
            adjustment_role=adjustment_role,
            temporal_role="baseline",
            window_start_hours=int(window_start_hours),
            window_end_hours=int(window_end_hours),
            aggregation=aggregation,
            leakage_risk="low" if low_risk_window else "review_required",
            status=status,
            domain=domain,
            concept_ids=tuple(int(value) for value in variable.get("concept_ids", ())),
            mapping_status=mapping_status,
        )

    @staticmethod
    def _excluded_from_adjustment(
        dag_variables: tuple[dict[str, Any], ...], hypothesis: dict[str, Any]
    ) -> tuple[str, ...]:
        """Audit trail for every `candidate_dag` variable/first-pass covariate
        that does not become a `CovariatePlan`, so exclusion is always
        recorded, never silent (same convention as `dag_tournament.
        MergeConflict`).
        """
        role_excluded = tuple(
            f"{variable['label']}: role={variable['role']} (excluded from covariate adjustment; "
            f"conditioning on a {variable['role']} can bias the causal estimate)"
            for variable in dag_variables
            if variable.get("role") in EXCLUDED_FROM_ADJUSTMENT_ROLES
        )
        rejected_excluded = tuple(
            f"{rejection['label']}: rejected by {rejection.get('reviewer', 'unknown reviewer')}"
            + (f" ({rejection['comment']})" if rejection.get("comment") else "")
            for rejection in hypothesis.get("rejected_covariates", ())
            if isinstance(rejection, dict)
        )
        return role_excluded + rejected_excluded

    @staticmethod
    def _unclassified_covariates(
        hypothesis: dict[str, Any], dag_variables: tuple[dict[str, Any], ...]
    ) -> tuple[str, ...]:
        """First-pass covariates with no `candidate_dag` role and no recorded
        rejection: neither approved into the plan nor audited as excluded,
        so they block `final_execution_allowed` via `unresolved_issues`
        instead of silently defaulting to a generic covariate plan.
        """
        classified_labels = {variable.get("label") for variable in dag_variables}
        rejected_labels = {
            rejection.get("label")
            for rejection in hypothesis.get("rejected_covariates", ())
            if isinstance(rejection, dict)
        }
        return tuple(
            label for label in hypothesis.get("candidate_confounders", ())
            if label not in classified_labels and label not in rejected_labels
        )

    @staticmethod
    def _required_text(value: dict[str, Any], field: str) -> str:
        result = value.get(field)
        if not isinstance(result, str) or not result.strip():
            raise ValueError(f"Missing {field}")
        return result.strip()

    @staticmethod
    def _required_int(value: dict[str, Any], field: str, minimum: int) -> int:
        result = value.get(field)
        if not isinstance(result, int) or isinstance(result, bool) or result < minimum:
            raise ValueError(f"{field} must be an integer >= {minimum}")
        return result
