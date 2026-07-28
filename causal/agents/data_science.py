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


@dataclass(frozen=True)
class ConceptCandidate:
    concept_id: int
    concept_name: str
    vocabulary_id: str
    domain_id: str


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
    temporal_role: str
    window_start_hours: int
    window_end_hours: int
    aggregation: str
    leakage_risk: str
    status: str = "definition_pending"


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


class DataScienceAgent:
    """Turn a normalized hypothesis into a reviewable, non-executing plan.

    Phenotypes are derived generically from `protocol["concept_sets"]`: each
    entry declares its own `role` and `domain` (see `_phenotype_from_concept_set`)
    instead of the agent hardcoding clinical names. This is what lets the same
    agent plan a sepsis/albumin study and an unrelated one without touching
    Python code -- only `concept_sets` changes.
    """

    def run(
        self,
        hypothesis: dict[str, Any],
        protocol: dict[str, Any],
        concept_lookup: ConceptLookup | None = None,
    ) -> AnalysisPlanArtifact:
        if hypothesis.get("schema_version") != "1.0":
            raise ValueError("Unsupported hypothesis schema_version")
        study_id = self._required_text(hypothesis, "study_id")
        concept_sets = protocol.get("concept_sets")
        if not isinstance(concept_sets, dict):
            raise ValueError("Protocol requires concept_sets")

        phenotypes = tuple(
            self._phenotype_from_concept_set(name, definition)
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
                name="index_event", role="time_zero", status="definition_pending",
                mode="standard", domain="standard",
                omop_tables=("visit_detail", "visit_occurrence"), concept_ids=(),
            ),)

        if concept_lookup is not None:
            phenotypes = self._attach_concept_candidates(phenotypes, concept_lookup)

        covariates = tuple(self._covariate(name) for name in hypothesis.get("candidate_confounders", ()))
        unresolved = tuple(hypothesis.get("unresolved_issues", ())) + tuple(
            f"phenotype:{item.name}:{item.status}" for item in phenotypes if item.approval_required and item.status != "approved"
        )
        required_tables = tuple(sorted({table for phenotype in phenotypes for table in phenotype.omop_tables} | {
            "person", "measurement", "procedure_occurrence", "drug_exposure", "condition_occurrence"
        }))
        source_fallback_available = all(
            item.mode in {"standard", "source_value_provisional"} for item in phenotypes
        )
        core_definitions_ready = all(
            item.status not in {"missing", "definition_pending"}
            for item in phenotypes
            if item.role in _REQUIRED_PHENOTYPE_ROLES
        )

        exposure_contrast = self._exposure_contrast(protocol)
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
            if phenotype.domain not in {"condition", "drug"} or phenotype.status == "approved":
                enriched.append(phenotype)
                continue
            term = phenotype.lookup_term or phenotype.name.replace("_", " ")
            response = (
                lookup.lookup_condition(term) if phenotype.domain == "condition"
                else lookup.lookup_drug(term)
            )
            enriched.append(replace(
                phenotype,
                concept_candidates=cls._parse_concept_candidates(response.text, phenotype.domain),
            ))
        return tuple(enriched)

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
            ))
        return tuple(candidates)

    @staticmethod
    def _phenotype_from_concept_set(name: str, definition: Any) -> PhenotypePlan:
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
        """
        if not isinstance(definition, dict):
            raise ValueError(f"concept_sets['{name}'] must be an object")
        role = definition.get("role")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"concept_sets['{name}'] requires a role")
        domain = definition.get("domain")
        if domain not in _PHENOTYPE_DOMAIN_TABLES:
            raise ValueError(
                f"concept_sets['{name}'] requires domain to be one of {tuple(_PHENOTYPE_DOMAIN_TABLES)}"
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
        return PhenotypePlan(
            name=name, role=role.strip(), status=status, mode=mode, domain=domain,
            omop_tables=_PHENOTYPE_DOMAIN_TABLES[domain], concept_ids=ids,
            source_value_rule=source_value_rule if mode == "source_value_provisional" else None,
            lookup_term=lookup_term,
        )

    @staticmethod
    def _covariate(name: str) -> CovariatePlan:
        normalized = str(name).strip()
        demographic = normalized.lower() in {"age", "sex", "race"}
        return CovariatePlan(
            name=normalized,
            temporal_role="baseline",
            window_start_hours=-24 if not demographic else -876000,
            window_end_hours=0,
            aggregation="value_at_index" if demographic else "definition_pending",
            leakage_risk="low" if demographic else "review_required",
            status="ready" if demographic else "definition_pending",
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
