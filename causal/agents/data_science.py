"""Deterministic analysis planner for an OMOP target-trial emulation."""

from __future__ import annotations

import csv
import io
from dataclasses import asdict, dataclass, replace
from typing import Any, Protocol

from causal.provenance import artifact_metadata


class ConceptLookup(Protocol):
    def lookup_drug(self, term: str) -> Any: ...
    def lookup_condition(self, term: str) -> Any: ...


@dataclass(frozen=True)
class ConceptCandidate:
    concept_id: int
    concept_name: str
    vocabulary_id: str
    domain_id: str


@dataclass(frozen=True)
class PhenotypePlan:
    name: str
    role: str
    status: str
    mode: str
    omop_tables: tuple[str, ...]
    concept_ids: tuple[int, ...]
    source_value_rule: str | None = None
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


class DataScienceAgent:
    """Turn a normalized hypothesis into a reviewable, non-executing plan."""

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

        phenotypes = (
            self._condition_phenotype("sepsis", "population", concept_sets),
            self._condition_phenotype("septic_shock", "effect_modifier", concept_sets),
            self._drug_phenotype("albumin", "treatment", concept_sets, r"(?i)^Albumin(?:, Human)?\b"),
            self._drug_phenotype(
                "crystalloids", "treatment_and_comparator", concept_sets,
                r"(?i)(Lactated Ringers|Sodium Chloride).*(Bag|Bottle|Injection)$",
            ),
            PhenotypePlan(
                name="all_cause_death", role="outcome", status="standard_table",
                mode="standard", omop_tables=("death",), concept_ids=(),
                approval_required=False,
            ),
            PhenotypePlan(
                name="icu_admission", role="time_zero", status="definition_pending",
                mode="standard", omop_tables=("visit_detail", "visit_occurrence"), concept_ids=(),
            ),
        )

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
            if item.role in {"population", "treatment", "treatment_and_comparator", "time_zero", "outcome"}
        )

        return AnalysisPlanArtifact(
            metadata=artifact_metadata(
                "data_science_agent", "0.1.0", {"hypothesis": hypothesis, "protocol": protocol}
            ),
            schema_version="1.0",
            study_id=study_id,
            status="draft_pending_review",
            data_source=self._required_text(protocol, "data_source"),
            design="active_comparator_target_trial_emulation",
            time_zero=self._required_text(hypothesis, "time_zero"),
            grace_period_hours=self._required_int(hypothesis, "grace_period_hours", 0),
            follow_up_days=self._required_int(hypothesis, "follow_up_days", 1),
            estimands=(self._required_text(hypothesis, "primary_estimand"), *tuple(hypothesis.get("secondary_estimands", ()))),
            phenotypes=phenotypes,
            covariates=covariates,
            required_tables=required_tables,
            feasibility_checks=(
                "eligible ICU stays and unique persons",
                "albumin exposure within grace period",
                "crystalloid exposure within grace period",
                "mutually exclusive treatment assignment",
                "28-day deaths by treatment arm",
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

    @classmethod
    def _attach_concept_candidates(
        cls,
        phenotypes: tuple[PhenotypePlan, ...],
        lookup: ConceptLookup,
    ) -> tuple[PhenotypePlan, ...]:
        terms = {
            "sepsis": ("condition", "sepsis"),
            "septic_shock": ("condition", "septic shock"),
            "albumin": ("drug", "albumin"),
            "crystalloids": ("drug", "sodium chloride"),
        }
        enriched = []
        for phenotype in phenotypes:
            lookup_spec = terms.get(phenotype.name)
            if lookup_spec is None or phenotype.status == "approved":
                enriched.append(phenotype)
                continue
            domain, term = lookup_spec
            response = (
                lookup.lookup_condition(term) if domain == "condition"
                else lookup.lookup_drug(term)
            )
            enriched.append(replace(
                phenotype,
                concept_candidates=cls._parse_concept_candidates(response.text, domain),
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
    def _condition_phenotype(name: str, role: str, concept_sets: dict[str, Any]) -> PhenotypePlan:
        definition = concept_sets.get(name, {})
        ids = tuple(int(value) for value in definition.get("concept_ids", ()))
        return PhenotypePlan(
            name=name, role=role, status=str(definition.get("status", "missing")),
            mode="standard", omop_tables=("condition_occurrence", "concept_ancestor"),
            concept_ids=ids,
        )

    @staticmethod
    def _drug_phenotype(name: str, role: str, concept_sets: dict[str, Any], rule: str) -> PhenotypePlan:
        definition = concept_sets.get(name, {})
        ids = tuple(int(value) for value in definition.get("concept_ids", ()))
        status = str(definition.get("status", "missing"))
        mode = "standard" if status == "approved" and ids else "source_value_provisional"
        return PhenotypePlan(
            name=name, role=role, status=status, mode=mode,
            omop_tables=("drug_exposure",), concept_ids=ids,
            source_value_rule=None if mode == "standard" else rule,
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
