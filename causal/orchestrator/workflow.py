"""Filesystem-backed workflow manager with explicit human approval gates."""

from __future__ import annotations

import json
import csv
import hashlib
import io
import os
import re
import shutil
import time
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from causal.orchestrator.state import ApprovalGate, NEXT_PHASE, RunState
import pandas as pd

from causal.agents import (
    DataScienceAgent,
    DiagnosticsAgent,
    EstimationAgent,
    ExecutionAgent,
    HypothesisAgent,
    QueryValidationAgent,
    TMLEAgent,
    analysis_plan_blockers,
)
from causal.agents.dag_identification import assess_identification, build_study_specification, review_dag
from causal.agents.estimation_procedure_suggestion import (
    DEFAULT_ESTIMATION_PROCEDURE_SUGGESTION_PROMPT_PATH,
    build_estimation_procedure_suggestion_prompt,
    deterministic_cate_suggestion,
    parse_estimation_procedure_suggestion_response,
)
from causal.agents.dag_point_revision import (
    DEFAULT_DAG_POINT_REVISION_PROMPT_PATH,
    apply_dag_point_operations,
    build_dag_point_revision_prompt,
    format_current_dag,
    parse_dag_point_revision_response,
)
from causal.agents.hypothesis import (
    DAG_VARIABLE_DOMAINS,
    DAG_VARIABLE_ROLES,
    DAG_VARIABLE_TEMPORAL_CLASSES,
)
from causal.agents.narrative_display import build_narrative_display, display_or_dash
from causal.agents.results_faithfulness import verify_narrative
from causal.agents.results_narrative import (
    DEFAULT_NARRATIVE_PROMPT_PATH,
    NARRATIVE_BOUNDARY_STATEMENT,
    NarrativeInput,
    NarrativeResult,
    build_narrative_prompt,
    narrative_result_from_payload,
    normalize_narrative_artifact,
    parse_narrative_response,
)
from causal.agents.user_question import GeneratedHypothesis
from causal.companion.companion import (
    DEFAULT_DESIGN_DIALOGUE_PROMPT_PATH,
    DEFAULT_ESTIMAND_DIALOGUE_PROMPT_PATH,
    DEFAULT_RESULTS_DIALOGUE_PROMPT_PATH,
    CompanionMessage,
    CompanionTurn,
    DesignDialogueContext,
    EstimandDialogueContext,
    ResultsCompanionTurn,
    build_design_dialogue_prompt,
    build_estimand_dialogue_prompt,
    build_results_dialogue_prompt,
    parse_companion_turn,
    parse_results_companion_turn,
)
from causal.companion.decision_memory import DecisionMemory, distill_decision_memory
from causal.companion.reviewer_memory import (
    ReviewerMemory,
    merge_run_into_memory,
    slugify_reviewer,
)
from causal.llm.base import LLMCallable
from causal.bigquery import BigQueryOMOPGateway
from causal.config import BigQueryDatasets
from causal.provenance import artifact_metadata
from causal.sensitivity import e_value
from causal.roadmap import (
    CENSORING_STRATEGIES,
    ESTIMATION_PROCEDURES,
    MISSING_DATA_STRATEGIES,
    CausalQuestion,
    StatisticalEstimand,
    TargetTrialProtocol,
)
from causal.data_builder import (
    AnalyticCohortBuilder,
    parse_cohort_csv_to_dataframe,
    write_analytic_dataset,
)
from causal.data_builder.analytic import AnalyticColumn
from causal.mapping.audit import (
    approved_concepts_missing_from_data_sql,
    concept_mapping_coverage_sql,
    concept_row_and_person_counts_sql,
    incidence_rate_sql,
    index_event_breakdown_sql,
    invalid_or_nonstandard_concepts_sql,
    observed_units_sql,
    orphan_concepts_sql,
)
from causal.mapping.registry import (
    MAPPING_STATUS_APPROVED,
    VALUE_DOMAINS,
    OMOPVariableDefinition,
    OMOPVariableRegistry,
    build_extraction_plan,
    extraction_plan_to_concept_set_spec,
    extraction_plan_to_covariate_spec,
    fixed_domain_to_concept_set_spec,
    is_fixed_domain,
    is_sql_executable,
    variable_from_covariate_plan,
)
from causal.mcp_client import OMCPClient
from causal.queries import (
    COVARIATE_AGGREGATIONS_BY_DOMAIN,
    DOMAIN_TABLES,
    EXPOSURE_CONTRASTS,
    OUTCOME_DOMAINS,
    ConceptSetSpec,
    CovariateSpec,
    feasibility_sql,
)
from causal.queries.negative_controls import (
    DEFAULT_NEGATIVE_CONTROL_PROMPT_PATH,
    build_negative_control_prompt,
    parse_negative_control_response,
    substitute_spec,
    variable_from_negative_control_proposal,
)


QUERY_ROLES = ("population", "treatment", "comparator", "outcome")

REQUIRED_GATES = {
    "target_trial_review": "target_trial",
    "dag_review": "dag_review",
    "identification_review": "identification_estimand",
    "concept_mapping": "clinical_mapping",
    "analysis_plan": "analysis_plan",
    "estimand_review": "statistical_estimand",
    "results_review": "results",
}

_ALL_GATE_NAMES = (
    "target_trial", "dag_review", "identification_estimand", "clinical_mapping", "analysis_plan",
    "statistical_estimand", "results",
)

# Runs created before `protocol_review` was split into target_trial_review /
# dag_review / identification_review (docs/causal_roadmap.md's governance
# split) have a `state.json` with a single "protocol" gate and a
# "protocol_review" phase instead of the three names above. Loading one of
# those runs under the current schema would otherwise KeyError the first
# time anything indexes `state.gates["target_trial"]` etc.
_LEGACY_PROTOCOL_GATE = "protocol"
_LEGACY_PROTOCOL_PHASE = "protocol_review"
_LEGACY_PROTOCOL_SPLIT_GATES = ("target_trial", "dag_review", "identification_estimand")


def _migrate_legacy_protocol_gate(state: RunState) -> None:
    legacy_gate = state.gates.pop(_LEGACY_PROTOCOL_GATE, None)
    for name in _LEGACY_PROTOCOL_SPLIT_GATES:
        if name in state.gates:
            continue
        if legacy_gate is not None:
            state.gates[name] = ApprovalGate(
                name,
                status=legacy_gate.status,
                reviewer=legacy_gate.reviewer,
                comment=legacy_gate.comment,
                decided_at=legacy_gate.decided_at,
            )
        else:
            state.gates[name] = ApprovalGate(name)
    if state.phase == _LEGACY_PROTOCOL_PHASE:
        state.phase = "target_trial_review"


def _migrate_missing_gates(state: RunState) -> None:
    """Runs created before a gate name was added to `_ALL_GATE_NAMES` (most
    recently `statistical_estimand`, for the `estimand_review` phase) have a
    `state.json` whose `gates` dict simply lacks that key -- `RunState.
    from_dict` never invents gates it was not told about. Left alone, the
    first `decide_gate(..., "statistical_estimand", ...)` call on such a run
    would raise "Unknown gate" even though the run is otherwise perfectly
    healthy. Defaulting the missing gate(s) to `pending` is exactly what a
    freshly `start()`-ed run already gets, so this only backfills, never
    resets, an existing decision.
    """
    for name in _ALL_GATE_NAMES:
        if name not in state.gates:
            state.gates[name] = ApprovalGate(name)

# How long a stale `.review.lock` file (left behind by a crashed/killed
# process -- this is advisory, not OS-enforced) is trusted before a new
# acquisition attempt steals it. Every locked critical section here is a
# handful of small JSON read/writes -- milliseconds -- so anything still
# holding the lock this long is not a legitimate in-progress review.
_LOCK_STALE_SECONDS = 30.0


class _RunLock:
    """Advisory, filesystem-based lock over one run's on-disk state, so two
    callers acting on the same run at the same time get a clear, bounded
    wait or a `TimeoutError` instead of the last writer silently discarding
    the other's change. Not a database transaction, just enough to turn a
    silent race into a loud one; `os.O_CREAT | os.O_EXCL` is atomic on both
    Windows and POSIX, so no external dependency is needed for this.

    Two distinct things share this one lock per run directory:
    `protocol.json`/`hypothesis.json` (read-modify-write `candidate_dag` --
    `review_covariate`, `map_covariate_concepts`,
    `review_covariate_aggregation`, etc.) and `state.json`
    (`decide_gate`, `advance`, `run_data_science_agent`,
    `run_execution_readiness`, `build_cohort_plan` -- all load-mutate-save
    `RunState`). Not reentrant: a caller that already holds this lock must
    never acquire it again on the same run_dir (deadlock, since acquisition
    just spin-polls `__enter__` until its own timeout). `build_cohort_plan`
    calls `run_execution_readiness` for this reason -- as a plain call that
    fully acquires and releases its own lock, never nested inside
    `build_cohort_plan`'s.
    """

    def __init__(self, run_dir: Path, timeout_seconds: float = 10.0, poll_seconds: float = 0.05):
        self._lock_path = run_dir / ".review.lock"
        self._timeout_seconds = timeout_seconds
        self._poll_seconds = poll_seconds

    def __enter__(self) -> "_RunLock":
        deadline = time.monotonic() + self._timeout_seconds
        while True:
            try:
                fd = os.open(str(self._lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return self
            except FileExistsError:
                self._steal_if_stale()
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Could not acquire the review lock for this run within "
                        f"{self._timeout_seconds}s -- another review is in progress"
                    ) from None
                time.sleep(self._poll_seconds)

    def _steal_if_stale(self) -> None:
        try:
            age_seconds = time.time() - self._lock_path.stat().st_mtime
        except FileNotFoundError:
            return
        if age_seconds > _LOCK_STALE_SECONDS:
            self._lock_path.unlink(missing_ok=True)

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._lock_path.unlink(missing_ok=True)


# Mirrors `omcp.governance.CausalExecutionContext._ID` exactly: `study_id` is
# later embedded verbatim in both `run_id` (see `_make_run_id`, checked by
# `WorkflowManager._state_path`) and `mcp_context.json`'s own `study_id`
# field, which OMCP independently re-validates against that same pattern
# before running any governed causal query. Validating it here, before any
# run directory or file is written, turns a charset mistake into one clear
# error at protocol-submission time instead of a `ValueError("Invalid
# run_id")` raised mid-`start()` (after `protocol.json` was already copied
# to an on-disk run directory with no `state.json`) or, worse, an opaque
# "Governed causal query rejected" failure deep inside the OMCP subprocess.
_STUDY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,127}$")

# `causal.roadmap.CENSORING_STRATEGIES` values whose `EstimationAgent` fit
# actually reads the per-row `censored` column (to filter rows, or to fit a
# censoring model from it) -- these must not be approved for the
# `statistical_estimand` gate while `censored` is still the always-NULL
# default (see `_load_censoring_definition_approved`); only
# `ignore_censoring_administrative_waiver` can be approved without it.
_CENSORING_STRATEGIES_REQUIRING_APPROVED_COLUMN = frozenset(
    {"exclude_censored_before_analysis", "ipcw_administrative_censoring"}
)


def _validate_study_id(study_id: str) -> str:
    study_id = study_id.strip()
    if not _STUDY_ID_RE.fullmatch(study_id):
        raise ValueError(
            f"Invalid study_id {study_id!r}: must start with a letter or digit and contain "
            "only letters, digits, '_' or '-' (3-128 characters) -- required by the shared "
            "run_id/OMCP causal-context contract"
        )
    return study_id


def _make_run_id(study_id: str) -> str:
    """Build a chronologically sortable, question-identifiable run id.

    Format: {UTC timestamp}_{study_id}-{short random hash}, e.g.
    20260723-143012_ppi_vs_h2ra_sup_mortality-f00fefd1
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}_{study_id}-{uuid.uuid4().hex[:8]}"


def _atomic_write_json(path: Path, data: Any) -> None:
    """Serialize `data` to `path` as indented JSON through a `.tmp` sidecar
    and an atomic `replace`, so a concurrent reader holding no `_RunLock`
    (e.g. the Streamlit UI polling a run) never observes a half-written
    file. Every persisted run artifact is written this way.
    """
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


#: Cap on how many free-text correction points one
#: `revise_candidate_dag_from_points` call accepts -- keeps the formatter
#: prompt bounded and matches how a reviewer actually works (a handful of
#: points, reviewed, then another pass if needed).
_MAX_DAG_REVISION_POINTS = 20

#: Fields of a targeted study-design correction worth replaying to a later
#: full regeneration (`UserQuestionHypothesisService.generate`'s
#: `prior_corrections`), so the same class of mistake is discouraged on a
#: fresh draft. Also an audit trail in `protocol.json`.
_MAX_CORRECTION_LOG_ENTRIES = 50


def _append_correction_log(
    protocol: dict[str, Any],
    *,
    kind: str,
    reviewer: str,
    comment: str,
    summary: str,
    detail: dict[str, Any] | None = None,
) -> None:
    """Append one targeted-correction record to `protocol["correction_log"]`
    (created if absent, capped at `_MAX_CORRECTION_LOG_ENTRIES`, oldest
    dropped first). Mutates `protocol` in place; the caller persists it.
    """
    log = protocol.get("correction_log")
    if not isinstance(log, list):
        log = []
    log.append(
        {
            "at": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "reviewer": reviewer,
            "comment": comment,
            "summary": summary,
            "detail": detail or {},
        }
    )
    protocol["correction_log"] = log[-_MAX_CORRECTION_LOG_ENTRIES:]


# treatment/comparator each have exactly one fallback role: `treatment_and_
# comparator`, a single shared-backbone phenotype used by both arms of an
# `add_on` design (e.g. a backbone therapy given to everyone, the treatment
# arm additionally getting an adjunct layered on top: "backbone" alone the
# comparator, "backbone" plus "adjunct" the treatment).
# population/outcome have no fallback role.
_ROLE_FALLBACK_ROLES: dict[str, tuple[str, ...]] = {
    "treatment": ("treatment_and_comparator",),
    "comparator": ("treatment_and_comparator",),
}


def _load_concept_specs(protocol: dict) -> dict[str, Any]:
    """Resolve the human-approved, per-question concept sets a cohort/
    feasibility query is built from -- population/treatment/comparator/
    outcome -- from `OMOPVariableRegistry.from_protocol(protocol)`, by role
    (`OMOPVariableRegistry.resolve_role`).

    A thin wrapper: the registry is the single source of truth for "which
    OMOP variable does this role resolve to" (see `causal.mapping.registry`
    module docstring); this function only adds the two things a registry
    entry alone cannot know -- which fallback role each query role accepts
    (`_ROLE_FALLBACK_ROLES`, protocol-shape domain knowledge, not generic
    registry logic) and the unmapped-outcome-defaults-to-death placeholder
    (`OMOPVariableRegistry.from_protocol`/`from_dict` never synthesizes a
    default entry that was not actually declared -- see that classmethod's
    docstring on rebuilding fresh, never inventing). Any `ValueError` from
    registry construction or resolution (an entry exists but is not yet
    approved, has no domain, etc.) is re-wrapped as `RuntimeError` here, the
    error type every caller of this function already expects.

    Previously read from a separate `query_concept_sets` store that a
    reviewer had to map independently from `concept_sets`, with nothing
    checking the two agreed -- a real divergence risk (the phenotype shown
    as "approved" in the review-only plan could differ from what a query
    actually selects) as well as duplicate review work for the same
    concept. Unified so there is exactly one place a concept gets approved
    for a role, read by every SQL generator (`feasibility_sql`,
    `AnalyticCohortBuilder.build`, `profile_sql`, `followup_audit_sql`) and
    by `DataScienceAgent`'s phenotype-review plan alike.
    """
    if not isinstance(protocol.get("concept_sets"), dict):
        raise RuntimeError(
            "Protocol requires concept_sets (population/treatment/comparator/outcome, "
            "each approved with a domain and concept_ids) before a cohort or feasibility "
            "query can be generated for this question"
        )
    # `OMOPVariableRegistry.from_protocol` requires a non-empty `study_id` --
    # a real invariant for a persisted registry, but irrelevant to resolving
    # concept specs (never read below), and every real `protocol.json`
    # already has one (`_validate_study_id`, enforced at `start()`). Only a
    # placeholder here, for callers (namely this function's own unit tests)
    # that build a minimal protocol dict without it.
    registry_protocol = protocol if isinstance(protocol.get("study_id"), str) and protocol["study_id"].strip() else {
        **protocol, "study_id": "unspecified",
    }
    try:
        registry = OMOPVariableRegistry.from_protocol(registry_protocol)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    exposure_contrast = _load_exposure_contrast(protocol)
    specs: dict[str, Any] = {}
    for role in QUERY_ROLES:
        fallback_roles = _ROLE_FALLBACK_ROLES.get(role, ())
        matched = registry.resolve_role(role, *fallback_roles)

        if role == "comparator" and exposure_contrast == "vs_unexposed":
            # Never raises here, matched or not, approved or not -- there is
            # no OMOP concept for "absence of a drug" (e.g. "aspirin vs no
            # aspirin"), so an unapproved/domain-less/absent comparator is
            # this design's correct, final state, not a blocker (mirrors
            # `_resolve_concept_spec`'s former carve-out one-for-one).
            if matched is None:
                specs[role] = ConceptSetSpec(domain="", concept_ids=())
            else:
                specs[role] = ConceptSetSpec(
                    domain=matched.domain or "",
                    concept_ids=matched.root_concept_ids,
                    include_descendants=matched.include_descendants,
                    excluded_concept_ids=matched.excluded_concept_ids,
                )
            continue

        if matched is None:
            if role == "outcome":
                # No role="outcome" variable declared at all: the same
                # "all-cause death via the standard OMOP `death` table"
                # default `DataScienceAgent.run` falls back to when it finds
                # no phenotype declaring role="outcome" either -- kept in
                # sync so both readers of `concept_sets` agree on what an
                # omitted outcome role means. A caller-level placeholder
                # (never persisted, never written back into the registry --
                # see this function's docstring).
                placeholder = OMOPVariableDefinition(
                    variable_id="outcome_all_cause_death_default",
                    dag_node_label="outcome",
                    variable_kind="phenotype",
                    causal_role="outcome",
                    observed=True,
                    domain="death",
                    mapping_status=MAPPING_STATUS_APPROVED,
                    provenance="protocol_declared",
                )
                specs[role] = fixed_domain_to_concept_set_spec(placeholder)
                continue
            wanted = f"role {role!r}" + (f" or {fallback_roles[0]!r}" if fallback_roles else "")
            raise RuntimeError(f"concept_sets requires an approved entry with {wanted}")

        try:
            if is_fixed_domain(matched.domain):
                specs[role] = fixed_domain_to_concept_set_spec(matched)
            else:
                specs[role] = extraction_plan_to_concept_set_spec(build_extraction_plan(matched))
        except ValueError as exc:
            raise RuntimeError(
                f"concept_sets role '{role}' must be approved with a domain before a query can be "
                f"generated for it ({exc})"
            ) from exc

    # `additional_population_concept_sets` (see `WorkflowManager.
    # add_population_concept_set`): names of further, independently-required
    # population-defining concept sets, ANDed onto `specs["population"]` by
    # `causal.queries.feasibility._cohort_ctes`'s `additional_population`
    # kwarg. Resolved by looking each name up directly in `registry.
    # variables` (matching `dag_node_label`, the raw `concept_sets` key) --
    # deliberately NOT via `registry.resolve_role`, which raises if two
    # variables share one `causal_role` and would wrongly reject a second
    # population-defining entry. Always present in `specs`, even when empty,
    # so callers can uniformly do `specs.get("population_additional", ())`.
    additional_names: list[str] = []
    additional_specs: list[ConceptSetSpec] = []
    for name in protocol.get("additional_population_concept_sets", ()) or ():
        name = str(name).strip()
        if not name:
            continue
        variable = next((v for v in registry.variables if v.dag_node_label == name), None)
        if variable is None:
            raise RuntimeError(
                f"additional_population_concept_sets entry '{name}' has no matching concept_sets entry"
            )
        try:
            if is_fixed_domain(variable.domain):
                spec = fixed_domain_to_concept_set_spec(variable)
            else:
                spec = extraction_plan_to_concept_set_spec(build_extraction_plan(variable))
        except ValueError as exc:
            raise RuntimeError(
                f"additional_population_concept_sets entry '{name}' must be approved with a domain "
                f"before a query can be generated for it ({exc})"
            ) from exc
        additional_names.append(name)
        additional_specs.append(spec)
    specs["population_additional"] = tuple(additional_specs)
    # Kept alongside (not folded into) `population_additional` so
    # `_mapping_coverage_roles` can label each extra concept set by its
    # `concept_sets` key for coverage-check error messages, without every
    # other reader of `specs["population_additional"]` (the 8 SQL-generation
    # call sites) having to unwrap a `(name, spec)` pair it doesn't need.
    specs["population_additional_names"] = tuple(additional_names)
    return specs


def _load_query_settings(protocol: dict) -> dict[str, Any]:
    settings = protocol.get("query_settings")
    if not isinstance(settings, dict):
        settings = {}
    return {
        "care_site_names": tuple(str(value) for value in settings.get("care_site_names", ())),
        "min_age_years": settings.get("min_age_years", 18),
    }


def _load_exposure_contrast(protocol: dict) -> str:
    """Read the human-approved treatment/comparator inclusion semantics for
    the analytic cohort (see `causal.queries.feasibility.EXPOSURE_CONTRASTS`).
    Kept separate from `_load_query_settings` because `feasibility_sql`
    reports all exposure combinations unfiltered and takes this only to know
    whether `comparator` needs to resolve to a real OMOP concept at all
    (`"vs_unexposed"` does not) -- `AnalyticCohortBuilder`'s person-level
    cohort is the one that actually filters by it.
    """
    settings = protocol.get("query_settings")
    if not isinstance(settings, dict):
        return "add_on"
    value = settings.get("exposure_contrast", "add_on")
    if value not in EXPOSURE_CONTRASTS:
        raise RuntimeError(f"query_settings.exposure_contrast must be one of {EXPOSURE_CONTRASTS}")
    return value


def _load_censoring_definition_approved(protocol: dict) -> bool:
    """Whether a reviewer has approved the real per-row `censored` column
    (see `causal.queries.feasibility.analytic_cohort_sql` and
    `WorkflowManager.approve_censoring_definition`). Defaults to `False`,
    same as `analytic_cohort_sql`'s own default -- an unreviewed run keeps
    `censored` as `NULL`, unchanged from before this flag existed."""
    settings = protocol.get("query_settings")
    if not isinstance(settings, dict):
        return False
    return bool(settings.get("censoring_definition_approved", False))


def _load_statistical_estimand_settings(protocol: dict) -> dict[str, str] | None:
    """Read the reviewer-set missing-data/censoring/estimation-procedure
    strategy (`protocol["statistical_estimand_settings"]`; see
    `WorkflowManager.set_statistical_estimand_strategy`). Kept in its own
    top-level protocol key rather than folded into `query_settings`: these
    are statistical modeling decisions consumed by `EstimationAgent`/
    `TMLEAgent`, not SQL-generation settings consumed by
    `analytic_cohort_sql`. Returns `None` (not a dict of empty strings) when
    no reviewer has set this yet, so callers can tell "not reviewed" apart
    from "reviewed as empty".

    `estimation_procedure` defaults to `"plug_in_standardization"` when
    absent from the persisted settings -- a run whose reviewer called
    `set_statistical_estimand_strategy` before this field existed keeps
    today's g-computation behavior rather than being forced back into
    re-review.
    """
    settings = protocol.get("statistical_estimand_settings")
    if not isinstance(settings, dict):
        return None
    required = ("missing_data_strategy", "censoring_strategy", "censoring_rationale")
    if not all(isinstance(settings.get(key), str) and settings.get(key) for key in required):
        return None
    estimation_procedure = settings.get("estimation_procedure", "plug_in_standardization")
    if estimation_procedure not in ESTIMATION_PROCEDURES:
        return None
    return {
        **{key: settings[key] for key in required},
        "estimation_procedure": estimation_procedure,
    }


# A concept whose text plainly appears in this many *unmapped* (concept_id=0)
# source_value rows, while barely appearing among *mapped* rows, is very
# likely present in the source data but never mapped by this OMOP build's
# ETL -- the pattern that first surfaced MIMIC-IV/OMOP's near-total oral-drug
# gap (RxNorm "Aspirin" matched 3 drug_exposure rows database-wide against
# ~180K unmapped rows whose drug_source_value read "Aspirin ..."). Below this
# row-count floor there just isn't enough text evidence either way to
# conclude anything -- a genuinely rare but well-mapped concept can have very
# few rows on both sides -- so the gate stays silent.
_MAPPING_COVERAGE_MIN_EVIDENCE_ROWS = 50
# Coverage = mapped / (mapped + unmapped_name_matched). Below this floor the
# vast majority of a concept's real-world occurrences in this dataset are
# unmapped, so a cohort built on it would silently undercount this role
# rather than fail loudly -- see `_check_mapping_coverage`.
_MAPPING_COVERAGE_MIN_RATIO = 0.05


def _mapping_coverage_roles(
    specs: dict[str, Any], exposure_contrast: str
) -> list[tuple[str, ConceptSetSpec]]:
    """Which query roles the mapping-coverage gate should check: every role
    whose spec actually resolves to a concept-based domain with concept_ids
    to query. Excludes `outcome` when it defaults to the `death` table (no
    concept_ids involved) and `comparator` under `"vs_unexposed"` (see
    `causal.queries.feasibility.EXPOSURE_CONTRASTS`; there is no OMOP concept
    for "absence of a drug" to check coverage for).

    Also covers every `additional_population_concept_sets` entry (see
    `_load_concept_specs`'s `population_additional`/`population_additional_names`),
    labeled by its own `concept_sets` key rather than a query role -- these
    entries have no single shared role name, unlike the four fixed ones."""
    roles: list[tuple[str, ConceptSetSpec]] = []
    for role in QUERY_ROLES:
        if role == "comparator" and exposure_contrast == "vs_unexposed":
            continue
        spec = specs.get(role)
        if spec is not None and spec.domain in DOMAIN_TABLES and spec.concept_ids:
            roles.append((role, spec))
    # `_load_concept_specs` appends to these two tuples in lockstep, so they are
    # always the same length; strict= makes a future drift a loud failure.
    for name, spec in zip(
        specs.get("population_additional_names", ()),
        specs.get("population_additional", ()),
        strict=True,
    ):
        if spec.domain in DOMAIN_TABLES and spec.concept_ids:
            roles.append((name, spec))
    return roles


def _check_mapping_coverage(role: str, spec: ConceptSetSpec, counts: dict[str, int]) -> None:
    """Raise if `role`'s concept set looks unmapped in this OMOP build rather
    than genuinely rare (see `_MAPPING_COVERAGE_MIN_EVIDENCE_ROWS`/
    `_MAPPING_COVERAGE_MIN_RATIO` for the thresholds and their rationale).
    Blocking here, before any cohort SQL runs, turns a silent zero-patient
    cohort into a clear, actionable error."""
    mapped = int(counts.get("mapped_row_count") or 0)
    unmapped = int(counts.get("unmapped_name_matched_row_count") or 0)
    total_evidence = mapped + unmapped
    if total_evidence < _MAPPING_COVERAGE_MIN_EVIDENCE_ROWS:
        return
    coverage = mapped / total_evidence
    if coverage >= _MAPPING_COVERAGE_MIN_RATIO:
        return
    raise RuntimeError(
        f"Mapping coverage gate failed for role '{role}' (concept_ids={list(spec.concept_ids)}): "
        f"only {mapped} row(s) in this OMOP build carry one of this concept's mapped standard "
        f"concept_ids, versus {unmapped} unmapped row(s) (concept_id=0) whose source_value text "
        f"still matches this concept's own vocabulary name ({coverage:.2%} coverage, below the "
        f"{_MAPPING_COVERAGE_MIN_RATIO:.0%} floor). This role is very likely present in the source "
        "data but not mapped by this dataset's ETL -- a cohort built on it would silently "
        "undercount rather than correctly report zero. Pick a different, better-mapped concept for "
        "this role, or a different data source, before continuing."
    )


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify_covariate_name(name: str) -> str:
    slug = _SLUG_RE.sub("_", name.strip().lower()).strip("_")
    return f"cov_{slug}" if slug else "cov_"


def _load_covariate_specs(analysis_plan: dict) -> tuple[CovariateSpec, ...]:
    """Ready, non-demographic covariates (see `causal.agents.data_science.
    CovariatePlan.status`) turned into `CovariateSpec`s for
    `AnalyticCohortBuilder.build()`, via `causal.mapping.registry` (see
    `docs/causal_roadmap.md` migration step 4).

    `status == "ready"` (computed by `DataScienceAgent._covariate` -- mapping
    approved, aggregation and window reviewed, domain known) stays the
    authoritative human-facing gate, exactly as before: a reviewer can still
    force a covariate back out of the cohort SQL by resetting `status`
    without touching its underlying mapping/aggregation/window fields, and
    that override must be respected here, not silently re-derived away.
    What changes is *how* a "ready" entry becomes a `CovariateSpec`: it is
    parsed into an `OMOPVariableDefinition` (`variable_from_covariate_plan`)
    and resolved via `build_extraction_plan`/`extraction_plan_to_covariate_spec`
    -- the same conversion path `causal.mapping.registry` uses everywhere
    else -- instead of this function building the dataclass directly from
    raw dict fields a second time. A covariate's human-entered `name` is
    slugified into a SQL column identifier here; `AnalyticCohortBuilder`
    still re-validates uniqueness/safety as a defense-in-depth check at the
    actual SQL-generation boundary (see `causal.queries.feasibility.
    _covariate_ctes_columns_and_joins`). Demographic covariates never reach
    here -- they keep `domain == ""` and are excluded by the `domain` check.
    """
    specs: list[CovariateSpec] = []
    seen_columns: set[str] = set()
    for item in analysis_plan.get("covariates", ()):
        if not isinstance(item, dict) or item.get("status") != "ready" or not item.get("domain"):
            continue
        variable = variable_from_covariate_plan(item)
        column_name = _slugify_covariate_name(variable.dag_node_label)
        if column_name in seen_columns:
            raise RuntimeError(
                f"Covariate '{variable.dag_node_label}' produces a duplicate SQL column name "
                f"'{column_name}' -- rename one of the colliding covariates"
            )
        seen_columns.add(column_name)
        plan = build_extraction_plan(variable)
        specs.append(extraction_plan_to_covariate_spec(variable, plan, column_name))
    return tuple(specs)


def _carry_forward_concept_candidates(
    artifact: dict[str, Any], previous_plan_path: Path
) -> tuple[dict[str, Any], bool]:
    """When `run_data_science_agent` runs without a live `concept_lookup`
    (no `--with-mcp`), the freshly rebuilt analysis plan has no
    `concept_candidates` at all -- `DataScienceAgent` never attempted a
    lookup. Without this, re-running `workflow plan` after having once run
    it `--with-mcp` would silently discard the candidate list a reviewer may
    still be working from. This only ever fills in a phenotype/covariate
    that has *no* candidates in the fresh artifact (i.e. no lookup was
    attempted this run) by name-matching the last plan that did have them;
    it never overwrites a genuine empty result from an actual live lookup,
    since this function is only called when `concept_lookup is None`. The
    approved mapping itself (`concept_ids`/`mapping_status` in
    `candidate_dag`) never depended on this file and is unaffected either
    way.
    """
    if not previous_plan_path.is_file():
        return artifact, False
    try:
        previous = json.loads(previous_plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return artifact, False

    carried_any = False

    def _carried(items: list[Any], previous_items: list[Any]) -> list[Any]:
        nonlocal carried_any
        previous_by_name = {
            item.get("name"): item.get("concept_candidates") or []
            for item in previous_items if isinstance(item, dict)
        }
        result = []
        for item in items:
            candidates = previous_by_name.get(item.get("name")) if isinstance(item, dict) else None
            if isinstance(item, dict) and not item.get("concept_candidates") and candidates:
                item = {**item, "concept_candidates": candidates}
                carried_any = True
            result.append(item)
        return result

    artifact = dict(artifact)
    artifact["phenotypes"] = _carried(list(artifact.get("phenotypes", ())), list(previous.get("phenotypes", ())))
    artifact["covariates"] = _carried(list(artifact.get("covariates", ())), list(previous.get("covariates", ())))
    return artifact, carried_any


def _mismatched_concept_domains(
    csv_text: str, concept_ids: list[int], expected_domain: str
) -> list[str]:
    """Parse a `Get_Concepts_By_Id` CSV response and return one
    human-readable note per concept_id that either wasn't found at all, or
    whose real OMOP `domain_id` doesn't match `expected_domain` -- e.g. a
    Drug-domain concept_id approved for a covariate declared
    domain="measurement". Every domain in `causal.agents.hypothesis.
    DAG_VARIABLE_DOMAINS` capitalizes exactly to its OMOP `domain_id`
    (condition->Condition, drug->Drug, ...), so no separate mapping table is
    needed here.
    """
    expected_domain_id = expected_domain.capitalize()
    found: dict[int, str] = {}
    for row in csv.DictReader(io.StringIO(csv_text)):
        try:
            found[int(row["concept_id"])] = row["domain_id"]
        except (KeyError, ValueError, TypeError):
            continue
    mismatches = []
    for concept_id in concept_ids:
        actual_domain_id = found.get(concept_id)
        if actual_domain_id is None:
            mismatches.append(f"{concept_id}: not found in the concept table")
        elif actual_domain_id != expected_domain_id:
            mismatches.append(
                f"{concept_id}: domain_id='{actual_domain_id}', expected '{expected_domain_id}' "
                f"for declared domain '{expected_domain}'"
            )
    return mismatches


def _signed_events(value: Any) -> str:
    """`+42` / `-16` / `0` for a signed events-per-1000 count in a
    `causal_report.md` bullet -- the sign carries the direction (more vs
    fewer outcome events with treatment), so it is always shown. `?` when
    the count is unavailable."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "?"
    value = int(value)
    return f"+{value}" if value > 0 else str(value)


def _censoring_strategy_report_note(estimand_block: dict[str, Any]) -> str:
    """Parenthetical detail for `causal_report.md`'s censoring-strategy line
    -- what actually happened to censored rows differs enough by strategy
    (dropped vs. reweighted vs. left alone) that one fixed phrase would
    misdescribe at least two of the three `causal.roadmap.
    CENSORING_STRATEGIES` values."""
    strategy = estimand_block.get("censoring_strategy")
    if strategy == "exclude_censored_before_analysis":
        return f" ({estimand_block.get('n_excluded_censored', 0)} row(s) excluded for censoring)"
    if strategy == "ipcw_administrative_censoring":
        return (
            f" ({estimand_block.get('n_censored', 0)} row(s) reweighted via IPCW; "
            f"censoring-model AUC {estimand_block.get('censoring_model_auc', '-')}, "
            f"mean/max weight {estimand_block.get('ipcw_weight_mean', '-')}/"
            f"{estimand_block.get('ipcw_weight_max', '-')})"
        )
    return ""


def _gather_assumptions(
    study_spec: dict[str, Any],
    readiness: dict[str, Any],
    cohort_manifest: dict[str, Any],
    dataset_manifest: dict[str, Any],
    diagnostics: dict[str, Any],
    estimand_block: dict[str, Any],
    cate: dict[str, Any] | None = None,
) -> list[str]:
    """Deduplicated assumptions/limitations for the causal report and the
    results narrative -- shared by `WorkflowManager.build_causal_report` and
    `WorkflowManager.run_results_narrative` so the two never drift apart on
    what a reviewer (human or LLM-assisted) sees as the caveats for the same
    estimate. Every artifact argument is optional (an empty dict when the
    corresponding file does not exist yet)."""
    assumptions: list[str] = []
    if study_spec:
        identification = study_spec.get("identification", {})
        assumptions.append(
            "Identification relies on the reviewed causal DAG and backdoor adjustment set "
            f"(identified_by_backdoor_adjustment="
            f"{identification.get('identified_by_backdoor_adjustment', '-')}); an unmeasured "
            "confounder omitted from the DAG would still bias this estimate."
        )
    for artifact, key in (
        (readiness, "warnings"), (cohort_manifest, "unresolved_issues"),
        (dataset_manifest, "warnings"), (diagnostics, "warnings"),
    ):
        assumptions.extend(artifact.get(key, ()) if artifact else ())
    assumptions.extend(estimand_block.get("warnings", ()) if estimand_block else ())
    assumptions = list(dict.fromkeys(assumptions))
    if cate and cate.get("estimand"):
        assumptions.append(
            "CATE/effect-modification estimates are reported above for the prespecified "
            "subgroup family only; the population ATE remains this study's primary estimand."
        )
    else:
        assumptions.append(
            "Out of scope (docs/causal_roadmap.md milestone #4): CATE/effect-modification "
            "estimation -- this run's estimator reports the population ATE only."
        )
    assumptions.append(
        "Censoring-aware (IPCW) adjustment is implemented for single-interval administrative "
        "censoring only, and only for the g-computation estimator -- see the statistical "
        "estimand's censoring_strategy above."
    )
    return assumptions


def _dispatch_estimator(
    statistical_estimand: dict[str, Any],
    dataframe: pd.DataFrame,
    *,
    study_id: str,
    covariate_columns: tuple[str, ...],
    n_bootstrap: int,
    random_seed: int,
) -> Any:
    """Fit `EstimationAgent` or `TMLEAgent` over `dataframe`, dispatched on
    `statistical_estimand`'s reviewed `estimation_procedure` -- the shared
    logic behind `WorkflowManager.run_estimation` (the real study's
    materialized dataset) and `WorkflowManager.run_negative_control_
    estimation` (a negative-control-substituted cohort), so the two never
    drift on which estimator a reviewed `estimation_procedure` maps to.
    `n_bootstrap` is ignored when TMLE is the reviewed procedure -- TMLE
    replaces the bootstrap with a closed-form efficient-influence-curve
    variance estimator.

    Only ever returns an `EstimationArtifact` or a `TMLEArtifact` -- never a
    `CATEArtifact`, even when `estimation_procedure == "causal_forest_dml"`
    (a CATE study's forced value, `roadmap.StatisticalEstimand.__post_init__
    `). This is deliberate, not an unhandled case: `causal.agents.cate.
    CATEEstimate` has no `effect_measure`/`point_estimate`/`ci_lower`/
    `ci_upper` fields (it reports a subgroup family, not one population
    contrast), so it cannot satisfy `run_estimation`'s/`run_negative_control_
    estimation`'s shared `GComputationEstimate`/`TMLEEstimate` contract --
    `WorkflowManager.run_cate` is the separate, CATE-shaped call path
    (`cate.json`) for that. Falling through to `EstimationAgent` (plug-in
    g-computation) here for a `"causal_forest_dml"` estimand instead gives a
    CATE study the same population-level ATE `estimation.json` every other
    study gets -- exactly what `build_causal_report`'s CATE section already
    assumes exists to cross-check `overall_ate_from_forest` against (see
    "Overall ATE from the forest (cross-check against the primary estimate
    above)"). `TMLEAgent` is not used for this fallback: TMLE's SuperLearner
    nuisance fit is materially more expensive, and this ATE is a secondary
    cross-check for a CATE run, not this run's primary reviewed estimate.
    """
    estimation_procedure = statistical_estimand.get("estimation_procedure", "plug_in_standardization")
    if estimation_procedure == "targeted_minimum_loss_based_estimation":
        return TMLEAgent().run(
            dataframe,
            study_id=study_id,
            covariate_columns=covariate_columns,
            censoring_strategy=statistical_estimand["censoring_strategy"],
            missing_data_strategy=statistical_estimand["missing_data_strategy"],
            random_seed=random_seed,
        )
    return EstimationAgent().run(
        dataframe,
        study_id=study_id,
        covariate_columns=covariate_columns,
        censoring_strategy=statistical_estimand["censoring_strategy"],
        missing_data_strategy=statistical_estimand["missing_data_strategy"],
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )


class WorkflowManager:
    def __init__(self, runs_dir: str | Path = "artifacts/runs") -> None:
        self.runs_dir = Path(runs_dir)

    def start(self, protocol_path: str | Path, mode: str = "exploratory") -> RunState:
        if mode not in {"exploratory", "final"}:
            raise ValueError("mode must be exploratory or final")
        source = Path(protocol_path)
        protocol = json.loads(source.read_text(encoding="utf-8"))
        study_id = protocol.get("study_id")
        if not isinstance(study_id, str) or not study_id.strip():
            raise ValueError("Protocol requires a non-empty study_id")
        study_id = _validate_study_id(study_id)

        run_id = _make_run_id(study_id)
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        snapshot = run_dir / "protocol.json"
        shutil.copyfile(source, snapshot)
        state = RunState(
            run_id=run_id,
            study_id=study_id,
            protocol_path=str(snapshot),
            mode=mode,
            gates={name: ApprovalGate(name) for name in _ALL_GATE_NAMES},
            artifacts=["protocol.json"],
        )
        state.record("run_started", f"mode={mode}")
        self.save(state)
        return state

    def start_generated(
        self,
        generated: GeneratedHypothesis,
        mode: str = "exploratory",
        model: str = "unknown",
    ) -> RunState:
        """Create a governed run from an LLM draft without bypassing gates."""
        if mode not in {"exploratory", "final"}:
            raise ValueError("mode must be exploratory or final")
        study_id = generated.protocol.get("study_id")
        if not isinstance(study_id, str) or not study_id.strip():
            raise ValueError("Generated protocol requires a non-empty study_id")
        study_id = _validate_study_id(study_id)
        run_id = _make_run_id(study_id)
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "protocol.json").write_text(
            json.dumps(generated.protocol, indent=2), encoding="utf-8"
        )
        (run_dir / "hypothesis.json").write_text(
            json.dumps(generated.hypothesis.to_dict(), indent=2), encoding="utf-8"
        )
        (run_dir / "llm_outputs.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "model": model,
                    "prompt_version": generated.prompt_version,
                    "contains_raw_llm_text": True,
                    "outputs": [item.to_dict() for item in generated.llm_outputs],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        state = RunState(
            run_id=run_id,
            study_id=study_id,
            protocol_path=str(run_dir / "protocol.json"),
            mode=mode,
            gates={name: ApprovalGate(name) for name in _ALL_GATE_NAMES},
            artifacts=["protocol.json", "hypothesis.json", "llm_outputs.json"],
        )
        state.record("run_started", f"mode={mode}; source=llm_draft")
        state.record(
            "agent_completed",
            f"gpt_hypothesis_draft; model={model}; raw_outputs={len(generated.llm_outputs)}",
        )
        self.save(state)
        return state

    def load(self, run_id: str) -> RunState:
        path = self._state_path(run_id)
        if not path.is_file():
            raise FileNotFoundError(f"Unknown run_id: {run_id}")
        state = RunState.from_dict(json.loads(path.read_text(encoding="utf-8")))
        _migrate_legacy_protocol_gate(state)
        _migrate_missing_gates(state)
        return state

    def save(self, state: RunState) -> None:
        path = self._state_path(state.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(path, state.to_dict())

    def decide_gate(self, run_id: str, gate: str, status: str, reviewer: str, comment: str | None = None) -> RunState:
        run_dir = self.runs_dir / run_id
        with _RunLock(run_dir):
            state = self.load(run_id)
            if gate not in state.gates:
                raise ValueError(f"Unknown gate: {gate}")
            if status not in {"approved", "rejected", "provisional"}:
                raise ValueError("status must be approved, rejected, or provisional")
            if gate == "target_trial" and status == "approved":
                self._block_if_target_trial_incomplete(run_id)
            if gate == "dag_review" and status == "approved":
                self._block_if_major_confounders_unresolved(run_id)
            if gate == "identification_estimand" and status == "approved":
                self._block_if_not_identified(run_id)
                self._write_study_specification_if_hypothesis_exists(run_id)
            if gate == "analysis_plan" and status == "approved":
                self._block_if_analysis_plan_incomplete(run_id)
            if gate == "statistical_estimand" and status == "approved":
                self._write_statistical_estimand(run_id)
            if gate == "results" and status == "approved":
                self._block_if_results_incomplete(run_id)
            state.gates[gate].decide(status, reviewer, comment)  # type: ignore[arg-type]
            state.record("gate_decided", f"{gate}={status} by {reviewer}")
            if gate == "identification_estimand" and status == "approved":
                if "study_specification.json" not in state.artifacts:
                    state.artifacts.append("study_specification.json")
            if gate == "statistical_estimand" and status == "approved":
                if "statistical_estimand.json" not in state.artifacts:
                    state.artifacts.append("statistical_estimand.json")
            self.save(state)
        return state

    def _invalidate_gate(self, state: RunState, gate: str, reason: str) -> None:
        """Reset an already-decided gate back to `pending` after a review
        action changes the content it approved.

        Without this, `ExecutionAgent`/OMCP governance (see
        `causal.agents.execution.ExecutionAgent.run`,
        `omcp.governance.authorize_causal_operation`) trust
        `state.gates[...].status` as a bare string with no link to the
        artifact's current content -- a gate approved once would keep
        authorizing execution even after the analysis plan or a covariate's
        clinical mapping was silently regenerated or edited underneath it.
        A no-op when the gate is not currently `approved`/`provisional`
        (the ordinary "review, then approve" flow never triggers this).
        """
        gate_obj = state.gates.get(gate)
        if gate_obj is None or gate_obj.status not in {"approved", "provisional"}:
            return
        previous_status = gate_obj.status
        gate_obj.status = "pending"
        gate_obj.reviewer = None
        gate_obj.comment = None
        gate_obj.decided_at = None
        state.record("gate_invalidated", f"{gate} reset from {previous_status} to pending: {reason}")

    def _block_if_analysis_plan_incomplete(self, run_id: str) -> None:
        """The `analysis_plan` gate cannot be approved while the plan itself
        still has unresolved issues or a covariate/phenotype that isn't
        ready -- otherwise approval is a rubber stamp that
        `causal.agents.execution.ExecutionAgent` would refuse to act on
        anyway (see `analysis_plan_blockers`), just without ever telling the
        reviewer what was actually missing.
        """
        plan_path = self.runs_dir / run_id / "analysis_plan.json"
        if not plan_path.is_file():
            raise ValueError(
                "Cannot approve the analysis_plan gate: run the Data Science Agent to "
                "generate analysis_plan.json first"
            )
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        unresolved, incomplete_covariates, incomplete_phenotypes = analysis_plan_blockers(plan)
        if not (unresolved or incomplete_covariates or incomplete_phenotypes):
            return
        details = []
        if unresolved:
            details.append(f"unresolved issues: {', '.join(unresolved)}")
        if incomplete_covariates:
            details.append(f"covariates not ready: {', '.join(incomplete_covariates)}")
        if incomplete_phenotypes:
            details.append(f"phenotypes not approved: {', '.join(incomplete_phenotypes)}")
        raise ValueError(
            "Cannot approve the analysis_plan gate while the plan is incomplete "
            f"({'; '.join(details)})."
        )

    def _block_if_results_incomplete(self, run_id: str) -> None:
        """The `results` gate (roadmap stage 9: results review) is the
        final, most consequential sign-off in the pipeline, but unlike every
        gate before it, `REQUIRED_GATES` has no entry for the `execution` or
        `diagnostics` phases -- `run_estimation` is an agent action a
        reviewer triggers, not a human review decision with its own gate,
        so nothing upstream forces it to have actually run before a run
        reaches `results_review`. Without this check, `results` could be
        approved -- a rubber stamp -- on a run that never fit a causal
        effect at all: `build_causal_report` degrades gracefully when
        `estimation.json` is absent (its `interpretation` section just says
        so), but that is a report-rendering fallback, not a substitute for
        this gate actually blocking -- the one rubber-stamp gap every other
        gate in this file (`_block_if_target_trial_incomplete`,
        `_block_if_major_confounders_unresolved`, `_block_if_not_identified`,
        `_block_if_analysis_plan_incomplete`) was already written to close.

        Deliberately does not also require `diagnostics.json`/
        `sensitivity_analysis.json`/CATE/negative controls: those are
        explicitly optional elsewhere in this module (`run_diagnostics` can
        run before or after `run_estimation`, see its own docstring) --
        widening this gate to require them would fight that existing design
        for no real gain, since the actual risk this closes is a study
        marked "approved" with no estimate at all.
        """
        if not (self.runs_dir / run_id / "estimation.json").is_file():
            raise ValueError(
                "Cannot approve the results gate: run_estimation has not been run for this "
                "run (estimation.json is missing) -- there is no estimate to approve"
            )

    def advance(self, run_id: str) -> RunState:
        run_dir = self.runs_dir / run_id
        with _RunLock(run_dir):
            state = self.load(run_id)
            gate_name = REQUIRED_GATES.get(state.phase)
            if gate_name:
                gate = state.gates[gate_name]
                accepted = gate.status == "approved" or (
                    state.mode == "exploratory" and gate.status == "provisional" and gate_name == "clinical_mapping"
                )
                if not accepted:
                    raise RuntimeError(f"Gate {gate_name} must be accepted before leaving {state.phase}")
            next_phase = NEXT_PHASE.get(state.phase)
            if next_phase is None:
                raise RuntimeError("Workflow is already at its final phase")
            previous = state.phase
            state.phase = next_phase
            state.record("phase_advanced", f"{previous}->{next_phase}")
            self.save(state)
        return state

    def run_hypothesis_agent(self, run_id: str) -> RunState:
        state = self.load(run_id)
        if state.phase not in {
            "target_trial_review", "dag_review", "identification_review", "concept_mapping",
        }:
            raise RuntimeError(
                "Hypothesis Agent can only run during target_trial_review, dag_review, "
                "identification_review, or concept_mapping"
            )
        protocol_path = self.runs_dir / run_id / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        self._write_hypothesis(run_id, protocol)
        if "hypothesis.json" not in state.artifacts:
            state.artifacts.append("hypothesis.json")
        state.record("agent_completed", "hypothesis_agent->hypothesis.json")
        self.save(state)
        return state

    def map_concept_set(
        self,
        run_id: str,
        name: str,
        domain: str,
        concept_ids: list[int],
        reviewer: str,
        status: str = "approved",
        comment: str | None = None,
        concept_lookup: OMCPClient | None = None,
    ) -> RunState:
        """Record a human's OMOP domain + concept(s) for a protocol-declared
        `concept_sets[name]` phenotype (population/treatment/comparator/
        outcome/... -- see `causal.agents.data_science.DataScienceAgent.
        _phenotype_from_concept_set`).

        This is the single place a concept gets approved for a role: the
        same `concept_sets` entry a reviewer maps here is what
        `_load_concept_specs` resolves query-generation `ConceptSetSpec`s
        from (see its docstring) -- there used to be a second, independent
        `query_concept_sets` mapping step for the same role with nothing
        checking the two agreed; that duplication is gone.

        Distinct from `map_covariate_concepts` (the equivalent for baseline
        covariates in `candidate_dag`): a `concept_sets` entry lives
        directly on `protocol.json`, keyed by an arbitrary clinical name
        rather than a fixed role (a study can declare more than one concept
        set per role, e.g. `treatment_and_comparator`), and its `status`
        also drives `HypothesisArtifact.unresolved_issues`/`executable`
        (see `HypothesisAgent._unresolved_concept_sets`), so
        `hypothesis.json` is regenerated here too, same as `review_covariate`.

        Only `domain`/`concept_ids`/`status`/`reviewer`/`provenance` are
        written here; `role`, `source_value_rule`, and `lookup_term`
        (protocol-author-set) are left untouched. `concept_lookup` is the
        same optional, opt-in domain cross-check as `map_covariate_concepts`
        -- see its docstring, including what it means for `provenance`.

        `domain='death'` is only valid for an entry whose declared `role` is
        `'outcome'` (see `OUTCOME_DOMAINS`) and needs no `concept_ids` -- the
        fixed OMOP `death` table, not a concept domain like the rest.
        """
        name = name.strip()
        if not name:
            raise ValueError("name is required")
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        if not isinstance(status, str) or not status.strip():
            raise ValueError("status is required")
        status = status.strip()
        try:
            concept_ids = [int(concept_id) for concept_id in concept_ids]
        except (TypeError, ValueError):
            raise ValueError("concept_ids must be a list of integers") from None

        state = self.load(run_id)
        run_dir = self.runs_dir / run_id
        protocol_path = run_dir / "protocol.json"
        if not (run_dir / "hypothesis.json").is_file():
            raise RuntimeError("Run the Hypothesis Agent before mapping concept sets")

        # Read-only peek at the entry's declared role, deliberately outside
        # `_RunLock` (same rationale as `map_covariate_concepts`'s
        # `declared_domain` peek): only an entry whose role is "outcome" may
        # use `domain='death'`, everything else must resolve to a real
        # concept domain.
        existing_entry = json.loads(protocol_path.read_text(encoding="utf-8")).get(
            "concept_sets", {}
        ).get(name)
        role = existing_entry.get("role") if isinstance(existing_entry, dict) else None
        allowed_domains = OUTCOME_DOMAINS if role == "outcome" else tuple(DOMAIN_TABLES)
        if domain not in allowed_domains:
            raise ValueError(f"domain for role {role!r} must be one of {allowed_domains}")
        if status == "approved" and domain != "death" and not concept_ids:
            raise ValueError("status='approved' requires at least one concept_id unless domain='death'")

        if concept_lookup is not None and concept_ids:
            response = concept_lookup.get_concepts_by_id(concept_ids)
            mismatches = _mismatched_concept_domains(response.text, concept_ids, domain)
            if mismatches:
                raise ValueError(
                    f"Concept domain check failed for concept_sets['{name}']: " + "; ".join(mismatches)
                )

        with _RunLock(run_dir):
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            concept_sets = protocol.get("concept_sets", {})
            entry = concept_sets.get(name)
            if not isinstance(entry, dict):
                raise RuntimeError(f"concept_sets has no entry named '{name}'")
            before = {
                "domain": entry.get("domain", ""),
                "concept_ids": entry.get("concept_ids", []),
                "status": entry.get("status", "missing"),
            }
            # `reviewer`/`provenance` mirror the fix in
            # `map_covariate_concepts` for `candidate_dag` -- without
            # writing them here, `causal.mapping.registry.
            # variable_from_concept_set` (which reads both back via
            # `definition.get("reviewer", "")`/`definition.get("provenance",
            # "protocol_declared")`) left every phenotype's `reviewer` blank
            # and `provenance` always defaulted to `"protocol_declared"`,
            # regardless of who actually approved it or whether a live
            # `concept_lookup` OMCP cross-check was used -- the same dead
            # safety net the covariate fix closed, just still open here.
            updated_entry = {
                **entry,
                "domain": domain,
                "concept_ids": concept_ids,
                "status": status,
                "reviewer": reviewer.strip(),
                "provenance": (
                    "omcp_candidate" if concept_lookup is not None
                    else entry.get("provenance", "protocol_declared")
                ),
            }
            protocol["concept_sets"] = {**concept_sets, name: updated_entry}
            _atomic_write_json(protocol_path, protocol)

            self._write_hypothesis(run_id, protocol)
        detail = f"{name}: domain={domain}, {len(concept_ids)} concept(s), status={status} by {reviewer}"
        if comment and comment.strip():
            detail += f" ({comment.strip()})"
        state.record(
            "concept_set_mapped", detail,
            changes={
                "before": before,
                "after": {"domain": domain, "concept_ids": concept_ids, "status": status},
            },
        )
        self._invalidate_gate(state, "clinical_mapping", f"concept_sets['{name}'] changed")
        self._invalidate_gate(state, "analysis_plan", f"concept_sets['{name}'] changed")
        self.save(state)
        return state

    def set_concept_set_lookup_terms(self, run_id: str, terms: dict[str, str]) -> RunState:
        """Update `concept_sets[name].lookup_term` for one or more entries.

        `lookup_term` is only a search hint: it seeds the default value of the
        OMOP concept-search box in the UI (and the term
        `causal.agents.data_science.DataScienceAgent` uses to pre-fetch
        candidate concepts). It is not part of any mapping decision -- it does
        not feed `_load_concept_specs`, cohort/feasibility SQL, or any gate --
        so this writes `protocol.json` under the lock and records the change,
        but invalidates no gate and does not regenerate `hypothesis.json`.

        Backs the "Suggest search terms" button in the Streamlit
        concept-mapping section (`causal.agents.concept_lookup_terms`); a name
        not present in `concept_sets` is ignored rather than raising, since
        the set can change between the suggestion and the save.
        """
        if not isinstance(terms, dict) or not all(
            isinstance(name, str) and isinstance(value, str) for name, value in terms.items()
        ):
            raise ValueError("terms must be a {concept_set_name: search_term} mapping of strings")
        cleaned = {name: value.strip() for name, value in terms.items() if value.strip()}
        if not cleaned:
            raise ValueError("terms contains no non-empty search term")

        state = self.load(run_id)
        run_dir = self.runs_dir / run_id
        protocol_path = run_dir / "protocol.json"
        with _RunLock(run_dir):
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            concept_sets = protocol.get("concept_sets", {})
            applied = {
                name: value for name, value in cleaned.items()
                if isinstance(concept_sets.get(name), dict)
            }
            if not applied:
                raise RuntimeError("none of the given names match a concept set on this run")
            protocol["concept_sets"] = {
                name: ({**entry, "lookup_term": applied[name]} if name in applied else entry)
                for name, entry in concept_sets.items()
            }
            _atomic_write_json(protocol_path, protocol)
        state.record(
            "concept_set_lookup_terms_set",
            "; ".join(f"{name}={value!r}" for name, value in sorted(applied.items())),
        )
        self.save(state)
        return state

    def add_population_concept_set(
        self,
        run_id: str,
        *,
        name: str,
        label: str,
        lookup_term: str = "",
        reviewer: str,
        comment: str | None = None,
    ) -> RunState:
        """Declare a second (or further) independently-required population-
        defining diagnosis -- e.g. a study whose population is "adults with
        type 2 diabetes AND COPD" needs both diagnoses separately OMOP-mapped
        and ANDed together, which the single `concept_sets["population"]`
        role cannot express on its own (see `causal.queries.feasibility.
        _cohort_ctes`'s `additional_population` kwarg and `_load_concept_specs`'s
        `population_additional` resolution, which is what actually ANDs it in).

        Scaffolds a new `concept_sets[name]` entry in the same
        `status="pending_omop_mapping"` shape `causal.agents.user_question.
        UserQuestionHypothesisService._pending_concept_sets` seeds for the
        primary four roles (`role="population_secondary"`, cosmetic --
        resolved by direct lookup in `_load_concept_specs`, never through
        `OMOPVariableRegistry.resolve_role`'s one-entry-per-role invariant),
        and lists `name` in `additional_population_concept_sets` so
        `_load_concept_specs` picks it up. The reviewer then maps/approves it
        through the existing, unmodified `map_concept_set` +
        `_render_concept_set_mapping_forms` UI, exactly like population/
        treatment/comparator/outcome today -- this method only creates the
        entry, it does not map it.
        """
        name = name.strip()
        if not name:
            raise ValueError("name is required")
        if not label.strip():
            raise ValueError("label is required")
        if not reviewer.strip():
            raise ValueError("reviewer is required")

        run_dir = self.runs_dir / run_id
        protocol_path = run_dir / "protocol.json"
        state = self.load(run_id)
        if not (run_dir / "hypothesis.json").is_file():
            raise RuntimeError("Run the Hypothesis Agent before adding a population concept set")

        with _RunLock(run_dir):
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            concept_sets = protocol.get("concept_sets", {})
            if name in concept_sets:
                raise ValueError(f"concept_sets already has an entry named '{name}'")
            protocol["concept_sets"] = {
                **concept_sets,
                name: {
                    "status": "pending_omop_mapping",
                    "concept_ids": [],
                    "label": label.strip(),
                    "lookup_term": lookup_term.strip(),
                    "role": "population_secondary",
                    "domain": "",
                },
            }
            existing_additional = tuple(protocol.get("additional_population_concept_sets", ()) or ())
            protocol["additional_population_concept_sets"] = list(existing_additional) + [name]
            _atomic_write_json(protocol_path, protocol)

            self._write_hypothesis(run_id, protocol)
        detail = f"'{name}' ({label.strip()}) added by {reviewer} as a required population concept set"
        if comment and comment.strip():
            detail += f" ({comment.strip()})"
        state.record("population_concept_set_added", detail)
        self._invalidate_gate(state, "clinical_mapping", f"additional population concept set '{name}' added")
        self._invalidate_gate(state, "analysis_plan", f"additional population concept set '{name}' added")
        self.save(state)
        return state

    def remove_population_concept_set(
        self, run_id: str, name: str, reviewer: str, comment: str | None = None,
    ) -> RunState:
        """Undo an `add_population_concept_set` call -- e.g. added by mistake,
        or the reviewer decides the second diagnosis should not gate
        population eligibility after all.

        Deletes `concept_sets[name]` outright (not merely unlisted from
        `additional_population_concept_sets`): an entry this method targets
        was always scaffolded by `add_population_concept_set` and is
        `status="pending_omop_mapping"` far more often than not, so there is
        rarely a real mapping worth preserving, and leaving a dangling,
        never-required entry around only invites confusion later. A reviewer
        who *has* already mapped it and wants to keep the mapping for
        reference should not call this -- there is currently no "required or
        not" toggle short of removal.
        """
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        name = name.strip()
        if not name:
            raise ValueError("name is required")

        run_dir = self.runs_dir / run_id
        protocol_path = run_dir / "protocol.json"
        state = self.load(run_id)

        with _RunLock(run_dir):
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            existing_additional = tuple(protocol.get("additional_population_concept_sets", ()) or ())
            if name not in existing_additional:
                raise ValueError(f"'{name}' is not a declared additional population concept set")
            protocol["additional_population_concept_sets"] = [
                item for item in existing_additional if item != name
            ]
            concept_sets = protocol.get("concept_sets", {})
            protocol["concept_sets"] = {key: value for key, value in concept_sets.items() if key != name}
            _atomic_write_json(protocol_path, protocol)

            if (run_dir / "hypothesis.json").is_file():
                self._write_hypothesis(run_id, protocol)
        detail = f"'{name}' removed by {reviewer} as a required population concept set"
        if comment and comment.strip():
            detail += f" ({comment.strip()})"
        state.record("population_concept_set_removed", detail)
        self._invalidate_gate(state, "clinical_mapping", f"additional population concept set '{name}' removed")
        self._invalidate_gate(state, "analysis_plan", f"additional population concept set '{name}' removed")
        self.save(state)
        return state

    def propose_negative_controls(
        self,
        run_id: str,
        llm: LLMCallable,
        *,
        prompt_path: Path = DEFAULT_NEGATIVE_CONTROL_PROMPT_PATH,
    ) -> RunState:
        """LLM-assisted negative-control candidate proposal (`docs.
        causal_roadmap` milestone 6, `causal.queries.negative_controls`).

        Clinical label + rationale only, never an OMOP concept -- concept
        mapping and approval are the separate, human-reviewed
        `map_negative_control` step below, the same "propose a label, map
        concepts separately" split `candidate_dag` covariates already use.
        Every candidate this writes has `mapping_status=
        "candidate_pending_human_approval"`; nothing in `causal.queries.
        negative_controls.parse_negative_control_response` can ever mark one
        approved, no matter what the raw LLM response claims (see that
        function's docstring) -- this call cannot make a negative control
        queryable by itself.

        Folds into the existing `clinical_mapping` gate/`concept_mapping`
        phase rather than a new gate, same as `map_concept_set`; only
        requires `hypothesis.json` to already exist, not a specific phase,
        so a reviewer can propose/re-propose at any point a mapping review
        is still open.
        """
        state = self.load(run_id)
        run_dir = self.runs_dir / run_id
        protocol_path = run_dir / "protocol.json"
        if not (run_dir / "hypothesis.json").is_file():
            raise RuntimeError("Run the Hypothesis Agent before proposing negative controls")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        concept_sets = protocol.get("concept_sets", {})
        # A `concept_sets` entry's clinical label is its dict key (e.g.
        # "proton_pump_inhibitor"), not a separate field.
        target = {
            role: next(
                (name for name, entry in concept_sets.items()
                 if isinstance(entry, dict) and entry.get("role") == role),
                "",
            )
            for role in QUERY_ROLES
        }
        prompt = build_negative_control_prompt(prompt_path, target)
        raw = llm(prompt)
        candidates = parse_negative_control_response(raw)

        with _RunLock(run_dir):
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            existing = protocol.get("negative_controls", {})
            already_approved = {
                control_id for control_id, entry in existing.items()
                if isinstance(entry, dict) and entry.get("mapping_status") == "approved"
            }
            updated = dict(existing)
            for candidate in candidates:
                if candidate["control_id"] in already_approved:
                    # Never overwrite an already-approved candidate with a
                    # fresh, unreviewed proposal of the same slugified label.
                    continue
                updated[candidate["control_id"]] = candidate
            protocol["negative_controls"] = updated
            _atomic_write_json(protocol_path, protocol)

            state = self.load(run_id)
            state.record(
                "negative_controls_proposed",
                f"{len(candidates)} candidate(s) proposed: "
                f"{', '.join(candidate['control_id'] for candidate in candidates)}",
            )
            self.save(state)
        return state

    def map_negative_control(
        self,
        run_id: str,
        control_id: str,
        domain: str,
        concept_ids: list[int],
        reviewer: str,
        status: str = "approved",
        comment: str | None = None,
        concept_lookup: OMCPClient | None = None,
    ) -> RunState:
        """Human review of one negative-control candidate: map it to a real
        OMOP domain/concept(s) and approve or reject it in one action --
        the same shape as `map_concept_set` (this is the one place a
        negative control actually becomes approved; `propose_negative_
        controls` above never can). `concept_lookup` is the same optional,
        opt-in domain cross-check `map_concept_set`/`map_covariate_concepts`
        already support.
        """
        control_id = control_id.strip()
        if not control_id:
            raise ValueError("control_id is required")
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        if not isinstance(status, str) or not status.strip():
            raise ValueError("status is required")
        status = status.strip()
        try:
            concept_ids = [int(concept_id) for concept_id in concept_ids]
        except (TypeError, ValueError):
            raise ValueError("concept_ids must be a list of integers") from None

        state = self.load(run_id)
        run_dir = self.runs_dir / run_id
        protocol_path = run_dir / "protocol.json"

        existing_entry = json.loads(protocol_path.read_text(encoding="utf-8")).get(
            "negative_controls", {}
        ).get(control_id)
        if not isinstance(existing_entry, dict):
            raise RuntimeError(f"negative_controls has no candidate named '{control_id}'")
        if domain not in DOMAIN_TABLES:
            raise ValueError(f"domain must be one of {tuple(DOMAIN_TABLES)}")
        if status == "approved" and not concept_ids:
            raise ValueError("status='approved' requires at least one concept_id")

        if concept_lookup is not None and concept_ids:
            response = concept_lookup.get_concepts_by_id(concept_ids)
            mismatches = _mismatched_concept_domains(response.text, concept_ids, domain)
            if mismatches:
                raise ValueError(
                    f"Concept domain check failed for negative_controls['{control_id}']: "
                    + "; ".join(mismatches)
                )

        with _RunLock(run_dir):
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            negative_controls = protocol.get("negative_controls", {})
            entry = negative_controls.get(control_id)
            if not isinstance(entry, dict):
                raise RuntimeError(f"negative_controls has no candidate named '{control_id}'")
            before = {
                "domain": entry.get("domain", ""),
                "concept_ids": entry.get("concept_ids", []),
                "mapping_status": entry.get("mapping_status", "candidate_pending_human_approval"),
            }
            updated_entry = {
                **entry,
                "domain": domain,
                "concept_ids": concept_ids,
                "mapping_status": status,
                "reviewer": reviewer.strip(),
                "provenance": "omcp_candidate" if concept_lookup is not None else entry.get("provenance", "llm_proposed"),
            }
            # Validated the same way every other registry entry is --
            # raises before anything is persisted if e.g. status="approved"
            # with a provenance in `_UNSUPERVISED_PROVENANCE` and no
            # reviewer (cannot happen here since reviewer is required
            # above, but this keeps exactly one validating code path).
            try:
                variable_from_negative_control_proposal(updated_entry)
            except ValueError as exc:
                raise ValueError(f"Cannot map negative control '{control_id}': {exc}") from exc
            protocol["negative_controls"] = {**negative_controls, control_id: updated_entry}
            _atomic_write_json(protocol_path, protocol)

        detail = f"{control_id}: domain={domain}, {len(concept_ids)} concept(s), status={status} by {reviewer}"
        if comment and comment.strip():
            detail += f" ({comment.strip()})"
        state.record(
            "negative_control_mapped", detail,
            changes={"before": before, "after": {"domain": domain, "concept_ids": concept_ids, "status": status}},
        )
        self.save(state)
        return state

    def review_covariate(
        self,
        run_id: str,
        label: str,
        decision: str,
        reviewer: str,
        comment: str | None = None,
        *,
        role: str | None = None,
        temporal_classification: str | None = None,
        pmid: str = "",
        observed: bool = True,
        certainty: str = "established",
        domain: str | None = None,
    ) -> RunState:
        """Record a human accept/reject decision for a pending first-pass covariate.

        "Accept" adds a role-classified `candidate_dag` entry (same shape the
        LLM/literature stage produces, minus a pmid); "reject" records an
        audited exclusion instead of leaving the covariate silently absent.
        Either way, `hypothesis.json` is regenerated so the decision is
        immediately reflected in `assess_identification`.

        `domain` is optional, unlike `role`/`temporal_classification`: a
        covariate can be fully DAG-reviewed before its OMOP domain is known,
        and today's callers overwhelmingly never set it. When given, it lets
        `DataScienceAgent` attach reviewable OMOP concept candidates to the
        resulting covariate (see `causal.agents.hypothesis.DAGVariable.domain`).

        Re-accepting an already-accepted label (reconsidering its role/
        temporal_classification/rationale) preserves whatever `concept_ids`/
        `mapping_status`/`aggregation`/`window_start_hours`/`window_end_hours`
        a prior `map_covariate_concepts`/`review_covariate_aggregation`/
        `review_covariate_window` call already recorded for it -- those are
        separate review actions (see their own docstrings) and this method
        must not silently discard that work just because it only received
        role/temporal_classification/rationale again. The one exception is
        `domain` actually changing to a different, non-empty value: mapping
        and aggregation are domain-specific (concept_ids were validated
        against the old domain; aggregation choices come from
        `COVARIATE_AGGREGATIONS_BY_DOMAIN`), so a real domain change resets
        them rather than carrying over values that no longer make sense.
        """
        if decision not in {"accept", "reject"}:
            raise ValueError("decision must be accept or reject")
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        label = label.strip()
        if not label:
            raise ValueError("label is required")

        state = self.load(run_id)
        run_dir = self.runs_dir / run_id
        protocol_path = run_dir / "protocol.json"
        if not (run_dir / "hypothesis.json").is_file():
            raise RuntimeError("Run the Hypothesis Agent before reviewing covariates")

        with _RunLock(run_dir):
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

            previous_entry = next(
                (item for item in protocol.get("candidate_dag", ()) if item.get("label") == label), None
            )
            candidate_dag = [item for item in protocol.get("candidate_dag", ()) if item.get("label") != label]
            rejected_covariates = [
                item for item in protocol.get("rejected_covariates", ()) if item.get("label") != label
            ]

            if decision == "accept":
                if role not in DAG_VARIABLE_ROLES:
                    raise ValueError(f"role must be one of {DAG_VARIABLE_ROLES}")
                if temporal_classification not in DAG_VARIABLE_TEMPORAL_CLASSES:
                    raise ValueError(
                        f"temporal_classification must be one of {DAG_VARIABLE_TEMPORAL_CLASSES}"
                    )
                if domain is not None and domain not in DAG_VARIABLE_DOMAINS:
                    raise ValueError(f"domain must be one of {DAG_VARIABLE_DOMAINS}")
                if not comment or not comment.strip():
                    raise ValueError("comment is required as the rationale when accepting a covariate")
                previous_domain = (previous_entry or {}).get("domain") or ""
                resolved_domain = domain if domain is not None else previous_domain
                domain_changed = bool(previous_domain) and resolved_domain != previous_domain
                new_entry = {
                    "label": label,
                    "role": role,
                    "temporal_classification": temporal_classification,
                    "rationale": comment.strip(),
                    "pmid": pmid,
                    "observed": observed,
                    "certainty": certainty,
                    "domain": resolved_domain,
                }
                if previous_entry and not domain_changed:
                    for field in (
                        "concept_ids", "mapping_status", "aggregation",
                        "window_start_hours", "window_end_hours",
                    ):
                        if field in previous_entry:
                            new_entry[field] = previous_entry[field]
                candidate_dag.append(new_entry)
            else:
                new_entry = {
                    "label": label,
                    "reviewer": reviewer.strip(),
                    "comment": (comment or "").strip(),
                    "decided_at": datetime.now(timezone.utc).isoformat(),
                }
                rejected_covariates.append(new_entry)

            protocol["candidate_dag"] = candidate_dag
            protocol["rejected_covariates"] = rejected_covariates
            _atomic_write_json(protocol_path, protocol)

            self._write_hypothesis(run_id, protocol)
        state.record(
            "covariate_reviewed", f"{label}={decision} by {reviewer}",
            changes={"before": previous_entry, "after": new_entry},
        )
        self._invalidate_gate(state, "clinical_mapping", f"covariate '{label}' {decision}ed")
        self._invalidate_gate(state, "analysis_plan", f"covariate '{label}' {decision}ed")
        self.save(state)
        return state

    def review_covariate_domain(
        self,
        run_id: str,
        label: str,
        domain: str,
        reviewer: str,
        comment: str | None = None,
    ) -> RunState:
        """Record a human's confirmation or correction of an already-classified
        covariate's OMOP domain.

        The DAG-drafting LLM (`causal/prompts/dag.md`) now proposes a
        `domain` alongside `role`/`temporal_classification`, but that
        proposal is exactly as provisional as the rest of the draft --
        `DataScienceAgent` only attempts a concept lookup once a domain is
        set (see `causal.agents.hypothesis.DAGVariable.domain`), so this is
        the general path to set or fix it for *any* already-accepted
        covariate, not only the ones a human classified from scratch via
        `review_covariate` (which also accepts a `domain`, but only at
        accept time). Distinct from `review_covariate` on purpose, same
        family as `review_covariate_aggregation`/`review_covariate_window`:
        it does not require re-supplying role/temporal_classification/
        rationale just to confirm or fix a domain.

        Changing `domain` to a value different from what was previously set
        invalidates whatever `concept_ids`/`mapping_status`/`aggregation` a
        prior `map_covariate_concepts`/`review_covariate_aggregation` call
        recorded (they were validated against the old domain) -- reset to
        their unset defaults rather than carrying over values that no
        longer make sense. `window_start_hours`/`window_end_hours` are left
        untouched: the ascertainment window is derived from
        `temporal_classification`, not `domain`.
        """
        label = label.strip()
        if not label:
            raise ValueError("label is required")
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        if domain not in DAG_VARIABLE_DOMAINS:
            raise ValueError(f"domain must be one of {DAG_VARIABLE_DOMAINS}")

        state = self.load(run_id)
        run_dir = self.runs_dir / run_id
        protocol_path = run_dir / "protocol.json"
        if not (run_dir / "hypothesis.json").is_file():
            raise RuntimeError("Run the Hypothesis Agent before reviewing a covariate domain")

        with _RunLock(run_dir):
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

            candidate_dag = protocol.get("candidate_dag", ())
            entry = next((item for item in candidate_dag if item.get("label") == label), None)
            if entry is None:
                raise RuntimeError(
                    f"'{label}' has no candidate_dag entry yet -- accept it via review_covariate first"
                )
            previous_domain = entry.get("domain") or ""
            updated_entry = {**entry, "domain": domain}
            if previous_domain and previous_domain != domain:
                updated_entry["concept_ids"] = []
                updated_entry["mapping_status"] = "unmapped"
                updated_entry["aggregation"] = "definition_pending"
            protocol["candidate_dag"] = [
                updated_entry if item.get("label") == label else item for item in candidate_dag
            ]
            _atomic_write_json(protocol_path, protocol)

            self._write_hypothesis(run_id, protocol)
        detail = f"{label}: domain={domain} by {reviewer}"
        if comment and comment.strip():
            detail += f" ({comment.strip()})"
        state.record(
            "covariate_domain_reviewed", detail,
            changes={"before": {"domain": previous_domain}, "after": {"domain": domain}},
        )
        self._invalidate_gate(state, "clinical_mapping", f"covariate '{label}' domain changed")
        self._invalidate_gate(state, "analysis_plan", f"covariate '{label}' domain changed")
        self.save(state)
        return state

    def map_covariate_concepts(
        self,
        run_id: str,
        label: str,
        concept_ids: list[int],
        reviewer: str,
        mapping_status: str = "approved",
        comment: str | None = None,
        concept_lookup: OMCPClient | None = None,
    ) -> RunState:
        """Record a human's chosen OMOP concept(s) for an already-classified covariate.

        Distinct from `review_covariate` on purpose: classifying a
        covariate's causal role (DAG review) and approving which OMOP
        concept represents it (concept-mapping review) are different review
        actions, often at different times, once `concept_candidates` exist
        to review at all -- bundling them would force re-supplying
        role/temporal_classification/rationale just to record a concept_id.
        The `candidate_dag` entry must already exist (i.e. the covariate was
        already accepted via `review_covariate`); every field but
        `concept_ids`/`mapping_status`/`mapping_reviewer`/`provenance` is
        left untouched.

        `concept_lookup` (optional) verifies each approved concept_id's real
        OMOP `domain_id` actually matches the covariate's declared `domain`
        before writing -- a reviewer can type a concept_id directly (not
        just pick one of the search results), so nothing else confirms it
        belongs to the expected domain. Skipped entirely when omitted, same
        as `run_data_science_agent`'s `concept_lookup`: this method stays
        filesystem-only/offline-testable by default. Whether it was passed
        also decides the `provenance` written onto the entry
        (`"omcp_candidate"` vs. whatever provenance the entry already
        carried) -- see `causal.mapping.registry._UNSUPERVISED_PROVENANCE`.
        """
        label = label.strip()
        if not label:
            raise ValueError("label is required")
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        if not isinstance(mapping_status, str) or not mapping_status.strip():
            raise ValueError("mapping_status is required")
        try:
            concept_ids = [int(concept_id) for concept_id in concept_ids]
        except (TypeError, ValueError):
            raise ValueError("concept_ids must be a list of integers") from None
        if mapping_status == "approved" and not concept_ids:
            raise ValueError("mapping_status='approved' requires at least one concept_id")

        state = self.load(run_id)
        run_dir = self.runs_dir / run_id
        protocol_path = run_dir / "protocol.json"
        if not (run_dir / "hypothesis.json").is_file():
            raise RuntimeError("Run the Hypothesis Agent before mapping covariate concepts")

        if concept_lookup is not None and mapping_status == "approved":
            # Read-only peek at the declared domain, deliberately outside
            # `_RunLock`: `concept_lookup.get_concepts_by_id` spawns an OMCP
            # subprocess (can take seconds), and the lock exists to guard
            # only the fast on-disk read-modify-write below, not a network
            # round trip. A benign race here (the domain changing between
            # this peek and the locked write) only makes this advisory
            # check stale -- the actual write is still fully protected by
            # the lock either way.
            declared_domain = next(
                (
                    item.get("domain")
                    for item in json.loads(protocol_path.read_text(encoding="utf-8")).get("candidate_dag", ())
                    if item.get("label") == label
                ),
                None,
            )
            if declared_domain:
                response = concept_lookup.get_concepts_by_id(concept_ids)
                mismatches = _mismatched_concept_domains(response.text, concept_ids, declared_domain)
                if mismatches:
                    raise ValueError(
                        f"Concept domain check failed for '{label}': " + "; ".join(mismatches)
                    )

        with _RunLock(run_dir):
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

            candidate_dag = protocol.get("candidate_dag", ())
            entry = next((item for item in candidate_dag if item.get("label") == label), None)
            if entry is None:
                raise RuntimeError(
                    f"'{label}' has no candidate_dag entry yet -- accept it via review_covariate first"
                )
            # `mapping_reviewer` is what `causal.mapping.registry.
            # variable_from_candidate_dag_entry` reads back out (see
            # `entry.get("mapping_reviewer", "")`) to populate
            # `OMOPVariableDefinition.reviewer` -- without writing it here,
            # that field (and the module's "an automated-provenance mapping
            # may never be approved without a named human reviewer" check)
            # stayed permanently blank for every real covariate this method
            # ever approved, even though `reviewer` was already required and
            # validated as an argument above.
            #
            # `provenance` mirrors `run_data_science_agent`'s own
            # `source = "omcp_candidates" if concept_lookup is not None
            # else ...` distinction: a live `concept_lookup` cross-check is
            # this codebase's one signal that a concept_id was OMCP-vetted
            # rather than purely human-supplied. Without writing this,
            # `causal.mapping.registry`'s `_UNSUPERVISED_PROVENANCE` check
            # (`provenance in ("llm_proposed", "omcp_candidate")` requires a
            # named reviewer) could never actually fire for a real
            # covariate -- every entry silently fell back to the adapter's
            # `"protocol_declared"` default regardless of how the mapping
            # was really produced, leaving that invariant a dead check (not
            # exploitable today only because `reviewer` above is already
            # unconditionally required). Falls back to whatever provenance
            # the entry already carried (not a blind reset to
            # "protocol_declared") so a legitimate prior value survives a
            # later re-review that happens not to use `concept_lookup`.
            updated_entry = {
                **entry,
                "concept_ids": concept_ids,
                "mapping_status": mapping_status.strip(),
                "mapping_reviewer": reviewer.strip(),
                "provenance": (
                    "omcp_candidate" if concept_lookup is not None
                    else entry.get("provenance", "protocol_declared")
                ),
            }
            protocol["candidate_dag"] = [
                updated_entry if item.get("label") == label else item for item in candidate_dag
            ]
            _atomic_write_json(protocol_path, protocol)

            self._write_hypothesis(run_id, protocol)
        detail = f"{label}: {len(concept_ids)} concept(s), status={mapping_status} by {reviewer}"
        if comment and comment.strip():
            detail += f" ({comment.strip()})"
        state.record(
            "covariate_concepts_mapped", detail,
            changes={
                "before": {
                    "concept_ids": entry.get("concept_ids", []),
                    "mapping_status": entry.get("mapping_status", "unmapped"),
                },
                "after": {"concept_ids": concept_ids, "mapping_status": mapping_status.strip()},
            },
        )
        self._invalidate_gate(state, "clinical_mapping", f"covariate '{label}' concept mapping changed")
        self._invalidate_gate(state, "analysis_plan", f"covariate '{label}' concept mapping changed")
        self.save(state)
        return state

    def review_covariate_aggregation(
        self,
        run_id: str,
        label: str,
        aggregation: str,
        reviewer: str,
        comment: str | None = None,
    ) -> RunState:
        """Record a human's chosen aggregation rule for an already-mapped covariate.

        Distinct from `map_covariate_concepts` on purpose, same as that
        method is distinct from `review_covariate`: knowing which OMOP
        concept(s) a covariate maps to answers "which rows"; this answers
        "how to combine repeated occurrences of them into one column"
        (see `causal.agents.hypothesis.DAGVariable.aggregation`) -- a
        separate review action, often at a different time. The
        `candidate_dag` entry must already exist and already have a
        `domain` set (aggregation choices are domain-specific -- see
        `COVARIATE_AGGREGATIONS_BY_DOMAIN`); every field but `aggregation`
        is left untouched.
        """
        label = label.strip()
        if not label:
            raise ValueError("label is required")
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        if not isinstance(aggregation, str) or not aggregation.strip():
            raise ValueError("aggregation is required")
        aggregation = aggregation.strip()

        state = self.load(run_id)
        run_dir = self.runs_dir / run_id
        protocol_path = run_dir / "protocol.json"
        if not (run_dir / "hypothesis.json").is_file():
            raise RuntimeError("Run the Hypothesis Agent before reviewing covariate aggregation")

        with _RunLock(run_dir):
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

            candidate_dag = protocol.get("candidate_dag", ())
            entry = next((item for item in candidate_dag if item.get("label") == label), None)
            if entry is None:
                raise RuntimeError(
                    f"'{label}' has no candidate_dag entry yet -- accept it via review_covariate first"
                )
            domain = entry.get("domain") or ""
            allowed = COVARIATE_AGGREGATIONS_BY_DOMAIN.get(domain)
            if not allowed:
                raise RuntimeError(
                    f"'{label}' has no OMOP domain set yet -- set one via review_covariate first"
                )
            if aggregation not in allowed:
                raise ValueError(f"aggregation for domain '{domain}' must be one of {allowed}")

            updated_entry = {**entry, "aggregation": aggregation}
            protocol["candidate_dag"] = [
                updated_entry if item.get("label") == label else item for item in candidate_dag
            ]
            _atomic_write_json(protocol_path, protocol)

            self._write_hypothesis(run_id, protocol)
        detail = f"{label}: aggregation={aggregation} by {reviewer}"
        if comment and comment.strip():
            detail += f" ({comment.strip()})"
        state.record(
            "covariate_aggregation_reviewed", detail,
            changes={
                "before": {"aggregation": entry.get("aggregation", "definition_pending")},
                "after": {"aggregation": aggregation},
            },
        )
        self._invalidate_gate(state, "clinical_mapping", f"covariate '{label}' aggregation changed")
        self._invalidate_gate(state, "analysis_plan", f"covariate '{label}' aggregation changed")
        self.save(state)
        return state

    def review_covariate_window(
        self,
        run_id: str,
        label: str,
        window_start_hours: int,
        window_end_hours: int,
        reviewer: str,
        comment: str | None = None,
    ) -> RunState:
        """Record a human's override of the auto-derived ascertainment window
        for an already-classified covariate.

        By default `causal.agents.data_science.DataScienceAgent._covariate`
        derives the window purely from `temporal_classification` (wide if
        `pre_treatment`, a narrow `-24h..0` otherwise) -- there was no way
        for a reviewer to correct it for a covariate that genuinely needs a
        different window (e.g. a lab value only meaningfully defined over
        the first 6h of an ICU stay). The `candidate_dag` entry must already
        exist; every field but `window_start_hours`/`window_end_hours` is
        left untouched. `window_end_hours <= 0` is enforced -- a baseline
        covariate window cannot extend past the index event without
        reintroducing the immortal-time bias this pipeline's grace-period
        design already exists to avoid.
        """
        label = label.strip()
        if not label:
            raise ValueError("label is required")
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        try:
            window_start_hours = int(window_start_hours)
            window_end_hours = int(window_end_hours)
        except (TypeError, ValueError):
            raise ValueError("window_start_hours and window_end_hours must be integers") from None
        if not window_start_hours <= window_end_hours <= 0:
            raise ValueError(
                "window must satisfy window_start_hours <= window_end_hours <= 0"
            )

        state = self.load(run_id)
        run_dir = self.runs_dir / run_id
        protocol_path = run_dir / "protocol.json"
        if not (run_dir / "hypothesis.json").is_file():
            raise RuntimeError("Run the Hypothesis Agent before reviewing a covariate window")

        with _RunLock(run_dir):
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

            candidate_dag = protocol.get("candidate_dag", ())
            entry = next((item for item in candidate_dag if item.get("label") == label), None)
            if entry is None:
                raise RuntimeError(
                    f"'{label}' has no candidate_dag entry yet -- accept it via review_covariate first"
                )
            updated_entry = {
                **entry, "window_start_hours": window_start_hours, "window_end_hours": window_end_hours,
            }
            protocol["candidate_dag"] = [
                updated_entry if item.get("label") == label else item for item in candidate_dag
            ]
            _atomic_write_json(protocol_path, protocol)

            self._write_hypothesis(run_id, protocol)
        detail = f"{label}: window={window_start_hours}h..{window_end_hours}h by {reviewer}"
        if comment and comment.strip():
            detail += f" ({comment.strip()})"
        state.record(
            "covariate_window_reviewed", detail,
            changes={
                "before": {
                    "window_start_hours": entry.get("window_start_hours"),
                    "window_end_hours": entry.get("window_end_hours"),
                },
                "after": {"window_start_hours": window_start_hours, "window_end_hours": window_end_hours},
            },
        )
        self._invalidate_gate(state, "clinical_mapping", f"covariate '{label}' window changed")
        self._invalidate_gate(state, "analysis_plan", f"covariate '{label}' window changed")
        self.save(state)
        return state

    def _block_if_target_trial_incomplete(self, run_id: str) -> None:
        """The `target_trial` gate (roadmap stage 1: question and target
        trial) cannot be approved unless `protocol["target_trial"]`
        constructs a valid `roadmap.CausalQuestion`/`TargetTrialProtocol` --
        works directly off `protocol.json` alone, before the Hypothesis
        Agent or any DAG stage has necessarily run, since target-trial
        completeness is scientifically prior to drafting a DAG (see
        `docs/causal_roadmap.md`'s normative stage order). Reuses those
        dataclasses' own validation (see `causal.agents.dag_identification.
        build_study_specification`, which does the same field mapping once
        `hypothesis.json` exists, for the fuller `CausalStudySpecification`).
        """
        protocol_path = self.runs_dir / run_id / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        target = protocol.get("target_trial")
        if not isinstance(target, dict):
            raise ValueError("Cannot approve the target_trial gate: protocol.json has no target_trial object")
        try:
            CausalQuestion(
                population=str(target.get("population", "")),
                treatment=str(target.get("treatment", "")),
                comparator=str(target.get("comparator", "")),
                outcome=str(target.get("outcome", "")),
            )
            TargetTrialProtocol(
                eligibility_criteria=tuple(str(item) for item in target.get("eligibility_criteria") or ()),
                treatment_strategy=str(target.get("treatment", "")),
                comparator_strategy=str(target.get("comparator", "")),
                assignment_procedure=str(target.get("assignment_procedure") or ""),
                time_zero=str(target.get("time_zero", "")),
                grace_period_hours=target.get("grace_period_hours"),
                follow_up_days=target.get("follow_up_days"),
                outcome_definition=str(target.get("outcome", "")),
                censoring_strategy=str(target.get("censoring_strategy") or ""),
                analysis_strategy=str(target.get("analysis_strategy") or ""),
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Cannot approve the target_trial gate: {exc}") from exc

    def set_target_trial_estimand(
        self,
        run_id: str,
        *,
        primary_estimand: str,
        secondary_estimands: tuple[str, ...] = (),
        reviewer: str,
        comment: str,
    ) -> RunState:
        """Let a reviewer directly correct `target_trial.primary_estimand`/
        `secondary_estimands` in place, for when the LLM draft got the
        estimand wrong and the reviewer already knows the fix -- an
        alternative to `_render_target_trial_regenerate`'s LLM round-trip
        (streamlit_app.py), which is slower and, being a fresh LLM call, not
        guaranteed to land on the same correction twice.

        Scoped to just these two fields, not the full `target_trial` object:
        population/treatment/comparator/outcome free text also seeds
        `literature_search` terms and `first_pass_covariates` (see
        `causal.agents.user_question.UserQuestionHypothesisService.generate`),
        so hand-editing those here would silently desynchronize
        already-generated downstream artifacts in a way the estimand fields,
        which nothing else derives from at this stage, do not.

        `primary_estimand` is restricted to `{"ATE", "ATT", "CATE"}` --
        looser at the LLM-drafting stage (`REQUIRED_TARGET_TRIAL_FIELDS` only
        checks non-empty), but this is exactly the set `roadmap.
        CausalEstimand.__post_init__` requires later at the
        `identification_estimand` gate, so accepting anything looser here
        would just defer the same rejection.

        Only allowed while `target_trial` is not yet `approved`: once
        approved, `dag_review` and later stages may already be under way, and
        retroactively changing the estimand would need a wider invalidation
        cascade than the single-gate reset `_invalidate_gate` does here
        (regenerate a new run instead, or reject the gate first).
        """
        if primary_estimand not in {"ATE", "ATT", "CATE"}:
            raise ValueError("primary_estimand must be one of ATE, ATT, CATE")
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        if not comment.strip():
            raise ValueError("comment is required to record why this was corrected")
        secondary_estimands = tuple(str(item).strip() for item in secondary_estimands if str(item).strip())

        run_dir = self.runs_dir / run_id
        protocol_path = run_dir / "protocol.json"
        state = self.load(run_id)
        if state.gates["target_trial"].status == "approved":
            raise ValueError(
                "Cannot manually edit the estimand after the target_trial gate is approved -- "
                "reject it first, or regenerate a new run."
            )

        with _RunLock(run_dir):
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            target = protocol.get("target_trial")
            if not isinstance(target, dict):
                raise ValueError("Cannot correct the estimand: protocol.json has no target_trial object")
            before = {
                "primary_estimand": target.get("primary_estimand"),
                "secondary_estimands": target.get("secondary_estimands"),
            }
            target["primary_estimand"] = primary_estimand
            target["secondary_estimands"] = list(secondary_estimands)
            _append_correction_log(
                protocol,
                kind="target_trial_estimand",
                reviewer=reviewer,
                comment=comment.strip(),
                summary=f"primary_estimand -> {primary_estimand}",
                detail={
                    "before": before,
                    "after": {
                        "primary_estimand": primary_estimand,
                        "secondary_estimands": list(secondary_estimands),
                    },
                },
            )
            _atomic_write_json(protocol_path, protocol)

        state.record(
            "target_trial_estimand_corrected",
            f"primary_estimand set to {primary_estimand} by {reviewer}: {comment.strip()}",
            changes={
                "before": before,
                "after": {"primary_estimand": primary_estimand, "secondary_estimands": list(secondary_estimands)},
            },
        )
        self._invalidate_gate(state, "target_trial", f"estimand manually corrected by {reviewer}")
        self.save(state)
        return state

    def set_target_trial_timing(
        self,
        run_id: str,
        *,
        time_zero: str | None = None,
        grace_period_hours: int | None = None,
        follow_up_days: int | None = None,
        reviewer: str,
        comment: str,
    ) -> RunState:
        """Let a reviewer directly correct `target_trial.time_zero` /
        `grace_period_hours` / `follow_up_days` in place, for when the LLM
        draft got a date or a follow-up window wrong and the reviewer already
        knows the fix -- the targeted counterpart of
        `_render_target_trial_regenerate` (`causal/ui/streamlit_app.py`), which
        discards the whole run and every review already done on it.

        Scoped to just these three timing fields, on the same rationale
        `set_target_trial_estimand` documents: `population`/`treatment`/
        `comparator`/`outcome` free text also seeds `literature_search` terms
        and `first_pass_covariates`, so hand-editing those here would silently
        desynchronize already-generated downstream artifacts; the timing
        fields feed the cohort/censoring machinery but nothing upstream of the
        DAG derives *from* them.

        Only the fields passed non-`None` change; at least one is required.
        `grace_period_hours`/`follow_up_days` are checked with the same rule
        `causal.agents.hypothesis.HypothesisAgent._positive_int` applies later
        (`grace_period_hours >= 0`, `follow_up_days >= 1`), so a value that
        would otherwise only be rejected at the `dag_review` gate is rejected
        here instead.

        Invalidates `target_trial` *and* `analysis_plan` /
        `statistical_estimand`: `follow_up_days` is read straight into
        `cohort.sql`, the censoring definition, and the feasibility snapshot
        (`observation_period_covers_follow_up`, `known_death_within_follow_up`),
        so a run that already reached those phases must re-review them.
        `hypothesis.json` is refreshed when it exists so `assess_identification`
        and every `hypothesis["follow_up_days"]` reader see the corrected value
        immediately, mirroring `review_covariate`.

        Only allowed while `target_trial` is not yet `approved` (reject it
        first, or regenerate a new run) -- same guard as
        `set_target_trial_estimand`.
        """
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        if not comment.strip():
            raise ValueError("comment is required to record why this was corrected")
        updates: dict[str, object] = {}
        if time_zero is not None:
            if not isinstance(time_zero, str) or not time_zero.strip():
                raise ValueError("time_zero must be a non-empty string")
            updates["time_zero"] = time_zero.strip()
        if grace_period_hours is not None:
            if (
                not isinstance(grace_period_hours, int)
                or isinstance(grace_period_hours, bool)
                or grace_period_hours < 0
            ):
                raise ValueError("grace_period_hours must be an integer >= 0")
            updates["grace_period_hours"] = grace_period_hours
        if follow_up_days is not None:
            if (
                not isinstance(follow_up_days, int)
                or isinstance(follow_up_days, bool)
                or follow_up_days < 1
            ):
                raise ValueError("follow_up_days must be an integer >= 1")
            updates["follow_up_days"] = follow_up_days
        if not updates:
            raise ValueError("Pass at least one of time_zero, grace_period_hours, follow_up_days")

        run_dir = self.runs_dir / run_id
        protocol_path = run_dir / "protocol.json"
        state = self.load(run_id)
        if state.gates["target_trial"].status == "approved":
            raise ValueError(
                "Cannot manually edit target-trial timing after the target_trial gate is "
                "approved -- reject it first, or regenerate a new run."
            )

        with _RunLock(run_dir):
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            target = protocol.get("target_trial")
            if not isinstance(target, dict):
                raise ValueError(
                    "Cannot correct target-trial timing: protocol.json has no target_trial object"
                )
            before = {field: target.get(field) for field in updates}
            target.update(updates)
            summary = ", ".join(f"{field}={value}" for field, value in updates.items())
            _append_correction_log(
                protocol,
                kind="target_trial_timing",
                reviewer=reviewer,
                comment=comment.strip(),
                summary=summary,
                detail={"before": before, "after": dict(updates)},
            )
            _atomic_write_json(protocol_path, protocol)
            if (run_dir / "hypothesis.json").is_file():
                self._write_hypothesis(run_id, protocol)

        state.record(
            "target_trial_timing_corrected",
            f"{summary} by {reviewer}: {comment.strip()}",
            changes={"before": before, "after": dict(updates)},
        )
        for gate in ("target_trial", "analysis_plan", "statistical_estimand"):
            self._invalidate_gate(state, gate, f"target-trial timing corrected by {reviewer}")
        self.save(state)
        return state

    def revise_candidate_dag_from_points(
        self,
        run_id: str,
        points: Sequence[str],
        llm: LLMCallable,
        reviewer: str,
        *,
        comment: str = "",
        prompt_path: Path = DEFAULT_DAG_POINT_REVISION_PROMPT_PATH,
    ) -> RunState:
        """Apply a reviewer's free-text DAG correction points as targeted
        edits, instead of regenerating the whole run.

        The `llm` here is a *formatter* (see
        `causal.agents.dag_point_revision`): it maps each point to an
        `add` / `reclassify` / `remove` operation. The reviewer is the
        grounding, so added/reclassified variables are expert-asserted
        (`pmid = ""`), exactly as `review_covariate` records a hand-added
        covariate -- this path is not the literature-grounded
        `_critique_and_revise_dag` loop and does not go through its
        citation allow-list.

        The revised `candidate_dag` is still validated by
        `HypothesisAgent().run(...)` (role/temporal legality, cycles,
        duplicate labels, window bounds) *before* anything is written: a
        revision that would not construct a valid graph raises and leaves
        `protocol.json` untouched. On success, `hypothesis.json` is
        refreshed and `dag_review` / `clinical_mapping` / `analysis_plan`
        are re-opened (a covariate reject/reclassify changes the adjustment
        set and every downstream artifact derived from it).

        Never blocks on the `dag_review` gate being approved -- unlike the
        target-trial timing/estimand fixes, a DAG correction after approval
        is the normal case (the reviewer spotted a missing confounder while
        mapping concepts); the gate re-opening is the safety mechanism.
        """
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        cleaned_points = [str(point).strip() for point in points if str(point).strip()]
        if not cleaned_points:
            raise ValueError("At least one non-empty correction point is required")
        if len(cleaned_points) > _MAX_DAG_REVISION_POINTS:
            raise ValueError(f"At most {_MAX_DAG_REVISION_POINTS} correction points at a time")

        run_dir = self.runs_dir / run_id
        protocol_path = run_dir / "protocol.json"
        state = self.load(run_id)
        if not (run_dir / "hypothesis.json").is_file():
            raise RuntimeError("Run the Hypothesis Agent before revising the DAG")

        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        target = protocol.get("target_trial")
        if not isinstance(target, dict):
            raise ValueError("Cannot revise the DAG: protocol.json has no target_trial object")
        current_dag = list(protocol.get("candidate_dag", []) or [])
        prompt = build_dag_point_revision_prompt(
            prompt_path,
            target=target,
            current_dag_text=format_current_dag(current_dag),
            points=cleaned_points,
        )
        revision = parse_dag_point_revision_response(llm(prompt))
        now = datetime.now(timezone.utc).isoformat()
        updated_protocol, apply_notes = apply_dag_point_operations(
            protocol, revision, reviewer.strip(), now=now
        )

        # Validate the revised graph before persisting anything.
        try:
            HypothesisAgent().run(updated_protocol)
        except ValueError as exc:
            raise ValueError(
                f"The revised DAG does not construct a valid graph: {exc}. "
                "Rephrase the correction point(s); nothing was changed."
            ) from exc

        before_labels = sorted(
            str(entry.get("label", "")).strip() for entry in current_dag if entry.get("label")
        )
        after_labels = sorted(
            str(entry.get("label", "")).strip()
            for entry in updated_protocol.get("candidate_dag", [])
            if entry.get("label")
        )
        summary = (
            f"{len(revision.operations)} operation(s) from {len(cleaned_points)} point(s)"
        )
        with _RunLock(run_dir):
            _append_correction_log(
                updated_protocol,
                kind="dag_points",
                reviewer=reviewer.strip(),
                comment=comment.strip(),
                summary=summary,
                detail={
                    "points": cleaned_points,
                    "operations": [
                        {
                            "op": operation.op,
                            "label": operation.label,
                            "role": operation.role,
                            "temporal_classification": operation.temporal_classification,
                        }
                        for operation in revision.operations
                    ],
                    "acknowledged_errors": list(revision.acknowledged_errors),
                    "unaddressed_points": list(revision.unaddressed_points),
                    "apply_notes": apply_notes,
                },
            )
            _atomic_write_json(protocol_path, updated_protocol)
            self._write_hypothesis(run_id, updated_protocol)

        detail_bits = [summary]
        if revision.unaddressed_points:
            detail_bits.append(f"{len(revision.unaddressed_points)} point(s) not applied")
        if apply_notes:
            detail_bits.append("; ".join(apply_notes))
        state.record(
            "candidate_dag_revised_from_points",
            f"{'; '.join(detail_bits)} by {reviewer.strip()}",
            changes={"before": before_labels, "after": after_labels},
        )
        for gate in ("dag_review", "clinical_mapping", "analysis_plan"):
            self._invalidate_gate(state, gate, f"DAG revised from reviewer points by {reviewer.strip()}")
        self.save(state)
        return state

    def _block_if_major_confounders_unresolved(self, run_id: str) -> None:
        """The `dag_review` gate (roadmap stage 2: causal DAG and clinical
        completeness) cannot be approved while a major confounder is still
        unresolved: the run may never have called the Hypothesis Agent yet,
        so absence of hypothesis.json is not itself a blocker here
        (protocol.json-only workflows never touch this gate's clinical
        content). Once hypothesis.json exists, its
        `unresolved_major_confounders` is authoritative.
        """
        hypothesis_path = self.runs_dir / run_id / "hypothesis.json"
        if not hypothesis_path.is_file():
            return
        hypothesis = json.loads(hypothesis_path.read_text(encoding="utf-8"))
        unresolved_major = hypothesis.get("unresolved_major_confounders") or []
        if unresolved_major:
            raise ValueError(
                "Cannot approve the dag_review gate while major confounders remain "
                f"unresolved: {', '.join(unresolved_major)}. Classify each one with a role and "
                "rationale, or explicitly reject it with a reviewer and rationale."
            )

    def _block_if_not_identified(self, run_id: str) -> None:
        """The `identification_estimand` gate (roadmap stage 3:
        identification and estimand) cannot be approved while the declared
        candidate_dag fails backdoor identification: the run may never have
        called the Hypothesis Agent yet, so absence of hypothesis.json is
        not itself a blocker here (mirrors
        `_block_if_major_confounders_unresolved`). Once hypothesis.json
        exists, `assess_identification` is authoritative. This does not
        replace `_block_if_major_confounders_unresolved`: an empty
        candidate_dag is trivially identified (see
        `causal.agents.dag_identification` for why), so this check mainly
        catches a major confounder that was classified but marked
        unobserved/latent -- no valid baseline adjustment set can exist for
        it -- or a DAG with more adjustment candidates than
        `assess_identification` can exhaustively verify.
        """
        hypothesis_path = self.runs_dir / run_id / "hypothesis.json"
        if not hypothesis_path.is_file():
            return
        protocol_path = self.runs_dir / run_id / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        hypothesis = HypothesisAgent().run(protocol)
        result = assess_identification(hypothesis)
        if not result.identified_by_backdoor_adjustment:
            detail = "; ".join(result.technical_notes) or (
                "no valid baseline adjustment set blocks every backdoor path "
                "(a classified confounder may be unobserved/latent)"
            )
            raise ValueError(
                "Cannot approve the identification_estimand gate: the declared causal DAG is not "
                f"identified by backdoor adjustment ({detail}). Run `workflow dag-export` "
                "to inspect the graph, or reclassify/resolve the blocking variable."
            )

    def _write_study_specification_if_hypothesis_exists(self, run_id: str) -> None:
        """Build and persist `study_specification.json` (the validated
        `roadmap.CausalStudySpecification` -- question, target trial,
        estimand, DAG) when approving the `identification_estimand` gate.

        Also acts as this gate's estimand-completeness check: `_block_if_
        not_identified` above only checks backdoor identification, not
        whether `CausalEstimand`/the rest of `TargetTrialProtocol` are
        internally valid (e.g. `horizon_days == follow_up_days`) -- this is
        the "wire CausalStudySpecification into the persistent gates"
        roadmap milestone (docs/causal_roadmap.md), landing at the point
        both the DAG and the target-trial content are already reviewed.
        Absence of hypothesis.json is not a blocker here, mirroring
        `_block_if_not_identified` -- nothing to build yet.
        """
        run_dir = self.runs_dir / run_id
        hypothesis_path = run_dir / "hypothesis.json"
        if not hypothesis_path.is_file():
            return
        protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
        hypothesis = HypothesisAgent().run(protocol)
        try:
            spec = build_study_specification(protocol, hypothesis)
        except ValueError as exc:
            raise ValueError(f"Cannot approve the identification_estimand gate: {exc}") from exc
        spec_path = run_dir / "study_specification.json"
        _atomic_write_json(spec_path, spec.to_dict())

    def _write_statistical_estimand(self, run_id: str) -> None:
        """Build and persist `statistical_estimand.json` (`roadmap.
        StatisticalEstimand`) when approving the `statistical_estimand`
        gate, and act as that gate's completeness check.

        Distinct from `study_specification.json`'s `CausalEstimand`: this
        fixes how the already-approved counterfactual contrast will actually
        be estimated -- the adjustment set as built into `cohort.sql` (not
        re-derived, `cohort_manifest.json`'s `baseline_covariate` columns are
        the single source of truth, same as `run_estimation`/`run_
        diagnostics`), and the reviewer's explicit missing-data/censoring
        strategy (`set_statistical_estimand_strategy`) -- rather than the
        g-formula and complete-case behavior staying implicit in `causal.
        agents.estimation.EstimationAgent`'s code with the analytic cohort's
        `censored` column silently unused, as before this gate existed.
        """
        run_dir = self.runs_dir / run_id
        study_spec_path = run_dir / "study_specification.json"
        if not study_spec_path.is_file():
            raise ValueError(
                "Cannot approve the statistical_estimand gate: run and approve "
                "identification_estimand first (study_specification.json is missing)"
            )
        manifest_path = run_dir / "cohort_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(
                "Cannot approve the statistical_estimand gate: run `workflow cohort-plan` first "
                "(cohort_manifest.json is missing) so the adjustment set actually built into "
                "cohort.sql is known"
            )
        protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
        strategy = _load_statistical_estimand_settings(protocol)
        if strategy is None:
            raise ValueError(
                "Cannot approve the statistical_estimand gate: run `workflow estimand-strategy` "
                "first to record a reviewed missing-data strategy, censoring strategy, and "
                "rationale"
            )
        if (
            strategy["censoring_strategy"] in _CENSORING_STRATEGIES_REQUIRING_APPROVED_COLUMN
            and not _load_censoring_definition_approved(protocol)
        ):
            raise ValueError(
                f"Cannot approve the statistical_estimand gate: censoring_strategy is "
                f"{strategy['censoring_strategy']!r} but the per-row `censored` column has not "
                "been approved yet (run `workflow approve-censoring-definition` first, or choose "
                "'ignore_censoring_administrative_waiver' with a stated rationale)"
            )
        study_spec = json.loads(study_spec_path.read_text(encoding="utf-8"))
        effect_measure = study_spec.get("estimand", {}).get("effect_measure")
        if effect_measure != "risk_difference":
            raise ValueError(
                "Cannot approve the statistical_estimand gate: only the risk_difference effect "
                "measure is implemented (causal.agents.estimation.EstimationAgent, causal.agents."
                f"tmle.TMLEAgent), but study_specification.json declares effect_measure="
                f"{effect_measure!r}. Either revise protocol.json's target_trial.effect_measure to "
                "risk_difference, or wait for an estimator that supports that scale."
            )
        # Passed straight through, not re-derived -- `population_summary`
        # was already fixed at the causal-question stage
        # (`CausalEstimand.population_summary`, `study_specification.json`
        # `estimand.population_summary`). Re-declaring or overriding it
        # here would let a reviewer silently change *what* is being
        # estimated at the estimation-procedure-review stage, which is not
        # this gate's job.
        population_summary = study_spec.get("estimand", {}).get("population_summary")
        cohort_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        adjustment_set = tuple(
            column["name"] for column in cohort_manifest["columns"]
            if column["role"] == "baseline_covariate"
        )
        state = self.load(run_id)
        if strategy["estimation_procedure"] in {
            "targeted_minimum_loss_based_estimation", "causal_forest_dml",
        }:
            outcome_model = "super_learner_ensemble"
        else:
            outcome_model = "logistic_regression_unpenalized"
        try:
            estimand = StatisticalEstimand(
                schema_version="1.2",
                study_id=state.study_id,
                identifying_functional="g_formula_standardization",
                outcome_model=outcome_model,
                adjustment_set=adjustment_set,
                effect_measure=effect_measure,
                missing_data_strategy=strategy["missing_data_strategy"],
                censoring_strategy=strategy["censoring_strategy"],
                censoring_rationale=strategy["censoring_rationale"],
                population_summary=population_summary or "ATE",
                estimation_procedure=strategy["estimation_procedure"],
            )
        except ValueError as exc:
            raise ValueError(f"Cannot approve the statistical_estimand gate: {exc}") from exc
        estimand_path = run_dir / "statistical_estimand.json"
        _atomic_write_json(estimand_path, estimand.to_dict())

    def _write_hypothesis(self, run_id: str, protocol: dict) -> None:
        artifact = HypothesisAgent().run(protocol).to_dict()
        artifact_path = self.runs_dir / run_id / "hypothesis.json"
        _atomic_write_json(artifact_path, artifact)

    def run_data_science_agent(
        self,
        run_id: str,
        concept_lookup: OMCPClient | None = None,
    ) -> RunState:
        state = self.load(run_id)
        # Same rationale as `build_cohort_plan`/`run_cate` etc.: regenerating
        # analysis_plan.json never needs OMCP and is derived purely from
        # hypothesis.json/protocol.json (see this method's own docstring),
        # so it must stay callable to pick up a post-hoc covariate reject/
        # remap even after `advance()` has moved the phase well past
        # analysis_plan.
        if state.phase not in {
            "concept_mapping", "feasibility", "analysis_plan",
            "estimand_review", "execution", "diagnostics", "results_review", "report",
        }:
            raise RuntimeError(
                "Data Science Agent requires concept_mapping, feasibility, analysis_plan, "
                "estimand_review, execution, diagnostics, results_review, or report phase"
            )
        run_dir = self.runs_dir / run_id
        hypothesis_path = run_dir / "hypothesis.json"
        if not hypothesis_path.is_file():
            raise RuntimeError("Run the Hypothesis Agent first")
        hypothesis = json.loads(hypothesis_path.read_text(encoding="utf-8"))
        protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
        # Rebuilt fresh from protocol, mirroring `_block_if_not_identified`,
        # rather than trusted from hypothesis.json: by the time this phase is
        # reachable the protocol gate is already approved, which guarantees
        # the DAG is backdoor-identified, so this call cannot fail here.
        adjustment = review_dag(HypothesisAgent().run(protocol)).adjustment.to_dict()
        artifact = DataScienceAgent().run(
            hypothesis, protocol, concept_lookup=concept_lookup, adjustment=adjustment
        ).to_dict()
        artifact_path = run_dir / "analysis_plan.json"
        carried_forward = False
        if concept_lookup is None:
            artifact, carried_forward = _carry_forward_concept_candidates(artifact, artifact_path)
        if concept_lookup is not None:
            source = "omcp_candidates"
        elif carried_forward:
            source = "carried_forward_candidates"
        else:
            source = "protocol_only"
        # `DataScienceAgent().run` above (LLM/OMCP calls) can take seconds --
        # deliberately outside the lock, same rationale as the
        # `concept_lookup` peek in `map_covariate_concepts`. Only the actual
        # file write and state bookkeeping below need to be atomic, and
        # `state` is reloaded fresh here (not reused from the top of this
        # method) so a concurrent `decide_gate`/review that completed while
        # the agent was running isn't silently overwritten by a stale save.
        with _RunLock(run_dir):
            state = self.load(run_id)
            _atomic_write_json(artifact_path, artifact)
            if "analysis_plan.json" not in state.artifacts:
                state.artifacts.append("analysis_plan.json")
            state.record(
                "agent_completed",
                f"data_science_agent->analysis_plan.json; mapping_source={source}",
            )
            self._invalidate_gate(state, "analysis_plan", "analysis_plan.json regenerated")
            self.save(state)
        return state

    def build_variable_registry(self, run_id: str) -> RunState:
        """Write `registry.json` -- the versioned `causal.mapping.registry.
        OMOPVariableRegistry` snapshot of every phenotype (`concept_sets`)
        and DAG covariate (`candidate_dag`) declared for this run, rebuilt
        fresh from `protocol.json` every time (same "rebuild, don't trust a
        stale copy" convention as `_write_hypothesis`).

        Only requires `protocol.json` -- unlike the analysis plan, the
        registry does not depend on the Data Science Agent having run, since
        it must never depend on OMOP data availability (see
        `docs/causal_roadmap.md`): an unmapped or unobservable variable is
        still a valid registry entry, only one `is_sql_executable(...)`
        reports as not-yet-executable.
        """
        state = self.load(run_id)
        run_dir = self.runs_dir / run_id
        protocol_path = run_dir / "protocol.json"
        if not protocol_path.is_file():
            raise RuntimeError("Run started without a protocol.json snapshot")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        registry = OMOPVariableRegistry.from_protocol(protocol)
        artifact_path = run_dir / "registry.json"
        _atomic_write_json(artifact_path, registry.to_dict())
        if "registry.json" not in state.artifacts:
            state.artifacts.append("registry.json")
        executable = sum(1 for variable in registry.variables if is_sql_executable(variable))
        state.record(
            "registry_built",
            f"registry.json; {len(registry.variables)} variable(s), {executable} SQL-executable",
        )
        self.save(state)
        return state

    def run_execution_readiness(self, run_id: str) -> RunState:
        """Create the execution policy artifact without running an estimator.

        The whole body runs under `_RunLock`: unlike the covariate-review
        methods, there is no slow network/LLM call here (`ExecutionAgent.run`
        is a pure policy function over already-loaded JSON), so there is no
        reason to narrow the critical section -- and reading
        `state.gates[...]` under the lock means a concurrent `decide_gate`
        can't be interleaved between that read and this method's own save.
        """
        run_dir = self.runs_dir / run_id
        with _RunLock(run_dir):
            state = self.load(run_id)
            # Same rationale as `run_cate`/`materialize_analytic_dataset`/
            # `run_estimation`: a pure policy re-check over already-loaded
            # JSON, not phase-specific state, so it must stay callable after
            # `advance()` moves the phase past `execution` without an
            # estimate ever having been computed.
            if state.phase not in {
                "feasibility", "analysis_plan", "estimand_review",
                "execution", "diagnostics", "results_review", "report",
            }:
                raise RuntimeError(
                    "Execution readiness requires feasibility, analysis_plan, estimand_review, "
                    "execution, diagnostics, results_review, or report phase"
                )
            plan_path = run_dir / "analysis_plan.json"
            if not plan_path.is_file():
                raise RuntimeError("Run the Data Science Agent first")
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            artifact = ExecutionAgent().run(
                plan,
                mode=state.mode,
                clinical_mapping_status=state.gates["clinical_mapping"].status,
                analysis_plan_status=state.gates["analysis_plan"].status,
            ).to_dict()
            artifact_path = run_dir / "execution_readiness.json"
            _atomic_write_json(artifact_path, artifact)
            mcp_context = {
                "run_id": state.run_id,
                "study_id": state.study_id,
                "mode": state.mode,
                "clinical_mapping_status": state.gates["clinical_mapping"].status,
                "analysis_plan_status": state.gates["analysis_plan"].status,
                "aggregate_feasibility_allowed": artifact["aggregate_feasibility_allowed"],
                "analytic_dataset_build_allowed": artifact["analytic_dataset_build_allowed"],
                "causal_estimation_allowed": artifact["causal_estimation_allowed"],
                "readiness_input_sha256": artifact["metadata"]["input_sha256"],
            }
            _atomic_write_json(run_dir / "mcp_context.json", mcp_context)
            if "execution_readiness.json" not in state.artifacts:
                state.artifacts.append("execution_readiness.json")
            if "mcp_context.json" not in state.artifacts:
                state.artifacts.append("mcp_context.json")
            state.record(
                "agent_completed",
                "execution_agent->execution_readiness.json+mcp_context.json; no causal effect estimated",
            )
            self.save(state)
        return state

    def build_cohort_plan(
        self,
        run_id: str,
        datasets: BigQueryDatasets | None = None,
    ) -> RunState:
        """Write cohort SQL and its governance manifest without executing it."""
        state = self.load(run_id)
        # Same rationale as `run_cate` etc.: purely re-derived from protocol/
        # analysis_plan.json, so must stay callable to rebuild cohort.sql
        # after a covariate is rejected/remapped post-hoc (a real, common
        # need) even once `advance()` has moved the phase past execution.
        if state.phase not in {
            "analysis_plan", "estimand_review", "execution", "diagnostics", "results_review", "report",
        }:
            raise RuntimeError("Cohort planning requires analysis_plan, estimand_review, or execution phase")
        run_dir = self.runs_dir / run_id
        self.run_execution_readiness(run_id)
        state = self.load(run_id)
        readiness = json.loads(
            (run_dir / "execution_readiness.json").read_text(encoding="utf-8")
        )
        protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
        hypothesis = json.loads((run_dir / "hypothesis.json").read_text(encoding="utf-8"))
        # `run_execution_readiness` above already requires analysis_plan.json
        # to exist, so it is guaranteed present here.
        analysis_plan = json.loads((run_dir / "analysis_plan.json").read_text(encoding="utf-8"))
        specs = _load_concept_specs(protocol)
        settings = _load_query_settings(protocol)
        exposure_contrast = _load_exposure_contrast(protocol)
        covariates = _load_covariate_specs(analysis_plan)
        censoring_definition_approved = _load_censoring_definition_approved(protocol)
        sql, manifest = AnalyticCohortBuilder().build(
            state.study_id,
            datasets or BigQueryDatasets.from_env(),
            population=specs["population"],
            treatment=specs["treatment"],
            comparator=specs["comparator"],
            outcome=specs["outcome"],
            grace_period_hours=hypothesis["grace_period_hours"],
            follow_up_days=hypothesis["follow_up_days"],
            execution_allowed=bool(readiness["analytic_dataset_build_allowed"]),
            exposure_contrast=exposure_contrast,
            covariates=covariates,
            censoring_definition_approved=censoring_definition_approved,
            additional_population=specs.get("population_additional", ()),
            **settings,
        )
        validation = QueryValidationAgent().run(sql, purpose="person_level_dataset")
        if not validation.passed:
            raise RuntimeError(
                f"Cohort SQL failed governance validation: {', '.join(validation.violations)}"
            )
        # `run_execution_readiness` above already acquired and released its
        # own `_RunLock`; taking a second one here (after it returned, not
        # nested inside it) protects only this method's own file writes and
        # state save without risking deadlock against a non-reentrant lock.
        # `state` is reloaded fresh so a concurrent gate decision or review
        # made while the (local, but non-trivial) SQL build above ran isn't
        # silently overwritten by a stale save.
        with _RunLock(run_dir):
            state = self.load(run_id)
            (run_dir / "cohort.sql").write_text(sql, encoding="utf-8")
            (run_dir / "cohort_manifest.json").write_text(
                json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
            )
            (run_dir / "cohort_query_validation.json").write_text(
                json.dumps(validation.to_dict(), indent=2), encoding="utf-8"
            )
            for artifact in ("cohort.sql", "cohort_manifest.json", "cohort_query_validation.json"):
                if artifact not in state.artifacts:
                    state.artifacts.append(artifact)
            state.record(
                "cohort_plan_generated",
                f"cohort.sql; execution_allowed={manifest.execution_allowed}; query_validation=passed",
            )
            self.save(state)
        return state

    def materialize_analytic_dataset(self, run_id: str, client: OMCPClient) -> RunState:
        """Execute the governed cohort SQL and write patient-level rows to
        `analytic_dataset.parquet` -- the one place in this codebase person-
        level data is ever pulled into memory (see `causal.data_builder.
        materialize`'s own docstring). Requires `cohort.sql`/`cohort_
        manifest.json` (`build_cohort_plan`) to already exist.

        Governance: reuses the already-signed `mcp_context.json` and calls
        `client.causal_select(cohort_sql, "cohort_build", context)` --
        `omcp.governance.authorize_causal_operation` already authorizes
        `"cohort_build"` off `analysis_plan_status == "approved"` and
        `context.analytic_dataset_build_allowed` (both re-derived fresh by
        `run_execution_readiness` below, not trusted from a stale readiness
        snapshot). There is deliberately no direct-database bypass for this
        call, unlike the dev-only escape hatch some aggregate feasibility
        paths retain -- person-level materialization always goes through
        OMCP's signed path.
        """
        state = self.load(run_id)
        # See `run_cate`'s identical comment: `advance()` can move the phase
        # past `execution`/`diagnostics` without ever having called this
        # method (nothing gates "advance" on an estimate actually existing),
        # which would otherwise permanently strand a run that got advanced
        # ahead of itself -- phases only move forward, never back.
        if state.phase not in {"execution", "diagnostics", "results_review", "report"}:
            raise RuntimeError(
                "Dataset materialization requires the execution, diagnostics, results_review, "
                "or report phase"
            )
        run_dir = self.runs_dir / run_id
        cohort_sql_path = run_dir / "cohort.sql"
        manifest_path = run_dir / "cohort_manifest.json"
        if not cohort_sql_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("Run build_cohort_plan first")
        self.run_execution_readiness(run_id)
        state = self.load(run_id)
        readiness = json.loads((run_dir / "execution_readiness.json").read_text(encoding="utf-8"))
        if not readiness["analytic_dataset_build_allowed"]:
            raise RuntimeError("Execution readiness does not permit building the analytic dataset")
        cohort_sql = cohort_sql_path.read_text(encoding="utf-8")
        cohort_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Defense in depth, cheap -- same rationale as build_cohort_plan's own
        # call: the SQL on disk should already be valid, but re-checking here
        # costs nothing and never trusts a file it did not just validate.
        validation = QueryValidationAgent().run(cohort_sql, purpose="person_level_dataset")
        if not validation.passed:
            raise RuntimeError(
                f"Cohort SQL failed re-validation before materialization: {', '.join(validation.violations)}"
            )
        context = json.loads((run_dir / "mcp_context.json").read_text(encoding="utf-8"))
        response = client.causal_select(cohort_sql, "cohort_build", context)
        columns = tuple(AnalyticColumn(**column) for column in cohort_manifest["columns"])
        dataframe = parse_cohort_csv_to_dataframe(response.text, columns)
        with _RunLock(run_dir):
            state = self.load(run_id)
            dataset_manifest = write_analytic_dataset(
                dataframe, run_dir,
                study_id=state.study_id, run_id=state.run_id,
                source_query_sha256=cohort_manifest["query_sha256"],
            )
            (run_dir / "dataset_manifest.json").write_text(
                json.dumps(dataset_manifest.to_dict(), indent=2), encoding="utf-8"
            )
            for artifact in ("analytic_dataset.parquet", "dataset_manifest.json"):
                if artifact not in state.artifacts:
                    state.artifacts.append(artifact)
            # Row count only -- this is the audit trail for the one place
            # patient-level rows are ever pulled into memory, so it must
            # never itself carry a value.
            state.record(
                "analytic_dataset_materialized",
                f"analytic_dataset.parquet; {dataset_manifest.row_count} row(s)",
            )
            self.save(state)
        return state

    def run_estimation(
        self,
        run_id: str,
        *,
        n_bootstrap: int = 1000,
        random_seed: int = 0,
    ) -> RunState:
        """Fit the ATE (`causal.agents.estimation.EstimationAgent`'s g-
        computation or `causal.agents.tmle.TMLEAgent`'s TMLE, dispatched on
        `statistical_estimand.json`'s reviewed `estimation_procedure`) over
        the materialized `analytic_dataset.parquet` and write
        `estimation.json`. Requires `materialize_analytic_dataset` to have
        already run in this run directory. `n_bootstrap` is ignored when
        TMLE is the reviewed procedure -- TMLE replaces the bootstrap with a
        closed-form efficient-influence-curve variance estimator.

        The covariate list is read from `cohort_manifest.json`'s own
        `baseline_covariate`-role columns rather than re-derived from
        `analysis_plan.json` -- that manifest is exactly the reviewed
        covariate set the materialized parquet was built from, so this is
        the single source of truth for "what is actually in this file",
        not a second, independently-computed list that could drift from it.
        """
        state = self.load(run_id)
        # See `run_cate`'s identical comment: re-derived entirely from
        # already-materialized files, so it must stay callable even after
        # `advance()` has moved the phase past diagnostics without an
        # estimate ever having been computed.
        if state.phase not in {"execution", "diagnostics", "results_review", "report"}:
            raise RuntimeError(
                "Estimation requires the execution, diagnostics, results_review, or report phase"
            )
        run_dir = self.runs_dir / run_id
        dataset_path = run_dir / "analytic_dataset.parquet"
        manifest_path = run_dir / "cohort_manifest.json"
        if not dataset_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("Run materialize_analytic_dataset first")
        estimand_path = run_dir / "statistical_estimand.json"
        if not estimand_path.is_file():
            raise RuntimeError("Run and approve the statistical_estimand gate first")
        self.run_execution_readiness(run_id)
        state = self.load(run_id)
        readiness = json.loads((run_dir / "execution_readiness.json").read_text(encoding="utf-8"))
        if not readiness["causal_estimation_allowed"]:
            raise RuntimeError("Execution readiness does not permit causal estimation")
        cohort_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        covariate_columns = tuple(
            column["name"] for column in cohort_manifest["columns"]
            if column["role"] == "baseline_covariate"
        )
        statistical_estimand = json.loads(estimand_path.read_text(encoding="utf-8"))
        if tuple(statistical_estimand["adjustment_set"]) != covariate_columns:
            raise RuntimeError(
                "statistical_estimand.json's adjustment_set no longer matches cohort_manifest."
                "json's baseline_covariate columns -- re-approve the statistical_estimand gate "
                "after the analysis plan or cohort plan changed"
            )
        dataframe = pd.read_parquet(dataset_path)
        artifact = _dispatch_estimator(
            statistical_estimand, dataframe,
            study_id=state.study_id, covariate_columns=covariate_columns,
            n_bootstrap=n_bootstrap, random_seed=random_seed,
        )
        with _RunLock(run_dir):
            state = self.load(run_id)
            (run_dir / "estimation.json").write_text(
                json.dumps(artifact.to_dict(), indent=2), encoding="utf-8"
            )
            if "estimation.json" not in state.artifacts:
                state.artifacts.append("estimation.json")
            estimand = artifact.estimand
            state.record(
                "estimation_completed",
                f"estimation.json; {estimand.effect_measure}="
                f"{estimand.point_estimate:.4f} "
                f"[{estimand.ci_lower:.4f}, {estimand.ci_upper:.4f}] "
                f"(n={estimand.n_total})",
            )
            self.save(state)
        return state

    def run_diagnostics(self, run_id: str) -> RunState:
        """Compute the standardized mean difference (SMD) covariate-balance
        table (`causal.agents.diagnostics.DiagnosticsAgent`) over the
        materialized `analytic_dataset.parquet` and write `diagnostics.json`.
        Needs no fitted model, so it can run before or after `run_estimation`
        -- only `materialize_analytic_dataset` is a prerequisite.
        """
        state = self.load(run_id)
        # Same rationale as `run_cate`: re-derived entirely from the
        # materialized dataset, callable "before or after run_estimation"
        # per this method's own docstring -- must stay callable after
        # `advance()` moves the phase past diagnostics too.
        if state.phase not in {"execution", "diagnostics", "results_review", "report"}:
            raise RuntimeError(
                "Diagnostics requires the execution, diagnostics, results_review, or report phase"
            )
        run_dir = self.runs_dir / run_id
        dataset_path = run_dir / "analytic_dataset.parquet"
        manifest_path = run_dir / "cohort_manifest.json"
        if not dataset_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("Run materialize_analytic_dataset first")
        cohort_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        covariate_columns = tuple(
            column["name"] for column in cohort_manifest["columns"]
            if column["role"] == "baseline_covariate"
        )
        covariate_data_types = {
            column["name"]: column["data_type"] for column in cohort_manifest["columns"]
            if column["role"] == "baseline_covariate"
        }
        dataframe = pd.read_parquet(dataset_path)
        artifact = DiagnosticsAgent().run(
            dataframe,
            study_id=state.study_id,
            covariate_columns=covariate_columns,
            covariate_data_types=covariate_data_types,
        )
        with _RunLock(run_dir):
            state = self.load(run_id)
            (run_dir / "diagnostics.json").write_text(
                json.dumps(artifact.to_dict(), indent=2), encoding="utf-8"
            )
            if "diagnostics.json" not in state.artifacts:
                state.artifacts.append("diagnostics.json")
            state.record(
                "diagnostics_completed",
                f"diagnostics.json; {artifact.n_covariates_flagged}/{len(artifact.balance)} "
                "covariate(s) flagged for imbalance",
            )
            self.save(state)
        return state

    def run_sensitivity_analysis(self, run_id: str) -> RunState:
        """Compute the E-value quantitative sensitivity analysis
        (`causal.sensitivity.e_value`, VanderWeele & Ding 2017) over an
        already-computed `estimation.json` and write
        `sensitivity_analysis.json`. Works identically for a g-computation
        or TMLE estimate -- both report a risk_ratio_point_estimate/
        risk_ratio_ci_lower/risk_ratio_ci_upper alongside their primary
        risk-difference estimate specifically for this. Needs no fitted
        model of its own, so -- like `run_diagnostics` -- it is callable any
        time after `run_estimation`, in either order relative to
        `run_diagnostics`.
        """
        state = self.load(run_id)
        # Same rationale as `run_cate`/`run_diagnostics`: re-derived from an
        # already-computed estimation.json, "callable any time after
        # run_estimation" per this method's own docstring -- must stay
        # callable after `advance()` moves the phase past diagnostics too.
        if state.phase not in {"execution", "diagnostics", "results_review", "report"}:
            raise RuntimeError(
                "Sensitivity analysis requires the execution, diagnostics, results_review, or "
                "report phase"
            )
        run_dir = self.runs_dir / run_id
        estimation_path = run_dir / "estimation.json"
        if not estimation_path.is_file():
            raise RuntimeError("Run run_estimation first")
        estimand = json.loads(estimation_path.read_text(encoding="utf-8"))["estimand"]
        try:
            result = e_value(
                estimand["risk_ratio_point_estimate"],
                estimand["risk_ratio_ci_lower"],
                estimand["risk_ratio_ci_upper"],
            )
        except ValueError as exc:
            raise RuntimeError(f"Cannot compute a sensitivity analysis: {exc}") from exc
        artifact = {
            "metadata": artifact_metadata(
                "sensitivity_agent", "0.1.0",
                {
                    "study_id": state.study_id,
                    "risk_ratio_point_estimate": estimand["risk_ratio_point_estimate"],
                    "risk_ratio_ci_lower": estimand["risk_ratio_ci_lower"],
                    "risk_ratio_ci_upper": estimand["risk_ratio_ci_upper"],
                },
            ),
            "schema_version": "1.0",
            "study_id": state.study_id,
            **result.to_dict(),
        }
        with _RunLock(run_dir):
            state = self.load(run_id)
            (run_dir / "sensitivity_analysis.json").write_text(
                json.dumps(artifact, indent=2), encoding="utf-8"
            )
            if "sensitivity_analysis.json" not in state.artifacts:
                state.artifacts.append("sensitivity_analysis.json")
            state.record(
                "sensitivity_analysis_completed",
                f"sensitivity_analysis.json; E-value(point)={result.e_value_point:.3f}, "
                f"E-value(CI)={result.e_value_ci:.3f}",
            )
            self.save(state)
        return state

    def run_negative_control_estimation(
        self,
        run_id: str,
        control_id: str,
        client: OMCPClient,
        *,
        datasets: BigQueryDatasets | None = None,
        n_bootstrap: int = 1000,
        random_seed: int = 0,
    ) -> RunState:
        """Run whichever estimator (`EstimationAgent`/`TMLEAgent`, per the
        already-reviewed `statistical_estimand.json`) was reviewed for the
        real study against a negative-control-substituted cohort instead
        (`docs/causal_roadmap.md` milestone 6, `causal.queries.
        negative_controls`). Requires `map_negative_control(..., status=
        "approved")` to have already run for `control_id`.

        `causal.queries.negative_controls.substitute_spec` swaps the
        approved negative control in for `treatment` (a negative-control
        exposure) or `outcome` (a negative-control outcome) -- population,
        comparator, the reviewed baseline covariates, censoring strategy,
        and exposure_contrast are all identical to the real study. The
        substituted cohort SQL is executed through the same governed OMCP
        `cohort_build` path `materialize_analytic_dataset` uses, but the
        resulting person-level DataFrame is never written to disk -- unlike
        the real study's `analytic_dataset.parquet`, this is a diagnostic
        side-analysis, not the run's primary governed dataset, so nothing
        is gained by persisting a second copy of patient-level rows for it.
        Only the negative-control cohort SQL (safe -- no patient-level
        values) and the aggregate estimate are written to disk.

        A diagnostic, not a gate: a null result here is evidence *for* the
        adjustment set; a non-null result is a warning `build_causal_report`
        surfaces, never something that blocks or changes any gate decision.
        """
        state = self.load(run_id)
        # Same rationale as `materialize_analytic_dataset` (the other method
        # here that also executes a fresh OMCP/BigQuery call rather than
        # just re-reading already-materialized files): `advance()` can move
        # the phase past `execution` without this ever having been called
        # for a given `control_id`, which would otherwise permanently stop
        # a reviewer from running a negative control they approved late, or
        # re-running one after the primary estimate is already in
        # `results_review`/`report`. A diagnostic side-analysis, not a
        # phase-specific action -- see this method's own docstring.
        if state.phase not in {"execution", "diagnostics", "results_review", "report"}:
            raise RuntimeError(
                "Negative control estimation requires the execution, diagnostics, "
                "results_review, or report phase"
            )
        run_dir = self.runs_dir / run_id
        protocol_path = run_dir / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        entry = protocol.get("negative_controls", {}).get(control_id)
        if not isinstance(entry, dict):
            raise RuntimeError(f"negative_controls has no candidate named '{control_id}'")
        if entry.get("mapping_status") != MAPPING_STATUS_APPROVED:
            raise RuntimeError(
                f"Negative control '{control_id}' is not approved (mapping_status="
                f"{entry.get('mapping_status')!r}) -- run map_negative_control first"
            )

        hypothesis_path = run_dir / "hypothesis.json"
        estimand_path = run_dir / "statistical_estimand.json"
        analysis_plan_path = run_dir / "analysis_plan.json"
        for path, label in (
            (hypothesis_path, "the Hypothesis Agent"),
            (estimand_path, "and approve the statistical_estimand gate"),
            (analysis_plan_path, "and approve the analysis_plan gate"),
        ):
            if not path.is_file():
                raise RuntimeError(f"Run {label} first")

        self.run_execution_readiness(run_id)
        state = self.load(run_id)
        readiness = json.loads((run_dir / "execution_readiness.json").read_text(encoding="utf-8"))
        if not readiness["analytic_dataset_build_allowed"]:
            raise RuntimeError("Execution readiness does not permit building an analytic dataset")

        control_variable = variable_from_negative_control_proposal(entry)
        try:
            if is_fixed_domain(control_variable.domain):
                control_spec = fixed_domain_to_concept_set_spec(control_variable)
            else:
                control_spec = extraction_plan_to_concept_set_spec(build_extraction_plan(control_variable))
        except ValueError as exc:
            raise RuntimeError(f"Cannot resolve negative control '{control_id}': {exc}") from exc

        specs = _load_concept_specs(protocol)
        substituted = substitute_spec(specs, control_variable.causal_role, control_spec)
        hypothesis = json.loads(hypothesis_path.read_text(encoding="utf-8"))
        analysis_plan = json.loads(analysis_plan_path.read_text(encoding="utf-8"))
        settings = _load_query_settings(protocol)
        exposure_contrast = _load_exposure_contrast(protocol)
        covariates = _load_covariate_specs(analysis_plan)
        censoring_definition_approved = _load_censoring_definition_approved(protocol)

        sql, manifest = AnalyticCohortBuilder().build(
            state.study_id,
            datasets or BigQueryDatasets.from_env(),
            population=substituted["population"],
            treatment=substituted["treatment"],
            comparator=substituted["comparator"],
            outcome=substituted["outcome"],
            grace_period_hours=hypothesis["grace_period_hours"],
            follow_up_days=hypothesis["follow_up_days"],
            execution_allowed=True,
            exposure_contrast=exposure_contrast,
            covariates=covariates,
            censoring_definition_approved=censoring_definition_approved,
            additional_population=substituted.get("population_additional", ()),
            **settings,
        )
        validation = QueryValidationAgent().run(sql, purpose="person_level_dataset")
        if not validation.passed:
            raise RuntimeError(
                f"Negative control cohort SQL failed governance validation: {', '.join(validation.violations)}"
            )
        context = json.loads((run_dir / "mcp_context.json").read_text(encoding="utf-8"))
        response = client.causal_select(sql, "cohort_build", context)
        dataframe = parse_cohort_csv_to_dataframe(response.text, manifest.columns)

        statistical_estimand = json.loads(estimand_path.read_text(encoding="utf-8"))
        covariate_columns = tuple(
            column.name for column in manifest.columns if column.role == "baseline_covariate"
        )
        artifact = _dispatch_estimator(
            statistical_estimand, dataframe,
            study_id=f"{state.study_id}_negative_control_{control_id}",
            covariate_columns=covariate_columns,
            n_bootstrap=n_bootstrap, random_seed=random_seed,
        )
        with _RunLock(run_dir):
            state = self.load(run_id)
            sql_filename = f"negative_control_{control_id}_cohort.sql"
            result_filename = f"negative_control_estimation_{control_id}.json"
            (run_dir / sql_filename).write_text(sql, encoding="utf-8")
            (run_dir / result_filename).write_text(
                json.dumps(artifact.to_dict(), indent=2), encoding="utf-8"
            )
            for artifact_name in (sql_filename, result_filename):
                if artifact_name not in state.artifacts:
                    state.artifacts.append(artifact_name)
            estimand = artifact.estimand
            # Row count only -- same discipline as materialize_analytic_
            # dataset's own audit-trail entry; the DataFrame itself is
            # never written to disk or logged.
            state.record(
                "negative_control_estimation_completed",
                f"{result_filename}; {len(dataframe)} row(s); {estimand.effect_measure}="
                f"{estimand.point_estimate:.4f} [{estimand.ci_lower:.4f}, {estimand.ci_upper:.4f}]",
            )
            self.save(state)
        return state

    def run_results_narrative(
        self,
        run_id: str,
        llm: LLMCallable,
        *,
        model: str = "unknown",
        provider: str = "unknown",
        prompt_path: Path = DEFAULT_NARRATIVE_PROMPT_PATH,
    ) -> RunState:
        """Optional, gracefully-degrading LLM interpretation of an already-
        computed estimate (`causal.agents.results_narrative`). Requires
        `estimation.json` to exist; builds `NarrativeInput` from on-disk JSON
        artifacts only -- `analytic_dataset.parquet` is never read here and
        never reaches `llm`.

        Fully separate from `build_causal_report`: if `llm(prompt)` raises
        (network, malformed response) this call raises too and no
        `results_narrative.json` is written, but the run and any previously
        built report are entirely unaffected -- `build_causal_report` only
        ever reads `results_narrative.json` if it happens to already exist
        (see its own docstring), it never calls an LLM itself.
        """
        state = self.load(run_id)
        narrative_input = self._results_narrative_input(state)
        prompt = build_narrative_prompt(prompt_path, narrative_input)
        result = parse_narrative_response(llm(prompt))
        return self._write_results_narrative(
            run_id, narrative_input, result,
            prompt=prompt, model=model, provider=provider, source="one_shot",
        )

    def _results_narrative_input(self, state: RunState) -> NarrativeInput:
        """Build the JSON-safe `NarrativeInput` for a run from its on-disk
        artifacts. Shared by `run_results_narrative` (one-shot),
        `run_results_companion_turn` (a dialogue turn), and
        `save_results_narrative_draft` (persisting a dialogue draft) so all
        three interpret -- and `verify_narrative` checks -- against the
        identical bundle. Raises if the phase is not `results_review` /
        `report` or `estimation.json` is missing.
        `analytic_dataset.parquet` is never read here.
        """
        if state.phase not in {"results_review", "report"}:
            raise RuntimeError("Results narrative requires the results_review or report phase")
        run_dir = self.runs_dir / state.run_id
        estimation_path = run_dir / "estimation.json"
        if not estimation_path.is_file():
            raise RuntimeError("Run run_estimation first")
        study_spec = self._read_optional_json(run_dir / "study_specification.json")
        dataset_manifest = self._read_optional_json(run_dir / "dataset_manifest.json")
        estimation = json.loads(estimation_path.read_text(encoding="utf-8"))
        diagnostics = self._read_optional_json(run_dir / "diagnostics.json")
        sensitivity_analysis = self._read_optional_json(run_dir / "sensitivity_analysis.json")
        readiness = self._read_optional_json(run_dir / "execution_readiness.json")
        cohort_manifest = self._read_optional_json(run_dir / "cohort_manifest.json")
        cate = self._read_optional_json(run_dir / "cate.json")
        assumptions = _gather_assumptions(
            study_spec, readiness, cohort_manifest, dataset_manifest, diagnostics,
            estimation.get("estimand", {}), cate,
        )
        # Deterministic, communication-ready rounding + derived clinical
        # quantities (events per 1000, relative risk change, NNT). Injected
        # under `estimation` so a claim can cite it (`estimation.
        # narrative_display....`) with the root already allow-listed, and so
        # every rounded value becomes a verbatim token `verify_narrative`
        # accepts. estimation.json on disk keeps full precision.
        estimation = {
            **estimation,
            "narrative_display": build_narrative_display(estimation, diagnostics, sensitivity_analysis),
        }
        return NarrativeInput(
            study_id=state.study_id,
            question=study_spec.get("question", {}),
            estimand=study_spec.get("estimand", {}),
            dataset_summary={
                "row_count": dataset_manifest.get("row_count"),
                "column_count": dataset_manifest.get("column_count"),
            },
            estimation=estimation,
            diagnostics=diagnostics,
            gate_history=tuple(
                {
                    "name": gate.name, "status": gate.status,
                    "reviewer": gate.reviewer, "decided_at": gate.decided_at,
                }
                for gate in state.gates.values()
            ),
            assumptions_and_limitations=tuple(assumptions),
        )

    def _write_results_narrative(
        self,
        run_id: str,
        narrative_input: NarrativeInput,
        result: NarrativeResult,
        *,
        prompt: str,
        model: str,
        provider: str,
        source: str,
    ) -> RunState:
        """Run the deterministic `verify_narrative` and persist a parsed
        narrative as `results_narrative.json` (`review_status: pending`).
        `source` is `one_shot` (`run_results_narrative`) or `dialogue`
        (`save_results_narrative_draft`) -- the artifacts are identical in
        shape and both still require `review_results_narrative` before the
        report.
        """
        verification = verify_narrative(result, narrative_input)
        run_dir = self.runs_dir / run_id
        artifact = {
            "schema_version": "2.0",
            "narrative": result.narrative_text,
            "claims": [claim.to_dict() for claim in result.claims],
            "verification": verification.to_dict(),
            "review_status": "pending",
            "reviewer": None,
            "reviewed_at": None,
            "review_note": None,
            "boundary_statement": NARRATIVE_BOUNDARY_STATEMENT,
            "model": model,
            "provider": provider,
            "source": source,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "metadata": artifact_metadata(
                "results_narrative_agent", "0.2.0",
                {"study_id": narrative_input.study_id, "model": model, "provider": provider},
            ),
        }
        with _RunLock(run_dir):
            state = self.load(run_id)
            (run_dir / "results_narrative.json").write_text(
                json.dumps(artifact, indent=2), encoding="utf-8"
            )
            if "results_narrative.json" not in state.artifacts:
                state.artifacts.append("results_narrative.json")
            state.record(
                "results_narrative_generated",
                f"results_narrative.json; source={source}; model={provider}:{model}; "
                f"verification={verification.status}; claims={len(result.claims)}",
            )
            self.save(state)
        return state

    def run_results_companion_turn(
        self,
        run_id: str,
        llm: LLMCallable,
        history: Sequence[CompanionMessage],
        *,
        prompt_path: Path = DEFAULT_RESULTS_DIALOGUE_PROMPT_PATH,
    ) -> ResultsCompanionTurn:
        """One turn of the conversational assist for interpreting the
        finished estimate (`causal.companion.companion`) -- the
        results-review counterpart of `run_companion_turn`. Owns the
        `llm(prompt)` call and the JSON-artifact reads; the pure
        prompt-build/parse lives in the companion module.

        Read-only with respect to run state: no `_RunLock`, nothing recorded
        in `state.history`, no file written. A turn may end with a
        `drafted_narrative`; the caller persists it with
        `save_results_narrative_draft` on an explicit reviewer confirm, and
        `review_results_narrative` is still the only way a narrative reaches
        `build_causal_report`.

        Same grounding as `run_results_narrative`: `NarrativeInput` from
        on-disk JSON only, so `analytic_dataset.parquet` never reaches `llm`.
        """
        state = self.load(run_id)
        narrative_input = self._results_narrative_input(state)
        prompt = build_results_dialogue_prompt(
            prompt_path, history=history, narrative_input=narrative_input
        )
        return parse_results_companion_turn(llm(prompt))

    def save_results_narrative_draft(
        self,
        run_id: str,
        drafted_narrative: dict[str, Any],
        history: Sequence[CompanionMessage],
        *,
        model: str = "unknown",
        provider: str = "unknown",
        prompt_path: Path = DEFAULT_RESULTS_DIALOGUE_PROMPT_PATH,
    ) -> RunState:
        """Persist a results-dialogue-drafted narrative as
        `results_narrative.json` (`review_status: pending`).

        `drafted_narrative` is the `{narrative, claims}` object a
        `run_results_companion_turn` turn carried; it is re-validated here
        (`narrative_result_from_payload`) and run through the same
        deterministic `verify_narrative` as a one-shot narrative. The
        reviewer still has to approve it with `review_results_narrative`
        before it enters the report. `prompt_sha256` records the dialogue
        prompt the draft was produced against, rebuilt from `history`.
        """
        state = self.load(run_id)
        narrative_input = self._results_narrative_input(state)
        result = narrative_result_from_payload(drafted_narrative, context="drafted_narrative")
        prompt = build_results_dialogue_prompt(
            prompt_path, history=history, narrative_input=narrative_input
        )
        return self._write_results_narrative(
            run_id, narrative_input, result,
            prompt=prompt, model=model, provider=provider, source="dialogue",
        )

    def review_results_narrative(
        self,
        run_id: str,
        decision: str,
        reviewer: str,
        note: str | None = None,
    ) -> RunState:
        """Human review of the optional LLM results-narrative interpretation
        -- entirely separate from the mandatory `results` gate, which still
        governs the statistical estimate itself and is untouched by this
        call (see `decide_gate`).

        `decision == "approved"` requires the deterministic faithfulness
        verification already stored on the artifact (`run_results_narrative`
        -> `causal.agents.results_faithfulness.verify_narrative`) to contain
        no `failed` check -- a reviewer cannot rubber-stamp a narrative the
        verifier has already flagged as unfaithful to the underlying
        artifacts; the only way past a failed check is `decision ==
        "excluded"`, which always succeeds and simply drops the narrative
        from `build_causal_report`.
        """
        if decision not in {"approved", "excluded"}:
            raise ValueError("decision must be approved or excluded")
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        run_dir = self.runs_dir / run_id
        narrative_path = run_dir / "results_narrative.json"
        if not narrative_path.is_file():
            raise RuntimeError("Run run_results_narrative first")
        with _RunLock(run_dir):
            state = self.load(run_id)
            if state.phase not in {"results_review", "report"}:
                raise RuntimeError("Results-narrative review requires the results_review or report phase")
            artifact = normalize_narrative_artifact(
                json.loads(narrative_path.read_text(encoding="utf-8"))
            )
            if decision == "approved":
                verification = artifact.get("verification") or {}
                checks = verification.get("checks", ())
                if verification.get("status") not in {"passed", "warning"} or any(
                    check.get("status") == "failed" for check in checks
                ):
                    raise ValueError(
                        "Cannot approve the results narrative: deterministic faithfulness "
                        f"verification status is '{verification.get('status', 'not_run')}'. "
                        "Exclude it from the report instead, or regenerate the narrative."
                    )
            artifact["review_status"] = decision
            artifact["reviewer"] = reviewer.strip()
            artifact["review_note"] = note.strip() if note else None
            artifact["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            narrative_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
            state.record("results_narrative_reviewed", f"results_narrative.json {decision} by {reviewer.strip()}")
            self.save(state)
        return state

    def run_cohort_profile_mcp(
        self,
        run_id: str,
        client: OMCPClient,
        datasets: BigQueryDatasets,
    ) -> RunState:
        """Validate the draft cohort using aggregate results only."""
        state = self.load(run_id)
        if state.phase not in {"feasibility", "analysis_plan"}:
            raise RuntimeError("Cohort profiling requires feasibility or analysis_plan phase")
        self.run_execution_readiness(run_id)
        state = self.load(run_id)
        run_dir = self.runs_dir / run_id
        protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
        hypothesis = json.loads((run_dir / "hypothesis.json").read_text(encoding="utf-8"))
        specs = _load_concept_specs(protocol)
        settings = _load_query_settings(protocol)
        exposure_contrast = _load_exposure_contrast(protocol)
        context = json.loads((run_dir / "mcp_context.json").read_text(encoding="utf-8"))
        response = client.causal_select(
            AnalyticCohortBuilder().profile_sql(
                state.study_id,
                datasets,
                population=specs["population"],
                treatment=specs["treatment"],
                comparator=specs["comparator"],
                outcome=specs["outcome"],
                grace_period_hours=hypothesis["grace_period_hours"],
                follow_up_days=hypothesis["follow_up_days"],
                exposure_contrast=exposure_contrast,
                additional_population=specs.get("population_additional", ()),
                **settings,
            ),
            "aggregate_feasibility",
            context,
        )
        rows = list(csv.DictReader(io.StringIO(response.text)))
        if len(rows) != 1:
            raise RuntimeError("OMCP cohort profile must return one aggregate row")
        profile: dict[str, Any] = {}
        for key, value in rows[0].items():
            if key is None or value in (None, ""):
                profile[key] = None
            elif "." in value:
                profile[key] = float(value)
            else:
                profile[key] = int(value)
        artifact = {
            "status": "aggregate_validation_only",
            "execution_transport": "omcp",
            "causal_effect_estimated": False,
            "patient_level_rows_returned": False,
            "profile": profile,
            "mcp_metadata": response.metadata,
            "warnings": [
                "The cohort still uses provisional phenotypes.",
                "Censoring is unresolved for every row in the draft contract.",
                "This profile validates mechanics and cannot support a causal conclusion.",
            ],
        }
        (run_dir / "cohort_profile.json").write_text(
            json.dumps(artifact, indent=2), encoding="utf-8"
        )
        if "cohort_profile.json" not in state.artifacts:
            state.artifacts.append("cohort_profile.json")
        state.record("cohort_profile_completed", "aggregate profile via OMCP")
        self.save(state)
        return state

    def run_followup_audit_mcp(
        self,
        run_id: str,
        client: OMCPClient,
        datasets: BigQueryDatasets,
    ) -> RunState:
        """Audit nominal follow-up completeness without returning patient rows."""
        state = self.load(run_id)
        if state.phase not in {"feasibility", "analysis_plan"}:
            raise RuntimeError("Follow-up audit requires feasibility or analysis_plan phase")
        self.run_execution_readiness(run_id)
        state = self.load(run_id)
        run_dir = self.runs_dir / run_id
        protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
        hypothesis = json.loads((run_dir / "hypothesis.json").read_text(encoding="utf-8"))
        specs = _load_concept_specs(protocol)
        settings = _load_query_settings(protocol)
        exposure_contrast = _load_exposure_contrast(protocol)
        context = json.loads((run_dir / "mcp_context.json").read_text(encoding="utf-8"))
        response = client.causal_select(
            AnalyticCohortBuilder().followup_audit_sql(
                state.study_id,
                datasets,
                population=specs["population"],
                treatment=specs["treatment"],
                comparator=specs["comparator"],
                outcome=specs["outcome"],
                grace_period_hours=hypothesis["grace_period_hours"],
                follow_up_days=hypothesis["follow_up_days"],
                exposure_contrast=exposure_contrast,
                additional_population=specs.get("population_additional", ()),
                **settings,
            ),
            "aggregate_feasibility",
            context,
        )
        rows = list(csv.DictReader(io.StringIO(response.text)))
        if len(rows) != 1:
            raise RuntimeError("OMCP follow-up audit must return one aggregate row")
        counts = {
            key: int(value) for key, value in rows[0].items()
            if key and value not in (None, "")
        }
        artifact = {
            "status": "aggregate_observability_audit_only",
            "execution_transport": "omcp",
            "causal_effect_estimated": False,
            "patient_level_rows_returned": False,
            "censoring_definition_approved": False,
            "counts": counts,
            "mcp_metadata": response.metadata,
            "warnings": [
                "OMOP observation periods may encode recorded encounters rather than complete mortality ascertainment.",
                "Nominal completeness is a diagnostic and is not an approved censoring definition.",
                "Clinical and data-owner validation is required before using this field analytically.",
            ],
        }
        (run_dir / "followup_audit.json").write_text(
            json.dumps(artifact, indent=2), encoding="utf-8"
        )
        if "followup_audit.json" not in state.artifacts:
            state.artifacts.append("followup_audit.json")
        state.record("followup_audit_completed", "aggregate audit via OMCP")
        self.save(state)
        return state

    def run_positivity_audit_mcp(
        self,
        run_id: str,
        client: OMCPClient,
        datasets: BigQueryDatasets,
    ) -> RunState:
        """Univariate treatment-arm overlap for every reviewed baseline
        covariate (see `causal.data_builder.analytic.AnalyticCohortBuilder.
        positivity_audit_sql` for the metric and its documented limits --
        one covariate at a time, not the full joint adjustment set).
        Requires `analysis_plan.json` (covariates come from there, via
        `_load_covariate_specs`), unlike `run_cohort_profile_mcp`/
        `run_followup_audit_mcp`, which only need population/treatment/
        comparator/outcome and so can run earlier, in the `feasibility`
        phase.
        """
        state = self.load(run_id)
        if state.phase not in {"analysis_plan", "execution"}:
            raise RuntimeError("Positivity audit requires analysis_plan or execution phase")
        self.run_execution_readiness(run_id)
        state = self.load(run_id)
        run_dir = self.runs_dir / run_id
        protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
        hypothesis = json.loads((run_dir / "hypothesis.json").read_text(encoding="utf-8"))
        # `run_execution_readiness` above already requires analysis_plan.json
        # to exist, so it is guaranteed present here.
        analysis_plan = json.loads((run_dir / "analysis_plan.json").read_text(encoding="utf-8"))
        specs = _load_concept_specs(protocol)
        settings = _load_query_settings(protocol)
        exposure_contrast = _load_exposure_contrast(protocol)
        covariates = _load_covariate_specs(analysis_plan)
        context = json.loads((run_dir / "mcp_context.json").read_text(encoding="utf-8"))
        response = client.causal_select(
            AnalyticCohortBuilder().positivity_audit_sql(
                state.study_id,
                datasets,
                population=specs["population"],
                treatment=specs["treatment"],
                comparator=specs["comparator"],
                outcome=specs["outcome"],
                grace_period_hours=hypothesis["grace_period_hours"],
                follow_up_days=hypothesis["follow_up_days"],
                exposure_contrast=exposure_contrast,
                covariates=covariates,
                additional_population=specs.get("population_additional", ()),
                **settings,
            ),
            "aggregate_feasibility",
            context,
        )
        rows = list(csv.DictReader(io.StringIO(response.text)))
        if len(rows) != 1:
            raise RuntimeError("OMCP positivity audit must return one aggregate row")
        counts = {
            key: (int(value) if "." not in value else float(value))
            for key, value in rows[0].items()
            if key and value not in (None, "")
        }
        artifact = {
            "status": "aggregate_observability_audit_only",
            "execution_transport": "omcp",
            "causal_effect_estimated": False,
            "patient_level_rows_returned": False,
            "counts": counts,
            "mcp_metadata": response.metadata,
            "warnings": [
                "Univariate overlap only -- does not check the full joint adjustment-set stratification.",
                "A zero *_treated_present or *_comparator_present count is an unambiguous positivity "
                "violation for that covariate; a non-zero count does not itself prove positivity holds.",
                "Value-covariate min/mean/max is a distributional summary, not a stratified overlap check.",
            ],
        }
        (run_dir / "positivity_audit.json").write_text(
            json.dumps(artifact, indent=2), encoding="utf-8"
        )
        if "positivity_audit.json" not in state.artifacts:
            state.artifacts.append("positivity_audit.json")
        state.record("positivity_audit_completed", "aggregate audit via OMCP")
        self.save(state)
        return state

    def run_covariate_missingness_audit_mcp(
        self,
        run_id: str,
        client: OMCPClient,
        datasets: BigQueryDatasets,
    ) -> RunState:
        """Per-reviewed-covariate missingness (see `causal.data_builder.
        analytic.AnalyticCohortBuilder.covariate_missingness_sql`),
        generalizing `run_cohort_profile_mcp`'s hardcoded age/gender/race-only
        missingness to every covariate `analysis_plan.json` marks ready.
        Same `analysis_plan.json` precondition as `run_positivity_audit_mcp`.
        """
        state = self.load(run_id)
        if state.phase not in {"analysis_plan", "execution"}:
            raise RuntimeError("Covariate missingness audit requires analysis_plan or execution phase")
        self.run_execution_readiness(run_id)
        state = self.load(run_id)
        run_dir = self.runs_dir / run_id
        protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
        hypothesis = json.loads((run_dir / "hypothesis.json").read_text(encoding="utf-8"))
        # `run_execution_readiness` above already requires analysis_plan.json
        # to exist, so it is guaranteed present here.
        analysis_plan = json.loads((run_dir / "analysis_plan.json").read_text(encoding="utf-8"))
        specs = _load_concept_specs(protocol)
        settings = _load_query_settings(protocol)
        exposure_contrast = _load_exposure_contrast(protocol)
        covariates = _load_covariate_specs(analysis_plan)
        context = json.loads((run_dir / "mcp_context.json").read_text(encoding="utf-8"))
        response = client.causal_select(
            AnalyticCohortBuilder().covariate_missingness_sql(
                state.study_id,
                datasets,
                population=specs["population"],
                treatment=specs["treatment"],
                comparator=specs["comparator"],
                outcome=specs["outcome"],
                grace_period_hours=hypothesis["grace_period_hours"],
                follow_up_days=hypothesis["follow_up_days"],
                exposure_contrast=exposure_contrast,
                covariates=covariates,
                additional_population=specs.get("population_additional", ()),
                **settings,
            ),
            "aggregate_feasibility",
            context,
        )
        rows = list(csv.DictReader(io.StringIO(response.text)))
        if len(rows) != 1:
            raise RuntimeError("OMCP covariate missingness audit must return one aggregate row")
        counts = {
            key: int(value) for key, value in rows[0].items()
            if key and value not in (None, "")
        }
        artifact = {
            "status": "aggregate_observability_audit_only",
            "execution_transport": "omcp",
            "causal_effect_estimated": False,
            "patient_level_rows_returned": False,
            "counts": counts,
            "mcp_metadata": response.metadata,
            "warnings": [
                "Event-domain (condition/drug/procedure) covariates are coalesced to 0 and can never "
                "be missing here by construction -- a nonzero *_missing count only applies to "
                "measurement/observation covariates.",
            ],
        }
        (run_dir / "covariate_missingness.json").write_text(
            json.dumps(artifact, indent=2), encoding="utf-8"
        )
        if "covariate_missingness.json" not in state.artifacts:
            state.artifacts.append("covariate_missingness.json")
        state.record("covariate_missingness_audit_completed", "aggregate audit via OMCP")
        self.save(state)
        return state

    def run_mapping_audit_mcp(
        self,
        run_id: str,
        variable_id: str,
        client: OMCPClient,
        datasets: BigQueryDatasets,
    ) -> RunState:
        """Run `causal.mapping.audit`'s aggregate-only SQL diagnostics for one
        `registry.json` variable through `OMCPClient.causal_select`, the same
        governed "Causal_Select_Query" path `run_cohort_profile_mcp`/
        `run_followup_audit_mcp` already execute through (no separate local
        `QueryValidationAgent` pre-check here either, matching those two --
        its final-projection column check is a blunt word-boundary search
        that cannot tell `COUNT(DISTINCT t.person_id)` apart from a bare
        `person_id` projection, and `concept_row_and_person_counts_sql`
        legitimately does the former).

        Before this method, nothing outside `tests/test_mapping_audit.py`
        ever called `causal.mapping.audit`'s SQL builders -- a reviewer had
        no way to actually run a mapping audit without writing ad-hoc code.
        This does not change what that module is: still diagnostic only, by
        design. `registry.json` is always rebuilt fresh from `protocol.json`
        (see `build_variable_registry`), so writing a finding back onto the
        registry would just be silently discarded on the next rebuild -- a
        reviewer reads the artifact this writes and acts on it through the
        existing `covariate-map-concepts`/`concept-set-map` review commands,
        exactly as before.

        Requires only that `variable_id` resolves to a valid
        `ExtractionPlan` (`build_extraction_plan`: approved, observed, a
        concept-based domain, at least one root concept_id) -- deliberately
        not the stricter `is_sql_executable` (which also demands a reviewed
        window/aggregation): checking whether approved concepts actually
        occur in the data is useful before that review is finished, not only
        after.
        """
        state = self.load(run_id)
        run_dir = self.runs_dir / run_id
        registry_path = run_dir / "registry.json"
        if not registry_path.is_file():
            raise RuntimeError("Run `workflow registry-build` first")
        registry = OMOPVariableRegistry.from_dict(
            json.loads(registry_path.read_text(encoding="utf-8"))
        )
        definition = registry.get(variable_id)
        if definition is None:
            raise RuntimeError(f"'{variable_id}' is not in registry.json")
        plan = build_extraction_plan(definition)

        sql_by_check = {
            "concept_row_and_person_counts": concept_row_and_person_counts_sql(datasets, plan),
            "approved_concepts_missing_from_data": approved_concepts_missing_from_data_sql(datasets, plan),
            "invalid_or_nonstandard_concepts": invalid_or_nonstandard_concepts_sql(datasets, plan),
            "orphan_concepts": orphan_concepts_sql(datasets, plan),
            "index_event_breakdown": index_event_breakdown_sql(datasets, plan),
            "incidence_rate": incidence_rate_sql(datasets, plan),
        }
        if plan.domain in VALUE_DOMAINS:
            sql_by_check["observed_units"] = observed_units_sql(datasets, plan)

        context = {
            "run_id": state.run_id,
            "study_id": state.study_id,
            "purpose": "mapping_audit",
            "variable_id": variable_id,
        }
        results: dict[str, list[dict[str, Any]]] = {}
        mcp_metadata: dict[str, Any] = {}
        for check_name, sql in sql_by_check.items():
            response = client.causal_select(sql, "aggregate_feasibility", context)
            results[check_name] = list(csv.DictReader(io.StringIO(response.text)))
            mcp_metadata[check_name] = response.metadata

        artifact = {
            "status": "aggregate_mapping_audit_only",
            "execution_transport": "omcp",
            "variable_id": variable_id,
            "mapping_status": definition.mapping_status,
            "checks": results,
            "mcp_metadata": mcp_metadata,
            "warnings": [
                "Diagnostic only: this does not change mapping_status or available_in_data, "
                "and does not block extraction.",
                "A reviewer must act on any finding here through the existing "
                "covariate-map-concepts/concept-set-map review commands.",
            ],
        }
        artifact_path = run_dir / f"mapping_audit_{variable_id}.json"
        _atomic_write_json(artifact_path, artifact)
        artifact_name = artifact_path.name
        if artifact_name not in state.artifacts:
            state.artifacts.append(artifact_name)
        state.record(
            "mapping_audit_completed",
            f"variable_id={variable_id}; checks={sorted(results)}",
        )
        self.save(state)
        return state

    def build_demo_report(self, run_id: str) -> Path:
        state = self.load(run_id)
        run_dir = self.runs_dir / run_id
        hypothesis = self._read_optional_json(run_dir / "hypothesis.json")
        plan = self._read_optional_json(run_dir / "analysis_plan.json")
        feasibility = self._read_optional_json(run_dir / "feasibility.json")
        cohort_manifest = self._read_optional_json(run_dir / "cohort_manifest.json")
        cohort_profile = self._read_optional_json(run_dir / "cohort_profile.json")
        followup_audit = self._read_optional_json(run_dir / "followup_audit.json")
        readiness = self._read_optional_json(run_dir / "execution_readiness.json")
        lines = [
            f"# CausalOMOP demonstration — {state.study_id}",
            "",
            "> Architecture and feasibility demonstration only. No causal effect has been estimated.",
            "",
            "## Workflow status",
            "",
            f"- Run: `{state.run_id}`",
            f"- Mode: `{state.mode}`",
            f"- Phase: `{state.phase}`",
            f"- Created: `{state.created_at}`",
            "",
            "## Approval gates",
            "",
            "| Gate | Status | Reviewer |",
            "|---|---|---|",
        ]
        for gate in state.gates.values():
            lines.append(
                f"| {gate.name} | {gate.status} | {gate.reviewer or '—'} |"
            )
        if hypothesis:
            lines.extend([
                "", "## Structured causal question", "",
                f"- Population: {hypothesis['population']}",
                f"- Treatment: {hypothesis['treatment']}",
                f"- Comparator: {hypothesis['comparator']}",
                f"- Outcome: {hypothesis['outcome']}",
                f"- Time zero: {hypothesis['time_zero']}",
                f"- Grace period: {hypothesis['grace_period_hours']} hours",
                f"- Follow-up: {hypothesis['follow_up_days']} days",
                f"- Primary estimand: {hypothesis['primary_estimand']}",
                "", "### Unresolved hypothesis issues", "",
            ])
            lines.extend(f"- `{issue}`" for issue in hypothesis.get("unresolved_issues", ()))
        if plan:
            lines.extend([
                "", "## Analysis plan", "",
                f"- Data source: `{plan['data_source']}`",
                f"- Design: `{plan['design']}`",
                f"- Exploratory execution allowed: `{plan['exploratory_execution_allowed']}`",
                f"- Final execution allowed: `{plan['final_execution_allowed']}`",
                "", "### Phenotypes", "",
                "| Name | Role | Mode | Status |",
                "|---|---|---|---|",
            ])
            for phenotype in plan.get("phenotypes", ()):
                lines.append(
                    f"| {phenotype['name']} | {phenotype['role']} | {phenotype['mode']} | {phenotype['status']} |"
                )
            lines.extend(["", "### Planned feasibility checks", ""])
            lines.extend(f"- {check}" for check in plan.get("feasibility_checks", ()))
        if feasibility:
            counts = feasibility.get("counts", {})
            lines.extend([
                "", "## Provisional feasibility counts", "",
                "> These counts use human-approved concept sets for this question's population/treatment/comparator/outcome. They are not a causal estimate.",
                "",
                f"- Eligible persons: **{counts.get('eligible_persons', '—')}**",
                f"- Treatment plus comparator: **{counts.get('treatment_and_comparator', '—')}**",
                f"- Comparator only: **{counts.get('comparator_only', '—')}**",
                f"- No recognized exposure: **{counts.get('neither_exposure', '—')}**",
                f"- Outcome events, all groups: **{counts.get('outcome_events', '—')}**",
                f"- Outcome events during grace period: **{counts.get('outcome_events_during_grace_period', '—')}**",
                "", "### Feasibility warnings", "",
            ])
            lines.extend(f"- {warning}" for warning in feasibility.get("warnings", ()))
        if cohort_manifest:
            lines.extend([
                "", "## Draft analytic cohort contract", "",
                f"- Status: `{cohort_manifest.get('status', 'unknown')}`",
                f"- Person-level data: `{cohort_manifest.get('contains_person_level_data', True)}`",
                f"- Execution allowed: `{cohort_manifest.get('execution_allowed', False)}`",
                f"- Delivery to an LLM allowed: `{cohort_manifest.get('delivery_to_llm_allowed', False)}`",
                "", "### Unresolved cohort issues", "",
            ])
            lines.extend(
                f"- `{issue}`" for issue in cohort_manifest.get("unresolved_issues", ())
            )
        if cohort_profile:
            profile = cohort_profile.get("profile", {})
            lines.extend([
                "", "## Aggregate cohort profile", "",
                "> This profile validates cohort-building mechanics only. It contains no patient-level rows and does not estimate a causal effect.",
                "",
                f"- Cohort persons: **{profile.get('cohort_persons', '—')}**",
                f"- Treated persons: **{profile.get('treated_persons', '—')}**",
                f"- Comparator persons: **{profile.get('comparator_persons', '—')}**",
                f"- Observed outcome events: **{profile.get('outcome_events', '—')}**",
                f"- Missing age: **{profile.get('missing_age', '—')}**",
                f"- Missing gender: **{profile.get('missing_gender', '—')}**",
                f"- Missing race: **{profile.get('missing_race', '—')}**",
                f"- Unresolved censoring: **{profile.get('unresolved_censoring', '—')}**",
                "", "### Cohort profile warnings", "",
            ])
            lines.extend(
                f"- {warning}" for warning in cohort_profile.get("warnings", ())
            )
        if followup_audit:
            counts = followup_audit.get("counts", {})
            lines.extend([
                "", "## Nominal follow-up audit", "",
                "> This is an aggregate observability diagnostic. The censoring definition is not clinically approved.",
                "",
                f"- Observation period covers follow-up: **{counts.get('observation_period_covers_follow_up', '—')}**",
                f"- Known death within follow-up: **{counts.get('known_death_within_follow_up', '—')}**",
                f"- Nominal follow-up complete: **{counts.get('nominal_followup_complete', '—')}**",
                f"- Nominal follow-up incomplete: **{counts.get('nominal_followup_incomplete', '—')}**",
                f"- No observation period at index: **{counts.get('no_observation_period_at_index', '—')}**",
                "", "### Follow-up audit warnings", "",
            ])
            lines.extend(f"- {warning}" for warning in followup_audit.get("warnings", ()))
        if readiness:
            lines.extend([
                "", "## Execution Agent decision", "",
                f"- Status: `{readiness.get('status', 'unknown')}`",
                f"- Aggregate feasibility allowed: `{readiness.get('aggregate_feasibility_allowed', False)}`",
                f"- Analytic dataset build allowed: `{readiness.get('analytic_dataset_build_allowed', False)}`",
                f"- Causal estimation allowed: `{readiness.get('causal_estimation_allowed', False)}`",
                "", "### Current execution blockers", "",
            ])
            lines.extend(f"- `{blocker}`" for blocker in readiness.get("blockers", ()))
        lines.extend([
            "", "## Interpretation", "",
            "The workflow contracts and audit trail are operational. Cohort feasibility refinement and causal estimates remain pending. "
            "Source-value phenotypes are restricted to development; final conclusions require clinical mapping approval.",
            "",
        ])
        report_path = run_dir / "demo_report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        if "demo_report.md" not in state.artifacts:
            state.artifacts.append("demo_report.md")
        state.record("report_generated", "demo_report.md")
        self.save(state)
        return report_path

    def build_causal_report(self, run_id: str) -> Path:
        """Assemble the auditable causal report (`docs/causal_roadmap.md`
        roadmap stage 11): the estimated effect, its diagnostics, and every
        assumption/limitation/gate decision it depends on, all linked back
        to the protocol/DAG/estimand contract and each artifact's own
        provenance. A distinct artifact from `build_demo_report` (left
        unchanged) -- that one is a pre-estimation feasibility snapshot;
        this one is only meaningful once an estimate exists.

        Callable from `results_review` (so a reviewer can preview the report
        before deciding the `results` gate) or `report`.
        """
        state = self.load(run_id)
        if state.phase not in {"results_review", "report"}:
            raise RuntimeError("Causal report requires the results_review or report phase")
        run_dir = self.runs_dir / run_id
        protocol = self._read_optional_json(run_dir / "protocol.json")
        study_spec = self._read_optional_json(run_dir / "study_specification.json")
        statistical_estimand = self._read_optional_json(run_dir / "statistical_estimand.json")
        dataset_manifest = self._read_optional_json(run_dir / "dataset_manifest.json")
        estimation = self._read_optional_json(run_dir / "estimation.json")
        diagnostics = self._read_optional_json(run_dir / "diagnostics.json")
        sensitivity_analysis = self._read_optional_json(run_dir / "sensitivity_analysis.json")
        negative_control_results = sorted(
            (
                (path.stem[len("negative_control_estimation_"):], self._read_optional_json(path))
                for path in run_dir.glob("negative_control_estimation_*.json")
            ),
            key=lambda item: item[0],
        )
        cate = self._read_optional_json(run_dir / "cate.json")
        readiness = self._read_optional_json(run_dir / "execution_readiness.json")
        cohort_manifest = self._read_optional_json(run_dir / "cohort_manifest.json")
        narrative = self._read_optional_json(run_dir / "results_narrative.json")

        results_gate = state.gates.get("results")
        if results_gate is not None and results_gate.status == "approved":
            banner = f"Results gate approved by {results_gate.reviewer} on {results_gate.decided_at}."
        else:
            status = results_gate.status if results_gate is not None else "pending"
            banner = f"PROVISIONAL -- the `results` gate is `{status}`, not yet approved."

        lines = [
            f"# CausalOMOP causal report -- {state.study_id}",
            "",
            f"> {banner}",
            "",
            "## Workflow status",
            "",
            f"- Run: `{state.run_id}`",
            f"- Mode: `{state.mode}`",
            f"- Phase: `{state.phase}`",
            f"- Created: `{state.created_at}`",
        ]

        if study_spec:
            question = study_spec.get("question", {})
            trial = study_spec.get("target_trial", {})
            estimand = study_spec.get("estimand", {})
            identification = study_spec.get("identification", {})
            lines.extend([
                "", "## Causal question, target trial, and estimand", "",
                f"- Population: {question.get('population', '-')}",
                f"- Treatment: {question.get('treatment', '-')}",
                f"- Comparator: {question.get('comparator', '-')}",
                f"- Outcome: {question.get('outcome', '-')}",
                f"- Time zero: {trial.get('time_zero', '-')}",
                f"- Follow-up: {trial.get('follow_up_days', '-')} days",
                f"- Primary estimand (population summary): {estimand.get('population_summary', '-')}",
                f"- Effect measure: {estimand.get('effect_measure', '-')}",
                f"- Identified by backdoor adjustment: "
                f"`{identification.get('identified_by_backdoor_adjustment', '-')}`",
                f"- Minimum adjustment set(s): {identification.get('minimum_adjustment_sets', '-')}",
            ])

        if statistical_estimand:
            lines.extend([
                "", "## Statistical estimand", "",
                "> The observed-data contract this run was actually estimated under -- distinct "
                "from the counterfactual estimand above, and reviewed via the "
                "`statistical_estimand` gate.",
                "",
                f"- Identifying functional: {statistical_estimand.get('identifying_functional', '-')}",
                f"- Outcome model: {statistical_estimand.get('outcome_model', '-')}",
                f"- Adjustment set: {list(statistical_estimand.get('adjustment_set', ()))}",
                f"- Missing-data strategy: {statistical_estimand.get('missing_data_strategy', '-')}",
                f"- Censoring strategy: {statistical_estimand.get('censoring_strategy', '-')}",
                f"- Censoring rationale: {statistical_estimand.get('censoring_rationale', '-')}",
            ])

        if dataset_manifest:
            lines.extend([
                "", "## Analytic dataset", "",
                "> Row/column counts and hashes only -- no patient-level value ever appears in "
                "this report.",
                "",
                f"- Rows: **{dataset_manifest.get('row_count', '-')}**",
                f"- Columns: **{dataset_manifest.get('column_count', '-')}**",
                f"- Parquet sha256: `{dataset_manifest.get('parquet_sha256', '-')}`",
                f"- Source cohort query sha256: `{dataset_manifest.get('source_cohort_query_sha256', '-')}`",
            ])

        estimand_block = estimation.get("estimand", {}) if estimation else {}
        if estimand_block:
            display = build_narrative_display(estimation, diagnostics, sensitivity_analysis)
            absolute = display.get("absolute_effect", {})
            relative = display.get("relative_effect", {})
            nnt = display.get("number_needed_to_treat", {})
            fit = display.get("model_fit", {})
            ci_pct = absolute.get("ci_level_pct") or int(estimand_block.get("ci_level", 0.95) * 100)
            per_n = absolute.get("events_per_n_patients", 1000)
            risk_ratio_line = (
                f"{display_or_dash(relative.get('risk_ratio'))} "
                f"[{display_or_dash(relative.get('ci_lower'))}, {display_or_dash(relative.get('ci_upper'))}]"
                if relative.get("available")
                else f"not computable ({relative.get('reason', '-')})"
            )
            lines.extend([
                "", "## Causal effect estimate", "",
                "> Numbers here are rounded for readability; the exact values are in "
                "`estimation.json` (see Provenance below).",
                "",
                f"- Method: {estimand_block.get('method', '-')} "
                f"({estimand_block.get('citation', '-')})",
                f"- Effect measure: {estimand_block.get('effect_measure', '-')} "
                "(absolute change in the probability of the outcome)",
                f"- Point estimate: **{display_or_dash(absolute.get('risk_difference', estimand_block.get('point_estimate', '-')))}** "
                f"({_signed_events(absolute.get('events_per_n_point'))} events per {per_n} patients treated)",
                f"- {ci_pct}% CI: "
                f"[{display_or_dash(absolute.get('ci_lower', estimand_block.get('ci_lower', '-')))}, "
                f"{display_or_dash(absolute.get('ci_upper', estimand_block.get('ci_upper', '-')))}]"
                + (" -- includes no effect" if absolute.get("interval_includes_no_effect") else ""),
                f"- Risk ratio: {risk_ratio_line}",
                f"- N (total / treated / comparator): "
                f"{estimand_block.get('n_total', '-')} / {estimand_block.get('n_treated', '-')} / "
                f"{estimand_block.get('n_comparator', '-')}",
                f"- Outcome events: {estimand_block.get('n_outcome_events', '-')}",
                f"- Bootstrap resamples used: {estimand_block.get('n_bootstrap', '-')} "
                f"(seed {estimand_block.get('bootstrap_seed', '-')})",
                f"- Outcome model: {estimand_block.get('outcome_model', '-')} "
                f"(AUC {display_or_dash(fit.get('outcome_model_auc', estimand_block.get('outcome_model_auc', '-')))}, "
                f"Brier {display_or_dash(fit.get('outcome_model_brier', estimand_block.get('outcome_model_brier', '-')))})",
                f"- Covariates adjusted for: {list(estimand_block.get('covariates_adjusted', ()))}",
                f"- Missing-data strategy: {estimand_block.get('missing_data_strategy', '-')}",
                f"- Censoring strategy: {estimand_block.get('censoring_strategy', '-')}"
                f"{_censoring_strategy_report_note(estimand_block)}",
            ])
            plain = [
                text
                for text in (
                    absolute.get("plain_statement"),
                    relative.get("plain_statement") if relative.get("available") else None,
                    nnt.get("plain_statement") if nnt.get("reported") else None,
                )
                if text
            ]
            if plain:
                lines.extend(["", "### In plain terms", "", "\n\n".join(plain)])
            lines.extend([
                "",
                "> This is the study's estimated average treatment effect under the assumptions "
                "listed below -- not a clinical recommendation.",
            ])

        if diagnostics:
            lines.extend([
                "", "## Diagnostics", "",
                f"### Covariate balance (SMD, {diagnostics.get('citation', 'Austin 2009')})", "",
                f"- {diagnostics.get('n_covariates_flagged', 0)} of "
                f"{len(diagnostics.get('balance', ()))} covariate(s) flagged "
                f"(|SMD| > {diagnostics.get('smd_flag_threshold', 0.1)})",
                "",
                "| Covariate | SMD | Treated mean | Comparator mean | Flagged |",
                "|---|---|---|---|---|",
            ])
            for entry in diagnostics.get("balance", ()):
                lines.append(
                    f"| {entry['column_name']} | {entry['smd']:.3f} | {entry['treated_mean']:.3f} | "
                    f"{entry['comparator_mean']:.3f} | {'yes' if entry['flagged'] else 'no'} |"
                )

        if sensitivity_analysis:
            lines.extend([
                "", "## Sensitivity analysis (E-value)", "",
                f"> {sensitivity_analysis.get('citation', '-')}",
                "",
                f"- Risk ratio: {sensitivity_analysis.get('risk_ratio', '-')}",
                f"- E-value for the point estimate: **{sensitivity_analysis.get('e_value_point', '-')}**",
                f"- E-value for the confidence interval: **{sensitivity_analysis.get('e_value_ci', '-')}**",
                "",
                f"> {sensitivity_analysis.get('interpretation', '-')}",
            ])

        if negative_control_results:
            lines.extend([
                "", "## Negative controls", "",
                "> A negative control substitutes an exposure or outcome believed to have no "
                "true causal effect, then fits this run's same reviewed estimator against it. A "
                "null result (CI crossing zero) is evidence *for* the adjustment set; a non-null "
                "result is a warning of possible unaddressed confounding -- a diagnostic, never a "
                "blocking gate.",
                "",
                "| Control | Point estimate | 95% CI | Crosses null |",
                "|---|---|---|---|",
            ])
            for control_id, result in negative_control_results:
                if not result:
                    continue
                control_estimand = result.get("estimand", {})
                point = control_estimand.get("point_estimate")
                ci_lower = control_estimand.get("ci_lower")
                ci_upper = control_estimand.get("ci_upper")
                crosses_null = isinstance(ci_lower, (int, float)) and isinstance(ci_upper, (int, float)) and ci_lower <= 0 <= ci_upper
                flag = "yes" if crosses_null else "**no -- possible residual confounding**"
                lines.append(f"| {control_id} | {point} | [{ci_lower}, {ci_upper}] | {flag} |")

        cate_estimand = cate.get("estimand", {}) if cate else {}
        if cate_estimand:
            lines.extend([
                "", "## CATE (conditional average treatment effect)", "",
                f"> {cate_estimand.get('citation', '-')} Adjusted CI/p-value use "
                f"{cate_estimand.get('multiple_testing_method', '-')} "
                f"({cate_estimand.get('multiple_testing_citation', '-')}) across the "
                f"{cate_estimand.get('n_prespecified_subgroups_tested', '-')} prespecified "
                "subgroup(s) below -- never the raw column.",
                "",
                f"- Overall ATE from the forest (cross-check against the primary estimate above): "
                f"{cate_estimand.get('overall_ate_from_forest', '-')} "
                f"[{cate_estimand.get('overall_ate_ci_lower', '-')}, "
                f"{cate_estimand.get('overall_ate_ci_upper', '-')}]",
                "",
                "### Prespecified subgroups (family-wise error controlled)", "",
                "| Subgroup | N | Point estimate | Raw CI | Raw p | Adjusted CI | Adjusted p |",
                "|---|---|---|---|---|---|---|",
            ])
            for entry in cate_estimand.get("prespecified_subgroup_results", ()):
                lines.append(
                    f"| {entry['subgroup_column']}={entry['stratum']} | {entry['n']} | "
                    f"{entry['point_estimate']:.4f} | [{entry['ci_lower']:.4f}, {entry['ci_upper']:.4f}] | "
                    f"{entry['pvalue']:.4f} | [{entry['adjusted_ci_lower']:.4f}, "
                    f"{entry['adjusted_ci_upper']:.4f}] | {entry['adjusted_pvalue']:.4f} |"
                )
            exploratory = cate_estimand.get("exploratory_subgroup_results", ())
            if exploratory:
                lines.extend([
                    "", "### Exploratory subgroups (hypothesis-generating only -- NOT family-wise "
                    "error controlled)", "",
                    "| Subgroup | N | Point estimate | Raw CI | Raw p |",
                    "|---|---|---|---|---|",
                ])
                for entry in exploratory:
                    lines.append(
                        f"| {entry['subgroup_column']}={entry['stratum']} | {entry['n']} | "
                        f"{entry['point_estimate']:.4f} | [{entry['ci_lower']:.4f}, "
                        f"{entry['ci_upper']:.4f}] | {entry['pvalue']:.4f} |"
                    )

        assumptions = _gather_assumptions(
            study_spec, readiness, cohort_manifest, dataset_manifest, diagnostics, estimand_block,
            cate,
        )
        lines.extend(["", "## Assumptions and limitations", ""])
        lines.extend(f"- {item}" for item in assumptions)

        lines.extend(["", "## LLM interpretive narrative", ""])
        if not narrative:
            lines.append("No LLM interpretation was requested for this run.")
        else:
            narrative = normalize_narrative_artifact(narrative)
            verification = narrative.get("verification") or {}
            verification_ok = verification.get("status") in {"passed", "warning"} and not any(
                check.get("status") == "failed" for check in verification.get("checks", ())
            )
            if narrative.get("review_status") == "excluded":
                lines.append("The LLM interpretation was excluded from this version of the report.")
                if narrative.get("reviewer"):
                    lines.append(f"_Excluded by {narrative['reviewer']} on {narrative.get('reviewed_at', '-')}._")
            elif narrative.get("review_status") == "approved" and verification_ok:
                lines.extend([
                    narrative["narrative"],
                    "",
                    f"> {narrative.get('boundary_statement', '')}",
                    "",
                    f"_Model: {narrative.get('provider', 'unknown')}:{narrative.get('model', 'unknown')}. "
                    f"Approved by {narrative.get('reviewer', '-')} on {narrative.get('reviewed_at', '-')}._",
                ])
            else:
                lines.append(
                    "An LLM interpretation was generated but is still pending human review (or failed "
                    "deterministic faithfulness verification) -- it is not included in this version of "
                    "the report."
                )

        lines.extend([
            "", "## Approval gates", "",
            "| Gate | Status | Reviewer | Decided |",
            "|---|---|---|---|",
        ])
        for gate in state.gates.values():
            lines.append(
                f"| {gate.name} | {gate.status} | {gate.reviewer or '-'} | {gate.decided_at or '-'} |"
            )

        decision_memory = self._decision_memory(state, protocol or {})
        (run_dir / "decision_memory.json").write_text(
            json.dumps(
                {
                    **decision_memory.to_dict(),
                    "metadata": artifact_metadata(
                        "decision_memory", "1.0", {"study_id": state.study_id}
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        lines.extend([
            "", "## Reviewer decisions", "",
            "> A deterministic digest (no LLM) of the covariate exclusions and settled choices "
            "distilled from the audit trail and the rejected gates -- the context the estimand-"
            "dialogue companion is given so it does not re-propose a ruled-out option. Snapshot "
            "written as `decision_memory.json`.",
            "",
            decision_memory.format_for_prompt(),
        ])

        lines.extend(["", "## Audit trail", "", "| At | Event | Detail |", "|---|---|---|"])
        for entry in state.history:
            lines.append(f"| {entry['at']} | {entry['event']} | {entry['detail']} |")

        lines.extend(["", "## Provenance", "", "| Artifact | Agent | Version | Generated (UTC) |", "|---|---|---|---|"])
        for name in (
            "hypothesis.json", "study_specification.json", "statistical_estimand.json",
            "cohort_manifest.json", "dataset_manifest.json", "estimation.json", "diagnostics.json",
            "sensitivity_analysis.json", "cate.json", "results_narrative.json", "decision_memory.json",
        ):
            payload = self._read_optional_json(run_dir / name)
            metadata = payload.get("metadata") if payload else None
            if not metadata:
                continue
            lines.append(
                f"| {name} | {metadata.get('agent_name', '-')} | {metadata.get('agent_version', '-')} | "
                f"{metadata.get('generated_at_utc', '-')} |"
            )
        for control_id, payload in negative_control_results:
            metadata = payload.get("metadata") if payload else None
            if not metadata:
                continue
            lines.append(
                f"| negative_control_estimation_{control_id}.json | {metadata.get('agent_name', '-')} | "
                f"{metadata.get('agent_version', '-')} | {metadata.get('generated_at_utc', '-')} |"
            )

        if estimand_block:
            interpretation = (
                "An average treatment effect has been estimated under the assumptions listed "
                "above. This is a governed technical result, not a clinical conclusion -- final "
                "sign-off is the `results` gate, decided by a human reviewer with the full "
                "assumptions list in view."
            )
        else:
            interpretation = (
                "No causal effect has been estimated for this run yet -- run `workflow estimate` "
                "(or the Streamlit Estimation & report tab) first. This report reflects only the "
                "protocol, gates, and audit trail collected so far."
            )
        lines.extend(["", "## Interpretation", "", interpretation, ""])

        report_path = run_dir / "causal_report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        with _RunLock(run_dir):
            state = self.load(run_id)
            for artifact_name in ("causal_report.md", "decision_memory.json"):
                if artifact_name not in state.artifacts:
                    state.artifacts.append(artifact_name)
            state.record("causal_report_generated", "causal_report.md")
            self.save(state)
        return report_path

    def set_exposure_contrast(
        self,
        run_id: str,
        exposure_contrast: str,
        reviewer: str,
        comment: str | None = None,
    ) -> RunState:
        """Record a human's choice of treatment/comparator inclusion semantics
        for this study's cohort queries (`protocol["query_settings"]
        ["exposure_contrast"]`; see `causal.queries.feasibility.
        EXPOSURE_CONTRASTS`).

        This is a real modeling decision, not a query-engine detail --
        `EXPOSURE_CONTRASTS`'s own comment warns that getting it wrong
        silently changes who is in the cohort -- so it is explicitly
        reviewer-set here rather than left to `_load_exposure_contrast`'s
        `"add_on"` default, which every run got silently until this method
        existed. Unlike `map_concept_set`/`review_covariate*`, this never
        touches `hypothesis.json`: `query_settings` only feeds SQL
        generation, not the DAG/identification artifact.
        """
        if exposure_contrast not in EXPOSURE_CONTRASTS:
            raise ValueError(f"exposure_contrast must be one of {EXPOSURE_CONTRASTS}")
        if not reviewer.strip():
            raise ValueError("reviewer is required")

        run_dir = self.runs_dir / run_id
        protocol_path = run_dir / "protocol.json"
        state = self.load(run_id)

        with _RunLock(run_dir):
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            settings = protocol.get("query_settings")
            settings = dict(settings) if isinstance(settings, dict) else {}
            before = settings.get("exposure_contrast", "add_on")
            settings["exposure_contrast"] = exposure_contrast
            protocol["query_settings"] = settings
            _atomic_write_json(protocol_path, protocol)

        detail = f"exposure_contrast: {before} -> {exposure_contrast} by {reviewer}"
        if comment and comment.strip():
            detail += f" ({comment.strip()})"
        state.record(
            "exposure_contrast_set", detail,
            changes={"before": before, "after": exposure_contrast},
        )
        self._invalidate_gate(state, "analysis_plan", "exposure_contrast changed")
        self.save(state)
        return state

    def approve_censoring_definition(
        self,
        run_id: str,
        reviewer: str,
        comment: str | None = None,
    ) -> RunState:
        """Record a human's approval of the real per-row `censored` column
        (`protocol["query_settings"]["censoring_definition_approved"]`; see
        `causal.queries.feasibility.analytic_cohort_sql`'s docstring for the
        exact administrative-censoring definition this then turns on).

        Requires `followup_audit.json` to already exist -- the same
        aggregate nominal-completeness numbers this per-row definition
        reuses -- so a reviewer approves this after having actually seen
        them, not blind. Same shape as `set_exposure_contrast`: lock, load
        `protocol.json`, flip the flag, atomic write, record, invalidate the
        `analysis_plan` gate (a cohort built before this approval has a
        different `censored` column than one built after).
        """
        run_dir = self.runs_dir / run_id
        if not (run_dir / "followup_audit.json").is_file():
            raise ValueError(
                "Cannot approve the censoring definition before running the follow-up audit "
                "(workflow followup-audit) -- review the nominal completeness numbers first"
            )
        if not reviewer.strip():
            raise ValueError("reviewer is required")

        protocol_path = run_dir / "protocol.json"
        state = self.load(run_id)

        with _RunLock(run_dir):
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            settings = protocol.get("query_settings")
            settings = dict(settings) if isinstance(settings, dict) else {}
            before = bool(settings.get("censoring_definition_approved", False))
            settings["censoring_definition_approved"] = True
            protocol["query_settings"] = settings
            _atomic_write_json(protocol_path, protocol)

        detail = f"censoring_definition_approved: {before} -> True by {reviewer}"
        if comment and comment.strip():
            detail += f" ({comment.strip()})"
        state.record(
            "censoring_definition_approved", detail,
            changes={"before": before, "after": True},
        )
        self._invalidate_gate(state, "analysis_plan", "censoring_definition_approved changed")
        self._invalidate_gate(state, "statistical_estimand", "censoring_definition_approved changed")
        self.save(state)
        return state

    def set_statistical_estimand_strategy(
        self,
        run_id: str,
        *,
        missing_data_strategy: str,
        censoring_strategy: str,
        censoring_rationale: str,
        reviewer: str,
        estimation_procedure: str = "plug_in_standardization",
        comment: str | None = None,
    ) -> RunState:
        """Record a human's explicit missing-data, censoring, and estimation-
        procedure strategy (`protocol["statistical_estimand_settings"]`) for
        the `statistical_estimand` gate (roadmap stage 8: the observed-data
        statistical estimand).

        Before this method existed, `EstimationAgent` always did complete-
        case analysis and always ignored the analytic cohort's per-row
        `censored` column -- a real modeling choice left implicit in
        estimator code, with the reviewer never asked. `censoring_rationale`
        is required non-empty text, not a checkbox: whichever strategy is
        picked, a human states *why* here, and `_write_statistical_estimand`
        persists it into `statistical_estimand.json` where `run_estimation`
        actually reads it back, so this is never a rubber stamp on an
        unread field. `estimation_procedure` (`causal.roadmap.
        ESTIMATION_PROCEDURES`) picks which reviewed estimator --
        `EstimationAgent`'s g-computation or `TMLEAgent`'s TMLE -- computes
        the shared `g_formula_standardization` functional; defaults to
        today's plug-in behavior so existing callers are unaffected.
        """
        if missing_data_strategy not in MISSING_DATA_STRATEGIES:
            raise ValueError(f"missing_data_strategy must be one of {MISSING_DATA_STRATEGIES}")
        if censoring_strategy not in CENSORING_STRATEGIES:
            raise ValueError(f"censoring_strategy must be one of {CENSORING_STRATEGIES}")
        if estimation_procedure not in ESTIMATION_PROCEDURES:
            raise ValueError(f"estimation_procedure must be one of {ESTIMATION_PROCEDURES}")
        if not censoring_rationale.strip():
            raise ValueError("censoring_rationale is required")
        if not reviewer.strip():
            raise ValueError("reviewer is required")

        run_dir = self.runs_dir / run_id
        protocol_path = run_dir / "protocol.json"
        state = self.load(run_id)

        with _RunLock(run_dir):
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            before = protocol.get("statistical_estimand_settings")
            protocol["statistical_estimand_settings"] = {
                "missing_data_strategy": missing_data_strategy,
                "censoring_strategy": censoring_strategy,
                "censoring_rationale": censoring_rationale.strip(),
                "estimation_procedure": estimation_procedure,
            }
            _atomic_write_json(protocol_path, protocol)

        detail = (
            f"statistical_estimand_settings set by {reviewer}: "
            f"missing_data_strategy={missing_data_strategy}, censoring_strategy={censoring_strategy}, "
            f"estimation_procedure={estimation_procedure}"
        )
        if comment and comment.strip():
            detail += f" ({comment.strip()})"
        state.record(
            "statistical_estimand_strategy_set", detail,
            changes={"before": before, "after": protocol["statistical_estimand_settings"]},
        )
        self._invalidate_gate(state, "statistical_estimand", "statistical_estimand_settings changed")
        self.save(state)
        return state

    def propose_estimation_procedure(
        self,
        run_id: str,
        llm: LLMCallable,
        *,
        prompt_path: Path = DEFAULT_ESTIMATION_PROCEDURE_SUGGESTION_PROMPT_PATH,
    ) -> RunState:
        """LLM-assisted (deterministic for CATE) `estimation_procedure`
        suggestion for the `statistical_estimand` gate, alongside `set_
        statistical_estimand_strategy` above -- an assist, never an
        approval: this only ever writes `protocol["statistical_estimand_
        suggestion"]`, a key entirely separate from `protocol["statistical_
        estimand_settings"]` (the human-approved one, written only by
        `set_statistical_estimand_strategy`), so a fresh suggestion can
        never clobber an already-approved run's settings, and no phase gate
        beyond requiring `study_specification.json` to exist is needed --
        mirrors `set_statistical_estimand_strategy` itself, which has none
        either, so a reviewer can propose/re-propose at any point this
        gate's form is open.

        When `study_specification.json`'s `estimand.population_summary ==
        "CATE"`, `StatisticalEstimand.__post_init__` already forces
        `estimation_procedure == "causal_forest_dml"` -- `deterministic_
        cate_suggestion` answers that directly and `llm()` is never called.
        Otherwise the LLM is asked to choose between `plug_in_
        standardization` and `targeted_minimum_loss_based_estimation`,
        grounded only in whichever of `feasibility.json`/`followup_audit.
        json`/`positivity_audit.json`/`covariate_missingness.json` already
        exist on disk (each read via `_read_optional_json`, which degrades
        to `{}`, never raising, when a diagnostic hasn't been run yet).
        """
        run_dir = self.runs_dir / run_id
        study_spec_path = run_dir / "study_specification.json"
        if not study_spec_path.is_file():
            raise RuntimeError(
                "Cannot propose an estimation procedure: study_specification.json is missing "
                "-- approve the identification_estimand gate first"
            )
        study_spec = json.loads(study_spec_path.read_text(encoding="utf-8"))
        estimand = study_spec.get("estimand", {})
        population_summary = estimand.get("population_summary")

        if population_summary == "CATE":
            analysis_plan = self._read_optional_json(run_dir / "analysis_plan.json")
            ready_effect_modifiers = [
                covariate.get("name") for covariate in analysis_plan.get("covariates", ())
                if isinstance(covariate, dict) and covariate.get("causal_role") == "effect_modifier"
                and covariate.get("status") == "ready"
            ]
            suggestion = deterministic_cate_suggestion(len(ready_effect_modifiers))
        else:
            feasibility = self._read_optional_json(run_dir / "feasibility.json")
            followup_audit = self._read_optional_json(run_dir / "followup_audit.json")
            positivity_audit = self._read_optional_json(run_dir / "positivity_audit.json")
            covariate_missingness = self._read_optional_json(run_dir / "covariate_missingness.json")
            prompt = build_estimation_procedure_suggestion_prompt(
                prompt_path,
                population_summary=population_summary or "",
                effect_measure=estimand.get("effect_measure", ""),
                feasibility=feasibility,
                followup_audit=followup_audit,
                positivity_audit=positivity_audit,
                covariate_missingness=covariate_missingness,
            )
            raw = llm(prompt)
            suggestion = parse_estimation_procedure_suggestion_response(raw)
            if suggestion.estimation_procedure == "causal_forest_dml":
                raise ValueError(
                    "LLM suggested causal_forest_dml but this study's population_summary is "
                    f"{population_summary!r}, not 'CATE' -- causal_forest_dml would violate "
                    "StatisticalEstimand's own population_summary cross-check"
                )

        run_dir = self.runs_dir / run_id
        protocol_path = run_dir / "protocol.json"
        with _RunLock(run_dir):
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            protocol["statistical_estimand_suggestion"] = {
                "estimation_procedure": suggestion.estimation_procedure,
                "rationale": suggestion.rationale,
                "provenance": suggestion.provenance,
                "population_summary_at_proposal": population_summary,
            }
            _atomic_write_json(protocol_path, protocol)

            state = self.load(run_id)
            state.record(
                "estimation_procedure_proposed",
                f"suggested {suggestion.estimation_procedure} ({suggestion.provenance})",
            )
            self.save(state)
        return state

    def run_design_companion_turn(
        self,
        run_id: str,
        llm: LLMCallable,
        history: Sequence[CompanionMessage],
        *,
        reviewer: str | None = None,
        prompt_path: Path = DEFAULT_DESIGN_DIALOGUE_PROMPT_PATH,
    ) -> CompanionTurn:
        """One turn of the study-design dialogue (`target_trial_review` /
        `dag_review` phases). Mirrors `run_companion_turn`: owns the
        `llm(prompt)` call and the JSON reads, records nothing, writes
        nothing, takes no lock. The returned `CompanionTurn.proposed_action`
        has been through `causal.companion.actions.validate_proposed_action`,
        so it can only ever name `set_target_trial_timing`,
        `set_target_trial_estimand`, or `revise_candidate_dag_from_points`
        with in-range arguments -- and the reviewer still confirms each card.
        """
        state = self.load(run_id)
        if state.phase not in {"target_trial_review", "dag_review"}:
            raise RuntimeError(
                "The study-design dialogue requires the target_trial_review or dag_review phase"
            )
        run_dir = self.runs_dir / run_id
        protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
        hypothesis = self._read_optional_json(run_dir / "hypothesis.json")
        unresolved = [
            str(item) for item in (hypothesis.get("unresolved_covariates") or ())
        ] if isinstance(hypothesis, dict) else []
        context = DesignDialogueContext(
            source_question=str(protocol.get("source_question", "")),
            phase=state.phase,
            target_trial_gate_approved=state.gates["target_trial"].status == "approved",
            target_trial=protocol.get("target_trial") or {},
            candidate_dag=list(protocol.get("candidate_dag") or []),
            first_pass_covariates=[str(item) for item in (protocol.get("first_pass_covariates") or ())],
            unresolved_covariates=unresolved,
            correction_log=list(protocol.get("correction_log") or []),
            decision_memory=self._decision_memory(state, protocol),
            reviewer_memory=self.reviewer_memory(reviewer) if reviewer else ReviewerMemory(),
        )
        prompt = build_design_dialogue_prompt(prompt_path, history=history, context=context)
        return parse_companion_turn(llm(prompt))

    def run_companion_turn(
        self,
        run_id: str,
        llm: LLMCallable,
        history: Sequence[CompanionMessage],
        *,
        reviewer: str | None = None,
        prompt_path: Path = DEFAULT_ESTIMAND_DIALOGUE_PROMPT_PATH,
    ) -> CompanionTurn:
        """One turn of the conversational assist at the `statistical_estimand`
        gate (`causal.companion.companion`). Owns the `llm(prompt)` call and
        the JSON-artifact reads; the pure prompt-build/parse lives in the
        companion module, matching `propose_estimation_procedure` above.

        When `reviewer` is given, that reviewer's cross-study memory
        (`causal.companion.reviewer_memory`) is loaded read-only and added to
        the prompt as advisory context -- their prior exclusions and estimand
        choices, never a decision already made here.

        Read-only with respect to run state: it takes no `_RunLock`, records
        nothing in `state.history`, and writes no file. The conversation
        transcript is the caller's to persist (`causal.companion.conversation`),
        and the one write this dialogue can lead to --
        `set_statistical_estimand_strategy` -- is a separate, explicitly
        confirmed call. The returned `CompanionTurn.proposed_action` has
        already been through `causal.companion.actions.validate_proposed_action`
        (inside `parse_companion_turn`), so it can only ever name that one
        allow-listed write with in-range arguments.

        `analytic_dataset.parquet` is never read here; the companion is fed
        the same class of on-disk JSON artifacts as
        `causal.agents.results_narrative`.
        """
        state = self.load(run_id)
        if state.phase != "estimand_review":
            raise RuntimeError("The estimand dialogue requires the estimand_review phase")
        run_dir = self.runs_dir / run_id
        study_spec_path = run_dir / "study_specification.json"
        if not study_spec_path.is_file():
            raise RuntimeError(
                "Cannot run the estimand dialogue: study_specification.json is missing "
                "-- approve the identification_estimand gate first"
            )
        protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
        query_settings = protocol.get("query_settings")
        context = EstimandDialogueContext(
            study_specification=json.loads(study_spec_path.read_text(encoding="utf-8")),
            current_settings=protocol.get("statistical_estimand_settings") or {},
            cohort_manifest=self._read_optional_json(run_dir / "cohort_manifest.json"),
            feasibility=self._read_optional_json(run_dir / "feasibility.json"),
            followup_audit=self._read_optional_json(run_dir / "followup_audit.json"),
            positivity_audit=self._read_optional_json(run_dir / "positivity_audit.json"),
            covariate_missingness=self._read_optional_json(run_dir / "covariate_missingness.json"),
            censoring_definition_approved=bool(
                (query_settings or {}).get("censoring_definition_approved", False)
            ),
            decision_memory=self._decision_memory(state, protocol),
            reviewer_memory=self.reviewer_memory(reviewer) if reviewer else ReviewerMemory(),
        )
        prompt = build_estimand_dialogue_prompt(prompt_path, history=history, context=context)
        return parse_companion_turn(llm(prompt))

    def decision_memory(self, run_id: str) -> DecisionMemory:
        """The reviewer-decision digest for a run
        (`causal.companion.decision_memory`) -- what the estimand dialogue is
        told the reviewer has already ruled on. Recomputed from `state.json`
        and `protocol.json` on every call; never cached and never written, so
        it is safe to call from a render path.
        """
        state = self.load(run_id)
        protocol_path = self.runs_dir / run_id / "protocol.json"
        protocol = (
            json.loads(protocol_path.read_text(encoding="utf-8"))
            if protocol_path.is_file()
            else {}
        )
        return self._decision_memory(state, protocol)

    def _decision_memory(self, state: RunState, protocol: dict[str, Any]) -> DecisionMemory:
        """Distil what the reviewer has already ruled on for this run
        (`causal.companion.decision_memory`) so a companion turn can be told
        not to re-propose an excluded covariate or re-open a settled choice.

        Pure derivation from `state.history`, the decided gates, and
        `protocol.json`'s `rejected_covariates` -- reads nothing new from
        disk and writes nothing, so a companion turn stays read-only.
        """
        return distill_decision_memory(
            history=state.history,
            gates={
                name: {
                    "status": gate.status,
                    "reviewer": gate.reviewer,
                    "comment": gate.comment,
                    "decided_at": gate.decided_at,
                }
                for name, gate in state.gates.items()
            },
            rejected_covariates=protocol.get("rejected_covariates") or [],
        )

    @property
    def _reviewers_dir(self) -> Path:
        return self.runs_dir.parent / "reviewers"

    def reviewer_memory(self, reviewer: str) -> ReviewerMemory:
        """This reviewer's cross-study memory
        (`causal.companion.reviewer_memory`), or an empty one if they have
        never saved anything. Read-only; safe from a render path."""
        path = self._reviewers_dir / slugify_reviewer(reviewer) / "memory.json"
        if not path.is_file():
            return ReviewerMemory(reviewer_id=slugify_reviewer(reviewer))
        return ReviewerMemory.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def remember_run_for_reviewer(self, run_id: str, reviewer: str) -> ReviewerMemory:
        """Fold this run's decisions (excluded covariates, the settled
        statistical-estimand strategy) into `reviewer`'s cross-study memory.

        Opt-in: nothing calls this automatically. Idempotent -- re-running it
        for the same study replaces that study's entries rather than
        appending (`merge_run_into_memory`). Writes only
        `artifacts/reviewers/<slug>/memory.json`; the run itself is untouched
        (no lock on the run, no `state.history` entry).
        """
        slug = slugify_reviewer(reviewer)
        state = self.load(run_id)
        run_dir = self.runs_dir / run_id
        protocol = self._read_optional_json(run_dir / "protocol.json")
        decision_memory = self._decision_memory(state, protocol)

        reviewer_dir = self._reviewers_dir / slug
        reviewer_dir.mkdir(parents=True, exist_ok=True)
        with _RunLock(reviewer_dir):
            current = self.reviewer_memory(reviewer)
            merged = merge_run_into_memory(
                current,
                study_id=state.study_id,
                decision_memory=decision_memory,
                estimand_settings=protocol.get("statistical_estimand_settings") or {},
            )
            merged = ReviewerMemory(
                reviewer_id=slug,
                exclusions=merged.exclusions,
                estimand_choices=merged.estimand_choices,
                updated_at=merged.updated_at,
            )
            _atomic_write_json(reviewer_dir / "memory.json", merged.to_dict())
        return merged

    def set_cate_cutpoint(self, run_id: str, covariate_name: str, cutpoint: float, reviewer: str) -> RunState:
        """Record a reviewer-entered cutpoint for splitting a continuous
        effect-modifier covariate into two CATE subgroup strata
        (`docs/causal_roadmap.md` milestone #4's CATE item) -- extends the
        existing human-review-gate pattern (the same pattern already used
        for covariate windows/aggregations) rather than inventing a
        clinical threshold or switching the multiple-testing unit to a
        continuous curve.

        Only a covariate with more than `causal.agents.cate.
        _MAX_CATEGORICAL_LEVELS` distinct values needs one -- a lower-
        cardinality covariate (e.g. a presence flag) is split by level
        automatically. `run_cate` raises a clear error naming which column
        still needs a cutpoint if one is missing, rather than silently
        dropping that subgroup from the prespecified family.
        """
        covariate_name = covariate_name.strip()
        if not covariate_name:
            raise ValueError("covariate_name is required")
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        run_dir = self.runs_dir / run_id
        protocol_path = run_dir / "protocol.json"
        state = self.load(run_id)
        column_name = _slugify_covariate_name(covariate_name)
        with _RunLock(run_dir):
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            cutpoints = dict(protocol.get("cate_cutpoints", {}))
            before = cutpoints.get(column_name)
            cutpoints[column_name] = float(cutpoint)
            protocol["cate_cutpoints"] = cutpoints
            _atomic_write_json(protocol_path, protocol)
        state.record(
            "cate_cutpoint_set",
            f"{covariate_name} ({column_name}): cutpoint={cutpoint} by {reviewer}",
            changes={"before": before, "after": cutpoint},
        )
        self.save(state)
        return state

    def run_cate(
        self,
        run_id: str,
        *,
        random_seed: int = 0,
    ) -> RunState:
        """Fit `CATEAgent` (a causal forest, Wager & Athey 2018) over the
        materialized `analytic_dataset.parquet` for a prespecified family
        of effect-modifier subgroups, and write `cate.json`.

        Gated on `statistical_estimand.json`'s `population_summary ==
        "CATE"` (which itself requires `estimation_procedure=
        "causal_forest_dml"`, enforced by `StatisticalEstimand`'s own
        cross-check) -- CATE is never the default; a study must have
        explicitly declared it as the primary estimand at the causal-
        question stage for this to be callable at all.

        The prespecified subgroup family resolves from `analysis_plan.
        json`'s `causal_role == "effect_modifier"`, `status == "ready"`
        covariates, matched to their materialized `cov_*` column names --
        never `hypothesis.json`'s free-text `candidate_effect_modifiers`,
        an early, ungated LLM hint, not a reviewed set fit for a family-
        wise-error-controlled test.
        """
        # Deferred, not a module-level import: `causal.agents.cate` pulls
        # in `econml`, which alone takes ~5s to import (numba/lightgbm/
        # statsmodels transitively) -- paying that on every `import causal.
        # orchestrator.workflow` (nearly the entire test suite) for a
        # method most runs never call would be a real, avoidable tax.
        from causal.agents.cate import CATEAgent

        state = self.load(run_id)
        # Matches `causal.ui.streamlit_app`'s `results_reviewable` -- CATE is
        # an optional, supplementary analysis (re-derived entirely from
        # already-materialized files, not from phase-specific state), so it
        # must stay callable after the primary estimate/report finishes and
        # advances the phase past diagnostics, not just during
        # execution/diagnostics. A narrower set here previously let the "Run
        # CATE analysis" button render enabled (per `results_reviewable`)
        # in a phase where this call would unconditionally reject it.
        if state.phase not in {"execution", "diagnostics", "results_review", "report"}:
            raise RuntimeError(
                "CATE estimation requires the execution, diagnostics, results_review, or report phase"
            )
        run_dir = self.runs_dir / run_id
        dataset_path = run_dir / "analytic_dataset.parquet"
        manifest_path = run_dir / "cohort_manifest.json"
        if not dataset_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("Run materialize_analytic_dataset first")
        estimand_path = run_dir / "statistical_estimand.json"
        if not estimand_path.is_file():
            raise RuntimeError("Run and approve the statistical_estimand gate first")
        statistical_estimand = json.loads(estimand_path.read_text(encoding="utf-8"))
        if statistical_estimand.get("population_summary") != "CATE":
            raise RuntimeError(
                "CATE estimation requires statistical_estimand.json's population_summary to be "
                f"'CATE' (found {statistical_estimand.get('population_summary')!r}) -- this "
                "study's primary estimand was not declared as CATE"
            )
        self.run_execution_readiness(run_id)
        state = self.load(run_id)
        readiness = json.loads((run_dir / "execution_readiness.json").read_text(encoding="utf-8"))
        if not readiness["causal_estimation_allowed"]:
            raise RuntimeError("Execution readiness does not permit causal estimation")
        cohort_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        covariate_columns = tuple(
            column["name"] for column in cohort_manifest["columns"]
            if column["role"] == "baseline_covariate"
        )
        if tuple(statistical_estimand["adjustment_set"]) != covariate_columns:
            raise RuntimeError(
                "statistical_estimand.json's adjustment_set no longer matches cohort_manifest."
                "json's baseline_covariate columns -- re-approve the statistical_estimand gate "
                "after the analysis plan or cohort plan changed"
            )
        analysis_plan_path = run_dir / "analysis_plan.json"
        if not analysis_plan_path.is_file():
            raise RuntimeError("Run the Data Science Agent first (analysis_plan.json is missing)")
        analysis_plan = json.loads(analysis_plan_path.read_text(encoding="utf-8"))
        subgroup_columns = tuple(
            _slugify_covariate_name(item["name"])
            for item in analysis_plan.get("covariates", ())
            if isinstance(item, dict) and item.get("status") == "ready"
            and item.get("causal_role") == "effect_modifier"
        )
        if not subgroup_columns:
            raise RuntimeError(
                "No reviewed effect-modifier covariates found in analysis_plan.json "
                "(causal_role='effect_modifier', status='ready') -- CATE requires at least one "
                "prespecified subgroup"
            )
        missing_subgroups = [column for column in subgroup_columns if column not in covariate_columns]
        if missing_subgroups:
            raise RuntimeError(
                f"Effect-modifier column(s) {missing_subgroups} are not in the materialized "
                "adjustment set -- re-approve the statistical_estimand gate after the analysis "
                "plan changed"
            )
        protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
        cutpoints = protocol.get("cate_cutpoints", {})
        dataframe = pd.read_parquet(dataset_path)
        try:
            artifact = CATEAgent().run(
                dataframe,
                study_id=state.study_id,
                covariate_columns=covariate_columns,
                subgroup_columns=subgroup_columns,
                cutpoints=cutpoints,
                random_seed=random_seed,
            )
        except ValueError as exc:
            raise RuntimeError(f"Cannot run CATE estimation: {exc}") from exc
        with _RunLock(run_dir):
            state = self.load(run_id)
            (run_dir / "cate.json").write_text(json.dumps(artifact.to_dict(), indent=2), encoding="utf-8")
            if "cate.json" not in state.artifacts:
                state.artifacts.append("cate.json")
            estimand = artifact.estimand
            state.record(
                "cate_completed",
                f"cate.json; overall_ate_from_forest={estimand.overall_ate_from_forest:.4f}; "
                f"{estimand.n_prespecified_subgroups_tested} prespecified subgroup(s) tested",
            )
            self.save(state)
        return state

    def run_feasibility(self, run_id: str, gateway: BigQueryOMOPGateway) -> RunState:
        state = self.load(run_id)
        if state.phase != "feasibility":
            raise RuntimeError("Feasibility execution requires the feasibility phase")
        if state.mode != "exploratory":
            raise RuntimeError("The provisional phenotype is restricted to exploratory runs")
        if state.gates["clinical_mapping"].status != "provisional":
            raise RuntimeError("clinical_mapping must be explicitly provisional")
        run_dir = self.runs_dir / run_id
        protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
        hypothesis = json.loads((run_dir / "hypothesis.json").read_text(encoding="utf-8"))
        specs = _load_concept_specs(protocol)
        settings = _load_query_settings(protocol)
        exposure_contrast = _load_exposure_contrast(protocol)
        for role, spec in _mapping_coverage_roles(specs, exposure_contrast):
            counts = gateway.mapping_coverage(spec, role)
            _check_mapping_coverage(role, spec, counts)
        result = gateway.feasibility(
            population=specs["population"],
            treatment=specs["treatment"],
            comparator=specs["comparator"],
            outcome=specs["outcome"],
            grace_period_hours=hypothesis["grace_period_hours"],
            follow_up_days=hypothesis["follow_up_days"],
            exposure_contrast=exposure_contrast,
            additional_population=specs.get("population_additional", ()),
            **settings,
        )
        artifact_path = run_dir / "feasibility.json"
        _atomic_write_json(artifact_path, result)
        if "feasibility.json" not in state.artifacts:
            state.artifacts.append("feasibility.json")
        state.record("feasibility_completed", "feasibility.json; no causal effect estimated")
        self.save(state)
        return state

    def run_feasibility_mcp(
        self,
        run_id: str,
        client: OMCPClient,
        datasets: BigQueryDatasets,
    ) -> RunState:
        """Run provisional aggregate feasibility through the shared OMCP server."""
        state = self.load(run_id)
        if state.phase != "feasibility":
            raise RuntimeError("Feasibility execution requires the feasibility phase")
        if state.mode != "exploratory":
            raise RuntimeError("The provisional phenotype is restricted to exploratory runs")
        if state.gates["clinical_mapping"].status != "provisional":
            raise RuntimeError("clinical_mapping must be explicitly provisional")

        self.run_execution_readiness(run_id)
        state = self.load(run_id)
        run_dir = self.runs_dir / run_id
        protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
        hypothesis = json.loads((run_dir / "hypothesis.json").read_text(encoding="utf-8"))
        specs = _load_concept_specs(protocol)
        settings = _load_query_settings(protocol)
        exposure_contrast = _load_exposure_contrast(protocol)
        grace_period_hours = hypothesis["grace_period_hours"]
        follow_up_days = hypothesis["follow_up_days"]
        context = json.loads((run_dir / "mcp_context.json").read_text(encoding="utf-8"))
        for role, spec in _mapping_coverage_roles(specs, exposure_contrast):
            coverage_sql = concept_mapping_coverage_sql(datasets, spec, role)
            coverage_response = client.causal_select(coverage_sql, "aggregate_feasibility", context)
            coverage_rows = list(csv.DictReader(io.StringIO(coverage_response.text)))
            if len(coverage_rows) != 1:
                detail = coverage_response.text.replace("\n", " ")[:500]
                raise RuntimeError(
                    f"OMCP mapping-coverage query for role '{role}' must return exactly one "
                    f"aggregate row; response={detail!r}"
                )
            coverage_counts = {
                key: int(value) for key, value in coverage_rows[0].items()
                if key and value not in (None, "")
            }
            _check_mapping_coverage(role, spec, coverage_counts)
        sql = feasibility_sql(
            datasets,
            population=specs["population"],
            treatment=specs["treatment"],
            comparator=specs["comparator"],
            outcome=specs["outcome"],
            grace_period_hours=grace_period_hours,
            follow_up_days=follow_up_days,
            exposure_contrast=exposure_contrast,
            additional_population=specs.get("population_additional", ()),
            **settings,
        )
        validation = QueryValidationAgent().run(sql, purpose="aggregate_feasibility")
        if not validation.passed:
            raise RuntimeError(
                f"Feasibility SQL failed governance validation: {', '.join(validation.violations)}"
            )
        response = client.causal_select(sql, "aggregate_feasibility", context)
        rows = list(csv.DictReader(io.StringIO(response.text)))
        if len(rows) != 1:
            detail = response.text.replace("\n", " ")[:500]
            raise RuntimeError(
                "OMCP feasibility query must return exactly one aggregate row; "
                f"response={detail!r}"
            )
        counts = {
            key: int(value) for key, value in rows[0].items()
            if key and value not in (None, "")
        }
        result = {
            "status": "provisional_cohort_feasibility",
            "execution_transport": "omcp",
            "causal_effect_estimated": False,
            "grace_period_hours": grace_period_hours,
            "follow_up_days": follow_up_days,
            "counts": counts,
            "mcp_metadata": response.metadata,
            "warnings": [
                "Concept sets reflect current human-approved mappings only; unmapped "
                "roles will fail with a clear error rather than silently degrading.",
                "Counts are for pipeline feasibility and must not be interpreted causally.",
                "outcome_events_during_grace_period counts events that occurred inside "
                "the exposure-ascertainment window itself, before treatment status was "
                "determined; a large share of outcome_events falling here warrants review "
                "before trusting the arm each such person was assigned to.",
            ],
        }
        artifact_path = run_dir / "feasibility.json"
        _atomic_write_json(artifact_path, result)
        if "feasibility.json" not in state.artifacts:
            state.artifacts.append("feasibility.json")
        state.record(
            "feasibility_completed",
            "feasibility.json via OMCP; no causal effect estimated",
        )
        self.save(state)
        return state

    @staticmethod
    def _read_optional_json(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def _state_path(self, run_id: str) -> Path:
        if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in run_id):
            raise ValueError("Invalid run_id")
        return self.runs_dir / run_id / "state.json"
