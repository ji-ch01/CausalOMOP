"""Filesystem-backed workflow manager with explicit human approval gates."""

from __future__ import annotations

import json
import csv
import io
import os
import re
import shutil
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from causal.orchestrator.state import ApprovalGate, NEXT_PHASE, RunState
from causal.agents import (
    DataScienceAgent,
    ExecutionAgent,
    HypothesisAgent,
    QueryValidationAgent,
    analysis_plan_blockers,
)
from causal.agents.dag_identification import assess_identification, build_study_specification, review_dag
from causal.agents.hypothesis import DAG_VARIABLE_DOMAINS, DAG_VARIABLE_ROLES, DAG_VARIABLE_TEMPORAL_CLASSES
from causal.agents.user_question import GeneratedHypothesis
from causal.bigquery import BigQueryOMOPGateway
from causal.config import BigQueryDatasets
from causal.roadmap import CausalQuestion, TargetTrialProtocol
from causal.data_builder import AnalyticCohortBuilder
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
    VALUE_DOMAINS,
    OMOPVariableRegistry,
    build_extraction_plan,
    extraction_plan_to_covariate_spec,
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


QUERY_ROLES = ("population", "treatment", "comparator", "outcome")

REQUIRED_GATES = {
    "target_trial_review": "target_trial",
    "dag_review": "dag_review",
    "identification_review": "identification_estimand",
    "concept_mapping": "clinical_mapping",
    "analysis_plan": "analysis_plan",
    "results_review": "results",
}

_ALL_GATE_NAMES = (
    "target_trial", "dag_review", "identification_estimand", "clinical_mapping", "analysis_plan", "results",
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
                    )
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
    20260723-143012_albumin_sepsis_28d_mimic_omop-f00fefd1
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}_{study_id}-{uuid.uuid4().hex[:8]}"


# treatment/comparator each have exactly one fallback role: `treatment_and_
# comparator`, a single shared-backbone phenotype used by both arms of an
# `add_on` design (e.g. "crystalloids" in
# `studies/albumin_sepsis_28d/protocol.json`, the comparator/backbone
# `crystalloids` alone, `crystalloids` plus `albumin` the treatment).
# population/outcome have no fallback role.
_ROLE_FALLBACK_ROLES: dict[str, tuple[str, ...]] = {
    "treatment": ("treatment_and_comparator",),
    "comparator": ("treatment_and_comparator",),
}


def _resolve_concept_spec(
    by_role: dict[str, list[dict]], role: str, exposure_contrast: str, *fallback_roles: str
) -> ConceptSetSpec | None:
    """Resolve one query role's `ConceptSetSpec` from `concept_sets`, by role.

    Tries `role` first, then each of `fallback_roles` in order (see
    `_ROLE_FALLBACK_ROLES`). Returns `None` only when *no* `concept_sets`
    entry declares any of these roles at all -- the caller decides what that
    means (a hard requirement for population/treatment/comparator, a
    `death`-table default for outcome). An entry that *does* declare the
    role but is not yet approved raises immediately instead of falling
    through to a fallback role or a default -- that would silently bypass
    the approval gate rather than genuinely finding no entry.

    Exception: `role="comparator"` under `exposure_contrast="vs_unexposed"`
    (see `causal.queries.feasibility.EXPOSURE_CONTRASTS`) never raises here,
    approved or not -- there is no OMOP concept for "absence of a drug" (e.g.
    "aspirin vs no aspirin"), so an unapproved/domain-less/`no_standard_
    concept` entry is that design's correct, final state, not a blocker.
    `_cohort_ctes` never reads this spec's `domain`/`concept_ids` in that
    mode, so whatever is (or isn't) recorded is safe to pass through as-is.
    """
    for candidate_role in (role, *fallback_roles):
        entries = by_role.get(candidate_role, [])
        if len(entries) > 1:
            raise RuntimeError(
                f"concept_sets has more than one entry declaring role '{candidate_role}' -- "
                "a query needs exactly one"
            )
        if len(entries) == 1:
            entry = entries[0]
            domain = entry.get("domain")
            domain_str = domain if isinstance(domain, str) else ""
            if candidate_role == "comparator" and exposure_contrast == "vs_unexposed":
                concept_ids = tuple(int(value) for value in entry.get("concept_ids", ()))
                return ConceptSetSpec(domain=domain_str, concept_ids=concept_ids)
            if entry.get("status") != "approved" or not domain_str:
                raise RuntimeError(
                    f"concept_sets role '{candidate_role}' must be approved with a domain "
                    "before a query can be generated for it"
                )
            concept_ids = tuple(int(value) for value in entry.get("concept_ids", ()))
            return ConceptSetSpec(domain=domain_str, concept_ids=concept_ids)
    return None


def _load_concept_specs(protocol: dict) -> dict[str, ConceptSetSpec]:
    """Resolve the human-approved, per-question concept sets a cohort/
    feasibility query is built from -- population/treatment/comparator/
    outcome -- directly from `concept_sets`, by role (see
    `_resolve_concept_spec`).

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
    concept_sets = protocol.get("concept_sets")
    if not isinstance(concept_sets, dict):
        raise RuntimeError(
            "Protocol requires concept_sets (population/treatment/comparator/outcome, "
            "each approved with a domain and concept_ids) before a cohort or feasibility "
            "query can be generated for this question"
        )
    by_role: dict[str, list[dict]] = defaultdict(list)
    for definition in concept_sets.values():
        if isinstance(definition, dict) and isinstance(definition.get("role"), str):
            by_role[definition["role"]].append(definition)

    exposure_contrast = _load_exposure_contrast(protocol)
    specs: dict[str, ConceptSetSpec] = {}
    for role in QUERY_ROLES:
        fallback_roles = _ROLE_FALLBACK_ROLES.get(role, ())
        spec = _resolve_concept_spec(by_role, role, exposure_contrast, *fallback_roles)
        if spec is None:
            if role == "outcome":
                # No role="outcome" concept set declared at all: the same
                # "all-cause death via the standard OMOP `death` table"
                # default `DataScienceAgent.run` falls back to when it finds
                # no phenotype declaring role="outcome" either -- kept in
                # sync so both readers of `concept_sets` agree on what an
                # omitted outcome role means.
                spec = ConceptSetSpec(domain="death", concept_ids=())
            elif role == "comparator" and exposure_contrast == "vs_unexposed":
                # No comparator entry declared at all -- fine under this
                # design (see `_resolve_concept_spec`'s docstring): there is
                # nothing to map in the first place.
                spec = ConceptSetSpec(domain="", concept_ids=())
            else:
                wanted = f"role {role!r}" + (f" or {fallback_roles[0]!r}" if fallback_roles else "")
                raise RuntimeError(f"concept_sets requires an approved entry with {wanted}")
        specs[role] = spec
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
    specs: dict[str, ConceptSetSpec], exposure_contrast: str
) -> list[tuple[str, ConceptSetSpec]]:
    """Which query roles the mapping-coverage gate should check: every role
    whose spec actually resolves to a concept-based domain with concept_ids
    to query. Excludes `outcome` when it defaults to the `death` table (no
    concept_ids involved) and `comparator` under `"vs_unexposed"` (see
    `causal.queries.feasibility.EXPOSURE_CONTRASTS`; there is no OMOP concept
    for "absence of a drug" to check coverage for)."""
    roles: list[tuple[str, ConceptSetSpec]] = []
    for role in QUERY_ROLES:
        if role == "comparator" and exposure_contrast == "vs_unexposed":
            continue
        spec = specs.get(role)
        if spec is not None and spec.domain in DOMAIN_TABLES and spec.concept_ids:
            roles.append((role, spec))
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
        return state

    def save(self, state: RunState) -> None:
        path = self._state_path(state.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(path)

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
            state.gates[gate].decide(status, reviewer, comment)  # type: ignore[arg-type]
            state.record("gate_decided", f"{gate}={status} by {reviewer}")
            if gate == "identification_estimand" and status == "approved":
                if "study_specification.json" not in state.artifacts:
                    state.artifacts.append("study_specification.json")
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
            raise ValueError("concept_ids must be a list of integers")

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
            temporary = protocol_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
            temporary.replace(protocol_path)

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
            temporary = protocol_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
            temporary.replace(protocol_path)

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
            temporary = protocol_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
            temporary.replace(protocol_path)

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
            raise ValueError("concept_ids must be a list of integers")
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
            temporary = protocol_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
            temporary.replace(protocol_path)

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
            temporary = protocol_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
            temporary.replace(protocol_path)

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
            raise ValueError("window_start_hours and window_end_hours must be integers")
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
            temporary = protocol_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
            temporary.replace(protocol_path)

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
        temporary = spec_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(spec_path)

    def _write_hypothesis(self, run_id: str, protocol: dict) -> None:
        artifact = HypothesisAgent().run(protocol).to_dict()
        artifact_path = self.runs_dir / run_id / "hypothesis.json"
        temporary = artifact_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        temporary.replace(artifact_path)

    def run_data_science_agent(
        self,
        run_id: str,
        concept_lookup: OMCPClient | None = None,
    ) -> RunState:
        state = self.load(run_id)
        if state.phase not in {"concept_mapping", "feasibility", "analysis_plan"}:
            raise RuntimeError("Data Science Agent requires concept_mapping, feasibility, or analysis_plan phase")
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
            temporary = artifact_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
            temporary.replace(artifact_path)
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
        temporary = artifact_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(registry.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(artifact_path)
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
            if state.phase not in {"feasibility", "analysis_plan", "execution"}:
                raise RuntimeError(
                    "Execution readiness requires feasibility, analysis_plan, or execution phase"
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
            temporary = artifact_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
            temporary.replace(artifact_path)
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
            context_path = run_dir / "mcp_context.json"
            context_temporary = context_path.with_suffix(".tmp")
            context_temporary.write_text(
                json.dumps(mcp_context, indent=2), encoding="utf-8"
            )
            context_temporary.replace(context_path)
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
        if state.phase not in {"analysis_plan", "execution"}:
            raise RuntimeError("Cohort planning requires analysis_plan or execution phase")
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
        temporary = artifact_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        temporary.replace(artifact_path)
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
                "", "## Normalized causal question", "",
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
            temporary = protocol_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
            temporary.replace(protocol_path)

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
            temporary = protocol_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
            temporary.replace(protocol_path)

        detail = f"censoring_definition_approved: {before} -> True by {reviewer}"
        if comment and comment.strip():
            detail += f" ({comment.strip()})"
        state.record(
            "censoring_definition_approved", detail,
            changes={"before": before, "after": True},
        )
        self._invalidate_gate(state, "analysis_plan", "censoring_definition_approved changed")
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
            **settings,
        )
        artifact_path = run_dir / "feasibility.json"
        temporary = artifact_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary.replace(artifact_path)
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
        temporary = artifact_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary.replace(artifact_path)
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
