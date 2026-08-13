"""DAG node -> OMOP variable definition -> deterministic SQL plan.

This module is the first version of the registry `docs/causal_roadmap.md`'s
migration step 4 calls for: *"Build an OMOP variable registry from approved
DAG nodes rather than fixed phenotype names."* Today an OMOP mapping is
scattered across `protocol.json["concept_sets"]` (population/treatment/
comparator/outcome/other phenotypes), `protocol.json["candidate_dag"]`
(baseline covariates -- see `causal.agents.hypothesis.DAGVariable`), and
`analysis_plan.json` (the review-only projection `causal.agents.data_science`
derives from both). Nothing ties "this DAG node" to "this OMOP definition" to
"this SQL" as one typed, versioned, serializable contract -- this module adds
that layer without replacing any of the above.

Design invariants (see `docs/causal_roadmap.md` and the module docstring of
`causal.queries.feasibility` for the sibling conventions this mirrors):

- A registry entry can exist in a fully unmapped, unobserved, or
  domain-unknown state -- the DAG must never depend on OMOP data
  availability (an unobservable confounder stays a modeled variable and an
  identification limitation, not a silently dropped one).
- A *candidate* concept (something a lookup tool proposed) is never confused
  with an *approved* one: `candidate_concept_ids` and `root_concept_ids` are
  separate fields, and only `root_concept_ids` ever reaches generated SQL.
- Nothing in this module ever sets `mapping_status="approved"` on its own.
  Every function here only *reads* a status that was already decided
  elsewhere (a human reviewer via `causal.orchestrator.workflow`, or a
  pre-existing approved `protocol.json` entry) -- see `parse_variable`'s
  `provenance`/`reviewer` check for the one place this is actually enforced,
  not just assumed.
- SQL is only ever generated from a `mapping_status="approved"`,
  `observed=True` entry (`build_extraction_plan` raises otherwise) -- knowing
  a mapping is *proposed* must never be enough to run it.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from causal.agents.data_science import EXCLUDED_FROM_ADJUSTMENT_ROLES
from causal.agents.hypothesis import DAG_VARIABLE_ROLES
from causal.queries.feasibility import (
    COVARIATE_AGGREGATIONS_BY_DOMAIN,
    DOMAIN_TABLES,
    PRESENCE_AGGREGATIONS,
    ConceptSetSpec,
    CovariateSpec,
)

SCHEMA_VERSION = "1.0"

# A registry entry is either a phenotype (population/treatment/comparator/
# outcome/... -- a `concept_sets` entry, whose temporal window is the
# protocol-level grace period/follow-up, not a per-variable one) or a
# covariate (a `candidate_dag` baseline variable, whose window/aggregation
# are reviewed individually). The two need different validation: only a
# covariate's `causal_role` is checked against `DAG_VARIABLE_ROLES`/
# forbidden-adjustment roles, since a phenotype's role
# (population/treatment/...) lives in a different, open-ended vocabulary
# `causal.agents.data_science._phenotype_from_concept_set` already treats as
# free text.
VARIABLE_KINDS = ("phenotype", "covariate")

# The two domains this codebase's SQL layer actually resolves a numeric value
# from (`t.value_as_number`, hardcoded in `causal.queries.feasibility`). Not
# duplicated from `COVARIATE_AGGREGATIONS_BY_DOMAIN` -- derived from it, so
# adding a new value-typed domain there does not require a second edit here.
VALUE_DOMAINS = tuple(
    domain for domain, aggregations in COVARIATE_AGGREGATIONS_BY_DOMAIN.items()
    if not set(aggregations) <= set(PRESENCE_AGGREGATIONS)
)

# Fixed OMOP tables/columns for the two roles that do not resolve through a
# concept domain (`causal.queries.feasibility.DOMAIN_TABLES`): all-cause
# mortality lives in the standard `death` table, and the study's time-zero
# anchor is a visit-level definition, not a concept set. Mirrors
# `causal.agents.data_science.DataScienceAgent.run`'s own
# `all_cause_death`/`index_event` scaffolding fallbacks.
_FIXED_DOMAIN_TABLES: dict[str, tuple[tuple[str, ...], str, str, str | None]] = {
    "death": (("death",), "", "death_date", None),
    "standard": (("visit_detail", "visit_occurrence"), "", "", None),
}

ALLOWED_DOMAINS = tuple(COVARIATE_AGGREGATIONS_BY_DOMAIN) + tuple(_FIXED_DOMAIN_TABLES)

# The single valid temporal anchor this first version supports -- every
# window in this codebase is relative to `index_datetime`
# (`causal.queries.feasibility._cohort_ctes`). A closed, one-element tuple
# rather than a free string so a future second anchor (e.g. "treatment_start"
# for a mediator ascertained relative to treatment, not baseline index) is a
# deliberate extension, not a silent typo.
TEMPORAL_ANCHORS = ("index_datetime",)

MAPPING_STATUS_APPROVED = "approved"

# Provenance values this module treats as "produced without a human in the
# loop" -- an entry claiming `mapping_status="approved"` from one of these
# must also carry a non-empty `reviewer`, which is the concrete, testable
# form of "no LLM/tool ever auto-approves a clinical mapping" (see module
# docstring). Anything else (e.g. "protocol_declared", the default for
# mappings already approved directly in `protocol.json` before this module
# existed) is trusted the same way `causal.agents.hypothesis`/
# `causal.agents.data_science` already trust an upstream `status="approved"`
# without re-demanding a reviewer name -- this module adds a new invariant
# going forward, it does not retroactively invalidate existing runs.
_UNSUPERVISED_PROVENANCE = ("llm_proposed", "omcp_candidate")


def default_missingness_rule(domain: str, aggregation: str) -> str:
    """The one clinically-correct missingness convention for an aggregation
    (see `causal.queries.feasibility._covariate_ctes_columns_and_joins`'s own
    docstring): presence/count aggregations `COALESCE` to 0 (absence of a
    coded event is a real "not present" fact); value aggregations stay NULL
    (a missing lab value must never be imputed to 0). Returns "" when
    `aggregation` is not yet reviewed (`"definition_pending"`) or `domain`
    has no aggregation semantics (`"death"`/`"standard"`) -- there is no
    missingness rule to state yet.
    """
    if domain not in COVARIATE_AGGREGATIONS_BY_DOMAIN:
        return ""
    if aggregation in PRESENCE_AGGREGATIONS:
        return "zero_when_absent"
    if aggregation in COVARIATE_AGGREGATIONS_BY_DOMAIN[domain]:
        return "null_when_missing"
    return ""


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(label: str) -> str:
    slug = _SLUG_RE.sub("_", label.strip().lower()).strip("_")
    return slug or "variable"


def _domain_table_columns(domain: str) -> tuple[tuple[str, ...], str, str, str | None]:
    """Resolve `(omop_tables, concept_id_column, datetime_column, value_column)`
    for a domain -- `causal.queries.feasibility.DOMAIN_TABLES` plus the two
    fixed, non-concept-based roles (`death`, `standard`)."""
    if domain in _FIXED_DOMAIN_TABLES:
        return _FIXED_DOMAIN_TABLES[domain]
    if domain not in DOMAIN_TABLES:
        raise ValueError(f"Unknown OMOP domain: {domain!r}; must be one of {ALLOWED_DOMAINS}")
    table, concept_col, dt_col = DOMAIN_TABLES[domain]
    value_col = "value_as_number" if domain in VALUE_DOMAINS else None
    return (table,), concept_col, dt_col, value_col


@dataclass(frozen=True)
class OMOPVariableDefinition:
    """One DAG node's OMOP operationalization: a fully serializable,
    versioned contract distinguishing a *proposed* mapping from an *approved*
    one, and "mapping approved" from "data available" and from "SQL can be
    generated" (see `is_data_availability_known`/`is_sql_executable`).

    Field groups, in the order the task brief lists them:

    - Identity: `variable_id` (stable, generated once -- see `_slugify` --
      and never recomputed from a later label change) vs `dag_node_label`
      (the current `candidate_dag` label or `concept_sets` name this entry
      operationalizes; may be renamed independently of `variable_id`).
    - Causal context: `variable_kind` (`"phenotype"` or `"covariate"` --
      see `VARIABLE_KINDS`), `causal_role`, `observed`.
    - OMOP shape: `domain`, `omop_tables`, `concept_id_column`,
      `datetime_column`, `value_column`.
    - Concept selection: `root_concept_ids` (approved roots only --
      candidates never belong here, see `candidate_concept_ids`),
      `include_descendants`, `excluded_concept_ids` (always wins over
      inclusion by descendants).
    - Timing: `window_start_hours`/`window_end_hours` (both-or-neither,
      `None` for most phenotypes -- their window is the protocol-level grace
      period/follow-up, not a per-variable one), `temporal_anchor`.
    - Extraction rules: `aggregation`, `expected_unit`, `missingness_rule`.
    - Review state: `mapping_status`, `reviewer`, `review_comment`,
      `provenance`.
    - Versioning: `schema_version`, `vocabulary_version`.
    - The two axes explicitly kept distinct from mapping approval:
      `candidate_concept_ids` (proposed, unreviewed) and
      `available_in_data` (set by an audit -- see `causal.mapping.audit` --
      `None` until one has actually run; a mapping can be approved and
      still turn out to have zero rows in a given CDM snapshot).
    """

    variable_id: str
    dag_node_label: str
    variable_kind: str
    causal_role: str
    observed: bool = True
    domain: str = ""
    omop_tables: tuple[str, ...] = ()
    concept_id_column: str = ""
    datetime_column: str = ""
    value_column: str | None = None
    root_concept_ids: tuple[int, ...] = ()
    include_descendants: bool = True
    excluded_concept_ids: tuple[int, ...] = ()
    window_start_hours: int | None = None
    window_end_hours: int | None = None
    temporal_anchor: str = "index_datetime"
    aggregation: str = "definition_pending"
    expected_unit: str | None = None
    missingness_rule: str = ""
    mapping_status: str = "unmapped"
    reviewer: str = ""
    review_comment: str = ""
    provenance: str = "unspecified"
    schema_version: str = SCHEMA_VERSION
    vocabulary_version: str | None = None
    candidate_concept_ids: tuple[int, ...] = ()
    available_in_data: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_variable(data: dict[str, Any]) -> OMOPVariableDefinition:
    """Validate a raw dict (JSON-loaded or hand-built) against every
    structural invariant this registry enforces, then construct the
    dataclass. The single validating constructor: both `OMOPVariableRegistry.
    from_dict` (deserialization) and the `variable_from_*` adapters below
    (which build a registry-shaped dict from an existing artifact, then call
    this) go through here, so there is exactly one place these rules live.

    Raises `ValueError` with a specific message on the first violation found
    -- mirrors `causal.agents.hypothesis.HypothesisAgent._candidate_dag`'s
    validate-then-construct convention.
    """
    variable_id = data.get("variable_id")
    if not isinstance(variable_id, str) or not variable_id.strip():
        raise ValueError("OMOPVariableDefinition requires a non-empty variable_id")
    dag_node_label = data.get("dag_node_label")
    if not isinstance(dag_node_label, str) or not dag_node_label.strip():
        raise ValueError("OMOPVariableDefinition requires a non-empty dag_node_label")
    variable_kind = data.get("variable_kind")
    if variable_kind not in VARIABLE_KINDS:
        raise ValueError(f"variable_kind must be one of {VARIABLE_KINDS}")
    causal_role = data.get("causal_role")
    if not isinstance(causal_role, str) or not causal_role.strip():
        raise ValueError(f"'{variable_id}' requires a non-empty causal_role")
    if variable_kind == "covariate" and causal_role not in DAG_VARIABLE_ROLES:
        raise ValueError(f"'{variable_id}' is a covariate; causal_role must be one of {DAG_VARIABLE_ROLES}")
    observed = data.get("observed", True)
    if not isinstance(observed, bool):
        raise ValueError(f"'{variable_id}'.observed must be a boolean")

    domain = data.get("domain", "") or ""
    if not isinstance(domain, str):
        raise ValueError(f"'{variable_id}'.domain must be a string")
    if domain and domain not in ALLOWED_DOMAINS:
        raise ValueError(f"'{variable_id}'.domain must be '' or one of {ALLOWED_DOMAINS}")

    omop_tables = tuple(data.get("omop_tables", ()) or ())
    concept_id_column = str(data.get("concept_id_column", "") or "")
    datetime_column = str(data.get("datetime_column", "") or "")
    value_column = data.get("value_column")
    if value_column is not None and not isinstance(value_column, str):
        raise ValueError(f"'{variable_id}'.value_column must be a string or null")
    if domain:
        expected_tables, expected_concept_col, expected_dt_col, expected_value_col = _domain_table_columns(domain)
        if omop_tables and tuple(omop_tables) != expected_tables:
            raise ValueError(
                f"'{variable_id}' declares domain '{domain}' but omop_tables={omop_tables!r}; "
                f"expected {expected_tables!r}"
            )
        omop_tables = expected_tables
        if concept_id_column and concept_id_column != expected_concept_col:
            raise ValueError(
                f"'{variable_id}' declares domain '{domain}' but concept_id_column={concept_id_column!r}; "
                f"expected {expected_concept_col!r}"
            )
        concept_id_column = expected_concept_col
        if datetime_column and datetime_column != expected_dt_col:
            raise ValueError(
                f"'{variable_id}' declares domain '{domain}' but datetime_column={datetime_column!r}; "
                f"expected {expected_dt_col!r}"
            )
        datetime_column = expected_dt_col
        if value_column and value_column != expected_value_col:
            raise ValueError(
                f"'{variable_id}' declares domain '{domain}' but value_column={value_column!r}; "
                f"only {expected_value_col!r} is supported by the existing SQL generator"
            )
        value_column = expected_value_col

    def _int_tuple(field: str) -> tuple[int, ...]:
        raw = data.get(field, ()) or ()
        try:
            return tuple(int(value) for value in raw)
        except (TypeError, ValueError):
            raise ValueError(f"'{variable_id}'.{field} must be a list of integers")

    root_concept_ids = _int_tuple("root_concept_ids")
    excluded_concept_ids = _int_tuple("excluded_concept_ids")
    candidate_concept_ids = _int_tuple("candidate_concept_ids")
    include_descendants = data.get("include_descendants", True)
    if not isinstance(include_descendants, bool):
        raise ValueError(f"'{variable_id}'.include_descendants must be a boolean")

    mapping_status = data.get("mapping_status", "unmapped")
    if not isinstance(mapping_status, str) or not mapping_status.strip():
        raise ValueError(f"'{variable_id}'.mapping_status must be a non-empty string")
    mapping_status = mapping_status.strip()
    if mapping_status == MAPPING_STATUS_APPROVED:
        if not observed:
            raise ValueError(
                f"'{variable_id}' cannot have mapping_status='approved' while observed=False -- "
                "an unobservable/latent variable is not executable"
            )
        if not domain:
            raise ValueError(f"'{variable_id}' cannot have mapping_status='approved' with no domain")
        if domain not in _FIXED_DOMAIN_TABLES and not root_concept_ids:
            raise ValueError(f"'{variable_id}' cannot have mapping_status='approved' with no root_concept_ids")

    reviewer = str(data.get("reviewer", "") or "")
    review_comment = str(data.get("review_comment", "") or "")
    provenance = str(data.get("provenance", "unspecified") or "unspecified")
    if mapping_status == MAPPING_STATUS_APPROVED and provenance in _UNSUPERVISED_PROVENANCE and not reviewer.strip():
        raise ValueError(
            f"'{variable_id}' has mapping_status='approved' with provenance={provenance!r} "
            "(an automated/candidate source) but no reviewer -- a clinical mapping may never be "
            "approved without a named human reviewer"
        )

    window_start_hours = data.get("window_start_hours")
    window_end_hours = data.get("window_end_hours")
    if (window_start_hours is None) != (window_end_hours is None):
        raise ValueError(f"'{variable_id}' must set both window_start_hours and window_end_hours, or neither")
    if window_start_hours is not None:
        try:
            window_start_hours = int(window_start_hours)
            window_end_hours = int(window_end_hours)
        except (TypeError, ValueError):
            raise ValueError(f"'{variable_id}'.window_start_hours/window_end_hours must be integers")
        if not window_start_hours <= window_end_hours <= 0:
            raise ValueError(
                f"'{variable_id}' window must satisfy window_start_hours <= window_end_hours <= 0"
            )

    temporal_anchor = data.get("temporal_anchor", "index_datetime")
    if temporal_anchor not in TEMPORAL_ANCHORS:
        raise ValueError(f"'{variable_id}'.temporal_anchor must be one of {TEMPORAL_ANCHORS}")

    aggregation = data.get("aggregation", "definition_pending")
    if not isinstance(aggregation, str) or not aggregation.strip():
        raise ValueError(f"'{variable_id}'.aggregation must be a non-empty string")
    aggregation = aggregation.strip()
    # `COVARIATE_AGGREGATIONS_BY_DOMAIN` (presence/count vs. value aggregates)
    # is a covariate-only vocabulary. A phenotype's ascertainment
    # (population/treatment/comparator/outcome) is always mechanically
    # "did a qualifying row occur in the window" -- `causal.queries.
    # feasibility._cohort_ctes` never references `value_as_number` for any of
    # them, regardless of OMOP domain -- so a measurement/observation-domain
    # phenotype (e.g. a lab-defined outcome) is not held to the same
    # domain/aggregation compatibility rule a covariate is.
    if variable_kind == "covariate" and aggregation != "definition_pending" and domain in COVARIATE_AGGREGATIONS_BY_DOMAIN:
        allowed = COVARIATE_AGGREGATIONS_BY_DOMAIN[domain]
        if aggregation not in allowed:
            raise ValueError(f"'{variable_id}' has domain '{domain}'; aggregation must be one of {allowed}")

    expected_unit = data.get("expected_unit")
    if expected_unit is not None and not isinstance(expected_unit, str):
        raise ValueError(f"'{variable_id}'.expected_unit must be a string or null")

    missingness_rule = data.get("missingness_rule", "") or ""
    if not isinstance(missingness_rule, str):
        raise ValueError(f"'{variable_id}'.missingness_rule must be a string")
    if variable_kind == "covariate" and aggregation != "definition_pending" and domain in COVARIATE_AGGREGATIONS_BY_DOMAIN:
        expected_missingness = default_missingness_rule(domain, aggregation)
        if missingness_rule and missingness_rule != expected_missingness:
            raise ValueError(
                f"'{variable_id}' aggregation '{aggregation}' requires missingness_rule "
                f"'{expected_missingness}' (the SQL generator's actual behavior), not '{missingness_rule}'"
            )
        missingness_rule = expected_missingness

    schema_version = str(data.get("schema_version", SCHEMA_VERSION))
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported OMOPVariableDefinition schema_version: {schema_version}")
    vocabulary_version = data.get("vocabulary_version")
    if vocabulary_version is not None and not isinstance(vocabulary_version, str):
        raise ValueError(f"'{variable_id}'.vocabulary_version must be a string or null")

    available_in_data = data.get("available_in_data")
    if available_in_data is not None and not isinstance(available_in_data, bool):
        raise ValueError(f"'{variable_id}'.available_in_data must be a boolean or null")

    return OMOPVariableDefinition(
        variable_id=variable_id.strip(),
        dag_node_label=dag_node_label.strip(),
        variable_kind=variable_kind,
        causal_role=causal_role.strip(),
        observed=observed,
        domain=domain,
        omop_tables=tuple(omop_tables),
        concept_id_column=concept_id_column,
        datetime_column=datetime_column,
        value_column=value_column,
        root_concept_ids=root_concept_ids,
        include_descendants=include_descendants,
        excluded_concept_ids=excluded_concept_ids,
        window_start_hours=window_start_hours,
        window_end_hours=window_end_hours,
        temporal_anchor=temporal_anchor,
        aggregation=aggregation,
        expected_unit=expected_unit,
        missingness_rule=missingness_rule,
        mapping_status=mapping_status,
        reviewer=reviewer,
        review_comment=review_comment,
        provenance=provenance,
        schema_version=schema_version,
        vocabulary_version=vocabulary_version,
        candidate_concept_ids=candidate_concept_ids,
        available_in_data=available_in_data,
    )


def is_sql_executable(definition: OMOPVariableDefinition) -> bool:
    """The technical capacity to generate SQL for this variable right now --
    distinct from `mapping_status == "approved"` alone (see module
    docstring): an approved mapping on an unobserved variable, or one with no
    domain, or a covariate with no reviewed window/aggregation, still cannot
    produce SQL. A pure, re-derived check (never a stored/stale field) so it
    can never drift from what `build_extraction_plan` would actually accept.
    """
    if not definition.observed:
        return False
    if definition.mapping_status != MAPPING_STATUS_APPROVED:
        return False
    if not definition.domain:
        return False
    if definition.domain not in _FIXED_DOMAIN_TABLES and not definition.root_concept_ids:
        return False
    if definition.variable_kind == "covariate":
        if definition.window_start_hours is None or definition.window_end_hours is None:
            return False
        if definition.aggregation == "definition_pending":
            return False
    return True


def variable_from_concept_set(name: str, definition: dict[str, Any]) -> OMOPVariableDefinition:
    """Adapter: one `protocol.json["concept_sets"][name]` entry (population/
    treatment/comparator/outcome/... -- see `causal.agents.data_science.
    DataScienceAgent._phenotype_from_concept_set`, which this mirrors) ->
    one registry entry.

    Phenotypes are always `observed=True` (a study cannot declare a
    population/exposure/outcome it cannot see) and always
    `aggregation="presence_in_window"` -- mechanically, `has_treatment`/
    `has_comparator`/`outcome_event` in `causal.queries.feasibility` are all
    "did a qualifying event occur", the same shape as a covariate's
    `presence_in_window`. Their window is left unset (`None`): the relevant
    windows (grace period, follow-up) are protocol-level, not per-variable,
    and are not duplicated here.

    `domain == ""` is otherwise rejected, with one exception: a
    `role="comparator"` entry whose `status` is `"no_standard_concept"` or
    unset (`"missing"`) -- the exact states `causal.orchestrator.workflow.
    _load_concept_specs`'s `vs_unexposed` carve-out tolerates for an
    `exposure_contrast="vs_unexposed"` design (e.g. "aspirin vs no aspirin"),
    where there is no OMOP concept for "absence of a drug" and so nothing
    could ever be mapped for this entry in the first place.
    """
    if not isinstance(definition, dict):
        raise ValueError(f"concept_sets['{name}'] must be an object")
    role = definition.get("role")
    if not isinstance(role, str) or not role.strip():
        raise ValueError(f"concept_sets['{name}'] requires a role")
    status = str(definition.get("status", "missing"))
    domain = definition.get("domain")
    domain_str = domain if isinstance(domain, str) else None
    comparator_domain_less_ok = role == "comparator" and status in {"no_standard_concept", "missing"}
    if domain_str is None or (
        domain_str not in ALLOWED_DOMAINS and not (domain_str == "" and comparator_domain_less_ok)
    ):
        raise ValueError(f"concept_sets['{name}'] requires domain to be one of {ALLOWED_DOMAINS}")
    domain = domain_str
    concept_ids = definition.get("concept_ids", ())
    mapping_status = MAPPING_STATUS_APPROVED if status == "approved" else status
    return parse_variable({
        "variable_id": _slugify(name),
        "dag_node_label": name,
        "variable_kind": "phenotype",
        "causal_role": role,
        "observed": True,
        "domain": domain,
        "root_concept_ids": concept_ids if mapping_status == MAPPING_STATUS_APPROVED else (),
        "candidate_concept_ids": concept_ids if mapping_status != MAPPING_STATUS_APPROVED else (),
        "include_descendants": bool(definition.get("include_descendants", True)),
        "excluded_concept_ids": definition.get("excluded_concept_ids", ()),
        "aggregation": "presence_in_window" if domain not in _FIXED_DOMAIN_TABLES else "definition_pending",
        "mapping_status": mapping_status,
        "reviewer": definition.get("reviewer", ""),
        "review_comment": definition.get("comment", ""),
        "provenance": definition.get("provenance", "protocol_declared"),
    })


def variable_from_candidate_dag_entry(
    entry: dict[str, Any], adjustment_role: str = ""
) -> OMOPVariableDefinition:
    """Adapter: one `protocol.json["candidate_dag"]` entry (see
    `causal.agents.hypothesis.DAGVariable`) -> one registry entry.

    `adjustment_role` (optional -- `"core"`/`"alternative"`/`"unused"`, see
    `causal.agents.dag_identification.AdjustmentSetSummary`) is carried
    through into `review_comment` purely for human-readable context in the
    registry dump; it is not itself validated here, since it is advisory
    (see that summary's own docstring on why "unused" is not "wrong"), not a
    gate.
    """
    if not isinstance(entry, dict):
        raise ValueError("candidate_dag entry must be an object")
    label = entry.get("label")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("candidate_dag entry requires a non-empty label")
    role = entry.get("role")
    domain = entry.get("domain") or ""
    mapping_status = entry.get("mapping_status", "unmapped")
    concept_ids = entry.get("concept_ids", ())
    aggregation = entry.get("aggregation", "definition_pending")
    comment = entry.get("rationale", "")
    if adjustment_role:
        comment = f"{comment} [adjustment_role={adjustment_role}]" if comment else f"[adjustment_role={adjustment_role}]"
    return parse_variable({
        "variable_id": _slugify(label),
        "dag_node_label": label,
        "variable_kind": "covariate",
        "causal_role": role,
        "observed": entry.get("observed", True),
        "domain": domain,
        "root_concept_ids": concept_ids if mapping_status == MAPPING_STATUS_APPROVED else (),
        "candidate_concept_ids": concept_ids if mapping_status != MAPPING_STATUS_APPROVED else (),
        "include_descendants": bool(entry.get("include_descendants", True)),
        "excluded_concept_ids": entry.get("excluded_concept_ids", ()),
        "window_start_hours": entry.get("window_start_hours"),
        "window_end_hours": entry.get("window_end_hours"),
        "aggregation": aggregation,
        "mapping_status": mapping_status,
        "reviewer": entry.get("mapping_reviewer", ""),
        "review_comment": comment,
        "provenance": entry.get("provenance", "protocol_declared"),
    })


def variable_from_covariate_plan(item: dict[str, Any]) -> OMOPVariableDefinition:
    """Adapter: one `analysis_plan.json["covariates"]` entry (see
    `causal.agents.data_science.CovariatePlan`) -> one registry entry.

    Distinct from `variable_from_candidate_dag_entry`: a `CovariatePlan` is
    already the fully-resolved projection `DataScienceAgent._covariate`
    computed -- its `window_start_hours`/`window_end_hours` are always
    concrete integers (auto-derived from `temporal_classification`, or a
    reviewer's explicit override; see that method's docstring), never the
    `None` a not-yet-reviewed `candidate_dag` entry would have. Reading from
    this already-resolved shape is what lets `causal.orchestrator.workflow.
    _load_covariate_specs` consume the registry's `is_sql_executable`/
    `build_extraction_plan` without this module re-deriving that
    auto-derivation logic itself (which would duplicate, and risk drifting
    from, `DataScienceAgent`'s own rule).

    `observed` is inferred from `status != "unobserved"` -- `CovariatePlan`
    has no separate boolean field for it (see that dataclass).
    """
    if not isinstance(item, dict):
        raise ValueError("covariate plan entry must be an object")
    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("covariate plan entry requires a non-empty name")
    domain = item.get("domain") or ""
    mapping_status = item.get("mapping_status", "unmapped")
    concept_ids = item.get("concept_ids", ())
    return parse_variable({
        "variable_id": _slugify(name),
        "dag_node_label": name,
        "variable_kind": "covariate",
        "causal_role": item.get("causal_role"),
        "observed": item.get("status") != "unobserved",
        "domain": domain,
        "root_concept_ids": concept_ids if mapping_status == MAPPING_STATUS_APPROVED else (),
        "candidate_concept_ids": concept_ids if mapping_status != MAPPING_STATUS_APPROVED else (),
        "window_start_hours": item.get("window_start_hours"),
        "window_end_hours": item.get("window_end_hours"),
        "aggregation": item.get("aggregation", "definition_pending"),
        "mapping_status": mapping_status,
        "provenance": "protocol_declared",
    })


@dataclass(frozen=True)
class OMOPVariableRegistry:
    """A versioned, serializable snapshot of every OMOP variable definition
    known for one study -- the single source of truth this module exists to
    provide (see module docstring). `from_protocol` is the backward-compatible
    adapter: it never requires `protocol.json` to change shape, so an
    existing run's `protocol.json` remains readable and a registry can always
    be rebuilt fresh from it (the same "rebuild, don't trust a stale copy"
    convention `causal.orchestrator.workflow._write_hypothesis` already uses
    for `hypothesis.json`).
    """

    schema_version: str
    study_id: str
    variables: tuple[OMOPVariableDefinition, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "variables": [variable.to_dict() for variable in self.variables],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OMOPVariableRegistry":
        study_id = data.get("study_id")
        if not isinstance(study_id, str) or not study_id.strip():
            raise ValueError("OMOPVariableRegistry requires a non-empty study_id")
        return cls(
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            study_id=study_id,
            variables=tuple(parse_variable(item) for item in data.get("variables", ())),
        )

    @classmethod
    def from_protocol(cls, protocol: dict[str, Any]) -> "OMOPVariableRegistry":
        study_id = protocol.get("study_id")
        if not isinstance(study_id, str) or not study_id.strip():
            raise ValueError("Protocol requires a non-empty study_id")
        concept_sets = protocol.get("concept_sets", {})
        if not isinstance(concept_sets, dict):
            raise ValueError("Protocol requires concept_sets")
        candidate_dag = protocol.get("candidate_dag", ())
        phenotypes = tuple(
            variable_from_concept_set(name, item)
            for name, item in concept_sets.items()
            if isinstance(item, dict)
        )
        covariates = tuple(
            variable_from_candidate_dag_entry(item)
            for item in candidate_dag
            if isinstance(item, dict)
        )
        return cls(schema_version=SCHEMA_VERSION, study_id=study_id.strip(), variables=phenotypes + covariates)

    def get(self, variable_id: str) -> OMOPVariableDefinition | None:
        return next((variable for variable in self.variables if variable.variable_id == variable_id), None)

    def resolve_role(self, role: str, *fallback_roles: str) -> OMOPVariableDefinition | None:
        """Resolve one query role (population/treatment/comparator/outcome,
        or a fallback role such as `treatment_and_comparator`) to its
        registry entry, indexed by `OMOPVariableDefinition.causal_role` --
        for a phenotype, `causal_role` *is* the query role (see
        `variable_from_concept_set`), the same value `causal.orchestrator.
        workflow._load_concept_specs` used to group a raw `concept_sets`
        dict by before this method existed.

        Tries `role` first, then each of `fallback_roles` in order, so an
        exact-role entry always takes priority over a shared-backbone
        fallback. Raises `RuntimeError` if more than one variable declares
        the same candidate role (a query needs exactly one); returns `None`
        only when *no* variable declares any of these roles at all -- the
        caller decides what that means (a hard requirement, or a default).
        """
        for candidate_role in (role, *fallback_roles):
            matches = tuple(variable for variable in self.variables if variable.causal_role == candidate_role)
            if len(matches) > 1:
                raise RuntimeError(
                    f"registry has more than one variable declaring causal_role {candidate_role!r} -- "
                    "a query needs exactly one"
                )
            if len(matches) == 1:
                return matches[0]
        return None


@dataclass(frozen=True)
class ExtractionPlan:
    """The fully-resolved, SQL-ready shape of one approved
    `OMOPVariableDefinition` -- the intermediate layer between the registry
    and the existing BigQuery builders (`OMOPVariableDefinition ->
    ExtractionPlan -> causal.queries.feasibility`). Deliberately a plain,
    smaller contract than `OMOPVariableDefinition`: it drops every
    review/provenance field that has no SQL-generation meaning, keeping only
    what `extraction_plan_to_concept_set_spec`/`extraction_plan_to_covariate_spec`
    need.
    """

    variable_id: str
    domain: str
    omop_tables: tuple[str, ...]
    concept_id_column: str
    datetime_column: str
    value_column: str | None
    root_concept_ids: tuple[int, ...]
    include_descendants: bool
    excluded_concept_ids: tuple[int, ...]
    window_start_hours: int | None
    window_end_hours: int | None
    aggregation: str
    missingness_rule: str


def build_extraction_plan(definition: OMOPVariableDefinition) -> ExtractionPlan:
    """Resolve an approved `OMOPVariableDefinition` into a SQL-ready
    `ExtractionPlan`. Requires `mapping_status == "approved"` and
    `observed == True` -- a proposed mapping or a declared-unobservable
    variable must never reach SQL generation (see module docstring and
    `is_sql_executable`).
    """
    if definition.mapping_status != MAPPING_STATUS_APPROVED:
        raise ValueError(
            f"'{definition.variable_id}' has mapping_status={definition.mapping_status!r}; "
            "a mapping must be approved before generating its extraction plan"
        )
    if not definition.observed:
        raise ValueError(
            f"'{definition.variable_id}' is not observed -- an unobservable/latent variable "
            "cannot be extracted, it remains a modeled identification limitation"
        )
    if definition.domain not in COVARIATE_AGGREGATIONS_BY_DOMAIN:
        raise ValueError(
            f"'{definition.variable_id}' has domain {definition.domain!r}; an extraction plan requires "
            f"a concept-based domain ({tuple(COVARIATE_AGGREGATIONS_BY_DOMAIN)})"
        )
    if not definition.root_concept_ids:
        raise ValueError(f"'{definition.variable_id}' has no approved root_concept_ids")
    return ExtractionPlan(
        variable_id=definition.variable_id,
        domain=definition.domain,
        omop_tables=definition.omop_tables,
        concept_id_column=definition.concept_id_column,
        datetime_column=definition.datetime_column,
        value_column=definition.value_column,
        root_concept_ids=definition.root_concept_ids,
        include_descendants=definition.include_descendants,
        excluded_concept_ids=definition.excluded_concept_ids,
        window_start_hours=definition.window_start_hours,
        window_end_hours=definition.window_end_hours,
        aggregation=definition.aggregation,
        missingness_rule=definition.missingness_rule,
    )


def extraction_plan_to_concept_set_spec(plan: ExtractionPlan) -> ConceptSetSpec:
    """`ExtractionPlan -> causal.queries.feasibility.ConceptSetSpec` -- feeds
    `feasibility_sql`/`AnalyticCohortBuilder` unchanged (see that module)."""
    return ConceptSetSpec(
        domain=plan.domain,
        concept_ids=plan.root_concept_ids,
        include_descendants=plan.include_descendants,
        excluded_concept_ids=plan.excluded_concept_ids,
    )


def is_fixed_domain(domain: str) -> bool:
    """Whether `domain` resolves through a fixed, non-concept-based OMOP
    table (`death`, `standard`) rather than an approved concept set -- see
    `fixed_domain_to_concept_set_spec`. Exposed as a function rather than the
    raw `_FIXED_DOMAIN_TABLES` dict so callers outside this module (e.g.
    `causal.orchestrator.workflow._load_concept_specs`) branch on the same
    single source of truth instead of importing a private constant."""
    return domain in _FIXED_DOMAIN_TABLES


def fixed_domain_to_concept_set_spec(definition: OMOPVariableDefinition) -> ConceptSetSpec:
    """`OMOPVariableDefinition -> causal.queries.feasibility.ConceptSetSpec`
    for the two roles that resolve through a fixed, non-concept-based OMOP
    table (`death`, `standard`) rather than an approved concept set.

    A sibling to `build_extraction_plan`, not a widened version of it:
    `build_extraction_plan` requires a concept-based domain and
    `root_concept_ids` (see its docstring), neither of which applies to a
    fixed-domain role, and its `ExtractionPlan` output shape (concept
    selection, window, aggregation) isn't meaningful here either. Still
    requires `mapping_status == "approved"` and `observed == True` -- a
    proposed mapping or a declared-unobservable variable must never reach
    SQL generation, the same invariant `build_extraction_plan` enforces.
    """
    if definition.mapping_status != MAPPING_STATUS_APPROVED:
        raise ValueError(
            f"'{definition.variable_id}' has mapping_status={definition.mapping_status!r}; "
            "a mapping must be approved before generating its concept-set spec"
        )
    if not definition.observed:
        raise ValueError(
            f"'{definition.variable_id}' is not observed -- an unobservable/latent variable "
            "cannot be extracted, it remains a modeled identification limitation"
        )
    if not is_fixed_domain(definition.domain):
        raise ValueError(
            f"'{definition.variable_id}' has domain {definition.domain!r}; "
            f"fixed_domain_to_concept_set_spec requires one of {tuple(_FIXED_DOMAIN_TABLES)}"
        )
    return ConceptSetSpec(domain=definition.domain, concept_ids=())


def extraction_plan_to_covariate_spec(definition: OMOPVariableDefinition, plan: ExtractionPlan, column_name: str) -> CovariateSpec:
    """`ExtractionPlan -> causal.queries.feasibility.CovariateSpec`.

    Requires `definition.causal_role` to be adjustment-eligible (see
    `causal.agents.data_science.EXCLUDED_FROM_ADJUSTMENT_ROLES`): a mediator,
    collider, selection variable, or instrument must never become a baseline
    adjustment covariate even if it happens to carry an approved mapping --
    conditioning on it would bias the estimate rather than help it. This is
    the concrete point in the pipeline where that leakage is actually
    prevented, not merely documented.
    """
    if definition.causal_role in EXCLUDED_FROM_ADJUSTMENT_ROLES:
        raise ValueError(
            f"'{definition.variable_id}' has causal_role={definition.causal_role!r}, which must never be "
            "used as a baseline adjustment covariate (conditioning on a mediator, collider, selection "
            "variable, or instrument can bias the causal estimate)"
        )
    if plan.window_start_hours is None or plan.window_end_hours is None:
        raise ValueError(f"'{definition.variable_id}' has no reviewed ascertainment window")
    if plan.aggregation == "definition_pending":
        raise ValueError(f"'{definition.variable_id}' has no reviewed aggregation rule")
    return CovariateSpec(
        column_name=column_name,
        domain=plan.domain,
        concept_ids=plan.root_concept_ids,
        window_start_hours=plan.window_start_hours,
        window_end_hours=plan.window_end_hours,
        aggregation=plan.aggregation,
        include_descendants=plan.include_descendants,
        excluded_concept_ids=plan.excluded_concept_ids,
    )
