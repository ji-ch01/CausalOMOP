"""DAG node -> OMOP variable definition -> deterministic SQL plan.

See `causal.mapping.registry` for the model/validation/adapters and
`causal.mapping.audit` for aggregate-only mapping diagnostics.
"""

from causal.mapping.registry import (
    ALLOWED_DOMAINS,
    MAPPING_STATUS_APPROVED,
    SCHEMA_VERSION,
    TEMPORAL_ANCHORS,
    VALUE_DOMAINS,
    VARIABLE_KINDS,
    ExtractionPlan,
    OMOPVariableDefinition,
    OMOPVariableRegistry,
    build_extraction_plan,
    default_missingness_rule,
    extraction_plan_to_concept_set_spec,
    extraction_plan_to_covariate_spec,
    is_sql_executable,
    parse_variable,
    variable_from_candidate_dag_entry,
    variable_from_concept_set,
    variable_from_covariate_plan,
)

__all__ = [
    "ALLOWED_DOMAINS",
    "MAPPING_STATUS_APPROVED",
    "SCHEMA_VERSION",
    "TEMPORAL_ANCHORS",
    "VALUE_DOMAINS",
    "VARIABLE_KINDS",
    "ExtractionPlan",
    "OMOPVariableDefinition",
    "OMOPVariableRegistry",
    "build_extraction_plan",
    "default_missingness_rule",
    "extraction_plan_to_concept_set_spec",
    "extraction_plan_to_covariate_spec",
    "is_sql_executable",
    "parse_variable",
    "variable_from_candidate_dag_entry",
    "variable_from_concept_set",
    "variable_from_covariate_plan",
]
