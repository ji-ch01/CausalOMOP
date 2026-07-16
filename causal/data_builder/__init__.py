# Causal Data Builder
# Transforms a causal question into a standardized dataset ready for estimation.
#
# Components:
#   cohort.py               — Phenotyping: extract (A, Y, index_date) per patient
#   structured_covariates.py — Structured Confounder Extractor from OMOP tables
#   leakage_guard.py        — Post-Treatment Leakage Guard
#   llm_covariates.py       — LLM Confounder Extractor from note_llm_log
#   builder.py              — Main pipeline orchestrator
from causal.data_builder.analytic import AnalyticCohortBuilder, CohortBuildManifest

__all__ = ["AnalyticCohortBuilder", "CohortBuildManifest"]
