"""Deterministic knowledge-graph lookups for DAG grounding (no LLM involved)."""

from causal.knowledge_graph.optimuskg_client import KGEdge, OptimusKGClient, OptimusKGConfig

__all__ = [
    "KGEdge",
    "OptimusKGClient",
    "OptimusKGConfig",
]
