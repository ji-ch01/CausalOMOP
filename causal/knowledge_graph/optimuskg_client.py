"""Deterministic lookups against OptimusKG (github.com/mims-harvard/OptimusKG).

Mirrors the shape of `causal.literature.openalex`/`pubmed` so a KG source can
sit behind the same "best-effort, never blocks the caller" contract as the
literature clients, but the underlying access pattern is different: OptimusKG
has no query API. `optimuskg.load_networkx()` downloads and caches ~22M edges
as Apache Parquet, then builds one in-memory `networkx.MultiDiGraph` — so this
client loads the graph once (lazily, on first lookup) and keeps it in memory
for the process lifetime rather than issuing one HTTP call per query.

The `optimuskg` package is an optional dependency (see
requirements-knowledge-graph.txt); importing this module never fails, but
constructing `OptimusKGClient` without it installed raises immediately, and
`is_available()` lets callers check first (same pattern `LiteratureOrchestrator.
is_configured()` uses for PubMed/OpenAlex).

Column names below (`name`, `category` for nodes; `subject`, `predicate`,
`object` for edges) follow the Biolink Model, which OptimusKG's BioCypher
pipeline targets — confirm against the installed dataset's actual schema
(`optimuskg.load_parquet("nodes/<type>.parquet").columns`) before relying on
this in production; the schema in the public docs was not fully enumerated at
the time this client was written.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KGEdge:
    source_id: str
    source_label: str
    predicate: str
    target_id: str
    target_label: str
    kg_source: str = "optimuskg"

    @property
    def citation_id(self) -> str:
        """Stable id for this edge, reusable anywhere a `pmid`-shaped string is expected."""
        return f"optimuskg:{self.source_id}|{self.predicate}|{self.target_id}"


@dataclass(frozen=True)
class OptimusKGConfig:
    lcc: bool = True
    doi: str | None = None
    max_matches_per_term: int = 5


class OptimusKGClient:
    """Loads the OptimusKG graph on first use and answers label-to-label relation lookups."""

    def __init__(self, config: OptimusKGConfig | None = None) -> None:
        self.config = config or OptimusKGConfig()
        self._graph: Any | None = None

    @staticmethod
    def is_available() -> bool:
        try:
            import optimuskg  # noqa: F401
        except ImportError:
            return False
        return True

    def find_relations(self, term_a: str, term_b: str) -> tuple[KGEdge, ...]:
        """Return KG edges directly connecting a node matching `term_a` to one matching `term_b`.

        Matching is case-insensitive substring matching on node `name`; both
        directions are checked since Biolink predicates are directional
        (e.g. "treats" vs "treated_by") but callers here only care whether a
        relation exists between the two clinical concepts, not which one the
        KG happened to record as subject.
        """
        term_a = term_a.strip()
        term_b = term_b.strip()
        if not term_a or not term_b:
            raise ValueError("Both terms are required")

        graph = self._load_graph()
        nodes_a = self._matching_node_ids(graph, term_a)
        nodes_b = self._matching_node_ids(graph, term_b)
        if not nodes_a or not nodes_b:
            return ()

        edges: list[KGEdge] = []
        for node_a in nodes_a:
            for node_b in nodes_b:
                edges.extend(self._edges_between(graph, node_a, node_b))
        return tuple(edges)

    def _load_graph(self) -> Any:
        if self._graph is not None:
            return self._graph
        try:
            import optimuskg
        except ImportError as exc:
            raise RuntimeError(
                "optimuskg is not installed; run `pip install -r requirements-knowledge-graph.txt`"
            ) from exc
        if self.config.doi:
            optimuskg.set_doi(self.config.doi)
        self._graph = optimuskg.load_networkx(lcc=self.config.lcc)
        return self._graph

    def _matching_node_ids(self, graph: Any, term: str) -> list[str]:
        needle = term.lower()
        matches = [
            node_id
            for node_id, data in graph.nodes(data=True)
            if needle in str(data.get("name", "")).lower()
        ]
        return matches[: self.config.max_matches_per_term]

    @staticmethod
    def _edges_between(graph: Any, node_a: str, node_b: str) -> list[KGEdge]:
        found: list[KGEdge] = []
        for source, target, data in graph.out_edges(node_a, data=True):
            if target == node_b:
                found.append(
                    KGEdge(
                        source_id=source,
                        source_label=str(graph.nodes[source].get("name", source)),
                        predicate=str(data.get("predicate", "related_to")),
                        target_id=target,
                        target_label=str(graph.nodes[target].get("name", target)),
                    )
                )
        for source, target, data in graph.out_edges(node_b, data=True):
            if target == node_a:
                found.append(
                    KGEdge(
                        source_id=source,
                        source_label=str(graph.nodes[source].get("name", source)),
                        predicate=str(data.get("predicate", "related_to")),
                        target_id=target,
                        target_label=str(graph.nodes[target].get("name", target)),
                    )
                )
        return found
