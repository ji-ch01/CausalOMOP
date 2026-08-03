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
pipeline targets — the schema in the public docs was not fully enumerated at
the time this client was written, so `_load_graph` runs `_validate_schema`
once per process against a small sample of the loaded graph and raises
immediately if `name`/`predicate` are missing, rather than every lookup
silently returning "no relations found". That check only catches a wrong
schema, not a right-schema-wrong-match: still confirm against the installed
dataset's actual columns (`optimuskg.load_parquet("nodes/<type>.parquet").
columns`) if lookups look suspicious.
"""

from __future__ import annotations

import itertools
import re
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

        Matching is case-insensitive whole-word/whole-phrase matching on node
        `name` (see `_matching_node_ids`); both directions are checked since
        Biolink predicates are directional (e.g. "treats" vs "treated_by")
        but callers here only care whether a relation exists between the two
        clinical concepts, not which one the KG happened to record as
        subject.
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
        graph = optimuskg.load_networkx(lcc=self.config.lcc)
        self._validate_schema(graph)
        self._graph = graph
        return self._graph

    @staticmethod
    def _validate_schema(graph: Any) -> None:
        """Fail loudly, once, if the installed OptimusKG release does not
        actually carry the Biolink-shaped `name`/`predicate` fields this
        client assumes (see module docstring: that schema was never
        confirmed against the real dataset at the time this client was
        written). Without this check, a schema drift would silently look
        like "no relations found" for every single lookup instead of one
        diagnosable error at load time.

        Sampled rather than exhaustive (a 100-node/100-edge prefix) since the
        full graph can hold ~22M edges; a genuine schema mismatch will show
        up in the first few rows regardless. An empty sample (graph has no
        nodes/edges at all, e.g. a test double) is not treated as a mismatch.
        """
        node_sample = list(itertools.islice(graph.nodes(data=True), 100))
        if node_sample and not any("name" in data for _, data in node_sample):
            raise RuntimeError(
                "OptimusKG graph nodes have no 'name' attribute in a 100-node sample; "
                "this client's Biolink-shaped schema assumption does not match the "
                "installed dataset -- check `optimuskg.load_parquet(...).columns` and "
                "update this client before trusting its lookups"
            )
        edge_sample = list(itertools.islice(graph.edges(data=True), 100))
        if edge_sample and not any("predicate" in data for _, _, data in edge_sample):
            raise RuntimeError(
                "OptimusKG graph edges have no 'predicate' attribute in a 100-edge sample; "
                "this client's Biolink-shaped schema assumption does not match the "
                "installed dataset -- check `optimuskg.load_parquet(...).columns` and "
                "update this client before trusting its lookups"
            )

    def _matching_node_ids(self, graph: Any, term: str) -> list[str]:
        """Node ids whose `name` contains `term` as a whole word/phrase.

        Whole-word matching, not a bare substring check: `needle in name`
        would treat e.g. "art" as matching inside "heart" or "arterial" --
        exactly the false-positive class this client's module docstring
        warns about. `\\b...\\b` still matches a multi-word term as one
        phrase (e.g. "myocardial infarction" matches inside "acute
        myocardial infarction"), but requires a non-word character (or
        string edge) on both sides, so it trades away matching a bare suffix
        (e.g. "infarction" no longer matches inside "infarctions") for far
        fewer spurious hits. It does not fix the complementary problem (the
        same clinical concept under a different synonym never matching at
        all) -- that needs concept-id-based matching, not a better string
        pattern.
        """
        pattern = re.compile(rf"\b{re.escape(term.lower())}\b")
        matches = [
            node_id
            for node_id, data in graph.nodes(data=True)
            if pattern.search(str(data.get("name", "")).lower())
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
