import unittest

from causal.knowledge_graph.optimuskg_client import OptimusKGClient, OptimusKGConfig


class _NodeView:
    """Duck-typed stand-in for `networkx`'s `NodeView`: supports both the
    callable form (`graph.nodes(data=True)`, used by `_matching_node_ids`/
    `_validate_schema`) and the subscript form (`graph.nodes[node_id]`, used
    by `_edges_between`) without depending on the real `networkx`/`optimuskg`
    optional packages."""

    def __init__(self, nodes: dict):
        self._nodes = nodes

    def __call__(self, data: bool = False):
        return list(self._nodes.items()) if data else list(self._nodes)

    def __getitem__(self, node_id: str):
        return self._nodes[node_id]

    def __iter__(self):
        return iter(self._nodes)


class _FakeGraph:
    """Minimal duck-typed `networkx.MultiDiGraph` stand-in exposing only the
    surface `causal.knowledge_graph.optimuskg_client` actually calls."""

    def __init__(self, nodes: dict, edges: list):
        self.nodes = _NodeView(nodes)
        self._edges = edges

    def out_edges(self, node_id: str, data: bool = False):
        matches = [(s, t, d) for s, t, d in self._edges if s == node_id]
        return matches if data else [(s, t) for s, t, _ in matches]

    def edges(self, data: bool = False):
        return list(self._edges) if data else [(s, t) for s, t, _ in self._edges]


_WELL_FORMED_NODES = {
    "n1": {"name": "heart failure", "category": "biolink:Disease"},
    "n2": {"name": "acute myocardial infarction", "category": "biolink:Disease"},
    "n3": {"name": "aspirin", "category": "biolink:Drug"},
}
_WELL_FORMED_EDGES = [
    ("n3", "n2", {"predicate": "treats"}),
]


class MatchingNodeIdsWholeWordTests(unittest.TestCase):
    """`_matching_node_ids` must match a whole word/phrase, not any substring
    -- a bare `needle in name` check would treat "art" as matching inside
    "heart failure", which is exactly the false-positive class flagged for
    this client."""

    def setUp(self):
        self.client = OptimusKGClient()
        self.graph = _FakeGraph(_WELL_FORMED_NODES, _WELL_FORMED_EDGES)

    def test_short_substring_does_not_match_a_longer_word(self):
        self.assertEqual(self.client._matching_node_ids(self.graph, "art"), [])

    def test_whole_word_matches(self):
        self.assertEqual(self.client._matching_node_ids(self.graph, "aspirin"), ["n3"])

    def test_multi_word_phrase_matches_as_one_unit(self):
        self.assertEqual(
            self.client._matching_node_ids(self.graph, "myocardial infarction"), ["n2"]
        )

    def test_matching_is_case_insensitive(self):
        self.assertEqual(self.client._matching_node_ids(self.graph, "HEART FAILURE"), ["n1"])

    def test_bare_suffix_no_longer_matches_the_containing_word(self):
        """Documented trade-off of moving to whole-word matching: a term that
        is a strict prefix of a node's word (e.g. searching "infarction"
        against a node literally named "infarctions") no longer matches.
        This is intentional -- see `_matching_node_ids`'s docstring."""
        graph = _FakeGraph({"n1": {"name": "myocardial infarctions"}}, [])
        self.assertEqual(self.client._matching_node_ids(graph, "infarction"), [])

    def test_respects_max_matches_per_term(self):
        nodes = {f"n{i}": {"name": "sepsis"} for i in range(10)}
        client = OptimusKGClient(OptimusKGConfig(max_matches_per_term=3))
        self.assertEqual(len(client._matching_node_ids(_FakeGraph(nodes, []), "sepsis")), 3)


class FindRelationsEndToEndTests(unittest.TestCase):
    """Exercises the full public `find_relations` path against a fake graph
    (bypassing `_load_graph`/the optional `optimuskg` dependency by setting
    `_graph` directly), confirming the false-positive fix holds end to end."""

    def setUp(self):
        self.client = OptimusKGClient()
        self.client._graph = _FakeGraph(_WELL_FORMED_NODES, _WELL_FORMED_EDGES)

    def test_finds_the_edge_between_two_whole_word_matches(self):
        edges = self.client.find_relations("aspirin", "myocardial infarction")
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].predicate, "treats")
        self.assertEqual(edges[0].source_label, "aspirin")
        self.assertEqual(edges[0].target_label, "acute myocardial infarction")

    def test_edge_is_found_regardless_of_argument_order(self):
        forward = self.client.find_relations("aspirin", "myocardial infarction")
        backward = self.client.find_relations("myocardial infarction", "aspirin")
        self.assertEqual(len(forward), 1)
        self.assertEqual(len(backward), 1)

    def test_a_substring_false_positive_term_finds_nothing(self):
        self.assertEqual(self.client.find_relations("art", "aspirin"), ())


class ValidateSchemaTests(unittest.TestCase):
    """`_validate_schema` is the defensive check added because the Biolink
    schema this client assumes was never confirmed against the real
    installed OptimusKG dataset -- it must fail loudly at load time instead
    of every lookup silently returning no relations."""

    def test_well_formed_schema_does_not_raise(self):
        graph = _FakeGraph(_WELL_FORMED_NODES, _WELL_FORMED_EDGES)
        OptimusKGClient._validate_schema(graph)  # must not raise

    def test_raises_when_nodes_have_no_name_attribute(self):
        graph = _FakeGraph({"n1": {"label": "heart failure"}}, [])
        with self.assertRaises(RuntimeError):
            OptimusKGClient._validate_schema(graph)

    def test_raises_when_edges_have_no_predicate_attribute(self):
        graph = _FakeGraph(
            _WELL_FORMED_NODES,
            [("n3", "n2", {"relation": "treats"})],
        )
        with self.assertRaises(RuntimeError):
            OptimusKGClient._validate_schema(graph)

    def test_empty_graph_is_not_treated_as_a_schema_mismatch(self):
        OptimusKGClient._validate_schema(_FakeGraph({}, []))  # must not raise


if __name__ == "__main__":
    unittest.main()
