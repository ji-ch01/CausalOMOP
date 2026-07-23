import unittest

from causal.literature.openalex import OpenAlexClient, OpenAlexConfig


SEARCH_JSON = b'{"results": [{"id": "https://openalex.org/W111"}, {"id": "https://openalex.org/W222"}]}'

WORKS_JSON = b"""
{
  "results": [
    {
      "id": "https://openalex.org/W111",
      "title": "Albumin for septic shock",
      "abstract_inverted_index": {
        "Albumin": [0],
        "resuscitation": [1],
        "improves": [2],
        "outcomes": [3]
      }
    }
  ]
}
"""


class OpenAlexClientTests(unittest.TestCase):
    def test_search_returns_short_work_ids(self):
        client = OpenAlexClient(OpenAlexConfig(), fetcher=lambda url: SEARCH_JSON)
        self.assertEqual(client.search("albumin sepsis"), ("W111", "W222"))

    def test_search_rejects_empty_term(self):
        client = OpenAlexClient(OpenAlexConfig(), fetcher=lambda url: SEARCH_JSON)
        with self.assertRaises(ValueError):
            client.search("   ")

    def test_search_rejects_out_of_range_retmax(self):
        client = OpenAlexClient(OpenAlexConfig(), fetcher=lambda url: SEARCH_JSON)
        with self.assertRaises(ValueError):
            client.search("albumin", retmax=0)
        with self.assertRaises(ValueError):
            client.search("albumin", retmax=101)

    def test_fetch_abstracts_reconstructs_text_from_inverted_index(self):
        client = OpenAlexClient(OpenAlexConfig(), fetcher=lambda url: WORKS_JSON)
        articles = client.fetch_abstracts(("W111",))
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].work_id, "W111")
        self.assertEqual(articles[0].title, "Albumin for septic shock")
        self.assertEqual(articles[0].abstract, "Albumin resuscitation improves outcomes")

    def test_fetch_abstracts_empty_input_does_not_call_fetcher(self):
        def _fail(url: str) -> bytes:
            raise AssertionError("fetcher should not be called for an empty id tuple")

        client = OpenAlexClient(OpenAlexConfig(), fetcher=_fail)
        self.assertEqual(client.fetch_abstracts(()), ())

    def test_url_includes_optional_config_fields(self):
        captured = {}

        def _capture(url: str) -> bytes:
            captured["url"] = url
            return SEARCH_JSON

        config = OpenAlexConfig(api_key="test-key", mailto="dev@example.org")
        OpenAlexClient(config, fetcher=_capture).search("albumin")
        self.assertIn("api_key=test-key", captured["url"])
        self.assertIn("mailto=dev%40example.org", captured["url"])


if __name__ == "__main__":
    unittest.main()
