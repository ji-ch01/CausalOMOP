import unittest

from causal.literature.pubmed import PubMedClient, PubMedConfig


ESEARCH_JSON = b'{"esearchresult": {"idlist": ["111", "222"]}}'

EFETCH_XML = b"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>111</PMID>
      <Article>
        <ArticleTitle>Albumin for septic shock</ArticleTitle>
        <Abstract>
          <AbstractText>Albumin resuscitation was associated with a difference in 90-day mortality.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


class PubMedClientTests(unittest.TestCase):
    def test_search_returns_pmids(self):
        client = PubMedClient(PubMedConfig(), fetcher=lambda url: ESEARCH_JSON)
        self.assertEqual(client.search("albumin sepsis"), ("111", "222"))

    def test_search_rejects_empty_term(self):
        client = PubMedClient(PubMedConfig(), fetcher=lambda url: ESEARCH_JSON)
        with self.assertRaises(ValueError):
            client.search("   ")

    def test_search_rejects_out_of_range_retmax(self):
        client = PubMedClient(PubMedConfig(), fetcher=lambda url: ESEARCH_JSON)
        with self.assertRaises(ValueError):
            client.search("albumin", retmax=0)
        with self.assertRaises(ValueError):
            client.search("albumin", retmax=101)

    def test_fetch_abstracts_parses_xml(self):
        client = PubMedClient(PubMedConfig(), fetcher=lambda url: EFETCH_XML)
        articles = client.fetch_abstracts(("111",))
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].pmid, "111")
        self.assertEqual(articles[0].title, "Albumin for septic shock")
        self.assertIn("90-day mortality", articles[0].abstract)

    def test_fetch_abstracts_empty_input_does_not_call_fetcher(self):
        def _fail(url: str) -> bytes:
            raise AssertionError("fetcher should not be called for an empty pmid tuple")

        client = PubMedClient(PubMedConfig(), fetcher=_fail)
        self.assertEqual(client.fetch_abstracts(()), ())

    def test_url_includes_optional_config_fields(self):
        captured = {}

        def _capture(url: str) -> bytes:
            captured["url"] = url
            return ESEARCH_JSON

        config = PubMedConfig(api_key="test-key", tool="causalomop", email="dev@example.org")
        PubMedClient(config, fetcher=_capture).search("albumin")
        self.assertIn("api_key=test-key", captured["url"])
        self.assertIn("email=dev%40example.org", captured["url"])


if __name__ == "__main__":
    unittest.main()
