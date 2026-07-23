import unittest

from causal.literature.openalex import OpenAlexClient, OpenAlexConfig
from causal.literature.orchestrator import LiteratureOrchestrator
from causal.literature.pubmed import PubMedClient, PubMedConfig


PUBMED_ESEARCH = b'{"esearchresult": {"idlist": ["111"]}}'
PUBMED_EFETCH = b"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>111</PMID>
      <Article>
        <ArticleTitle>Albumin for septic shock</ArticleTitle>
        <Abstract><AbstractText>Albumin resuscitation was associated with lower mortality.</AbstractText></Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""

SYSTEMATIC_REVIEW_ESEARCH = b'{"esearchresult": {"idlist": ["222"]}}'
SYSTEMATIC_REVIEW_EFETCH = b"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>222</PMID>
      <Article>
        <ArticleTitle>Albumin versus crystalloids for fluid resuscitation (Review)</ArticleTitle>
        <Abstract><AbstractText>Systematic review of albumin resuscitation trials.</AbstractText></Abstract>
      </Article>
      <PubmedData>
        <PublicationTypeList>
          <PublicationType>Systematic Review</PublicationType>
        </PublicationTypeList>
      </PubmedData>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""

MIXED_ESEARCH = b'{"esearchresult": {"idlist": ["111", "222"]}}'
MIXED_EFETCH = b"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>111</PMID>
      <Article>
        <ArticleTitle>Albumin for septic shock</ArticleTitle>
        <Abstract><AbstractText>Albumin resuscitation was associated with lower mortality.</AbstractText></Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>222</PMID>
      <Article>
        <ArticleTitle>Albumin versus crystalloids for fluid resuscitation (Review)</ArticleTitle>
        <Abstract><AbstractText>Systematic review of albumin resuscitation trials.</AbstractText></Abstract>
      </Article>
      <PubmedData>
        <PublicationTypeList>
          <PublicationType>Systematic Review</PublicationType>
        </PublicationTypeList>
      </PubmedData>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""

REVIEW_HEAVY_ESEARCH = b'{"esearchresult": {"idlist": ["222", "333", "444"]}}'
REVIEW_HEAVY_EFETCH = b"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>222</PMID>
      <Article>
        <ArticleTitle>Albumin systematic review one</ArticleTitle>
        <Abstract><AbstractText>Systematic review of albumin resuscitation trials.</AbstractText></Abstract>
      </Article>
      <PubmedData>
        <PublicationTypeList><PublicationType>Systematic Review</PublicationType></PublicationTypeList>
      </PubmedData>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>333</PMID>
      <Article>
        <ArticleTitle>Albumin systematic review two</ArticleTitle>
        <Abstract><AbstractText>Another systematic review of albumin resuscitation trials.</AbstractText></Abstract>
      </Article>
      <PubmedData>
        <PublicationTypeList><PublicationType>Meta-Analysis</PublicationType></PublicationTypeList>
      </PubmedData>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>444</PMID>
      <Article>
        <ArticleTitle>Albumin RCT with baseline severity characteristics</ArticleTitle>
        <Abstract><AbstractText>A randomized trial reporting SOFA score and vasopressor use at baseline.</AbstractText></Abstract>
      </Article>
      <PubmedData>
        <PublicationTypeList><PublicationType>Randomized Controlled Trial</PublicationType></PublicationTypeList>
      </PubmedData>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""

OPENALEX_SEARCH = b'{"results": [{"id": "https://openalex.org/W222"}]}'
OPENALEX_WORKS = b"""
{
  "results": [
    {
      "id": "https://openalex.org/W222",
      "title": "Crystalloids versus albumin in sepsis",
      "abstract_inverted_index": {"Baseline": [0], "severity": [1], "matters": [2]}
    }
  ]
}
"""

OPENALEX_DUPLICATE_TITLE = b"""
{
  "results": [
    {
      "id": "https://openalex.org/W333",
      "title": "Albumin for septic shock",
      "abstract_inverted_index": {"Duplicate": [0]}
    }
  ]
}
"""

def fake_pubmed() -> PubMedClient:
    def fetcher(url: str) -> bytes:
        if "esearch.fcgi" in url:
            return PUBMED_ESEARCH
        return PUBMED_EFETCH

    return PubMedClient(PubMedConfig(), fetcher=fetcher)


def fake_pubmed_mixed_evidence() -> PubMedClient:
    """Unclassified article (111) listed before a systematic review (222)."""

    def fetcher(url: str) -> bytes:
        if "esearch.fcgi" in url:
            return MIXED_ESEARCH
        return MIXED_EFETCH

    return PubMedClient(PubMedConfig(), fetcher=fetcher)


def fake_pubmed_review_heavy() -> PubMedClient:
    """Two systematic reviews plus one RCT, more reviews than the limit under test."""

    def fetcher(url: str) -> bytes:
        if "esearch.fcgi" in url:
            return REVIEW_HEAVY_ESEARCH
        return REVIEW_HEAVY_EFETCH

    return PubMedClient(PubMedConfig(), fetcher=fetcher)


def fake_pubmed_systematic_review() -> PubMedClient:
    def fetcher(url: str) -> bytes:
        if "esearch.fcgi" in url:
            return SYSTEMATIC_REVIEW_ESEARCH
        return SYSTEMATIC_REVIEW_EFETCH

    return PubMedClient(PubMedConfig(), fetcher=fetcher)


def fake_openalex(works_json: bytes = OPENALEX_WORKS) -> OpenAlexClient:
    def fetcher(url: str) -> bytes:
        return OPENALEX_SEARCH if "search=" in url else works_json

    return OpenAlexClient(OpenAlexConfig(), fetcher=fetcher)


class LiteratureOrchestratorTests(unittest.TestCase):
    def test_not_configured_without_any_source(self):
        self.assertFalse(LiteratureOrchestrator().is_configured())

    def test_configured_with_at_least_one_source(self):
        self.assertTrue(LiteratureOrchestrator(pubmed=fake_pubmed()).is_configured())

    def test_merges_articles_from_both_sources(self):
        orchestrator = LiteratureOrchestrator(pubmed=fake_pubmed(), openalex=fake_openalex())
        articles = orchestrator.search_and_fetch("albumin sepsis")
        self.assertEqual(len(articles), 2)
        self.assertEqual({a.source for a in articles}, {"pubmed", "openalex"})
        self.assertEqual({a.citation_id for a in articles}, {"111", "W222"})

    def test_deduplicates_same_title_across_sources(self):
        orchestrator = LiteratureOrchestrator(
            pubmed=fake_pubmed(), openalex=fake_openalex(OPENALEX_DUPLICATE_TITLE)
        )
        articles = orchestrator.search_and_fetch("albumin sepsis")
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].source, "pubmed")

    def test_pubmed_article_without_publication_type_is_unclassified(self):
        orchestrator = LiteratureOrchestrator(pubmed=fake_pubmed())
        articles = orchestrator.search_and_fetch("albumin sepsis")
        self.assertEqual(articles[0].evidence_level, "unclassified")

    def test_systematic_review_publication_type_is_classified(self):
        orchestrator = LiteratureOrchestrator(pubmed=fake_pubmed_systematic_review())
        articles = orchestrator.search_and_fetch("albumin sepsis")
        self.assertEqual(articles[0].evidence_level, "systematic_review")

    def test_openalex_articles_are_always_unclassified(self):
        orchestrator = LiteratureOrchestrator(openalex=fake_openalex())
        articles = orchestrator.search_and_fetch("albumin sepsis")
        self.assertEqual(articles[0].evidence_level, "unclassified")

    def test_limit_keeps_highest_evidence_tier_even_if_listed_later(self):
        orchestrator = LiteratureOrchestrator(pubmed=fake_pubmed_mixed_evidence())
        articles = orchestrator.search_and_fetch("albumin sepsis", limit=1)
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].citation_id, "222")
        self.assertEqual(articles[0].evidence_level, "systematic_review")

    def test_limit_falls_back_to_lower_tiers_to_fill_quota(self):
        orchestrator = LiteratureOrchestrator(pubmed=fake_pubmed_mixed_evidence())
        articles = orchestrator.search_and_fetch("albumin sepsis", limit=10)
        self.assertEqual({a.citation_id for a in articles}, {"111", "222"})

    def test_limit_does_not_let_reviews_crowd_out_every_rct(self):
        orchestrator = LiteratureOrchestrator(pubmed=fake_pubmed_review_heavy())
        articles = orchestrator.search_and_fetch("albumin sepsis", limit=2)
        self.assertIn("444", {a.citation_id for a in articles})

    def test_limit_does_not_let_unclassified_noise_crowd_out_graded_evidence(self):
        orchestrator = LiteratureOrchestrator(
            pubmed=fake_pubmed_review_heavy(), openalex=fake_openalex()
        )
        articles = orchestrator.search_and_fetch("albumin sepsis", limit=3)
        self.assertEqual({a.citation_id for a in articles}, {"222", "333", "444"})

    def test_transport_errors_are_not_swallowed(self):
        def broken_fetcher(url: str) -> bytes:
            raise RuntimeError("network down")

        broken_openalex = OpenAlexClient(OpenAlexConfig(), fetcher=broken_fetcher)
        orchestrator = LiteratureOrchestrator(pubmed=fake_pubmed(), openalex=broken_openalex)
        with self.assertRaises(RuntimeError):
            orchestrator.search_and_fetch("albumin sepsis")


if __name__ == "__main__":
    unittest.main()
