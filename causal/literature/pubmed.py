"""Deterministic PubMed retrieval (NCBI E-utilities), no LLM involved here.

This exists to ground a future Hypothesis Agent DAG/confounder draft in real,
retrievable abstracts instead of an LLM's parametric memory: fetch real
articles first, then only let the LLM cite PMIDs that were actually returned
here. That citation-grounding check is not wired up yet — this module is only
the retrieval primitive it will depend on.

No patient-level data is ever sent to PubMed; only literature search terms
derived from the drafted causal question (population/treatment/outcome
labels), the same text already sent to the LLM provider.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from causal.llm.base import load_dotenv_if_available


Fetcher = Callable[[str], bytes]


@dataclass(frozen=True)
class PubMedArticle:
    pmid: str
    title: str
    abstract: str


@dataclass(frozen=True)
class PubMedConfig:
    base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    api_key: str | None = None
    tool: str = "causalomop"
    email: str | None = None
    timeout_seconds: float = 20.0

    @classmethod
    def from_env(cls) -> "PubMedConfig":
        load_dotenv_if_available()
        return cls(
            api_key=os.getenv("NCBI_API_KEY") or None,
            email=os.getenv("NCBI_CONTACT_EMAIL") or None,
        )


class PubMedClient:
    """Thin, dependency-free client for the NCBI esearch/efetch endpoints."""

    def __init__(self, config: PubMedConfig | None = None, fetcher: Fetcher | None = None) -> None:
        self.config = config or PubMedConfig.from_env()
        self._fetch = fetcher or self._http_fetch

    def search(self, term: str, retmax: int = 10) -> tuple[str, ...]:
        """Return PMIDs matching a free-text term. Does not fetch content."""
        term = term.strip()
        if not term:
            raise ValueError("A non-empty search term is required")
        if not 1 <= retmax <= 100:
            raise ValueError("retmax must be between 1 and 100")
        payload = json.loads(
            self._fetch(self._url("esearch.fcgi", {"db": "pubmed", "term": term, "retmode": "json", "retmax": retmax}))
        )
        ids = payload.get("esearchresult", {}).get("idlist")
        if not isinstance(ids, list):
            raise RuntimeError("Unexpected PubMed esearch response shape")
        return tuple(str(pmid) for pmid in ids)

    def fetch_abstracts(self, pmids: tuple[str, ...]) -> tuple[PubMedArticle, ...]:
        """Return title/abstract text for a set of PMIDs already returned by search()."""
        if not pmids:
            return ()
        raw = self._fetch(
            self._url("efetch.fcgi", {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "rettype": "abstract"})
        )
        return _parse_efetch_xml(raw)

    def _url(self, endpoint: str, params: dict[str, object]) -> str:
        query = dict(params)
        if self.config.api_key:
            query["api_key"] = self.config.api_key
        if self.config.tool:
            query["tool"] = self.config.tool
        if self.config.email:
            query["email"] = self.config.email
        return f"{self.config.base_url}/{endpoint}?{urlencode(query)}"

    def _http_fetch(self, url: str) -> bytes:
        request = Request(url, headers={"User-Agent": "CausalOMOP/0.1"})
        with urlopen(request, timeout=self.config.timeout_seconds) as response:  # noqa: S310 (fixed NCBI host)
            return response.read()


def _parse_efetch_xml(raw: bytes) -> tuple[PubMedArticle, ...]:
    root = ElementTree.fromstring(raw)
    articles = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        if pmid_el is None or not (pmid_el.text or "").strip():
            continue
        title_el = article.find(".//ArticleTitle")
        abstract_parts = [
            "".join(node.itertext()).strip()
            for node in article.findall(".//Abstract/AbstractText")
        ]
        articles.append(
            PubMedArticle(
                pmid=pmid_el.text.strip(),
                title="".join(title_el.itertext()).strip() if title_el is not None else "",
                abstract=" ".join(part for part in abstract_parts if part),
            )
        )
    return tuple(articles)
