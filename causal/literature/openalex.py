"""Deterministic OpenAlex retrieval (Works API), no LLM involved here.

Mirrors `causal.literature.pubmed`'s two-step search()/fetch_abstracts()
shape so both clients can sit behind the same interface in
`causal.literature.orchestrator.LiteratureOrchestrator`. OpenAlex is free and
requires no key; `OPENALEX_API_KEY` only raises the rate limit (premium
tier), and `mailto` only opts into the faster "polite pool".
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from causal.llm.base import load_dotenv_if_available


Fetcher = Callable[[str], bytes]


@dataclass(frozen=True)
class OpenAlexArticle:
    work_id: str
    title: str
    abstract: str


@dataclass(frozen=True)
class OpenAlexConfig:
    base_url: str = "https://api.openalex.org"
    api_key: str | None = None
    mailto: str | None = None
    timeout_seconds: float = 20.0

    @classmethod
    def from_env(cls) -> "OpenAlexConfig":
        load_dotenv_if_available()
        return cls(
            api_key=os.getenv("OPENALEX_API_KEY") or None,
            mailto=os.getenv("OPENALEX_MAILTO") or os.getenv("NCBI_CONTACT_EMAIL") or None,
        )


class OpenAlexClient:
    """Thin, dependency-free client for the OpenAlex Works endpoint."""

    def __init__(self, config: OpenAlexConfig | None = None, fetcher: Fetcher | None = None) -> None:
        self.config = config or OpenAlexConfig.from_env()
        self._fetch = fetcher or self._http_fetch

    def search(self, term: str, retmax: int = 10) -> tuple[str, ...]:
        """Return OpenAlex work ids (e.g. "W2741809807") matching a free-text term."""
        term = term.strip()
        if not term:
            raise ValueError("A non-empty search term is required")
        if not 1 <= retmax <= 100:
            raise ValueError("retmax must be between 1 and 100")
        payload = json.loads(
            self._fetch(self._url("works", {"search": term, "per-page": retmax, "select": "id"}))
        )
        results = payload.get("results")
        if not isinstance(results, list):
            raise RuntimeError("Unexpected OpenAlex works response shape")
        ids = []
        for item in results:
            work_id = _short_id(item.get("id")) if isinstance(item, dict) else None
            if work_id:
                ids.append(work_id)
        return tuple(ids)

    def fetch_abstracts(self, work_ids: tuple[str, ...]) -> tuple[OpenAlexArticle, ...]:
        """Return title/abstract text for OpenAlex work ids already returned by search()."""
        if not work_ids:
            return ()
        payload = json.loads(
            self._fetch(
                self._url(
                    "works",
                    {
                        "filter": "ids.openalex:" + "|".join(work_ids),
                        "select": "id,title,abstract_inverted_index",
                        "per-page": len(work_ids),
                    },
                )
            )
        )
        results = payload.get("results")
        if not isinstance(results, list):
            raise RuntimeError("Unexpected OpenAlex works response shape")
        articles = []
        for item in results:
            if not isinstance(item, dict):
                continue
            work_id = _short_id(item.get("id"))
            if not work_id:
                continue
            articles.append(
                OpenAlexArticle(
                    work_id=work_id,
                    title=str(item.get("title") or ""),
                    abstract=_reconstruct_abstract(item.get("abstract_inverted_index")),
                )
            )
        return tuple(articles)

    def _url(self, endpoint: str, params: dict[str, object]) -> str:
        query = dict(params)
        if self.config.api_key:
            query["api_key"] = self.config.api_key
        if self.config.mailto:
            query["mailto"] = self.config.mailto
        return f"{self.config.base_url}/{endpoint}?{urlencode(query)}"

    def _http_fetch(self, url: str) -> bytes:
        request = Request(url, headers={"User-Agent": "CausalOMOP/0.1"})
        with urlopen(request, timeout=self.config.timeout_seconds) as response:  # noqa: S310 (fixed OpenAlex host)
            return response.read()


def _short_id(openalex_url: object) -> str | None:
    """"https://openalex.org/W2741809807" -> "W2741809807"."""
    if not isinstance(openalex_url, str) or not openalex_url:
        return None
    return openalex_url.rsplit("/", 1)[-1] or None


def _reconstruct_abstract(inverted_index: object) -> str:
    """OpenAlex ships abstracts as {word: [position, ...]}; rebuild plain text."""
    if not isinstance(inverted_index, dict) or not inverted_index:
        return ""
    positions: dict[int, str] = {}
    for word, occurrences in inverted_index.items():
        if not isinstance(occurrences, list):
            continue
        for position in occurrences:
            if isinstance(position, int):
                positions[position] = word
    return " ".join(positions[index] for index in sorted(positions))
