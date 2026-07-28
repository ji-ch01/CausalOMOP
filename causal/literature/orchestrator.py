"""Multi-source literature retrieval orchestration for DAG grounding.

Queries every configured source (currently PubMed and OpenAlex) with the same
free-text term, normalizes their heterogeneous article shapes into one
`LiteratureArticle`, deduplicates by title, and hands the merged evidence set
back to `causal.agents.user_question` for LLM-driven DAG drafting.

A source is queried best-effort: if it fails validation (e.g. empty term) or
returns nothing, that degrades to zero articles *from that source only* — it
never blocks the others, matching the existing "never blocks producing a
protocol draft" contract in `UserQuestionHypothesisService`.

The `citation_id` returned per article is stored downstream in
`DAGEdge.pmid` unmodified. That field name predates multi-source support but
stays safe to reuse: PubMed ids are digit-only and OpenAlex work ids always
start with "W", so citations from either source can never collide.

Each article also carries an `evidence_level`, derived from PubMed's
`PublicationTypeList` (the standard evidence-hierarchy pyramid: meta-analyses
and systematic reviews outrank RCTs, which outrank other observational
designs) so downstream consumers can weight citations by evidence strength
rather than by which journal or database happened to return them. OpenAlex
does not expose a comparable publication-type taxonomy, so its articles are
always `"unclassified"`.

`search_and_fetch(..., limit=N)` selects the N articles used for DAG/
hypothesis grounding round-robin across the *graded* evidence tiers
(systematic reviews/meta-analyses, RCTs, observational studies — all
derived from PubMed's own `PublicationTypeList`), not by a flat sort. A flat
sort would let a topic with many indexed systematic reviews crowd out every
RCT/cohort study, even though those primary studies are usually the ones
that actually describe the treatment-assignment and confounding structure a
DAG needs (reviews report pooled treatment-outcome effects, rarely the
baseline-characteristics detail behind them).

`"unclassified"` (PubMed hits with no recognized publication type, and every
OpenAlex hit, since OpenAlex exposes no comparable taxonomy) deliberately
does *not* take part in that round-robin: OpenAlex's free-text relevance
ranking is weaker than PubMed's, so unclassified hits are more likely to be
tangentially related (e.g. matching on a shared lab value like albumin
across an unrelated liver-disease paper). They only fill quota left over
after every graded article has had a chance, so topical noise can never
bump a real systematic review or RCT out of the selection.
"""

from __future__ import annotations

from dataclasses import dataclass

from causal.literature.openalex import OpenAlexClient
from causal.literature.pubmed import PubMedClient


#: PubMed `PublicationType` values that map to each evidence-hierarchy tier,
#: checked in this order so an article tagged with several types resolves to
#: its strongest one (e.g. a "Meta-Analysis" that is also a "Review").
_SYSTEMATIC_REVIEW_TYPES = {"Meta-Analysis", "Systematic Review"}
_RCT_TYPES = {"Randomized Controlled Trial", "Controlled Clinical Trial", "Pragmatic Clinical Trial"}
_OBSERVATIONAL_TYPES = {"Observational Study", "Comparative Study", "Multicenter Study", "Case Reports"}

#: Evidence tiers in priority order, strongest first.
_EVIDENCE_TIERS = ("systematic_review", "rct", "observational", "unclassified")

#: Tiers that participate in round-robin selection; "unclassified" is a
#: fallback filler only (see module docstring), never guaranteed a seat.
_GRADED_TIERS = ("systematic_review", "rct", "observational")
_UNCLASSIFIED_TIER = "unclassified"


@dataclass(frozen=True)
class LiteratureArticle:
    source: str
    citation_id: str
    title: str
    abstract: str
    evidence_level: str = "unclassified"


@dataclass(frozen=True)
class LiteratureOrchestrator:
    pubmed: PubMedClient | None = None
    openalex: OpenAlexClient | None = None

    def is_configured(self) -> bool:
        return self.pubmed is not None or self.openalex is not None

    def search_and_fetch(
        self, term: str, retmax_per_source: int = 8, limit: int | None = None
    ) -> tuple[LiteratureArticle, ...]:
        articles: list[LiteratureArticle] = []
        if self.pubmed is not None:
            articles.extend(self._search_pubmed(term, retmax_per_source))
        if self.openalex is not None:
            articles.extend(self._search_openalex(term, retmax_per_source))
        deduped = _deduplicate(articles)
        if limit is None:
            return deduped
        return _select_by_evidence_tier(deduped, limit)

    def _search_pubmed(self, term: str, retmax: int) -> tuple[LiteratureArticle, ...]:
        try:
            pmids = self.pubmed.search(term, retmax=retmax)
        except ValueError:
            return ()
        return tuple(
            LiteratureArticle(
                "pubmed",
                article.pmid,
                article.title,
                article.abstract,
                _evidence_level(article.publication_types),
            )
            for article in self.pubmed.fetch_abstracts(pmids)
        )

    def _search_openalex(self, term: str, retmax: int) -> tuple[LiteratureArticle, ...]:
        try:
            work_ids = self.openalex.search(term, retmax=retmax)
        except ValueError:
            return ()
        return tuple(
            LiteratureArticle("openalex", article.work_id, article.title, article.abstract)
            for article in self.openalex.fetch_abstracts(work_ids)
        )


def _evidence_level(publication_types: tuple[str, ...]) -> str:
    types = set(publication_types)
    if types & _SYSTEMATIC_REVIEW_TYPES:
        return "systematic_review"
    if types & _RCT_TYPES:
        return "rct"
    if types & _OBSERVATIONAL_TYPES:
        return "observational"
    return "unclassified"


def select_by_evidence_tier(
    articles: tuple[LiteratureArticle, ...], limit: int
) -> tuple[LiteratureArticle, ...]:
    """Public entry point onto the same evidence-tier prioritization
    `search_and_fetch(..., limit=N)` applies internally, for a caller that
    merges articles from more than one retrieval pass (e.g.
    `causal.agents.user_question`'s per-covariate targeted searches) and
    needs to rank the leftover pool the same way after reserving some slots.
    """
    return _select_by_evidence_tier(articles, limit)


def _select_by_evidence_tier(
    articles: tuple[LiteratureArticle, ...], limit: int
) -> tuple[LiteratureArticle, ...]:
    """Round-robin the graded tiers; unclassified only fills leftover quota."""
    by_tier: dict[str, list[LiteratureArticle]] = {tier: [] for tier in _EVIDENCE_TIERS}
    for article in articles:
        by_tier[article.evidence_level].append(article)
    selected: list[LiteratureArticle] = []
    while len(selected) < limit and any(by_tier[tier] for tier in _GRADED_TIERS):
        for tier in _GRADED_TIERS:
            if len(selected) >= limit:
                break
            if by_tier[tier]:
                selected.append(by_tier[tier].pop(0))
    for article in by_tier[_UNCLASSIFIED_TIER]:
        if len(selected) >= limit:
            break
        selected.append(article)
    return tuple(selected)


def _deduplicate(articles: list[LiteratureArticle]) -> tuple[LiteratureArticle, ...]:
    """Drop later duplicates of the same normalized title (e.g. indexed in both sources)."""
    seen_titles: set[str] = set()
    deduped: list[LiteratureArticle] = []
    for article in articles:
        key = " ".join(article.title.lower().split())
        if key:
            if key in seen_titles:
                continue
            seen_titles.add(key)
        deduped.append(article)
    return tuple(deduped)
