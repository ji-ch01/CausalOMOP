"""Literature-grounded accuracy metrics for clinical-term -> OMOP concept
lookup, evaluated against a reviewer-curated reference set.

Every clinical term the pipeline maps (a phenotype/covariate name, e.g.
"sepsis") goes through `causal.agents.data_science.DataScienceAgent.
_lookup_candidates`, which calls an OMCP `Lookup_Condition`/`Lookup_Drug`/
`Lookup_Concept` tool (see `omcp.concept_lookup`) and returns a ranked list
of candidate `concept_id`s -- never an auto-approved mapping (a human
reviewer always picks the final one). Until this module, that search step
had no accuracy benchmark anywhere in the repo: no precision/recall, no
regression check when the search or ranking logic changes. This module
closes that gap the same way `causal.agents.dag_metrics` closed the
equivalent gap for DAG structure (SID/SHD) -- established, citable
information-retrieval metrics, not an invented heuristic:

- `recall_at_k`: whether the reviewer-approved concept is present anywhere
  in the top `k` results. Standard IR evaluation metric (Manning, C.D.,
  Raghavan, P., & Schuetze, H. (2008). Introduction to Information
  Retrieval. Cambridge University Press, Ch. 8).
- `reciprocal_rank` (and its mean across a case set, MRR): 1/rank of the
  first correct result, 0 if absent. Standard for evaluating a ranked
  retrieval list against one known-correct answer (Voorhees, E.M. (1999).
  The TREC-8 Question Answering Track Report. In Proceedings of the 8th
  Text Retrieval Conference).

`evaluate_lookup_accuracy` runs both over a list of `LookupCase`s and
separates cases with no achievable standard-concept mapping today (an
upstream ETL/vocabulary gap -- e.g. a source value that stays at
`concept_id = 0` in the current conversion, see
`docs/vocabulary_mapping_coverage.md`) from real search misses, so the
search algorithm is never penalized for a mapping that cannot exist yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence


def recall_at_k(retrieved: Sequence[int], expected: int, k: int) -> bool:
    """Whether `expected` appears among the first `k` items of `retrieved`
    (already assumed to be ranked best-first, e.g. by OMCP's
    `name_edit_distance` ordering)."""
    if k <= 0:
        raise ValueError("k must be positive")
    return expected in retrieved[:k]


def reciprocal_rank(retrieved: Sequence[int], expected: int) -> float:
    """1/rank of `expected`'s first occurrence in `retrieved` (rank 1 =
    first result), or 0.0 if `expected` is absent -- the per-case term
    averaged into Mean Reciprocal Rank (Voorhees, 1999)."""
    for rank, concept_id in enumerate(retrieved, start=1):
        if concept_id == expected:
            return 1.0 / rank
    return 0.0


@dataclass(frozen=True)
class LookupCase:
    """One reference case: searching `term` in `domain` is expected to
    surface `expected_concept_id` among the candidates. `expected_concept_id
    = None` marks a documented gap where no standard OMOP concept exists yet
    (see module docstring) -- excluded from recall/MRR, reported separately.
    `note` records the provenance of that judgment (e.g. a citation to
    `concept_audit.md`), not a workflow approval status."""

    term: str
    domain: str
    expected_concept_id: int | None
    note: str = ""


@dataclass(frozen=True)
class LookupAccuracyReport:
    case_count: int
    evaluated_case_count: int
    known_gap_case_count: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mean_reciprocal_rank: float
    misses: tuple[str, ...]


def evaluate_lookup_accuracy(
    cases: Sequence[LookupCase],
    lookup: Callable[[LookupCase], Sequence[int]],
) -> LookupAccuracyReport:
    """Run `lookup` (e.g. a thin wrapper around `OMCPClient`/`ConceptLookup`
    that returns ranked `concept_id`s for one case) over every case with a
    real `expected_concept_id`, and aggregate recall@1/5/10 and MRR.

    Cases with `expected_concept_id is None` (a known upstream mapping gap)
    are counted in `known_gap_case_count` and skipped by every metric below,
    so a search-algorithm regression is never masked -- or falsely blamed --
    by a gap it cannot fix.
    """
    evaluated = [case for case in cases if case.expected_concept_id is not None]
    known_gaps = len(cases) - len(evaluated)
    if not evaluated:
        return LookupAccuracyReport(
            case_count=len(cases), evaluated_case_count=0, known_gap_case_count=known_gaps,
            recall_at_1=0.0, recall_at_5=0.0, recall_at_10=0.0, mean_reciprocal_rank=0.0, misses=(),
        )

    hits_at_1 = hits_at_5 = hits_at_10 = 0
    reciprocal_ranks = []
    misses = []
    for case in evaluated:
        retrieved = list(lookup(case))
        expected = case.expected_concept_id
        hits_at_1 += recall_at_k(retrieved, expected, 1)
        hits_at_5 += recall_at_k(retrieved, expected, 5)
        hits_at_10 += recall_at_k(retrieved, expected, 10)
        rr = reciprocal_rank(retrieved, expected)
        reciprocal_ranks.append(rr)
        if rr == 0.0:
            misses.append(case.term)

    n = len(evaluated)
    return LookupAccuracyReport(
        case_count=len(cases),
        evaluated_case_count=n,
        known_gap_case_count=known_gaps,
        recall_at_1=hits_at_1 / n,
        recall_at_5=hits_at_5 / n,
        recall_at_10=hits_at_10 / n,
        mean_reciprocal_rank=sum(reciprocal_ranks) / n,
        misses=tuple(misses),
    )
