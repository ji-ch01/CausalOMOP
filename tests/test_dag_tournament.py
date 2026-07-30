import unittest

from causal.agents.dag_critique import build_pre_check
from causal.agents.dag_tournament import (
    MAX_TOURNAMENT_SIZE,
    TournamentCandidate,
    dedup_candidates,
    rank_candidates,
    run_tournament,
    structural_hamming_distance,
)


def _protocol() -> dict:
    return {
        "study_id": "test_study",
        "source_question": "Does treatment affect outcome?",
        "target_trial": {
            "population": "population", "treatment": "treatment",
            "comparator": "comparator", "time_zero": "index",
            "grace_period_hours": 24, "outcome": "outcome",
            "follow_up_days": 28, "primary_estimand": "ATE",
        },
        "concept_sets": {"treatment": {"status": "pending"}},
    }


def _variable(
    label: str, role: str = "confounder", temporal: str = "pre_treatment", pmid: str = "111",
    observed: bool = True,
) -> dict:
    return {
        "label": label,
        "role": role,
        "rationale": f"{label} affects both treatment choice and outcome.",
        "temporal_classification": temporal,
        "pmid": pmid,
        "observed": observed,
    }


def _relation(source: str, target: str, pmid: str = "111") -> dict:
    return {
        "source": source,
        "target": target,
        "rationale": f"{source} affects {target}.",
        "pmid": pmid,
    }


def _candidate(index: int, variables: list[dict], relations: list[dict] | None = None) -> TournamentCandidate:
    review = build_pre_check(_protocol(), variables, relations or [])
    assert review is not None
    return TournamentCandidate(
        index=index, variables=tuple(variables), relations=tuple(relations or []),
        unresolved=(), review=review, note="",
    )


class StructuralHammingDistanceTests(unittest.TestCase):
    def test_identical_graphs_have_zero_distance(self):
        a = _candidate(0, [_variable("age")])
        b = _candidate(1, [_variable("age")])
        self.assertEqual(
            structural_hamming_distance(a.review.identification.dag, b.review.identification.dag), 0
        )

    def test_disagreeing_role_changes_distance(self):
        a = _candidate(0, [_variable("age", role="confounder")])
        b = _candidate(1, [_variable("age", role="effect_modifier")])
        # confounder wires age->exposure and age->outcome; effect_modifier
        # wires only age->outcome -- exactly one pair (age, exposure)
        # differs (edge vs. none).
        self.assertEqual(
            structural_hamming_distance(a.review.identification.dag, b.review.identification.dag), 1
        )

    def test_extra_variable_in_one_graph_counts_its_pairs_as_mismatches(self):
        a = _candidate(0, [_variable("age")])
        b = _candidate(1, [_variable("age"), _variable("severity")])
        # "severity" contributes two edges (severity->exposure,
        # severity->outcome) absent from graph a, and the (age, severity)
        # pair is "none" in both -- so exactly 2 pairs differ.
        self.assertEqual(
            structural_hamming_distance(a.review.identification.dag, b.review.identification.dag), 2
        )


class RankCandidatesTests(unittest.TestCase):
    def test_identified_candidate_outranks_unidentified_one(self):
        # An unobserved confounder cannot be adjusted for, so its backdoor
        # path stays open -- unlike an *empty* candidate_dag (trivially
        # "identified" per DAGIdentificationResult's own docstring), this
        # candidate is genuinely not identified.
        identified = _candidate(0, [_variable("age")])
        unidentified = _candidate(1, [_variable("severity", observed=False)])
        self.assertTrue(identified.review.identification.identified_by_backdoor_adjustment)
        self.assertFalse(unidentified.review.identification.identified_by_backdoor_adjustment)
        ranked = rank_candidates((unidentified, identified))
        self.assertEqual(ranked[0].index, 0)

    def test_stable_order_preserved_on_a_full_tie(self):
        first = _candidate(0, [_variable("age")])
        second = _candidate(1, [_variable("age")])
        ranked = rank_candidates((first, second))
        self.assertEqual([c.index for c in ranked], [0, 1])


class DedupCandidatesTests(unittest.TestCase):
    def test_drops_structurally_identical_candidate_keeping_the_higher_ranked_one(self):
        best = _candidate(0, [_variable("age")])
        duplicate = _candidate(1, [_variable("age")])
        survivors, dropped = dedup_candidates(rank_candidates((best, duplicate)))
        self.assertEqual(len(survivors), 1)
        self.assertEqual(dropped, 1)

    def test_keeps_structurally_different_candidates(self):
        a = _candidate(0, [_variable("age")])
        b = _candidate(1, [_variable("severity")])
        survivors, dropped = dedup_candidates(rank_candidates((a, b)))
        self.assertEqual(len(survivors), 2)
        self.assertEqual(dropped, 0)


class RunTournamentTests(unittest.TestCase):
    def test_disjoint_candidates_are_unioned(self):
        candidates_raw = (
            ([_variable("age")], [], [], ""),
            ([_variable("severity")], [], [], ""),
        )
        result = run_tournament(_protocol(), candidates_raw)
        self.assertEqual(len(result.survivors), 2)
        self.assertEqual(result.dropped_duplicate_count, 0)
        labels = {variable["label"] for variable in result.merged_variables}
        self.assertEqual(labels, {"age", "severity"})
        self.assertEqual(result.conflicts, ())

    def test_duplicate_candidates_are_deduped_before_merge(self):
        candidates_raw = (
            ([_variable("age")], [], [], ""),
            ([_variable("age")], [], [], ""),
        )
        result = run_tournament(_protocol(), candidates_raw)
        self.assertEqual(result.dropped_duplicate_count, 1)
        self.assertEqual(len(result.merged_variables), 1)

    def test_disagreeing_role_is_a_conflict_and_excludes_the_label(self):
        candidates_raw = (
            ([_variable("age", role="confounder")], [], [], ""),
            ([_variable("age", role="effect_modifier")], [], [], ""),
        )
        result = run_tournament(_protocol(), candidates_raw)
        self.assertEqual([conflict.kind for conflict in result.conflicts], ["role_conflict"])
        self.assertEqual(result.merged_variables, ())

    def test_disagreeing_relation_direction_is_a_conflict_and_drops_the_relation(self):
        shared_variables = [_variable("age"), _variable("severity")]
        candidates_raw = (
            (shared_variables, [_relation("age", "severity")], [], ""),
            (shared_variables, [_relation("severity", "age")], [], ""),
        )
        result = run_tournament(_protocol(), candidates_raw)
        self.assertEqual([conflict.kind for conflict in result.conflicts], ["direction_conflict"])
        self.assertEqual(result.merged_relations, ())
        # Both candidates still agree on the variables themselves.
        self.assertEqual({v["label"] for v in result.merged_variables}, {"age", "severity"})

    def test_three_way_cycle_across_candidates_is_broken_and_flagged(self):
        shared_variables = [_variable("a"), _variable("b"), _variable("c")]
        candidates_raw = (
            (shared_variables, [_relation("a", "b")], [], ""),
            (shared_variables, [_relation("b", "c")], [], ""),
            (shared_variables, [_relation("c", "a")], [], ""),
        )
        result = run_tournament(_protocol(), candidates_raw)
        self.assertEqual(len(result.merged_relations), 2)
        self.assertEqual(len(result.conflicts), 1)
        self.assertEqual(result.conflicts[0].kind, "cycle_conflict")
        # The surviving merge must actually build (no residual cycle).
        review = build_pre_check(_protocol(), list(result.merged_variables), list(result.merged_relations))
        self.assertIsNotNone(review)

    def test_relation_touching_a_role_conflicted_label_is_dropped_without_a_second_conflict(self):
        candidates_raw = (
            (
                [_variable("age", role="confounder"), _variable("severity", role="confounder")],
                [_relation("age", "severity")],
                [],
                "",
            ),
            ([_variable("age", role="effect_modifier"), _variable("severity", role="confounder")], [], [], ""),
        )
        result = run_tournament(_protocol(), candidates_raw)
        kinds = [conflict.kind for conflict in result.conflicts]
        self.assertEqual(kinds, ["role_conflict"])
        self.assertEqual(result.merged_relations, ())

    def test_unresolved_entries_are_deduped_by_label_across_candidates(self):
        unresolved_entry = {"label": "bmi", "reason": "no clear role"}
        candidates_raw = (
            ([_variable("age")], [], [unresolved_entry], ""),
            ([_variable("age")], [], [dict(unresolved_entry)], ""),
        )
        result = run_tournament(_protocol(), candidates_raw)
        self.assertEqual(result.merged_unresolved, (unresolved_entry,))

    def test_empty_population_yields_an_empty_merge_without_raising(self):
        result = run_tournament(_protocol(), (([], [], [], ""), ([], [], [], "")))
        self.assertEqual(result.merged_variables, ())
        self.assertEqual(result.merged_relations, ())
        self.assertEqual(result.conflicts, ())


class MaxTournamentSizeTests(unittest.TestCase):
    def test_is_a_small_positive_bound(self):
        self.assertGreaterEqual(MAX_TOURNAMENT_SIZE, 2)
        self.assertLessEqual(MAX_TOURNAMENT_SIZE, 10)


if __name__ == "__main__":
    unittest.main()
