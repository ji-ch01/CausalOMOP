import unittest

from causal.mapping.metrics import (
    LookupCase,
    evaluate_lookup_accuracy,
    reciprocal_rank,
    recall_at_k,
)


class RecallAtKTests(unittest.TestCase):
    def test_true_when_expected_is_within_the_top_k(self):
        self.assertTrue(recall_at_k([5, 132797, 9], 132797, k=5))

    def test_false_when_expected_is_outside_the_top_k(self):
        self.assertFalse(recall_at_k([1, 2, 132797], 132797, k=2))

    def test_false_when_expected_is_entirely_absent(self):
        self.assertFalse(recall_at_k([1, 2, 3], 132797, k=10))

    def test_rejects_a_non_positive_k(self):
        with self.assertRaises(ValueError):
            recall_at_k([1], 1, k=0)


class ReciprocalRankTests(unittest.TestCase):
    def test_first_result_scores_one(self):
        self.assertEqual(reciprocal_rank([132797, 1, 2], 132797), 1.0)

    def test_third_result_scores_one_third(self):
        self.assertAlmostEqual(reciprocal_rank([1, 2, 132797], 132797), 1 / 3)

    def test_absent_result_scores_zero(self):
        self.assertEqual(reciprocal_rank([1, 2, 3], 132797), 0.0)


class EvaluateLookupAccuracyTests(unittest.TestCase):
    def test_perfect_lookup_scores_one_on_every_metric(self):
        cases = [
            LookupCase("sepsis", "condition", 132797),
            LookupCase("septic shock", "condition", 196236),
        ]
        report = evaluate_lookup_accuracy(cases, lookup=lambda case: [case.expected_concept_id])
        self.assertEqual(report.evaluated_case_count, 2)
        self.assertEqual(report.recall_at_1, 1.0)
        self.assertEqual(report.recall_at_5, 1.0)
        self.assertEqual(report.recall_at_10, 1.0)
        self.assertEqual(report.mean_reciprocal_rank, 1.0)
        self.assertEqual(report.misses, ())

    def test_a_result_ranked_third_still_counts_toward_recall_at_5_but_not_recall_at_1(self):
        cases = [LookupCase("sepsis", "condition", 132797)]
        report = evaluate_lookup_accuracy(cases, lookup=lambda case: [1, 2, 132797])
        self.assertEqual(report.recall_at_1, 0.0)
        self.assertEqual(report.recall_at_5, 1.0)
        self.assertAlmostEqual(report.mean_reciprocal_rank, 1 / 3)

    def test_a_miss_is_recorded_by_term_and_lowers_every_metric(self):
        cases = [
            LookupCase("sepsis", "condition", 132797),
            LookupCase("unfindable term", "condition", 999),
        ]
        report = evaluate_lookup_accuracy(
            cases, lookup=lambda case: [case.expected_concept_id] if case.term == "sepsis" else [1, 2, 3]
        )
        self.assertEqual(report.recall_at_10, 0.5)
        self.assertEqual(report.misses, ("unfindable term",))

    def test_known_gap_cases_are_excluded_from_every_metric_not_penalized(self):
        cases = [
            LookupCase("sepsis", "condition", 132797),
            LookupCase("chartevents item 220045", "measurement", None, note="known ETL gap: bare source itemid, concept_id=0"),
        ]
        report = evaluate_lookup_accuracy(cases, lookup=lambda case: [case.expected_concept_id])
        self.assertEqual(report.case_count, 2)
        self.assertEqual(report.evaluated_case_count, 1)
        self.assertEqual(report.known_gap_case_count, 1)
        self.assertEqual(report.recall_at_1, 1.0)

    def test_only_known_gap_cases_reports_zero_valued_metrics_without_dividing_by_zero(self):
        cases = [LookupCase("chartevents item 220045", "measurement", None)]
        report = evaluate_lookup_accuracy(cases, lookup=lambda case: [])
        self.assertEqual(report.evaluated_case_count, 0)
        self.assertEqual(report.known_gap_case_count, 1)
        self.assertEqual(report.recall_at_1, 0.0)
        self.assertEqual(report.mean_reciprocal_rank, 0.0)


if __name__ == "__main__":
    unittest.main()
