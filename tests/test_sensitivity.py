import math
import unittest

from causal.sensitivity import e_value


class EValueFormulaTests(unittest.TestCase):
    """Validated against closed-form/analytically-verifiable properties of
    VanderWeele & Ding 2017's own published formula, rather than against a
    remembered rounded worked example from the paper -- these properties
    (E(1)=1, symmetry under RR -> 1/RR, the exact RR=2 closed form) are
    stated in the paper itself and are safe to hardcode."""

    def test_null_risk_ratio_has_e_value_one(self):
        result = e_value(1.0, 1.0, 1.0)
        self.assertEqual(result.e_value_point, 1.0)
        self.assertEqual(result.e_value_ci, 1.0)

    def test_matches_the_closed_form_for_risk_ratio_two(self):
        result = e_value(2.0, 2.0, 2.0)
        self.assertAlmostEqual(result.e_value_point, 2.0 + math.sqrt(2.0), places=9)

    def test_is_symmetric_under_inversion(self):
        above = e_value(3.0, 3.0, 3.0)
        below = e_value(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        self.assertAlmostEqual(above.e_value_point, below.e_value_point, places=9)

    def test_e_value_increases_as_risk_ratio_moves_away_from_one(self):
        near_null = e_value(1.2, 1.2, 1.2)
        far_from_null = e_value(4.0, 4.0, 4.0)
        self.assertLess(near_null.e_value_point, far_from_null.e_value_point)

    def test_ci_bound_closest_to_the_null_is_used_above_one(self):
        result = e_value(3.0, 2.0, 5.0)
        self.assertAlmostEqual(result.e_value_ci, 2.0 + math.sqrt(2.0 * 1.0), places=9)

    def test_ci_bound_closest_to_the_null_is_used_below_one(self):
        result = e_value(0.4, 0.2, 0.6)
        expected_rr = 1.0 / 0.6
        self.assertAlmostEqual(result.e_value_ci, expected_rr + math.sqrt(expected_rr * (expected_rr - 1.0)), places=9)

    def test_ci_crossing_the_null_gives_e_value_ci_of_one(self):
        result = e_value(3.0, 0.8, 5.0)
        self.assertEqual(result.e_value_ci, 1.0)

    def test_protective_risk_ratio_ci_crossing_the_null_gives_e_value_ci_of_one(self):
        result = e_value(0.4, 0.3, 1.5)
        self.assertEqual(result.e_value_ci, 1.0)

    def test_rejects_non_positive_risk_ratio(self):
        with self.assertRaises(ValueError):
            e_value(0.0, 0.5, 1.5)
        with self.assertRaises(ValueError):
            e_value(-1.0, 0.5, 1.5)

    def test_rejects_a_nan_risk_ratio(self):
        """A legitimate value for `GComputationEstimate`/`TMLEEstimate`'s
        `risk_ratio_point_estimate` when an arm's standardized risk was
        exactly 0 -- must raise explicitly rather than silently pass the
        `<= 0` check (NaN compares False to everything) and produce a NaN
        E-value."""
        with self.assertRaises(ValueError):
            e_value(float("nan"), 0.5, 1.5)
        with self.assertRaises(ValueError):
            e_value(2.0, float("nan"), 1.5)

    def test_rejects_an_inverted_confidence_interval(self):
        with self.assertRaises(ValueError):
            e_value(2.0, 3.0, 1.0)

    def test_reports_citation_and_interpretation(self):
        result = e_value(2.0, 1.5, 2.8)
        self.assertIn("VanderWeele", result.citation)
        self.assertIn("2017", result.citation)
        self.assertIn("E-value", result.interpretation)
        self.assertIn("unmeasured confounder", result.interpretation)

    def test_to_dict_round_trips_every_field(self):
        result = e_value(2.0, 1.5, 2.8)
        as_dict = result.to_dict()
        self.assertEqual(set(as_dict), {"risk_ratio", "e_value_point", "e_value_ci", "citation", "interpretation"})


if __name__ == "__main__":
    unittest.main()
