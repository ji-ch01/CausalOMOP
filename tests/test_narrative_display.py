import json
import unittest

from causal.agents.narrative_display import build_narrative_display, display_or_dash
from causal.agents.results_faithfulness import verify_narrative
from causal.agents.results_narrative import Claim, NarrativeInput, NarrativeResult

_ESTIMATION = {
    "estimand": {
        "point_estimate": 0.0418732,
        "ci_lower": -0.0158321,
        "ci_upper": 0.0982004,
        "ci_level": 0.95,
        "effect_measure": "risk_difference",
        "risk_ratio_point_estimate": 1.348219,
        "risk_ratio_ci_lower": 0.883,
        "risk_ratio_ci_upper": 2.051,
        "outcome_model_auc": 0.7413,
        "outcome_model_brier": 0.1822,
        "n_total": 812,
        "n_treated": 402,
        "n_comparator": 410,
        "n_outcome_events": 143,
    }
}
_DIAGNOSTICS = {
    "n_covariates_flagged": 2,
    "smd_flag_threshold": 0.1,
    "balance": [
        {"column_name": "age", "smd": 0.1421, "treated_mean": 1, "comparator_mean": 1, "flagged": True},
        {"column_name": "sex", "smd": -0.0312, "treated_mean": 1, "comparator_mean": 1, "flagged": False},
    ],
}
_SENSITIVITY = {"e_value_point": 1.9432, "e_value_ci": 1.2071}


class RoundingTests(unittest.TestCase):
    def test_risk_difference_and_ci_are_rounded(self):
        block = build_narrative_display(_ESTIMATION)["absolute_effect"]
        self.assertEqual(block["risk_difference"], 0.0419)
        self.assertEqual(block["ci_lower"], -0.0158)
        self.assertEqual(block["ci_upper"], 0.0982)
        self.assertEqual(block["ci_level_pct"], 95)

    def test_events_per_1000_are_whole_signed_integers(self):
        block = build_narrative_display(_ESTIMATION)["absolute_effect"]
        self.assertEqual(block["events_per_n_patients"], 1000)
        self.assertEqual(block["events_per_n_point"], 42)
        self.assertEqual(block["events_per_n_ci_lower"], -16)
        self.assertEqual(block["events_per_n_ci_upper"], 98)

    def test_percentage_points_rounded_to_one_decimal(self):
        block = build_narrative_display(_ESTIMATION)["absolute_effect"]
        self.assertEqual(block["percentage_points_point"], 4.2)

    def test_risk_ratio_rounded_to_two_decimals_with_relative_change(self):
        block = build_narrative_display(_ESTIMATION)["relative_effect"]
        self.assertTrue(block["available"])
        self.assertEqual(block["risk_ratio"], 1.35)
        self.assertEqual(block["ci_lower"], 0.88)
        self.assertEqual(block["ci_upper"], 2.05)
        self.assertEqual(block["relative_change_pct"], 35)

    def test_model_fit_and_evalue_rounded_to_two_decimals(self):
        block = build_narrative_display(_ESTIMATION, _DIAGNOSTICS, _SENSITIVITY)
        self.assertEqual(block["model_fit"]["outcome_model_auc"], 0.74)
        self.assertEqual(block["model_fit"]["outcome_model_brier"], 0.18)
        self.assertEqual(block["unmeasured_confounding_evalue"]["e_value_point"], 1.94)
        self.assertEqual(block["unmeasured_confounding_evalue"]["e_value_ci"], 1.21)

    def test_largest_absolute_smd_is_reported_rounded(self):
        block = build_narrative_display(_ESTIMATION, _DIAGNOSTICS)["covariate_balance"]
        self.assertEqual(block["largest_absolute_smd"], 0.14)
        self.assertEqual(block["n_covariates_flagged"], 2)
        self.assertEqual(block["n_covariates_checked"], 2)

    def test_balance_statement_is_framed_as_crude_pre_adjustment(self):
        # `run_diagnostics` computes SMD on the raw analytic cohort, and
        # g-computation/TMLE adjust through the outcome model rather than by
        # rebalancing covariates -- so this SMD is a crude, before-adjustment
        # quantity and the plain statement must not imply otherwise (it is
        # copied verbatim into the narrative prompt and companion dialogue).
        flagged = build_narrative_display(_ESTIMATION, _DIAGNOSTICS)["covariate_balance"]
        self.assertIn("before outcome-model adjustment", flagged["plain_statement"])
        self.assertNotIn("After adjustment", flagged["plain_statement"])
        self.assertNotIn("after adjustment", flagged["plain_statement"])

        balanced = build_narrative_display(
            _ESTIMATION, {**_DIAGNOSTICS, "n_covariates_flagged": 0}
        )["covariate_balance"]
        self.assertIn("before outcome-model adjustment", balanced["plain_statement"])
        self.assertNotIn("after adjustment", balanced["plain_statement"])


class DerivedQuantityTests(unittest.TestCase):
    def test_nnt_omitted_when_interval_includes_no_effect(self):
        block = build_narrative_display(_ESTIMATION)["number_needed_to_treat"]
        self.assertFalse(block["reported"])
        self.assertIn("includes no effect", block["reason"])

    def test_nnt_reported_when_interval_excludes_no_effect(self):
        estimation = {"estimand": {**_ESTIMATION["estimand"], "point_estimate": -0.04, "ci_lower": -0.07, "ci_upper": -0.01}}
        block = build_narrative_display(estimation)["number_needed_to_treat"]
        self.assertTrue(block["reported"])
        self.assertEqual(block["patients_per_event"], 25)
        self.assertIn("1 fewer outcome event for every 25 patients", block["plain_statement"])

    def test_direction_wording_uses_a_recognised_keyword(self):
        higher = build_narrative_display(_ESTIMATION)["absolute_effect"]["plain_statement"]
        self.assertIn("a higher risk of the outcome", higher)
        lower = build_narrative_display(
            {"estimand": {**_ESTIMATION["estimand"], "point_estimate": -0.04, "ci_lower": -0.07, "ci_upper": -0.01}}
        )["absolute_effect"]["plain_statement"]
        self.assertIn("a lower risk of the outcome", lower)

    def test_risk_ratio_unavailable_when_not_a_number(self):
        estimation = {"estimand": {**_ESTIMATION["estimand"], "risk_ratio_point_estimate": float("nan")}}
        block = build_narrative_display(estimation)["relative_effect"]
        self.assertFalse(block["available"])
        self.assertIn("only the absolute", block["reason"])

    def test_no_optional_blocks_without_their_inputs(self):
        block = build_narrative_display(_ESTIMATION)
        self.assertNotIn("covariate_balance", block)
        self.assertNotIn("unmeasured_confounding_evalue", block)

    def test_handles_a_missing_point_estimate(self):
        block = build_narrative_display({"estimand": {}})
        self.assertFalse(block["absolute_effect"]["available"])


class DisplayOrDashTests(unittest.TestCase):
    def test_none_and_nan_render_as_dash(self):
        self.assertEqual(display_or_dash(None), "-")
        self.assertEqual(display_or_dash(float("nan")), "-")
        self.assertEqual(display_or_dash(float("inf")), "-")

    def test_a_real_number_passes_through(self):
        self.assertEqual(display_or_dash(0.42), 0.42)
        self.assertEqual(display_or_dash(0), 0)


class FaithfulnessInteractionTests(unittest.TestCase):
    """The whole point of the block: a narrative that quotes only the
    rounded `narrative_display` values must pass `verify_narrative`, which
    rejects any number not present verbatim in the JSON it was shown."""

    def _input(self) -> NarrativeInput:
        estimation = {
            **_ESTIMATION,
            "narrative_display": build_narrative_display(_ESTIMATION, _DIAGNOSTICS, _SENSITIVITY),
        }
        return NarrativeInput(
            study_id="s",
            question={"population": "sepsis", "treatment": "albumin", "comparator": "crystalloid", "outcome": "death"},
            estimand={"population_summary": "ATE", "effect_measure": "risk_difference"},
            dataset_summary={"row_count": 812, "column_count": 12},
            estimation=estimation,
            diagnostics=_DIAGNOSTICS,
            gate_history=(),
            assumptions_and_limitations=("An unmeasured confounder omitted from the DAG would still bias this estimate.",),
        )

    def test_a_fully_rounded_narrative_passes_every_check(self):
        narrative_input = self._input()
        display = narrative_input.estimation["narrative_display"]
        narrative = NarrativeResult(
            narrative_text=(
                display["absolute_effect"]["plain_statement"]
                + " "
                + display["relative_effect"]["plain_statement"]
                + " The analysis includes 812 patients with 143 outcome events. "
                + display["covariate_balance"]["plain_statement"]
                + " An unmeasured confounder omitted from the DAG would still bias this estimate."
            ),
            claims=(
                Claim(
                    text=display["absolute_effect"]["plain_statement"],
                    claim_type="effect_direction",
                    source_fields=("estimation.narrative_display.absolute_effect.plain_statement",),
                ),
                Claim(
                    text="The 95% compatibility interval runs from 16 fewer to 98 more per 1000 and includes the possibility of no effect.",
                    claim_type="uncertainty",
                    source_fields=(
                        "estimation.narrative_display.absolute_effect.events_per_n_ci_lower",
                        "estimation.narrative_display.absolute_effect.events_per_n_ci_upper",
                    ),
                ),
                Claim(
                    text="The analysis includes 812 patients.",
                    claim_type="sample_size",
                    source_fields=("estimation.estimand.n_total",),
                ),
                Claim(
                    text=display["covariate_balance"]["plain_statement"],
                    claim_type="diagnostic_warning",
                    source_fields=("diagnostics.n_covariates_flagged",),
                ),
                Claim(
                    text="An unmeasured confounder omitted from the DAG would still bias this estimate.",
                    claim_type="assumption_or_limitation",
                    source_fields=("assumptions_and_limitations",),
                ),
            ),
        )
        report = verify_narrative(narrative, narrative_input)
        self.assertEqual(report.status, "passed", [c.to_dict() for c in report.checks])

    def test_the_block_is_json_serialisable_and_finite(self):
        block = build_narrative_display(_ESTIMATION, _DIAGNOSTICS, _SENSITIVITY)
        text = json.dumps(block)  # would raise on a non-serialisable value
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)


if __name__ == "__main__":
    unittest.main()
