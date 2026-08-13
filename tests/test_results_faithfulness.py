import unittest

from causal.agents.results_faithfulness import FaithfulnessCheck, FaithfulnessReport, verify_narrative
from causal.agents.results_narrative import Claim, NarrativeInput, NarrativeResult


def _narrative_input(
    *,
    point_estimate: float = 0.0421,
    ci_lower: float = -0.0158,
    ci_upper: float = 0.0983,
    ci_level: float = 0.95,
    effect_measure: str = "risk_difference",
    n_total: int = 300,
    n_covariates_flagged: int = 1,
    assumptions: tuple[str, ...] = ("Assumes no unmeasured confounding.", "Complete-case analysis."),
) -> NarrativeInput:
    return NarrativeInput(
        study_id="test_study",
        question={"population": "sepsis", "treatment": "albumin", "comparator": "crystalloid", "outcome": "death"},
        estimand={"population_summary": "ATE", "effect_measure": effect_measure},
        dataset_summary={"row_count": n_total, "column_count": 10},
        estimation={
            "estimand": {
                "point_estimate": point_estimate,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "ci_level": ci_level,
                "effect_measure": effect_measure,
                "n_total": n_total,
            }
        },
        diagnostics={
            "n_covariates_flagged": n_covariates_flagged,
            "warnings": ("Some covariates are imbalanced.",) if n_covariates_flagged else (),
        },
        gate_history=(),
        assumptions_and_limitations=assumptions,
    )


def _find(report: FaithfulnessReport, subject: str) -> FaithfulnessCheck:
    for check in report.checks:
        if check.subject == subject:
            return check
    raise AssertionError(f"no check with subject={subject!r} in {[c.subject for c in report.checks]}")


class FullyFaithfulNarrativeTests(unittest.TestCase):
    """A narrative that plays entirely by the rules should pass every check."""

    def test_all_checks_pass(self):
        narrative_input = _narrative_input()
        narrative = NarrativeResult(
            narrative_text=(
                "The estimated effect is 0.0421, a slightly increased risk, with a confidence interval of "
                "[-0.0158, 0.0983] that crosses the null across 300 patients. 1 covariate is imbalanced."
            ),
            claims=(
                Claim(
                    text="The point estimate of 0.0421 indicates an increase in risk.",
                    claim_type="effect_direction",
                    source_fields=("estimation.estimand.point_estimate",),
                ),
                Claim(
                    text="The confidence interval [-0.0158, 0.0983] crosses the null.",
                    claim_type="uncertainty",
                    source_fields=("estimation.estimand.ci_lower", "estimation.estimand.ci_upper"),
                ),
                Claim(
                    text="The analysis covers 300 patients in total.",
                    claim_type="sample_size",
                    source_fields=("estimation.estimand.n_total",),
                ),
                Claim(
                    text="1 covariate shows an imbalance flagged by the diagnostics.",
                    claim_type="diagnostic_warning",
                    source_fields=("diagnostics.n_covariates_flagged",),
                ),
                Claim(
                    text="The analysis assumes no unmeasured confounding.",
                    claim_type="assumption_or_limitation",
                    source_fields=("assumptions_and_limitations",),
                ),
            ),
        )
        report = verify_narrative(narrative, narrative_input)
        self.assertEqual(report.status, "passed")
        for check in report.checks:
            self.assertEqual(check.status, "passed", msg=f"{check.subject}: {check.reason}")


class NumberFaithfulnessTests(unittest.TestCase):
    def test_invented_number_in_narrative_text_fails(self):
        narrative_input = _narrative_input()
        narrative = NarrativeResult(
            narrative_text="The effect is estimated at 42.7, far outside anything in the source data.",
            claims=(
                Claim(
                    text="Increase in risk.", claim_type="effect_direction",
                    source_fields=("estimation.estimand.point_estimate",),
                ),
            ),
        )
        report = verify_narrative(narrative, narrative_input)
        self.assertEqual(report.status, "failed")
        check = _find(report, "narrative_text")
        self.assertEqual(check.status, "failed")
        self.assertIn("42.7", check.reason)

    def test_a_differently_rounded_number_fails(self):
        """0.0421 rounded to 0.04 is not a verbatim copy -- the prompt
        explicitly forbids re-rounding, so this must be caught."""
        narrative_input = _narrative_input()
        narrative = NarrativeResult(
            narrative_text="The point estimate is approximately 0.04.",
            claims=(
                Claim(
                    text="Increase in risk.", claim_type="effect_direction",
                    source_fields=("estimation.estimand.point_estimate",),
                ),
            ),
        )
        report = verify_narrative(narrative, narrative_input)
        check = _find(report, "narrative_text")
        self.assertEqual(check.status, "failed")
        self.assertIn("0.04", check.reason)

    def test_a_verbatim_number_passes(self):
        narrative_input = _narrative_input()
        narrative = NarrativeResult(
            narrative_text="The point estimate is 0.0421 among 300 patients.",
            claims=(
                Claim(
                    text="Increase in risk.", claim_type="effect_direction",
                    source_fields=("estimation.estimand.point_estimate",),
                ),
            ),
        )
        report = verify_narrative(narrative, narrative_input)
        self.assertEqual(_find(report, "narrative_text").status, "passed")

    def test_ci_level_percentage_phrasing_is_accepted(self):
        """0.95 (ci_level) conventionally read aloud as "95%" is not an
        invented number -- it is a faithful restatement of a field already
        in the artifacts."""
        narrative_input = _narrative_input(ci_level=0.95)
        narrative = NarrativeResult(
            narrative_text="The 95% confidence interval is reported below.",
            claims=(
                Claim(
                    text="Increase in risk.", claim_type="effect_direction",
                    source_fields=("estimation.estimand.point_estimate",),
                ),
            ),
        )
        report = verify_narrative(narrative, narrative_input)
        self.assertEqual(_find(report, "narrative_text").status, "passed")


class EffectDirectionTests(unittest.TestCase):
    def test_wrong_direction_fails(self):
        narrative_input = _narrative_input(point_estimate=0.0421)  # positive => increase
        narrative = NarrativeResult(
            narrative_text="See claims.",
            claims=(
                Claim(
                    text="The treatment is associated with a decrease in risk.",
                    claim_type="effect_direction",
                    source_fields=("estimation.estimand.point_estimate",),
                ),
            ),
        )
        report = verify_narrative(narrative, narrative_input)
        claim_check = _find(report, "claim_0")
        self.assertEqual(claim_check.status, "failed")
        self.assertIn("decrease", claim_check.reason)
        self.assertEqual(report.status, "failed")

    def test_correct_direction_passes(self):
        narrative_input = _narrative_input(point_estimate=0.0421)
        narrative = NarrativeResult(
            narrative_text="See claims.",
            claims=(
                Claim(
                    text="The treatment is associated with an increase in risk.",
                    claim_type="effect_direction",
                    source_fields=("estimation.estimand.point_estimate",),
                ),
            ),
        )
        report = verify_narrative(narrative, narrative_input)
        self.assertEqual(_find(report, "claim_0").status, "passed")

    def test_unrecognized_direction_wording_is_a_warning_not_a_failure(self):
        narrative_input = _narrative_input(point_estimate=0.0421)
        narrative = NarrativeResult(
            narrative_text="See claims.",
            claims=(
                Claim(
                    text="Something happened to the risk.",
                    claim_type="effect_direction",
                    source_fields=("estimation.estimand.point_estimate",),
                ),
            ),
        )
        report = verify_narrative(narrative, narrative_input)
        self.assertEqual(_find(report, "claim_0").status, "warning")


class UncertaintyTests(unittest.TestCase):
    def test_ci_crossing_null_described_as_significant_fails(self):
        # ci_lower/ci_upper straddle 0 -> crosses the null.
        narrative_input = _narrative_input(ci_lower=-0.02, ci_upper=0.10)
        narrative = NarrativeResult(
            narrative_text="See claims.",
            claims=(
                Claim(
                    text="The effect is statistically significant.",
                    claim_type="uncertainty",
                    source_fields=("estimation.estimand.ci_lower", "estimation.estimand.ci_upper"),
                ),
            ),
        )
        report = verify_narrative(narrative, narrative_input)
        claim_check = _find(report, "claim_0")
        self.assertEqual(claim_check.status, "failed")
        self.assertEqual(report.status, "failed")

    def test_ci_excluding_null_described_correctly_passes(self):
        narrative_input = _narrative_input(ci_lower=-0.11, ci_upper=-0.01)
        narrative = NarrativeResult(
            narrative_text="See claims.",
            claims=(
                Claim(
                    text="The confidence interval does not cross the null.",
                    claim_type="uncertainty",
                    source_fields=("estimation.estimand.ci_lower", "estimation.estimand.ci_upper"),
                ),
            ),
        )
        report = verify_narrative(narrative, narrative_input)
        self.assertEqual(_find(report, "claim_0").status, "passed")

    def test_ci_excluding_null_described_as_crossing_fails(self):
        narrative_input = _narrative_input(ci_lower=-0.11, ci_upper=-0.01)
        narrative = NarrativeResult(
            narrative_text="See claims.",
            claims=(
                Claim(
                    text="The confidence interval crosses the null.",
                    claim_type="uncertainty",
                    source_fields=("estimation.estimand.ci_lower", "estimation.estimand.ci_upper"),
                ),
            ),
        )
        report = verify_narrative(narrative, narrative_input)
        self.assertEqual(_find(report, "claim_0").status, "failed")


class DiagnosticWarningCoverageTests(unittest.TestCase):
    def test_omitted_diagnostic_warning_fails(self):
        narrative_input = _narrative_input(n_covariates_flagged=2)
        narrative = NarrativeResult(
            narrative_text="The effect estimate is 0.0421.",
            claims=(
                Claim(
                    text="Increase in risk.", claim_type="effect_direction",
                    source_fields=("estimation.estimand.point_estimate",),
                ),
            ),
        )
        report = verify_narrative(narrative, narrative_input)
        check = _find(report, "diagnostic_warning_coverage")
        self.assertEqual(check.status, "failed")
        self.assertEqual(report.status, "failed")

    def test_covered_via_a_typed_claim_passes(self):
        narrative_input = _narrative_input(n_covariates_flagged=2)
        narrative = NarrativeResult(
            narrative_text="The effect estimate is 0.0421.",
            claims=(
                Claim(
                    text="Two covariates are imbalanced between arms.",
                    claim_type="diagnostic_warning",
                    source_fields=("diagnostics.n_covariates_flagged",),
                ),
            ),
        )
        report = verify_narrative(narrative, narrative_input)
        self.assertEqual(_find(report, "diagnostic_warning_coverage").status, "passed")

    def test_covered_via_narrative_prose_alone_passes(self):
        narrative_input = _narrative_input(n_covariates_flagged=2)
        narrative = NarrativeResult(
            narrative_text="Two covariates show meaningful imbalance between the treatment arms.",
            claims=(
                Claim(
                    text="Increase in risk.", claim_type="effect_direction",
                    source_fields=("estimation.estimand.point_estimate",),
                ),
            ),
        )
        report = verify_narrative(narrative, narrative_input)
        self.assertEqual(_find(report, "diagnostic_warning_coverage").status, "passed")

    def test_no_flagged_covariates_needs_no_mention(self):
        narrative_input = _narrative_input(n_covariates_flagged=0)
        narrative = NarrativeResult(
            narrative_text="The effect estimate is 0.0421.",
            claims=(
                Claim(
                    text="Increase in risk.", claim_type="effect_direction",
                    source_fields=("estimation.estimand.point_estimate",),
                ),
            ),
        )
        report = verify_narrative(narrative, narrative_input)
        self.assertEqual(_find(report, "diagnostic_warning_coverage").status, "passed")


class AssumptionsCoverageTests(unittest.TestCase):
    def test_omitted_assumptions_are_a_warning_not_a_failure(self):
        narrative_input = _narrative_input(
            n_covariates_flagged=0, assumptions=("Assumes no unmeasured confounding.",)
        )
        narrative = NarrativeResult(
            narrative_text="The effect estimate is 0.0421.",
            claims=(
                Claim(
                    text="Increase in risk.", claim_type="effect_direction",
                    source_fields=("estimation.estimand.point_estimate",),
                ),
            ),
        )
        report = verify_narrative(narrative, narrative_input)
        check = _find(report, "assumptions_coverage")
        self.assertEqual(check.status, "warning")
        self.assertEqual(report.status, "warning")

    def test_covered_via_a_typed_claim_passes(self):
        narrative_input = _narrative_input(assumptions=("Assumes no unmeasured confounding.",))
        narrative = NarrativeResult(
            narrative_text="The effect estimate is 0.0421.",
            claims=(
                Claim(
                    text="This assumes no unmeasured confounder was omitted.",
                    claim_type="assumption_or_limitation",
                    source_fields=("assumptions_and_limitations",),
                ),
            ),
        )
        report = verify_narrative(narrative, narrative_input)
        self.assertEqual(_find(report, "assumptions_coverage").status, "passed")

    def test_contradicting_an_assumption_fails(self):
        narrative_input = _narrative_input(assumptions=("Assumes no unmeasured confounding.",))
        narrative = NarrativeResult(
            narrative_text="This result definitively proves causation with no possible confounding.",
            claims=(
                Claim(
                    text="Increase in risk.", claim_type="effect_direction",
                    source_fields=("estimation.estimand.point_estimate",),
                ),
            ),
        )
        report = verify_narrative(narrative, narrative_input)
        check = _find(report, "assumptions_coverage")
        self.assertEqual(check.status, "failed")
        self.assertEqual(report.status, "failed")

    def test_no_assumptions_recorded_needs_no_mention(self):
        narrative_input = _narrative_input(assumptions=())
        narrative = NarrativeResult(
            narrative_text="The effect estimate is 0.0421.",
            claims=(
                Claim(
                    text="Increase in risk.", claim_type="effect_direction",
                    source_fields=("estimation.estimand.point_estimate",),
                ),
            ),
        )
        report = verify_narrative(narrative, narrative_input)
        self.assertEqual(_find(report, "assumptions_coverage").status, "passed")


class SourceFieldAllowListTests(unittest.TestCase):
    def test_field_outside_allow_listed_roots_fails(self):
        narrative_input = _narrative_input()
        narrative = NarrativeResult(
            narrative_text="See claims.",
            claims=(
                Claim(
                    text="Increase in risk.", claim_type="effect_direction",
                    source_fields=("analytic_dataset.row_0.value",),
                ),
            ),
        )
        report = verify_narrative(narrative, narrative_input)
        check = _find(report, "claim_0")
        self.assertEqual(check.status, "failed")
        self.assertIn("analytic_dataset.row_0.value", check.reason)

    def test_field_that_does_not_resolve_fails(self):
        narrative_input = _narrative_input()
        narrative = NarrativeResult(
            narrative_text="See claims.",
            claims=(
                Claim(
                    text="Increase in risk.", claim_type="effect_direction",
                    source_fields=("estimation.estimand.made_up_field",),
                ),
            ),
        )
        report = verify_narrative(narrative, narrative_input)
        check = _find(report, "claim_0")
        self.assertEqual(check.status, "failed")
        self.assertIn("made_up_field", check.reason)

    def test_field_that_resolves_is_recorded_as_evidence(self):
        narrative_input = _narrative_input()
        narrative = NarrativeResult(
            narrative_text="See claims.",
            claims=(
                Claim(
                    text="Increase in risk.", claim_type="effect_direction",
                    source_fields=("estimation.estimand.point_estimate",),
                ),
            ),
        )
        report = verify_narrative(narrative, narrative_input)
        check = _find(report, "claim_0")
        self.assertEqual(check.status, "passed")
        self.assertIn("estimation.estimand.point_estimate", check.evidence)


class ReportSerializationTests(unittest.TestCase):
    def test_to_dict_round_trips_status_and_checks(self):
        narrative_input = _narrative_input()
        narrative = NarrativeResult(
            narrative_text="The point estimate is 0.0421.",
            claims=(
                Claim(
                    text="Increase in risk.", claim_type="effect_direction",
                    source_fields=("estimation.estimand.point_estimate",),
                ),
            ),
        )
        report = verify_narrative(narrative, narrative_input)
        payload = report.to_dict()
        self.assertEqual(payload["status"], report.status)
        self.assertEqual(len(payload["checks"]), len(report.checks))
        self.assertIn("checked_at_utc", payload)
        for check_payload in payload["checks"]:
            self.assertEqual(
                set(check_payload.keys()), {"subject", "claim_type", "text", "status", "reason", "evidence"}
            )


if __name__ == "__main__":
    unittest.main()
