"""Coverage for `causal.companion.reviewer_memory` -- the pure fold of one
run's decisions into a reviewer's cross-study memory, and its rendering. No
I/O, no LLM."""

import json
import unittest

from causal.companion.decision_memory import DecisionMemory, RejectedCovariate
from causal.companion.reviewer_memory import (
    RememberedEstimandChoice,
    RememberedExclusion,
    ReviewerMemory,
    merge_run_into_memory,
    slugify_reviewer,
)

_NOW = "2026-08-28T12:00:00+00:00"


class SlugifyReviewerTests(unittest.TestCase):
    def test_normalises_case_and_punctuation(self):
        self.assertEqual(slugify_reviewer("  Dr. Chen  "), "dr_chen")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            slugify_reviewer("  ...  ")


class MergeRunIntoMemoryTests(unittest.TestCase):
    def _decision_memory(self, *labels: str) -> DecisionMemory:
        return DecisionMemory(
            rejected_covariates=tuple(
                RejectedCovariate(label, "Dr. Chen", f"{label} is post-treatment", _NOW)
                for label in labels
            )
        )

    def test_adds_exclusions_and_the_estimand_choice(self):
        merged = merge_run_into_memory(
            ReviewerMemory(reviewer_id="dr_chen"),
            study_id="aspirin_ami",
            decision_memory=self._decision_memory("baseline_lactate"),
            estimand_settings={
                "missing_data_strategy": "complete_case_analysis",
                "censoring_strategy": "ignore_censoring_administrative_waiver",
                "estimation_procedure": "targeted_minimum_loss_based_estimation",
            },
            now=_NOW,
        )
        self.assertEqual([e.label for e in merged.exclusions], ["baseline_lactate"])
        self.assertEqual(merged.exclusions[0].study_id, "aspirin_ami")
        self.assertEqual(len(merged.estimand_choices), 1)
        self.assertEqual(
            merged.estimand_choices[0].estimation_procedure,
            "targeted_minimum_loss_based_estimation",
        )

    def test_no_estimand_choice_when_settings_incomplete(self):
        merged = merge_run_into_memory(
            ReviewerMemory(),
            study_id="s1",
            decision_memory=self._decision_memory(),
            estimand_settings={"missing_data_strategy": "complete_case_analysis"},
            now=_NOW,
        )
        self.assertEqual(merged.estimand_choices, ())

    def test_re_merging_the_same_study_replaces_not_appends(self):
        first = merge_run_into_memory(
            ReviewerMemory(),
            study_id="aspirin_ami",
            decision_memory=self._decision_memory("baseline_lactate", "prior_mi"),
            now=_NOW,
        )
        second = merge_run_into_memory(
            first,
            study_id="aspirin_ami",
            decision_memory=self._decision_memory("baseline_lactate"),
            now=_NOW,
        )
        self.assertEqual([e.label for e in second.exclusions], ["baseline_lactate"])

    def test_other_studies_are_left_untouched(self):
        first = merge_run_into_memory(
            ReviewerMemory(),
            study_id="study_a",
            decision_memory=self._decision_memory("cov_a"),
            now="2026-08-01T00:00:00+00:00",
        )
        second = merge_run_into_memory(
            first,
            study_id="study_b",
            decision_memory=self._decision_memory("cov_b"),
            now=_NOW,
        )
        self.assertEqual({e.study_id for e in second.exclusions}, {"study_a", "study_b"})


class FormatForPromptTests(unittest.TestCase):
    def test_empty_sentence(self):
        self.assertEqual(
            ReviewerMemory().format_for_prompt(),
            "No cross-study conventions recorded for this reviewer yet.",
        )

    def test_renders_both_sections(self):
        memory = ReviewerMemory(
            exclusions=(RememberedExclusion("lactate", "post-treatment", "aspirin_ami", _NOW),),
            estimand_choices=(
                RememberedEstimandChoice(
                    "aspirin_ami", "complete_case_analysis",
                    "ipcw_administrative_censoring", "plug_in_standardization", _NOW,
                ),
            ),
        )
        rendered = memory.format_for_prompt()
        self.assertIn("excluded in earlier studies", rendered)
        self.assertIn('lactate: "post-treatment"', rendered)
        self.assertIn("strategies you have chosen before", rendered)
        self.assertIn("ipcw_administrative_censoring", rendered)


class RoundTripTests(unittest.TestCase):
    def test_to_dict_from_dict(self):
        memory = merge_run_into_memory(
            ReviewerMemory(reviewer_id="dr_chen"),
            study_id="aspirin_ami",
            decision_memory=DecisionMemory(
                rejected_covariates=(RejectedCovariate("lactate", "Dr. Chen", "post-tx", _NOW),)
            ),
            estimand_settings={
                "missing_data_strategy": "complete_case_analysis",
                "censoring_strategy": "ignore_censoring_administrative_waiver",
                "estimation_procedure": "plug_in_standardization",
            },
            now=_NOW,
        )
        restored = ReviewerMemory.from_dict(json.loads(json.dumps(memory.to_dict())))
        self.assertEqual(restored, memory)


if __name__ == "__main__":
    unittest.main()
