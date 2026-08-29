"""Coverage for `causal.companion.decision_memory` -- the pure, deterministic
distillation of a run's reviewer decisions into a replayable digest. No LLM,
no I/O, so it is fully exercised here with hand-built history/gate data."""

import unittest

from causal.companion.decision_memory import (
    DecisionMemory,
    RecordedDecision,
    RejectedCovariate,
    distill_decision_memory,
)

_FIXED = "2026-08-20T09:00:00+00:00"


class DistillDecisionMemoryTests(unittest.TestCase):
    def test_empty_run_yields_empty_memory(self):
        memory = distill_decision_memory(history=[], gates={}, rejected_covariates=[])
        self.assertTrue(memory.is_empty())
        self.assertEqual(memory.rejected_covariates, ())
        self.assertEqual(memory.decisions, ())

    def test_rejected_covariates_are_carried_through(self):
        memory = distill_decision_memory(
            history=[],
            rejected_covariates=[
                {
                    "label": "baseline_lactate",
                    "reviewer": "Dr. Chen",
                    "comment": "measured after the exposure; post-treatment",
                    "decided_at": _FIXED,
                }
            ],
        )
        self.assertEqual(len(memory.rejected_covariates), 1)
        self.assertEqual(memory.rejected_covariates[0].label, "baseline_lactate")
        self.assertFalse(memory.is_empty())

    def test_rejected_covariate_without_a_label_is_dropped(self):
        memory = distill_decision_memory(
            history=[], rejected_covariates=[{"reviewer": "Dr. Chen", "comment": "x"}]
        )
        self.assertEqual(memory.rejected_covariates, ())

    def test_only_decision_shaped_history_events_are_kept(self):
        history = [
            {"at": "2026-08-20T09:00:00+00:00", "event": "run_started", "detail": "mode=exploratory"},
            {"at": "2026-08-20T09:01:00+00:00", "event": "agent_completed", "detail": "hypothesis_agent"},
            {
                "at": "2026-08-20T09:02:00+00:00",
                "event": "covariate_window_reviewed",
                "detail": "SOFA score: window=-24h..0h by Dr. Lee (standard pre-index window)",
            },
            {
                "at": "2026-08-20T09:03:00+00:00",
                "event": "estimation_procedure_proposed",
                "detail": "suggested targeted_minimum_loss_based_estimation (heuristic)",
            },
        ]
        memory = distill_decision_memory(history=history, gates={}, rejected_covariates=[])
        self.assertEqual(len(memory.decisions), 1)
        self.assertEqual(memory.decisions[0].category, "covariate")
        self.assertIn("SOFA score", memory.decisions[0].summary)

    def test_history_event_with_no_detail_is_skipped(self):
        memory = distill_decision_memory(
            history=[{"at": _FIXED, "event": "covariate_reviewed", "detail": "  "}],
            gates={},
            rejected_covariates=[],
        )
        self.assertEqual(memory.decisions, ())

    def test_only_rejected_gates_are_replayed_from_the_gate_objects(self):
        gates = {
            "dag_review": {
                "status": "rejected",
                "reviewer": "Dr. Chen",
                "comment": "adjustment set is missing renal function",
                "decided_at": "2026-08-20T10:00:00+00:00",
            },
            "target_trial": {
                "status": "approved",
                "reviewer": "Dr. Chen",
                "comment": "looks right",
                "decided_at": "2026-08-20T09:00:00+00:00",
            },
            "identification_estimand": {
                "status": "pending", "reviewer": None, "comment": None, "decided_at": None,
            },
        }
        memory = distill_decision_memory(history=[], gates=gates, rejected_covariates=[])
        self.assertEqual(len(memory.decisions), 1)
        decision = memory.decisions[0]
        self.assertEqual(decision.category, "gate")
        self.assertIn("dag_review rejected by Dr. Chen", decision.summary)
        self.assertIn("renal function", decision.summary)

    def test_gate_invalidation_is_kept_as_a_decision(self):
        history = [
            {
                "at": "2026-08-20T11:00:00+00:00",
                "event": "gate_invalidated",
                "detail": "analysis_plan reset from approved to pending: covariate 'SOFA' rejected",
            }
        ]
        memory = distill_decision_memory(history=history, gates={}, rejected_covariates=[])
        self.assertEqual(len(memory.decisions), 1)
        self.assertEqual(memory.decisions[0].category, "gate")
        self.assertIn("reset from approved to pending", memory.decisions[0].summary)

    def test_a_rejected_covariates_history_line_is_not_listed_twice(self):
        history = [
            {
                "at": "2026-08-20T09:00:00+00:00",
                "event": "covariate_reviewed",
                "detail": "baseline_lactate=reject by Dr. Chen",
            },
            {
                "at": "2026-08-20T09:01:00+00:00",
                "event": "covariate_reviewed",
                "detail": "SOFA score=accept by Dr. Lee",
            },
        ]
        memory = distill_decision_memory(
            history=history,
            gates={},
            rejected_covariates=[
                {"label": "baseline_lactate", "reviewer": "Dr. Chen", "comment": "post-treatment",
                 "decided_at": "2026-08-20T09:00:00+00:00"}
            ],
        )
        self.assertEqual([c.label for c in memory.rejected_covariates], ["baseline_lactate"])
        summaries = [d.summary for d in memory.decisions]
        self.assertEqual(summaries, ["SOFA score=accept by Dr. Lee"])

    def test_decisions_are_merged_in_timestamp_order(self):
        history = [
            {
                "at": "2026-08-20T09:05:00+00:00",
                "event": "censoring_definition_approved",
                "detail": "censoring_definition_approved: False -> True by Dr. Lee",
            }
        ]
        gates = {
            "dag_review": {
                "status": "rejected",
                "reviewer": "Dr. Chen",
                "comment": "",
                "decided_at": "2026-08-20T09:02:00+00:00",
            }
        }
        memory = distill_decision_memory(history=history, gates=gates, rejected_covariates=[])
        stamps = [decision.at for decision in memory.decisions]
        self.assertEqual(stamps, sorted(stamps))
        self.assertEqual(memory.decisions[0].category, "gate")
        self.assertEqual(memory.decisions[1].category, "estimand")

    def test_generated_at_is_injectable(self):
        memory = distill_decision_memory(
            history=[], gates={}, rejected_covariates=[], generated_at=_FIXED
        )
        self.assertEqual(memory.generated_at, _FIXED)


class FormatForPromptTests(unittest.TestCase):
    def test_empty_memory_has_a_stable_sentence(self):
        rendered = DecisionMemory().format_for_prompt()
        self.assertEqual(rendered, "No reviewer decisions have been recorded for this run yet.")

    def test_rejected_covariates_render_with_reason_and_attribution(self):
        memory = DecisionMemory(
            rejected_covariates=(
                RejectedCovariate("baseline_lactate", "Dr. Chen", "post-treatment", _FIXED),
            )
        )
        rendered = memory.format_for_prompt()
        self.assertIn("Covariates the reviewer has excluded:", rendered)
        self.assertIn('baseline_lactate: "post-treatment"', rendered)
        self.assertIn("Dr. Chen / 2026-08-20", rendered)

    def test_rejected_covariate_without_a_reason_still_renders(self):
        memory = DecisionMemory(
            rejected_covariates=(RejectedCovariate("x", "", "", ""),)
        )
        self.assertIn("no rationale recorded", memory.format_for_prompt())

    def test_decisions_render_by_category(self):
        memory = DecisionMemory(
            decisions=(
                RecordedDecision(_FIXED, "gate", "dag_review approved by Dr. Chen"),
                RecordedDecision(_FIXED, "estimand", "censoring waiver by Dr. Lee"),
            )
        )
        rendered = memory.format_for_prompt()
        self.assertIn("Decisions already recorded this run:", rendered)
        self.assertIn("[gate] dag_review approved by Dr. Chen (2026-08-20)", rendered)
        self.assertIn("[estimand] censoring waiver by Dr. Lee", rendered)

    def test_deterministic_render(self):
        memory = distill_decision_memory(
            history=[
                {
                    "at": _FIXED,
                    "event": "covariate_reviewed",
                    "detail": "SOFA=reject by Dr. Lee (post-treatment)",
                }
            ],
            gates={},
            rejected_covariates=[],
            generated_at=_FIXED,
        )
        self.assertEqual(memory.format_for_prompt(), memory.format_for_prompt())


class DecisionMemoryRoundTripTests(unittest.TestCase):
    def test_to_dict_from_dict_round_trip(self):
        memory = DecisionMemory(
            rejected_covariates=(
                RejectedCovariate("baseline_lactate", "Dr. Chen", "post-treatment", _FIXED),
            ),
            decisions=(RecordedDecision(_FIXED, "gate", "dag_review approved by Dr. Chen"),),
            generated_at=_FIXED,
        )
        restored = DecisionMemory.from_dict(memory.to_dict())
        self.assertEqual(restored, memory)

    def test_to_dict_is_json_safe(self):
        import json

        payload = distill_decision_memory(
            history=[{"at": _FIXED, "event": "exposure_contrast_set", "detail": "vs_unexposed"}],
            gates={
                "dag_review": {
                    "status": "rejected",
                    "reviewer": "Dr. Chen",
                    "comment": "needs renal function",
                    "decided_at": _FIXED,
                }
            },
            rejected_covariates=[],
            generated_at=_FIXED,
        ).to_dict()
        self.assertEqual(json.loads(json.dumps(payload)), payload)
        self.assertEqual(len(payload["decisions"]), 2)


if __name__ == "__main__":
    unittest.main()
