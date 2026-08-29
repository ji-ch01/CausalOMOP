"""Coverage for `causal.companion.actions` -- the allow-list between an LLM
turn's free-text `proposed_action` and any real `WorkflowManager` call.

The security-relevant cases: an unknown action name, an unexpected argument
(`reviewer` / `run_id`), and an out-of-range enum value must all raise rather
than pass through."""

import unittest

from causal.companion.actions import (
    ALLOWED_ACTION_NAMES,
    ActionValidationError,
    ProposedAction,
    validate_proposed_action,
)

_VALID = {
    "action": "set_statistical_estimand_strategy",
    "args": {
        "missing_data_strategy": "complete_case_analysis",
        "censoring_strategy": "ignore_censoring_administrative_waiver",
        "estimation_procedure": "plug_in_standardization",
        "censoring_rationale": "  Administrative censoring judged non-informative for this cohort.  ",
    },
}


def _valid(**arg_overrides):
    payload = {"action": _VALID["action"], "args": dict(_VALID["args"])}
    payload["args"].update(arg_overrides)
    return payload


class ValidateProposedActionTests(unittest.TestCase):
    def test_valid_action_is_accepted_and_rationale_stripped(self):
        result = validate_proposed_action(_VALID)
        self.assertIsInstance(result, ProposedAction)
        self.assertEqual(result.action, "set_statistical_estimand_strategy")
        self.assertEqual(
            result.args["censoring_rationale"],
            "Administrative censoring judged non-informative for this cohort.",
        )
        self.assertIn("complete_case_analysis", result.summary)

    def test_none_and_empty_dict_mean_no_action(self):
        self.assertIsNone(validate_proposed_action(None))
        self.assertIsNone(validate_proposed_action({}))

    def test_non_dict_raises(self):
        with self.assertRaises(ActionValidationError):
            validate_proposed_action("set_statistical_estimand_strategy")

    def test_unknown_action_name_raises(self):
        with self.assertRaises(ActionValidationError):
            validate_proposed_action({"action": "decide_gate", "args": {}})

    def test_unexpected_arg_is_rejected(self):
        # The companion must not be able to inject the reviewer identity or run id.
        for injected in ("reviewer", "run_id", "comment"):
            with self.assertRaises(ActionValidationError):
                validate_proposed_action(_valid(**{injected: "attacker"}))

    def test_out_of_range_enum_value_raises(self):
        with self.assertRaises(ActionValidationError):
            validate_proposed_action(_valid(censoring_strategy="delete_all_rows"))

    def test_missing_required_enum_field_raises(self):
        payload = _valid()
        del payload["args"]["estimation_procedure"]
        with self.assertRaises(ActionValidationError):
            validate_proposed_action(payload)

    def test_blank_rationale_raises(self):
        with self.assertRaises(ActionValidationError):
            validate_proposed_action(_valid(censoring_rationale="   "))

    def test_over_long_rationale_raises(self):
        with self.assertRaises(ActionValidationError):
            validate_proposed_action(_valid(censoring_rationale="x" * 2001))

    def test_args_not_a_dict_raises(self):
        with self.assertRaises(ActionValidationError):
            validate_proposed_action({"action": "set_statistical_estimand_strategy", "args": []})

    def test_every_allowed_action_name_is_a_workflowmanager_method(self):
        from causal.orchestrator.workflow import WorkflowManager

        for name in ALLOWED_ACTION_NAMES:
            self.assertTrue(callable(getattr(WorkflowManager, name, None)), name)


if __name__ == "__main__":
    unittest.main()
