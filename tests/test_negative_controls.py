import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from causal.agents.estimation import EstimationAgent
from causal.config import BigQueryDatasets
from causal.mapping.registry import MAPPING_STATUS_APPROVED, OMOPVariableDefinition
from causal.queries import ConceptSetSpec, analytic_cohort_sql
from causal.queries.negative_controls import (
    NEGATIVE_CONTROL_ROLES,
    DEFAULT_NEGATIVE_CONTROL_PROMPT_PATH,
    build_negative_control_prompt,
    parse_negative_control_response,
    substitute_spec,
    variable_from_negative_control_proposal,
)

SEPSIS = ConceptSetSpec(domain="condition", concept_ids=(132797,))
ALBUMIN = ConceptSetSpec(domain="drug", concept_ids=(1,))
CRYSTALLOID = ConceptSetSpec(domain="drug", concept_ids=(2,))
DEATH = ConceptSetSpec(domain="death")
VITAMIN_D = ConceptSetSpec(domain="drug", concept_ids=(999,))


def _real_specs() -> dict[str, ConceptSetSpec]:
    return {"population": SEPSIS, "treatment": ALBUMIN, "comparator": CRYSTALLOID, "outcome": DEATH}


class SubstituteSpecTests(unittest.TestCase):
    def test_negative_control_exposure_substitutes_treatment_only(self):
        substituted = substitute_spec(_real_specs(), "negative_control_exposure", VITAMIN_D)
        self.assertEqual(substituted["treatment"], VITAMIN_D)
        self.assertEqual(substituted["population"], SEPSIS)
        self.assertEqual(substituted["comparator"], CRYSTALLOID)
        self.assertEqual(substituted["outcome"], DEATH)

    def test_negative_control_outcome_substitutes_outcome_only(self):
        fracture = ConceptSetSpec(domain="condition", concept_ids=(444,))
        substituted = substitute_spec(_real_specs(), "negative_control_outcome", fracture)
        self.assertEqual(substituted["outcome"], fracture)
        self.assertEqual(substituted["treatment"], ALBUMIN)

    def test_rejects_an_unknown_control_role(self):
        with self.assertRaises(ValueError):
            substitute_spec(_real_specs(), "not_a_real_role", VITAMIN_D)

    def test_rejects_incomplete_specs(self):
        with self.assertRaises(ValueError):
            substitute_spec({"population": SEPSIS}, "negative_control_exposure", VITAMIN_D)


class SubstitutedCohortSqlTests(unittest.TestCase):
    """The "SQL reuse" this module provides: `substitute_spec` plus the
    existing `analytic_cohort_sql` -- no new SQL-generation function --
    must produce cohort SQL that actually reflects the substituted
    exposure/outcome, not the real study's."""

    def test_negative_control_exposure_sql_references_the_control_concept_not_the_real_treatment(self):
        substituted = substitute_spec(_real_specs(), "negative_control_exposure", VITAMIN_D)
        sql = analytic_cohort_sql(
            BigQueryDatasets(),
            population=substituted["population"], treatment=substituted["treatment"],
            comparator=substituted["comparator"], outcome=substituted["outcome"],
        )
        self.assertIn("999", sql)
        real_treatment_sql = analytic_cohort_sql(
            BigQueryDatasets(), population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
        )
        self.assertNotIn("999", real_treatment_sql)

    def test_negative_control_outcome_sql_uses_the_controls_domain(self):
        fracture = ConceptSetSpec(domain="condition", concept_ids=(444,))
        substituted = substitute_spec(_real_specs(), "negative_control_outcome", fracture)
        sql = analytic_cohort_sql(
            BigQueryDatasets(),
            population=substituted["population"], treatment=substituted["treatment"],
            comparator=substituted["comparator"], outcome=substituted["outcome"],
        )
        # A condition-domain outcome resolves through condition_occurrence,
        # not the death table -- unlike the real study's DEATH outcome.
        self.assertIn("condition_concept_id", sql)
        self.assertIn("444", sql)
        real_outcome_sql = analytic_cohort_sql(
            BigQueryDatasets(), population=SEPSIS, treatment=ALBUMIN, comparator=CRYSTALLOID, outcome=DEATH,
        )
        self.assertNotIn("444", real_outcome_sql)


class VariableFromNegativeControlProposalTests(unittest.TestCase):
    def test_converts_a_pending_proposal(self):
        proposal = {
            "control_id": "vitamin_d", "control_type": "negative_control_exposure",
            "label": "Vitamin D supplementation", "rationale": "no known mechanism",
            "domain": "", "concept_ids": [], "mapping_status": "candidate_pending_human_approval",
            "provenance": "llm_proposed",
        }
        variable = variable_from_negative_control_proposal(proposal)
        self.assertEqual(variable.variable_kind, "phenotype")
        self.assertEqual(variable.causal_role, "negative_control_exposure")
        self.assertNotEqual(variable.mapping_status, MAPPING_STATUS_APPROVED)
        self.assertEqual(variable.root_concept_ids, ())

    def test_converts_an_approved_mapping(self):
        proposal = {
            "control_id": "vitamin_d", "control_type": "negative_control_exposure",
            "label": "Vitamin D supplementation", "domain": "drug", "concept_ids": [999],
            "mapping_status": "approved", "reviewer": "Clinician", "provenance": "omcp_candidate",
        }
        variable = variable_from_negative_control_proposal(proposal)
        self.assertEqual(variable.mapping_status, MAPPING_STATUS_APPROVED)
        self.assertEqual(variable.root_concept_ids, (999,))

    def test_rejects_an_unknown_control_type(self):
        with self.assertRaises(ValueError):
            variable_from_negative_control_proposal({"control_type": "treatment", "label": "x"})

    def test_rejects_a_blank_label(self):
        with self.assertRaises(ValueError):
            variable_from_negative_control_proposal({"control_type": "negative_control_exposure", "label": "  "})


class BuildNegativeControlPromptTests(unittest.TestCase):
    def test_substitutes_all_required_placeholders(self):
        prompt = build_negative_control_prompt(DEFAULT_NEGATIVE_CONTROL_PROMPT_PATH, {
            "population": "sepsis", "treatment": "albumin",
            "comparator": "crystalloids", "outcome": "28-day mortality",
        })
        self.assertNotIn("{{POPULATION}}", prompt)
        self.assertNotIn("{{TREATMENT}}", prompt)
        self.assertIn("albumin", prompt)
        self.assertIn("28-day mortality", prompt)

    def test_missing_placeholder_in_template_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_template = Path(tmp) / "bad.md"
            bad_template.write_text("no placeholders here", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_negative_control_prompt(bad_template, {"population": "x"})


class ParseNegativeControlResponseTests(unittest.TestCase):
    """The concrete, testable form of "no LLM auto-approves a clinical
    mapping": every code path through this parser must be checked to
    confirm it cannot return an approved candidate, no matter what the raw
    response claims."""

    def test_parses_valid_candidates(self):
        raw = (
            '{"candidates": [{"control_type": "negative_control_exposure", '
            '"label": "Vitamin D", "rationale": "no known mechanism"}]}'
        )
        candidates = parse_negative_control_response(raw)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["control_type"], "negative_control_exposure")
        self.assertEqual(candidates[0]["label"], "Vitamin D")

    def test_no_candidate_is_ever_approved_even_if_the_raw_response_claims_it(self):
        raw = (
            '{"candidates": [{"control_type": "negative_control_exposure", "label": "Vitamin D", '
            '"rationale": "no known mechanism", "mapping_status": "approved", '
            '"domain": "drug", "concept_ids": [999], "provenance": "protocol_declared"}]}'
        )
        candidates = parse_negative_control_response(raw)
        self.assertEqual(candidates[0]["mapping_status"], "candidate_pending_human_approval")
        self.assertEqual(candidates[0]["provenance"], "llm_proposed")
        # concept_ids/domain are structurally never read from the LLM at all.
        self.assertEqual(candidates[0]["concept_ids"], [])
        self.assertEqual(candidates[0]["domain"], "")

    def test_caps_the_number_of_candidates(self):
        items = [
            {"control_type": "negative_control_exposure", "label": f"drug {i}", "rationale": "x"}
            for i in range(20)
        ]
        raw = json.dumps({"candidates": items})
        candidates = parse_negative_control_response(raw)
        self.assertLessEqual(len(candidates), 6)

    def test_skips_malformed_candidates_but_keeps_valid_ones(self):
        raw = (
            '{"candidates": ['
            '{"control_type": "not_a_role", "label": "x", "rationale": "y"},'
            '{"control_type": "negative_control_outcome", "label": "", "rationale": "y"},'
            '{"control_type": "negative_control_outcome", "label": "Ankle fracture", "rationale": "y"}'
            ']}'
        )
        candidates = parse_negative_control_response(raw)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["label"], "Ankle fracture")

    def test_rejects_empty_response(self):
        with self.assertRaises(ValueError):
            parse_negative_control_response("")

    def test_rejects_non_json_response(self):
        with self.assertRaises(ValueError):
            parse_negative_control_response("not json")

    def test_rejects_response_with_no_candidates_key(self):
        with self.assertRaises(ValueError):
            parse_negative_control_response('{"foo": "bar"}')

    def test_rejects_when_every_candidate_is_malformed(self):
        with self.assertRaises(ValueError):
            parse_negative_control_response('{"candidates": [{"control_type": "not_a_role"}]}')

    def test_negative_control_roles_are_exactly_two(self):
        self.assertEqual(set(NEGATIVE_CONTROL_ROLES), {"negative_control_exposure", "negative_control_outcome"})


def _synthetic_negative_control_cohort(n: int, true_effect: float, seed: int) -> pd.DataFrame:
    """Shapes a DataFrame the way a negative-control-substituted
    materialization would (same A/Y/age columns `EstimationAgent` reads),
    with `A` standing in for the negative control's exposure/outcome
    indicator. `true_effect=0.0` is the "known-null" case a negative
    control is supposed to look like; a nonzero value is the "known-
    confounded" case a real residual-confounding warning should catch."""
    rng = np.random.default_rng(seed)
    age = rng.normal(0, 1, size=n)
    a = rng.integers(0, 2, size=n)
    baseline_risk = 0.3 + 0.05 * age
    risk = np.clip(baseline_risk + true_effect * a, 0.02, 0.98)
    y = rng.binomial(1, risk)
    return pd.DataFrame({"A": a, "Y": y, "age": age})


class NegativeControlEstimateInterpretationTests(unittest.TestCase):
    """The statistical property `build_causal_report`'s "Crosses null"
    column relies on: a known-null negative control's CI should cross
    zero, a known-confounded one should not. Exercises `EstimationAgent`
    directly on cohorts shaped like a negative-control-substituted
    materialization, rather than the full governed OMCP pipeline."""

    def test_known_null_negative_control_ci_crosses_zero(self):
        dataset = _synthetic_negative_control_cohort(3000, true_effect=0.0, seed=1)
        artifact = EstimationAgent().run(
            dataset, study_id="negative_control_test", covariate_columns=("age",),
            n_bootstrap=100, random_seed=1,
        )
        estimand = artifact.estimand
        self.assertLessEqual(estimand.ci_lower, 0.0)
        self.assertGreaterEqual(estimand.ci_upper, 0.0)

    def test_known_confounded_negative_control_ci_does_not_cross_zero(self):
        dataset = _synthetic_negative_control_cohort(3000, true_effect=0.2, seed=2)
        artifact = EstimationAgent().run(
            dataset, study_id="negative_control_test", covariate_columns=("age",),
            n_bootstrap=100, random_seed=2,
        )
        estimand = artifact.estimand
        self.assertGreater(estimand.ci_lower, 0.0)


if __name__ == "__main__":
    unittest.main()
