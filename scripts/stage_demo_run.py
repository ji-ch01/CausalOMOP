"""Stage a fully synthetic causal-workflow run straight to a late phase, for
offline UI/dialogue testing.

WHY THIS EXISTS
---------------
The estimand dialogue (phase ``estimand_review``) and the
results-interpretation dialogue (phase ``results_review`` / ``report``) are
the two conversation surfaces in ``causal/ui/streamlit_app.py`` that only
appear late in a run. Reaching them through the real pipeline needs a working
OMCP subprocess and BigQuery access. This script fakes every one of those
steps -- synthetic protocol, auto-approved gates, a fabricated in-memory
cohort -- so a developer can open the Streamlit app and click through the
dialogues without any external service.

WHAT IT IS NOT
--------------
Not part of the governed workflow. Every gate it "approves" is approved by
the fake reviewer ``demo-staging``; the cohort is random numbers, not
patients. Never point ``--runs-dir`` at a directory that holds real studies
you care about (it only ever *adds* a new run, but keep demo and real runs
apart anyway). Nothing here is importable by ``causal/`` and it is excluded
from the test suite.

USAGE
-----
    python scripts/stage_demo_run.py --to estimand_review
    python scripts/stage_demo_run.py --to results_review
    streamlit run causal/ui/streamlit_app.py     # then open the printed run id
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from causal.config import BigQueryDatasets  # noqa: E402
from causal.mcp_client import MCPToolResponse  # noqa: E402
from causal.orchestrator.workflow import WorkflowManager  # noqa: E402

_REVIEWER = "demo-staging"

_TARGET_TRIAL = {
    "population": "adults admitted to the ICU with sepsis",
    "treatment": "intravenous albumin",
    "comparator": "crystalloids alone",
    "time_zero": "ICU admission",
    "grace_period_hours": 24,
    "outcome": "28-day all-cause mortality",
    "follow_up_days": 28,
    "primary_estimand": "ATE",
    "eligibility_criteria": ["adult", "sepsis diagnosis at admission"],
    "assignment_procedure": "Emulated baseline assignment at ICU admission",
    "censoring_strategy": "Administrative censoring at end of 28-day follow-up",
    "analysis_strategy": "Intention-to-treat analogue",
    "effect_measure": "risk_difference",
    "treatment_value": "albumin",
    "comparator_value": "crystalloids",
}

_PROTOCOL = {
    "study_id": "demo_albumin_sepsis",
    "source_question": (
        "Among ICU adults with sepsis, does intravenous albumin versus crystalloids "
        "alone change 28-day mortality?"
    ),
    "target_trial": _TARGET_TRIAL,
    "data_source": "demo.synthetic",
    "first_pass_covariates": ["SOFA score"],
    "concept_sets": {
        "population": {"role": "population", "status": "approved", "domain": "condition", "concept_ids": [132797]},
        "treatment": {"role": "treatment", "status": "approved", "domain": "drug", "concept_ids": [1]},
        "comparator": {"role": "comparator", "status": "approved", "domain": "drug", "concept_ids": [2]},
        "outcome": {"role": "outcome", "status": "approved", "domain": "condition", "concept_ids": [316866]},
    },
}


class _FakeCohortRowsClient:
    """Duck-typed OMCP stand-in: returns the same fabricated cohort CSV for
    every ``causal_select`` call, so ``materialize_analytic_dataset`` has
    something to parse without a real OMCP checkout."""

    def __init__(self, csv_text: str) -> None:
        self._csv_text = csv_text

    def causal_select(self, query: str, operation: str, causal_context: dict) -> MCPToolResponse:
        return MCPToolResponse(text=self._csv_text, metadata={"rows": self._csv_text.count("\n") - 1})


def _synthetic_cohort_csv(n: int = 300, seed: int = 11) -> str:
    rng = random.Random(seed)
    lines = [
        "person_id,A,Y,index_datetime,follow_up_days,censored,age,"
        "gender_concept_id,race_concept_id,cov_sofa_score"
    ]
    for i in range(n):
        treatment = rng.randint(0, 1)
        age = rng.randint(18, 90)
        risk = min(max(0.2 + 0.15 * treatment + 0.002 * age, 0.01), 0.99)
        outcome = 1 if rng.random() < risk else 0
        gender = rng.choice((8507, 8532))
        race = rng.choice((8527, 8516))
        sofa = round(rng.uniform(0, 20), 1)
        lines.append(
            f"{100 + i},{treatment},{outcome},2024-01-01T00:00:00,28,,{age},{gender},{race},{sofa}"
        )
    return "\n".join(lines) + "\n"


def _stage_to_estimand_review(manager: WorkflowManager, datasets: BigQueryDatasets) -> str:
    """Mirror ``tests.test_workflow._bring_run_to_estimand_review``: a
    single-covariate run advanced through every gate to ``estimand_review``
    with ``study_specification.json`` and ``cohort_manifest.json`` written,
    stopping before any statistical-estimand decision (the dialogue's job)."""
    with tempfile.NamedTemporaryFile(
        "w", suffix="_demo_protocol.json", delete=False, encoding="utf-8"
    ) as handle:
        json.dump(_PROTOCOL, handle)
        protocol_path = Path(handle.name)

    state = manager.start(protocol_path)
    run_id = state.run_id
    run_dir = manager.runs_dir / run_id
    manager.run_hypothesis_agent(run_id)

    manager.review_covariate(
        run_id, "SOFA score", "accept", _REVIEWER,
        "Higher SOFA is associated with both treatment choice and mortality.",
        role="confounder", temporal_classification="pre_treatment", domain="measurement",
    )
    manager.map_covariate_concepts(run_id, "SOFA score", [4301868], _REVIEWER)
    manager.review_covariate_aggregation(run_id, "SOFA score", "last_value_before_index", _REVIEWER)

    manager.decide_gate(run_id, "target_trial", "approved", _REVIEWER)
    manager.decide_gate(run_id, "dag_review", "approved", _REVIEWER)
    manager.decide_gate(run_id, "identification_estimand", "approved", _REVIEWER)
    for _ in range(3):
        manager.advance(run_id)
    manager.decide_gate(run_id, "clinical_mapping", "approved", _REVIEWER)
    manager.advance(run_id)
    manager.run_data_science_agent(run_id)
    manager.advance(run_id)

    plan_path = run_dir / "analysis_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["exploratory_execution_allowed"] = True
    plan["unresolved_issues"] = []
    for phenotype in plan["phenotypes"]:
        phenotype["status"] = "approved"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    manager.decide_gate(run_id, "analysis_plan", "approved", _REVIEWER)

    state = manager.advance(run_id)
    assert state.phase == "estimand_review", state.phase
    manager.build_cohort_plan(run_id, datasets)
    return run_id


def _stage_to_results_review(manager: WorkflowManager, datasets: BigQueryDatasets) -> str:
    """Continue past ``estimand_review``: record a plug-in strategy, approve
    the ``statistical_estimand`` gate, materialize the fabricated cohort, run
    the estimator and balance diagnostics, and advance to ``results_review``
    (mirrors ``tests.test_workflow._ResultsNarrativeFixture``)."""
    run_id = _stage_to_estimand_review(manager, datasets)
    manager.set_statistical_estimand_strategy(
        run_id,
        missing_data_strategy="complete_case_analysis",
        censoring_strategy="ignore_censoring_administrative_waiver",
        censoring_rationale="Follow-up is a fixed 28-day administrative window in this demo cohort.",
        reviewer=_REVIEWER,
    )
    manager.decide_gate(run_id, "statistical_estimand", "approved", _REVIEWER)
    manager.advance(run_id)
    manager.materialize_analytic_dataset(run_id, _FakeCohortRowsClient(_synthetic_cohort_csv()))
    manager.run_estimation(run_id, n_bootstrap=200, random_seed=1)
    manager.run_diagnostics(run_id)
    manager.advance(run_id)
    state = manager.advance(run_id)
    assert state.phase == "results_review", state.phase
    return run_id


_STAGERS = {
    "estimand_review": _stage_to_estimand_review,
    "results_review": _stage_to_results_review,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--to", choices=tuple(_STAGERS), default="estimand_review",
        help="phase to stage the run up to (default: estimand_review)",
    )
    parser.add_argument(
        "--runs-dir", default=str(_REPO_ROOT / "artifacts" / "runs"),
        help="run directory the Streamlit app reads (default: artifacts/runs)",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    manager = WorkflowManager(runs_dir)
    datasets = BigQueryDatasets()

    print(f"Staging a synthetic run to '{args.to}' under {runs_dir} ...", file=sys.stderr)
    run_id = _STAGERS[args.to](manager, datasets)

    print(run_id)
    print(
        f"\nOpen it:  streamlit run causal/ui/streamlit_app.py\n"
        f"          then pick run '{run_id}' from the sidebar.\n"
        f"The {'estimand' if args.to == 'estimand_review' else 'results-interpretation'} "
        f"dialogue is in the left conversation panel.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
