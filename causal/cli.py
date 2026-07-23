"""Command-line entry point for infrastructure and data readiness checks."""

from __future__ import annotations

import argparse
import json
import sys

from causal.bigquery import BigQueryOMOPGateway, make_google_client
from causal.config import BigQueryDatasets
from causal.agents import HypothesisAgent, UserQuestionHypothesisService
from causal.agents.dag_identification import assess_identification
from causal.agents.hypothesis import DAG_VARIABLE_ROLES
from causal.agents.user_question import LLMRawOutput
from causal.literature import LiteratureOrchestrator, OpenAlexClient, PubMedClient
from causal.llm import build_hypothesis_llm
from causal.mcp_client import OMCPClient
from causal.orchestrator import WorkflowManager
from causal.roadmap import to_dagitty


def _print_raw_llm_output(output: LLMRawOutput) -> None:
    print(f"--- RAW LLM OUTPUT: {output.stage} ---", file=sys.stderr)
    print(output.raw_text, file=sys.stderr)
    print(f"--- END RAW LLM OUTPUT: {output.stage} ---", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(prog="causalomop")
    sub = parser.add_subparsers(dest="command", required=True)
    config_cmd = sub.add_parser("config", help="print resolved dataset roles")
    config_cmd.add_argument("--json", action="store_true")
    preflight = sub.add_parser("preflight", help="validate the converted OMOP dataset")
    preflight.add_argument("--count-persons", action="store_true")
    concepts = sub.add_parser("concepts", help="search candidate OMOP concepts")
    concepts.add_argument("term")
    concepts.add_argument("--limit", type=int, default=20)
    usage = sub.add_parser("concept-usage", help="search concepts actually used in clinical rows")
    usage.add_argument("domain", choices=("condition", "drug"))
    usage.add_argument("term")
    usage.add_argument("--limit", type=int, default=20)
    source_usage = sub.add_parser("source-usage", help="search original source values, including unmapped rows")
    source_usage.add_argument("domain", choices=("condition", "drug"))
    source_usage.add_argument("term")
    source_usage.add_argument("--limit", type=int, default=20)
    visit_audit = sub.add_parser("visit-audit", help="audit visit_detail locations for ICU time zero")
    visit_audit.add_argument("--limit", type=int, default=100)
    workflow = sub.add_parser("workflow", help="manage a persistent causal workflow")
    workflow_sub = workflow.add_subparsers(dest="workflow_command", required=True)
    workflow_start = workflow_sub.add_parser("start")
    workflow_start.add_argument("protocol")
    workflow_start.add_argument("--mode", choices=("exploratory", "final"), default="exploratory")
    workflow_question = workflow_sub.add_parser(
        "start-question", help="create a governed draft from a clinical question using GPT"
    )
    workflow_question.add_argument("question")
    workflow_question.add_argument("--model", default=None)
    workflow_question.add_argument("--mode", choices=("exploratory", "final"), default="exploratory")
    workflow_question.add_argument(
        "--skip-literature", action="store_true",
        help="skip PubMed/OpenAlex retrieval and the literature-grounded DAG draft",
    )
    workflow_status = workflow_sub.add_parser("status")
    workflow_status.add_argument("run_id")
    workflow_gate = workflow_sub.add_parser("gate")
    workflow_gate.add_argument("run_id")
    workflow_gate.add_argument("gate", choices=("protocol", "clinical_mapping", "analysis_plan", "results"))
    workflow_gate.add_argument("status", choices=("approved", "rejected", "provisional"))
    workflow_gate.add_argument("--reviewer", required=True)
    workflow_gate.add_argument("--comment")
    workflow_covariate_review = workflow_sub.add_parser(
        "covariate-review",
        help="accept (with a causal role) or reject a pending first-pass covariate",
    )
    workflow_covariate_review.add_argument("run_id")
    workflow_covariate_review.add_argument("label")
    workflow_covariate_review.add_argument("decision", choices=("accept", "reject"))
    workflow_covariate_review.add_argument("--reviewer", required=True)
    workflow_covariate_review.add_argument(
        "--role", choices=DAG_VARIABLE_ROLES,
        help="required when decision=accept",
    )
    workflow_covariate_review.add_argument(
        "--comment", help="rationale; required when decision=accept"
    )
    workflow_covariate_review.add_argument("--pmid", default="")
    workflow_covariate_review.add_argument(
        "--unobserved", action="store_true", help="mark the accepted variable as latent/unmeasured"
    )
    workflow_covariate_review.add_argument(
        "--certainty", choices=("established", "plausible", "contested", "unknown_direction"),
        default="established",
    )
    workflow_advance = workflow_sub.add_parser("advance")
    workflow_advance.add_argument("run_id")
    workflow_hypothesis = workflow_sub.add_parser("hypothesis")
    workflow_hypothesis.add_argument("run_id")
    workflow_plan = workflow_sub.add_parser("plan")
    workflow_plan.add_argument("run_id")
    workflow_plan.add_argument(
        "--with-mcp", action="store_true",
        help="retrieve review-only OMOP concept candidates through OMCP",
    )
    workflow_report = workflow_sub.add_parser("demo-report")
    workflow_report.add_argument("run_id")
    workflow_feasibility = workflow_sub.add_parser("feasibility")
    workflow_feasibility.add_argument("run_id")
    workflow_feasibility.add_argument(
        "--transport", choices=("mcp", "direct"), default="mcp"
    )
    workflow_readiness = workflow_sub.add_parser(
        "execution-readiness", help="evaluate what the Execution Agent may run"
    )
    workflow_readiness.add_argument("run_id")
    workflow_cohort_plan = workflow_sub.add_parser(
        "cohort-plan", help="generate governed analytic-cohort SQL without executing it"
    )
    workflow_cohort_plan.add_argument("run_id")
    workflow_cohort_profile = workflow_sub.add_parser(
        "cohort-profile", help="run aggregate-only cohort validation through OMCP"
    )
    workflow_cohort_profile.add_argument("run_id")
    workflow_followup_audit = workflow_sub.add_parser(
        "followup-audit", help="audit aggregate 28-day follow-up observability through OMCP"
    )
    workflow_followup_audit.add_argument("run_id")
    workflow_dag_export = workflow_sub.add_parser(
        "dag-export",
        help="export the literature-grounded DAG as dagitty text (paste into dagitty.net)",
    )
    workflow_dag_export.add_argument("run_id")
    workflow_dag_export.add_argument(
        "--highlight-adjustment-set", action="store_true",
        help="mark the computed minimum backdoor adjustment set as [adjusted] in the export",
    )
    args = parser.parse_args()
    datasets = BigQueryDatasets.from_env()

    if args.command == "workflow":
        manager = WorkflowManager()
        if args.workflow_command == "start":
            state = manager.start(args.protocol, args.mode)
        elif args.workflow_command == "start-question":
            resolved_llm = build_hypothesis_llm(model=args.model)
            literature = (
                None if args.skip_literature
                else LiteratureOrchestrator(pubmed=PubMedClient(), openalex=OpenAlexClient())
            )
            generated = UserQuestionHypothesisService().generate(
                args.question,
                resolved_llm.call,
                data_source=datasets.fq_dataset("omop"),
                literature=literature,
                on_llm_output=_print_raw_llm_output,
            )
            state = manager.start_generated(
                generated, args.mode, f"{resolved_llm.provider}:{resolved_llm.model}"
            )
        elif args.workflow_command == "status":
            state = manager.load(args.run_id)
        elif args.workflow_command == "gate":
            state = manager.decide_gate(args.run_id, args.gate, args.status, args.reviewer, args.comment)
        elif args.workflow_command == "hypothesis":
            state = manager.run_hypothesis_agent(args.run_id)
        elif args.workflow_command == "covariate-review":
            state = manager.review_covariate(
                args.run_id,
                args.label,
                args.decision,
                args.reviewer,
                args.comment,
                role=args.role,
                pmid=args.pmid,
                observed=not args.unobserved,
                certainty=args.certainty,
            )
        elif args.workflow_command == "plan":
            lookup = (
                OMCPClient(project=datasets.project, dataset=datasets.omop)
                if args.with_mcp else None
            )
            state = manager.run_data_science_agent(args.run_id, lookup)
        elif args.workflow_command == "demo-report":
            report = manager.build_demo_report(args.run_id)
            print(report)
            return 0
        elif args.workflow_command == "feasibility":
            if args.transport == "mcp":
                state = manager.run_feasibility_mcp(
                    args.run_id,
                    OMCPClient(project=datasets.project, dataset=datasets.omop),
                    datasets,
                )
            else:
                client = make_google_client(datasets.project)
                state = manager.run_feasibility(
                    args.run_id, BigQueryOMOPGateway(datasets, client)
                )
        elif args.workflow_command == "execution-readiness":
            state = manager.run_execution_readiness(args.run_id)
        elif args.workflow_command == "cohort-plan":
            state = manager.build_cohort_plan(args.run_id, datasets)
        elif args.workflow_command == "cohort-profile":
            state = manager.run_cohort_profile_mcp(
                args.run_id,
                OMCPClient(project=datasets.project, dataset=datasets.omop),
                datasets,
            )
        elif args.workflow_command == "followup-audit":
            state = manager.run_followup_audit_mcp(
                args.run_id,
                OMCPClient(project=datasets.project, dataset=datasets.omop),
                datasets,
            )
        elif args.workflow_command == "dag-export":
            protocol_path = manager.runs_dir / args.run_id / "protocol.json"
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            hypothesis = HypothesisAgent().run(protocol)
            result = assess_identification(hypothesis)
            adjustment_set = (
                result.minimum_adjustment_sets[0]
                if args.highlight_adjustment_set and result.minimum_adjustment_sets
                else ()
            )
            print(to_dagitty(result.dag, adjustment_set))
            print(
                f"// identified_by_backdoor_adjustment={result.identified_by_backdoor_adjustment} "
                f"confounder_count={result.confounder_count}",
                file=sys.stderr,
            )
            for warning in result.warnings:
                print(f"// warning: {warning}", file=sys.stderr)
            return 0
        else:
            state = manager.advance(args.run_id)
        print(json.dumps(state.to_dict(), indent=2))
        return 0

    if args.command == "config":
        values = {role: datasets.fq_dataset(role) for role in ("omop", "vocabulary", "etl", "metrics")}
        print(json.dumps(values, indent=2) if args.json else "\n".join(f"{k}: {v}" for k, v in values.items()))
        return 0

    gateway = BigQueryOMOPGateway(datasets, make_google_client(datasets.project))
    if args.command == "concepts":
        print(json.dumps(gateway.search_concepts(args.term, args.limit), indent=2, default=str))
        return 0
    if args.command == "concept-usage":
        print(json.dumps(gateway.search_used_concepts(args.domain, args.term, args.limit), indent=2, default=str))
        return 0
    if args.command == "source-usage":
        print(json.dumps(gateway.search_source_values(args.domain, args.term, args.limit), indent=2, default=str))
        return 0
    if args.command == "visit-audit":
        print(json.dumps(gateway.audit_visit_details(args.limit), indent=2, default=str))
        return 0
    result = gateway.preflight(count_persons=args.count_persons)
    print(json.dumps({
        "ready": result.ready,
        "accessible_tables": result.accessible_tables,
        "missing_tables": result.missing_tables,
        "person_count": result.person_count,
    }, indent=2))
    return 0 if result.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
