"""Command-line entry point for infrastructure and data readiness checks."""

from __future__ import annotations

import argparse
import json
import sys

from causal.bigquery import BigQueryOMOPGateway, make_google_client
from causal.config import BigQueryDatasets
from causal.agents import HypothesisAgent, UserQuestionHypothesisService
from causal.agents.dag_identification import review_dag
from causal.agents.hypothesis import (
    DAG_VARIABLE_AGGREGATIONS,
    DAG_VARIABLE_DOMAINS,
    DAG_VARIABLE_ROLES,
    DAG_VARIABLE_TEMPORAL_CLASSES,
)
from causal.agents.user_question import LLMRawOutput
from causal.knowledge_graph import OptimusKGClient
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
    workflow_question.add_argument(
        "--skip-knowledge-graph", action="store_true",
        help="skip OptimusKG retrieval when proposing the candidate DAG",
    )
    workflow_question.add_argument(
        "--skip-critique", action="store_true",
        help="skip the independent critique/revision round after the DAG draft",
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
        "--temporal-classification", choices=DAG_VARIABLE_TEMPORAL_CLASSES,
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
    workflow_covariate_review.add_argument(
        "--domain", choices=DAG_VARIABLE_DOMAINS,
        help="optional OMOP domain; enables an OMCP concept lookup for this covariate later",
    )
    workflow_concept_set_map = workflow_sub.add_parser(
        "concept-set-map",
        help="record the OMOP domain + concept(s) for a protocol concept_sets entry",
    )
    workflow_concept_set_map.add_argument("run_id")
    workflow_concept_set_map.add_argument("name", help="concept_sets key, e.g. 'population'")
    workflow_concept_set_map.add_argument(
        "--domain", required=True,
        help=(
            "one of condition/drug/procedure/measurement/observation, or 'death' "
            "(only valid for an entry whose role is 'outcome') -- this is also what "
            "feasibility/cohort SQL generation reads for this role, there is no separate mapping step"
        ),
    )
    workflow_concept_set_map.add_argument(
        "--concept-ids", type=int, nargs="+", default=[],
        help="one or more OMOP concept_id values; required unless --status is non-approved or --domain death",
    )
    workflow_concept_set_map.add_argument("--reviewer", required=True)
    workflow_concept_set_map.add_argument(
        "--status", default="approved",
        help="e.g. approved, no_standard_concept (free text; 'approved' requires --concept-ids unless --domain death)",
    )
    workflow_concept_set_map.add_argument("--comment")
    workflow_concept_set_map.add_argument(
        "--with-mcp", action="store_true",
        help="verify each concept_id's real OMOP domain matches --domain through OMCP",
    )
    workflow_covariate_map_concepts = workflow_sub.add_parser(
        "covariate-map-concepts",
        help="record the OMOP concept(s) chosen for an already-accepted covariate",
    )
    workflow_covariate_map_concepts.add_argument("run_id")
    workflow_covariate_map_concepts.add_argument("label")
    workflow_covariate_map_concepts.add_argument(
        "--concept-ids", type=int, nargs="+", default=[],
        help="one or more OMOP concept_id values; required unless --mapping-status is non-approved",
    )
    workflow_covariate_map_concepts.add_argument("--reviewer", required=True)
    workflow_covariate_map_concepts.add_argument(
        "--mapping-status", default="approved",
        help="e.g. approved, no_standard_concept (free text; 'approved' requires --concept-ids)",
    )
    workflow_covariate_map_concepts.add_argument("--comment")
    workflow_covariate_map_concepts.add_argument(
        "--with-mcp", action="store_true",
        help="verify each concept_id's real OMOP domain matches the covariate's declared domain through OMCP",
    )
    workflow_covariate_review_domain = workflow_sub.add_parser(
        "covariate-review-domain",
        help="confirm or correct the OMOP domain for an already-accepted covariate",
    )
    workflow_covariate_review_domain.add_argument("run_id")
    workflow_covariate_review_domain.add_argument("label")
    workflow_covariate_review_domain.add_argument("domain", choices=DAG_VARIABLE_DOMAINS)
    workflow_covariate_review_domain.add_argument("--reviewer", required=True)
    workflow_covariate_review_domain.add_argument("--comment")
    workflow_covariate_review_aggregation = workflow_sub.add_parser(
        "covariate-review-aggregation",
        help="record how to combine repeated occurrences of an already-mapped covariate",
    )
    workflow_covariate_review_aggregation.add_argument("run_id")
    workflow_covariate_review_aggregation.add_argument("label")
    workflow_covariate_review_aggregation.add_argument("aggregation", choices=DAG_VARIABLE_AGGREGATIONS)
    workflow_covariate_review_aggregation.add_argument("--reviewer", required=True)
    workflow_covariate_review_aggregation.add_argument("--comment")
    workflow_covariate_review_window = workflow_sub.add_parser(
        "covariate-review-window",
        help="override the auto-derived ascertainment window for an already-accepted covariate",
    )
    workflow_covariate_review_window.add_argument("run_id")
    workflow_covariate_review_window.add_argument("label")
    workflow_covariate_review_window.add_argument("window_start_hours", type=int)
    workflow_covariate_review_window.add_argument("window_end_hours", type=int)
    workflow_covariate_review_window.add_argument("--reviewer", required=True)
    workflow_covariate_review_window.add_argument("--comment")
    workflow_advance = workflow_sub.add_parser("advance")
    workflow_advance.add_argument("run_id")
    workflow_hypothesis = workflow_sub.add_parser("hypothesis")
    workflow_hypothesis.add_argument("run_id")
    workflow_registry_build = workflow_sub.add_parser(
        "registry-build", help="write registry.json from protocol.json's concept_sets/candidate_dag"
    )
    workflow_registry_build.add_argument("run_id")
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
            knowledge_graph = (
                None if args.skip_knowledge_graph or not OptimusKGClient.is_available()
                else OptimusKGClient()
            )
            generated = UserQuestionHypothesisService().generate(
                args.question,
                resolved_llm.call,
                data_source=datasets.fq_dataset("omop"),
                literature=literature,
                knowledge_graph=knowledge_graph,
                on_llm_output=_print_raw_llm_output,
                enable_critique=not args.skip_critique,
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
        elif args.workflow_command == "registry-build":
            state = manager.build_variable_registry(args.run_id)
        elif args.workflow_command == "concept-set-map":
            concept_lookup = (
                OMCPClient(project=datasets.project, dataset=datasets.omop)
                if args.with_mcp else None
            )
            state = manager.map_concept_set(
                args.run_id,
                args.name,
                args.domain,
                args.concept_ids,
                args.reviewer,
                status=args.status,
                comment=args.comment,
                concept_lookup=concept_lookup,
            )
        elif args.workflow_command == "covariate-review":
            state = manager.review_covariate(
                args.run_id,
                args.label,
                args.decision,
                args.reviewer,
                args.comment,
                role=args.role,
                temporal_classification=args.temporal_classification,
                pmid=args.pmid,
                observed=not args.unobserved,
                certainty=args.certainty,
                domain=args.domain,
            )
        elif args.workflow_command == "covariate-map-concepts":
            concept_lookup = (
                OMCPClient(project=datasets.project, dataset=datasets.omop)
                if args.with_mcp else None
            )
            state = manager.map_covariate_concepts(
                args.run_id,
                args.label,
                args.concept_ids,
                args.reviewer,
                mapping_status=args.mapping_status,
                comment=args.comment,
                concept_lookup=concept_lookup,
            )
        elif args.workflow_command == "covariate-review-domain":
            state = manager.review_covariate_domain(
                args.run_id,
                args.label,
                args.domain,
                args.reviewer,
                comment=args.comment,
            )
        elif args.workflow_command == "covariate-review-aggregation":
            state = manager.review_covariate_aggregation(
                args.run_id,
                args.label,
                args.aggregation,
                args.reviewer,
                comment=args.comment,
            )
        elif args.workflow_command == "covariate-review-window":
            state = manager.review_covariate_window(
                args.run_id,
                args.label,
                args.window_start_hours,
                args.window_end_hours,
                args.reviewer,
                comment=args.comment,
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
            review = review_dag(hypothesis)
            result = review.identification
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
            print(
                f"// adjustment_sets={review.adjustment.set_count} "
                f"set_size={review.adjustment.set_size} "
                f"core={list(review.adjustment.core_variable_labels)} "
                f"alternatives={list(review.adjustment.alternative_variable_labels)} "
                f"unused_confounders={list(review.adjustment.unused_confounder_labels)}",
                file=sys.stderr,
            )
            print(
                f"// evidence: literature_only={review.evidence.literature_only} "
                f"kg_only={review.evidence.kg_only} corroborated={review.evidence.corroborated} "
                f"uncited={review.evidence.uncited} "
                f"cited_fraction={review.evidence.cited_fraction:.2f}",
                file=sys.stderr,
            )
            print(
                f"// confidence (GRADE-inspired): high={review.confidence.high} "
                f"moderate={review.confidence.moderate} low={review.confidence.low} "
                f"very_low={review.confidence.very_low} "
                f"high_or_moderate_fraction={review.confidence.high_or_moderate_fraction:.2f}",
                file=sys.stderr,
            )
            for warning in result.technical_notes:
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
