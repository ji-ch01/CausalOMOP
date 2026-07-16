"""Provider-neutral bridge from a user question to the Hypothesis Agent."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from causal.agents.hypothesis import DAG_RELATIONS, HypothesisAgent, HypothesisArtifact
from causal.literature.pubmed import PubMedArticle, PubMedClient
from causal.llm.base import LLMCallable


_STUDY_ID = re.compile(r"^[a-z][a-z0-9_]{2,79}$")
_LITERATURE_TERM_FIELDS = (
    "population_terms", "treatment_terms", "comparator_terms", "outcome_terms"
)
_ABSTRACT_PREVIEW_CHARS = 600
_MAX_TERMS_PER_ROLE = 3
_MAX_TERM_CHARS = 80


@dataclass(frozen=True)
class LLMRawOutput:
    stage: str
    raw_text: str

    def to_dict(self) -> dict[str, str]:
        return {"stage": self.stage, "raw_text": self.raw_text}


LLMOutputCallback = Callable[[LLMRawOutput], None]


@dataclass(frozen=True)
class GeneratedHypothesis:
    protocol: dict
    hypothesis: HypothesisArtifact
    llm_outputs: tuple[LLMRawOutput, ...]
    prompt_version: str = "1.0"


class UserQuestionHypothesisService:
    """Call an injected LLM, validate its JSON, then use the deterministic agent."""

    def __init__(
        self,
        prompt_path: str | Path = "causal/prompts/hypothesis.md",
        dag_prompt_path: str | Path = "causal/prompts/dag.md",
    ) -> None:
        self.prompt_path = Path(prompt_path)
        self.dag_prompt_path = Path(dag_prompt_path)

    def build_prompt(self, question: str) -> str:
        question = question.strip()
        if not question:
            raise ValueError("User question is required")
        template = self.prompt_path.read_text(encoding="utf-8")
        if "{{USER_QUESTION}}" not in template:
            raise ValueError("Hypothesis prompt is missing {{USER_QUESTION}}")
        return template.replace("{{USER_QUESTION}}", question)

    def generate(
        self,
        question: str,
        llm: LLMCallable,
        data_source: str | None = None,
        pubmed: PubMedClient | None = None,
        on_llm_output: LLMOutputCallback | None = None,
    ) -> GeneratedHypothesis:
        raw = llm(self.build_prompt(question))
        first_output = LLMRawOutput("target_trial_draft", raw)
        llm_outputs = [first_output]
        if on_llm_output is not None:
            on_llm_output(first_output)
        protocol = self.parse_response(question, raw)
        if data_source:
            protocol["data_source"] = data_source
        if pubmed is not None:
            protocol = self._attach_literature_dag(
                protocol, llm, pubmed, llm_outputs, on_llm_output
            )
        return GeneratedHypothesis(
            protocol,
            HypothesisAgent().run(protocol),
            tuple(llm_outputs),
        )

    def _attach_literature_dag(
        self,
        protocol: dict,
        llm: LLMCallable,
        pubmed: PubMedClient,
        llm_outputs: list[LLMRawOutput],
        on_llm_output: LLMOutputCallback | None,
    ) -> dict:
        """Best-effort literature grounding: never blocks producing a protocol draft.

        A missing/empty search result or a malformed second-stage LLM response
        degrades to an empty candidate_dag with an auditable note, instead of
        failing the whole hypothesis draft. Genuine PubMed/LLM transport errors
        (network, auth, ...) are not caught here and propagate normally.
        """
        target = protocol.get("target_trial", {})
        articles: tuple[PubMedArticle, ...] = ()
        term = ""
        try:
            term = self._literature_search_term(protocol)
            pmids = pubmed.search(term, retmax=8)
            articles = pubmed.fetch_abstracts(pmids) if pmids else ()
        except ValueError:
            articles = ()

        edges: list[dict] = []
        rejected = 0
        if not articles:
            note = "No PubMed literature could be retrieved for this question; candidate_dag is empty."
        else:
            try:
                raw = llm(self._build_dag_prompt(target, articles))
                dag_output = LLMRawOutput("literature_dag_draft", raw)
                llm_outputs.append(dag_output)
                if on_llm_output is not None:
                    on_llm_output(dag_output)
                edges, rejected = self._parse_dag_response(raw, {article.pmid for article in articles})
            except ValueError:
                edges, rejected = [], 0
            note = (
                f"{len(edges)} literature-grounded candidate DAG edge(s) proposed "
                f"from {len(articles)} retrieved article(s)"
            )
            if rejected:
                note += f"; {rejected} edge(s) rejected for ungrounded or invalid content"

        protocol["candidate_dag"] = edges
        protocol["literature_evidence"] = {
            "search_query": term,
            "retrieved_pmids": [article.pmid for article in articles],
        }
        protocol.setdefault("design_constraints", [])
        protocol["design_constraints"] = list(protocol["design_constraints"]) + [note]
        return protocol

    @staticmethod
    def _literature_search_term(protocol: dict) -> str:
        terms = UserQuestionHypothesisService._validate_literature_search(
            protocol.get("literature_search")
        )

        def group(values: list[str]) -> str:
            return "(" + " OR ".join(
                f'"{value}"[Title/Abstract]' for value in values
            ) + ")"

        query_groups = [
            group(terms["population_terms"]),
            group(terms["treatment_terms"]),
        ]
        context = terms["comparator_terms"] + terms["outcome_terms"]
        if context:
            query_groups.append(group(context))
        return " AND ".join(query_groups)

    @staticmethod
    def _validate_literature_search(value: object) -> dict[str, list[str]]:
        if not isinstance(value, dict):
            raise ValueError("literature_search must be an object")
        normalized: dict[str, list[str]] = {}
        for field in _LITERATURE_TERM_FIELDS:
            items = value.get(field)
            if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
                raise ValueError(f"literature_search.{field} must be a list of strings")
            if len(items) > _MAX_TERMS_PER_ROLE:
                raise ValueError(
                    f"literature_search.{field} must contain at most {_MAX_TERMS_PER_ROLE} terms"
                )
            cleaned = []
            for item in items:
                term = " ".join(item.split())
                if (
                    not term
                    or len(term) > _MAX_TERM_CHARS
                    or any(char in term for char in ('"', "[", "]", "\\"))
                    or any(ord(char) < 32 for char in term)
                ):
                    raise ValueError(f"Invalid PubMed term in literature_search.{field}")
                cleaned.append(term)
            normalized[field] = cleaned
        for required in ("population_terms", "treatment_terms", "outcome_terms"):
            if not normalized[required]:
                raise ValueError(f"literature_search.{required} requires at least one term")
        return normalized

    def _build_dag_prompt(self, target: dict, articles: tuple[PubMedArticle, ...]) -> str:
        template = self.dag_prompt_path.read_text(encoding="utf-8")
        listed = "\n".join(
            f"- PMID {article.pmid}: {article.title} — {article.abstract[:_ABSTRACT_PREVIEW_CHARS]}"
            for article in articles
        )
        replacements = {
            "{{POPULATION}}": str(target.get("population", "")),
            "{{TREATMENT}}": str(target.get("treatment", "")),
            "{{COMPARATOR}}": str(target.get("comparator", "")),
            "{{OUTCOME}}": str(target.get("outcome", "")),
            "{{ARTICLES}}": listed,
        }
        for placeholder, value in replacements.items():
            if placeholder not in template:
                raise ValueError(f"DAG prompt is missing {placeholder}")
            template = template.replace(placeholder, value)
        return template

    @staticmethod
    def _parse_dag_response(raw: str, known_pmids: set[str]) -> tuple[list[dict], int]:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("LLM returned an empty DAG response")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("DAG LLM response must be one valid JSON object") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("candidate_dag"), list):
            raise ValueError("DAG response must contain a candidate_dag list")

        accepted: list[dict] = []
        rejected = 0
        for item in payload["candidate_dag"]:
            if (
                isinstance(item, dict)
                and isinstance(item.get("confounder"), str) and item["confounder"].strip()
                and item.get("relation") in DAG_RELATIONS
                and isinstance(item.get("pmid"), str) and item["pmid"] in known_pmids
                and isinstance(item.get("rationale"), str) and item["rationale"].strip()
            ):
                accepted.append({
                    "confounder": item["confounder"].strip(),
                    "relation": item["relation"],
                    "pmid": item["pmid"],
                    "rationale": item["rationale"].strip(),
                })
            else:
                rejected += 1
        return accepted, rejected

    def parse_response(self, original_question: str, raw: str) -> dict:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("LLM returned an empty response")
        try:
            protocol = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM response must be one valid JSON object") from exc
        if not isinstance(protocol, dict):
            raise ValueError("LLM response must be a JSON object")
        if protocol.get("source_question") != original_question.strip():
            raise ValueError("LLM must preserve the original source_question exactly")
        study_id = protocol.get("study_id")
        if not isinstance(study_id, str) or not _STUDY_ID.fullmatch(study_id):
            raise ValueError("LLM produced an invalid study_id")
        assumptions = protocol.get("assumptions_requiring_review", [])
        if not isinstance(assumptions, list) or not all(isinstance(item, str) for item in assumptions):
            raise ValueError("assumptions_requiring_review must be a list of strings")
        protocol["literature_search"] = self._validate_literature_search(
            protocol.get("literature_search")
        )
        protocol["concept_sets"] = self._pending_concept_sets(protocol)
        protocol.setdefault("design_constraints", [])
        protocol["design_constraints"] = list(protocol["design_constraints"]) + [
            "LLM-generated draft requires human protocol approval",
            "OMOP availability and concept mappings must be verified before execution",
        ]
        return protocol

    @staticmethod
    def _pending_concept_sets(protocol: dict) -> dict:
        target = protocol.get("target_trial", {})
        return {
            role: {
                "status": "pending_omop_mapping",
                "concept_ids": [],
                "label": target.get(role, ""),
            }
            for role in ("population", "treatment", "comparator", "outcome")
        }
