"""Provider-neutral bridge from a user question to the Hypothesis Agent."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from causal.agents.dag_critique import (
    build_critique_prompt,
    build_pre_check,
    build_revision_prompt,
    format_structural_findings,
    parse_critique_response,
)
from causal.agents.hypothesis import (
    DAG_VARIABLE_ROLES,
    ROLE_ALLOWED_TEMPORAL_CLASSES,
    HypothesisAgent,
    HypothesisArtifact,
)
from causal.knowledge_graph import KGEdge, OptimusKGClient
from causal.literature.orchestrator import LiteratureArticle, LiteratureOrchestrator, select_by_evidence_tier
from causal.llm.base import LLMCallable
from causal.roadmap import EDGE_CERTAINTY


def _parsed_certainty(item: dict) -> str:
    certainty = item.get("certainty", "established")
    return certainty if certainty in EDGE_CERTAINTY else "established"


def _kg_reinforcement(
    label: str, anchors: tuple[str, ...], kg_edges: tuple[KGEdge, ...], accepted_pmid: str
) -> str:
    """Return a KG edge id that independently confirms `label`'s link to an anchor term.

    Only meaningful when `accepted_pmid` is itself a literature citation: a
    variable already grounded solely by a KG edge (no abstract) has that same
    edge in `pmid`, not here — this field exists to flag *agreement between
    two independent sources*, not to restate the one source already cited.

    `label` and `anchor` must match *opposite* endpoints of the same edge.
    Matching both on the same endpoint (e.g. `label` and `anchor` both
    landing in `source_label` because they happen to share a substring) is
    not evidence of anything: the edge's other endpoint was never shown to
    relate to the anchor at all in that case, just coincidentally
    substring-matched alongside it.
    """
    if not accepted_pmid or accepted_pmid.startswith("optimuskg:"):
        return ""
    needle = label.strip().lower()
    if not needle:
        return ""
    for edge in kg_edges:
        source, target = edge.source_label.lower(), edge.target_label.lower()
        label_on_source = needle in source
        label_on_target = needle in target
        if not label_on_source and not label_on_target:
            continue
        for anchor in anchors:
            anchor_needle = anchor.strip().lower()
            if not anchor_needle:
                continue
            if (label_on_source and anchor_needle in target) or (
                label_on_target and anchor_needle in source
            ):
                return edge.citation_id
    return ""


def _format_articles(articles: tuple[LiteratureArticle, ...]) -> str:
    return "\n".join(
        f"- [{article.source}, evidence={article.evidence_level}] {article.citation_id}: "
        f"{article.title} — {article.abstract[:_ABSTRACT_PREVIEW_CHARS]}"
        for article in articles
    )


def _format_kg_relations(kg_edges: tuple[KGEdge, ...]) -> str:
    return "\n".join(
        f"- [optimuskg] {edge.citation_id}: {edge.source_label} --{edge.predicate}--> {edge.target_label}"
        for edge in kg_edges
    )


def _sanitize_query_text(value: str) -> str:
    """Best-effort cleanup of free-text (a first-pass covariate label) into a
    safe PubMed/OpenAlex search term.

    Unlike `_validate_literature_search` (which rejects a malformed
    LLM-authored term outright), this never raises: a covariate label that
    cannot be turned into a safe term just yields no targeted evidence for
    that one covariate, rather than failing the whole draft.
    """
    term = " ".join(value.split())
    term = "".join(char for char in term if char not in ('"', "[", "]", "\\") and ord(char) >= 32)
    return " ".join(term.split())[:_MAX_TERM_CHARS]


def _covariate_search_term(
    covariate: str, population_terms: tuple[str, ...], anchor_terms: tuple[str, ...]
) -> str:
    """Pair one first-pass covariate with the population terms *and* the
    treatment/outcome anchor terms already validated in `literature_search`.

    Population terms are required, not merely another OR option alongside
    treatment/outcome: a covariate paired only with a generic outcome term
    like "mortality" is satisfied by almost any critical-care severity or
    biomarker study, regardless of whether it has anything to do with this
    question's population (observed in practice: a post-cardiac-arrest BMI/
    mortality study and a severe-fever D-dimer/mortality study both matched
    an ICU arterial-catheter question's covariate searches purely because
    both mention "mortality", despite neither population having anything to
    do with arterial catheters).
    """
    cleaned = _sanitize_query_text(covariate)
    if not cleaned or not population_terms or not anchor_terms:
        return ""
    population_group = "(" + " OR ".join(f'"{term}"[Title/Abstract]' for term in population_terms) + ")"
    anchor_group = "(" + " OR ".join(f'"{term}"[Title/Abstract]' for term in anchor_terms) + ")"
    return f'"{cleaned}"[Title/Abstract] AND {population_group} AND {anchor_group}'


def _search_covariate_evidence(
    protocol: dict, literature: LiteratureOrchestrator
) -> dict[str, tuple[LiteratureArticle, ...]]:
    """Best-effort, per-covariate targeted search (see `_MAX_COVARIATES_SEARCHED`).

    Returns only covariates that actually matched something; a covariate
    absent from the result simply gets no dedicated evidence (it can still
    be grounded incidentally by the general P/T/C/O search, or end up
    `unresolved` -- unchanged from before this ran).
    """
    literature_search = protocol.get("literature_search")
    if not isinstance(literature_search, dict):
        return {}
    population_terms = tuple(literature_search.get("population_terms", ()))
    anchor_terms = tuple(literature_search.get("treatment_terms", ())) + tuple(
        literature_search.get("outcome_terms", ())
    )
    if not population_terms or not anchor_terms:
        return {}
    covariates = [
        str(item).strip() for item in protocol.get("first_pass_covariates", ()) if str(item).strip()
    ][:_MAX_COVARIATES_SEARCHED]
    hits: dict[str, tuple[LiteratureArticle, ...]] = {}
    for covariate in covariates:
        term = _covariate_search_term(covariate, population_terms, anchor_terms)
        if not term:
            continue
        try:
            found = literature.search_and_fetch(
                term,
                retmax_per_source=_COVARIATE_RETMAX_PER_SOURCE,
                limit=_MAX_COVARIATE_ARTICLES_PER_LABEL,
            )
        except ValueError:
            found = ()
        if found:
            hits[covariate] = found
    return hits


def _merge_with_covariate_coverage(
    main_pool: tuple[LiteratureArticle, ...],
    covariate_hits: dict[str, tuple[LiteratureArticle, ...]],
    limit: int,
) -> tuple[LiteratureArticle, ...]:
    """Reserve a slot for each covariate's targeted evidence before filling
    the remaining budget from the general P/T/C/O search's evidence-tier
    ranking.

    Without this reservation, a covariate's single targeted-but-
    `unclassified` article can be crowded out by several generic
    `systematic_review`-tier articles that never actually discuss it --
    exactly the failure mode `_search_covariate_evidence` exists to fix, so
    merging naively back into one flat tier-ranked pool would undo it.
    """
    selected: list[LiteratureArticle] = []
    seen_ids: set[str] = set()
    for hits in covariate_hits.values():
        for article in hits:
            if len(selected) >= limit:
                return tuple(selected)
            if article.citation_id in seen_ids:
                continue
            seen_ids.add(article.citation_id)
            selected.append(article)
    remaining = limit - len(selected)
    if remaining <= 0:
        return tuple(selected)
    filtered_main = tuple(article for article in main_pool if article.citation_id not in seen_ids)
    selected.extend(select_by_evidence_tier(filtered_main, remaining))
    return tuple(selected)


_STUDY_ID = re.compile(r"^[a-z][a-z0-9_]{2,79}$")
_STUDY_ID_INVALID_CHARS = re.compile(r"[^a-z0-9_]+")
_LITERATURE_TERM_FIELDS = (
    "population_terms", "treatment_terms", "comparator_terms", "outcome_terms"
)
_ABSTRACT_PREVIEW_CHARS = 600
_MAX_TERMS_PER_ROLE = 3
_MAX_TERM_CHARS = 80
_MAX_DAG_ARTICLES = 16
_MAX_KG_RELATIONS = 10
_KG_ANCHOR_FIELDS = ("population", "treatment", "comparator", "outcome")
#: A classic confounder (age, an illness-severity score, ...) is otherwise
#: only ever grounded if it happens to be mentioned in whatever the generic
#: population/treatment/comparator/outcome search returns -- which in
#: practice is often dominated by articles that merely mention the treatment
#: term in passing. `_search_covariate_evidence` runs one small, targeted
#: search per first-pass covariate so it has a chance to be cited even when
#: the generic search never surfaces it. Capped to bound the extra network
#: calls this adds per draft (each covariate costs up to one esearch/efetch
#: round trip per configured literature source).
_MAX_COVARIATES_SEARCHED = 12
_COVARIATE_RETMAX_PER_SOURCE = 4
_MAX_COVARIATE_ARTICLES_PER_LABEL = 1


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
        dag_critique_prompt_path: str | Path = "causal/prompts/dag_critique.md",
    ) -> None:
        self.prompt_path = Path(prompt_path)
        self.dag_prompt_path = Path(dag_prompt_path)
        self.dag_critique_prompt_path = Path(dag_critique_prompt_path)

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
        literature: LiteratureOrchestrator | None = None,
        knowledge_graph: OptimusKGClient | None = None,
        on_llm_output: LLMOutputCallback | None = None,
        enable_critique: bool = True,
    ) -> GeneratedHypothesis:
        raw = llm(self.build_prompt(question))
        first_output = LLMRawOutput("target_trial_draft", raw)
        llm_outputs = [first_output]
        if on_llm_output is not None:
            on_llm_output(first_output)
        protocol = self.parse_response(question, raw)
        if data_source:
            protocol["data_source"] = data_source
        literature_configured = literature is not None and literature.is_configured()
        if literature_configured or knowledge_graph is not None:
            protocol = self._attach_literature_dag(
                protocol,
                llm,
                literature if literature_configured else None,
                knowledge_graph,
                llm_outputs,
                on_llm_output,
                enable_critique,
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
        literature: LiteratureOrchestrator | None,
        knowledge_graph: OptimusKGClient | None,
        llm_outputs: list[LLMRawOutput],
        on_llm_output: LLMOutputCallback | None,
        enable_critique: bool = True,
    ) -> dict:
        """Best-effort literature/KG grounding: never blocks producing a protocol draft.

        A missing/empty search result or a malformed second-stage LLM response
        degrades to an empty candidate_dag with an auditable note, instead of
        failing the whole hypothesis draft. Genuine transport errors from a
        literature source or the LLM (network, auth, ...) are not caught here
        and propagate normally; only per-source retrieval failures are
        absorbed, inside `LiteratureOrchestrator`/`OptimusKGClient` themselves.

        A variable can be accepted on knowledge-graph evidence alone, with no
        retrieved abstract discussing it — the gate (`_parse_dag_response`)
        checks `pmid` against the union of literature and KG citation ids, not
        literature only. When *both* sources independently support the same
        variable, that agreement is recorded as `kg_evidence_id` (a second,
        reinforcing evidence id) rather than silently discarded once the
        `pmid` slot is already filled by the literature citation.

        When `enable_critique` and the draft is non-empty, a third LLM call
        (see `_critique_and_revise_dag`) independently challenges the draft
        before it reaches `protocol["candidate_dag"]`, with at most one
        bounded revision round. This never approves or rejects anything --
        the result is exactly as provisional as an uncritiqued draft would
        have been, still gated by `HypothesisAgent` and the human `protocol`
        approval gate in `causal.orchestrator.workflow`.
        """
        target = protocol.get("target_trial", {})
        articles: tuple[LiteratureArticle, ...] = ()
        kg_edges: tuple[KGEdge, ...] = ()
        covariate_hits: dict[str, tuple[LiteratureArticle, ...]] = {}
        term = ""
        try:
            term = self._literature_search_term(protocol)
            if literature is not None:
                pool = literature.search_and_fetch(term)
                covariate_hits = _search_covariate_evidence(protocol, literature)
                articles = _merge_with_covariate_coverage(pool, covariate_hits, _MAX_DAG_ARTICLES)
        except ValueError:
            articles = ()
        if knowledge_graph is not None:
            try:
                kg_edges = self._fetch_kg_relations(protocol, knowledge_graph)
            except RuntimeError:
                kg_edges = ()

        variables: list[dict] = []
        relations: list[dict] = []
        rejected = 0
        unresolved: list[dict] = []
        if not articles and not kg_edges:
            note = "No literature could be retrieved for this question; candidate_dag is empty."
        else:
            treatment = str(target.get("treatment", ""))
            outcome = str(target.get("outcome", ""))
            known_citation_ids = {article.citation_id for article in articles} | {
                edge.citation_id for edge in kg_edges
            }
            listed = _format_articles(articles)
            kg_listed = _format_kg_relations(kg_edges)
            dag_prompt = self._build_dag_prompt(target, articles, kg_edges)
            try:
                raw = llm(dag_prompt)
                dag_output = LLMRawOutput("literature_dag_draft", raw)
                llm_outputs.append(dag_output)
                if on_llm_output is not None:
                    on_llm_output(dag_output)
                variables, relations, rejected, unresolved = self._parse_dag_response(
                    raw, known_citation_ids, kg_edges, treatment, outcome,
                )
            except ValueError:
                variables, relations, rejected, unresolved = [], [], 0, []
            reinforced = sum(1 for variable in variables if variable.get("kg_evidence_id"))
            note = (
                f"{len(variables)} candidate DAG variable(s) and "
                f"{len(relations)} covariate relation(s) proposed from {len(articles)} retrieved "
                f"article(s) ({len(covariate_hits)} first-pass covariate(s) had dedicated targeted "
                f"evidence) and {len(kg_edges)} knowledge-graph relation(s)"
            )
            if reinforced:
                note += (
                    f"; {reinforced} variable(s) independently corroborated by both "
                    "literature and the knowledge graph"
                )
            if rejected:
                note += f"; {rejected} entr(y/ies) rejected for ungrounded or invalid content"
            if unresolved:
                note += (
                    f"; {len(unresolved)} first-pass covariate(s) flagged unresolved "
                    "by the drafting LLM"
                )
            if enable_critique and variables:
                variables, relations, extra_rejected, unresolved, critique_note = self._critique_and_revise_dag(
                    protocol, target, listed, kg_listed, dag_prompt,
                    variables, relations, unresolved, known_citation_ids, kg_edges,
                    treatment, outcome, llm, llm_outputs, on_llm_output,
                )
                rejected += extra_rejected
                note += critique_note

        protocol["candidate_dag"] = variables
        protocol["covariate_relations"] = relations
        protocol["candidate_dag_unresolved"] = unresolved
        protocol["literature_evidence"] = {
            "search_query": term,
            "retrieved_citations": [
                {"source": article.source, "id": article.citation_id} for article in articles
            ] + [
                {"source": edge.kg_source, "id": edge.citation_id} for edge in kg_edges
            ],
        }
        protocol.setdefault("design_constraints", [])
        protocol["design_constraints"] = list(protocol["design_constraints"]) + [note]
        return protocol

    def _critique_and_revise_dag(
        self,
        protocol: dict,
        target: dict,
        listed: str,
        kg_listed: str,
        dag_prompt: str,
        variables: list[dict],
        relations: list[dict],
        unresolved: list[dict],
        known_citation_ids: set[str],
        kg_edges: tuple[KGEdge, ...],
        treatment: str,
        outcome: str,
        llm: LLMCallable,
        llm_outputs: list[LLMRawOutput],
        on_llm_output: LLMOutputCallback | None,
    ) -> tuple[list[dict], list[dict], int, list[dict], str]:
        """Independently challenge a non-empty DAG draft, with at most one
        bounded revision round, before it reaches `protocol["candidate_dag"]`.

        Never raises: a malformed critique or revision response degrades to
        the pre-step DAG with an explanatory note appended to the caller's
        `note`, exactly like the surrounding "never blocks the draft"
        contract in `_attach_literature_dag`. A revised proposal is parsed
        by the same `_parse_dag_response` gate as the original draft -- the
        critique loop never gets a separate, looser trust path.
        """
        review = build_pre_check(protocol, variables, relations)
        findings_text = format_structural_findings(review)
        proposed_dag_json = json.dumps(
            {"candidate_dag": variables, "covariate_relations": relations, "unresolved": unresolved},
            indent=2,
        )
        critique_prompt = build_critique_prompt(
            self.dag_critique_prompt_path, target, listed, kg_listed, proposed_dag_json, findings_text,
        )
        try:
            raw = llm(critique_prompt)
            critique_output = LLMRawOutput("literature_dag_critique", raw)
            llm_outputs.append(critique_output)
            if on_llm_output is not None:
                on_llm_output(critique_output)
            critique = parse_critique_response(raw)
        except ValueError:
            return variables, relations, 0, unresolved, (
                "; DAG critique step failed or returned malformed output, "
                "using pre-critique candidate_dag"
            )

        if critique.verdict == "approve":
            return variables, relations, 0, unresolved, (
                "; candidate_dag reviewed by an independent critique step and approved without changes"
            )

        revision_prompt = build_revision_prompt(dag_prompt, proposed_dag_json, critique)
        try:
            raw_revision = llm(revision_prompt)
            revision_output = LLMRawOutput("literature_dag_revision", raw_revision)
            llm_outputs.append(revision_output)
            if on_llm_output is not None:
                on_llm_output(revision_output)
            rev_variables, rev_relations, rev_rejected, rev_unresolved = self._parse_dag_response(
                raw_revision, known_citation_ids, kg_edges, treatment, outcome,
            )
        except ValueError:
            return variables, relations, 0, unresolved, (
                f"; DAG critique requested {len(critique.concerns)} change(s) but the revision "
                "step failed or returned malformed output, using pre-revision candidate_dag"
            )

        return rev_variables, rev_relations, rev_rejected, rev_unresolved, (
            f"; candidate_dag revised after critique ({len(critique.concerns)} concern(s) addressed)"
        )

    @staticmethod
    def _fetch_kg_relations(protocol: dict, knowledge_graph: OptimusKGClient) -> tuple[KGEdge, ...]:
        """KG relation lookup grounding both the trial's core concepts and its
        first-pass covariates.

        OptimusKG has no free-text relevance search (see
        `causal.knowledge_graph.optimuskg_client`): a relation is only found
        by querying two concept labels directly. `candidate_dag` variable
        labels do not exist yet at this point in drafting — they are what the
        LLM proposes *from* this evidence — so this cannot query them
        directly either. But `first_pass_covariates` (the declared candidate
        confounder labels) *are* already known here, same premise as the
        literature side's `_search_covariate_evidence`, so each is paired
        against treatment/outcome directly. Without this, a KG edge could
        only ever ground a candidate_dag variable whose label happened to be
        a substring of one of the four anchor terms themselves — effectively
        never, for a real confounder like "baseline severity" or "age", which
        made the "a variable can be accepted on KG evidence alone" claim in
        `_attach_literature_dag`'s docstring true only for the four core
        concepts, not for the covariates the DAG most needs grounding for.

        Covariate-anchor pairs are queried first and get first claim on the
        shared `_MAX_KG_RELATIONS` budget (mirrors
        `_merge_with_covariate_coverage`'s reservation on the literature
        side): the four-anchor pairwise sweep below is useful in its own
        right (e.g. grounding the treatment-outcome relation itself) but is
        not what this method exists to fix, so it must not be able to
        exhaust the budget before covariate grounding gets a chance.
        Covariate-to-covariate relations remain out of scope: at this point
        only the ungrounded first-pass list exists, not `candidate_dag`.
        """
        target = protocol.get("target_trial", {})
        edges: list[KGEdge] = []
        seen_ids: set[str] = set()

        def _extend(found: tuple[KGEdge, ...]) -> bool:
            """Append newly-seen edges; return True once the budget is full."""
            for edge in found:
                if edge.citation_id not in seen_ids:
                    seen_ids.add(edge.citation_id)
                    edges.append(edge)
                    if len(edges) >= _MAX_KG_RELATIONS:
                        return True
            return False

        treatment = str(target.get("treatment", "")).strip()
        outcome = str(target.get("outcome", "")).strip()
        covariates = [
            str(item).strip() for item in protocol.get("first_pass_covariates", ()) if str(item).strip()
        ][:_MAX_COVARIATES_SEARCHED]
        for covariate in covariates:
            for anchor in (treatment, outcome):
                if not anchor:
                    continue
                try:
                    found = knowledge_graph.find_relations(covariate, anchor)
                except ValueError:
                    continue
                if _extend(found):
                    return tuple(edges)

        anchors = [str(target.get(field, "")).strip() for field in _KG_ANCHOR_FIELDS]
        anchors = [term for term in anchors if term]
        for index, term_a in enumerate(anchors):
            for term_b in anchors[index + 1:]:
                try:
                    found = knowledge_graph.find_relations(term_a, term_b)
                except ValueError:
                    continue
                if _extend(found):
                    return tuple(edges)
        return tuple(edges)

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

    def _build_dag_prompt(
        self, target: dict, articles: tuple[LiteratureArticle, ...], kg_edges: tuple[KGEdge, ...] = ()
    ) -> str:
        template = self.dag_prompt_path.read_text(encoding="utf-8")
        listed = _format_articles(articles)
        kg_listed = _format_kg_relations(kg_edges)
        # {{KG_RELATIONS}} is optional: older/custom dag prompt templates that
        # do not reference it still work unchanged (kg_edges is simply
        # dropped), matching this module's "never blocks" contract.
        required = {
            "{{POPULATION}}": str(target.get("population", "")),
            "{{TREATMENT}}": str(target.get("treatment", "")),
            "{{COMPARATOR}}": str(target.get("comparator", "")),
            "{{OUTCOME}}": str(target.get("outcome", "")),
            "{{ARTICLES}}": listed,
        }
        for placeholder, value in required.items():
            if placeholder not in template:
                raise ValueError(f"DAG prompt is missing {placeholder}")
            template = template.replace(placeholder, value)
        if "{{KG_RELATIONS}}" in template:
            template = template.replace("{{KG_RELATIONS}}", kg_listed)
        return template

    @staticmethod
    def _parse_dag_response(
        raw: str,
        known_citation_ids: set[str],
        kg_edges: tuple[KGEdge, ...] = (),
        treatment: str = "",
        outcome: str = "",
    ) -> tuple[list[dict], list[dict], int, list[dict]]:
        """Returns (accepted_variables, accepted_relations, rejected_count, unresolved).

        `unresolved` are first-pass covariates the LLM explicitly says it
        could not confidently classify (see `causal/prompts/dag.md`'s
        `unresolved` field) -- kept distinct from `rejected`, which counts
        entries that failed *this* module's structural/grounding gate
        (bad role, uncited pmid, ...). An `unresolved` entry is not a
        grounding failure, so it is not counted toward `rejected`.
        """
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("LLM returned an empty DAG response")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("DAG LLM response must be one valid JSON object") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("candidate_dag"), list):
            raise ValueError("DAG response must contain a candidate_dag list")

        rejected = 0
        accepted_variables: list[dict] = []
        known_labels: set[str] = set()
        for item in payload["candidate_dag"]:
            role = item.get("role") if isinstance(item, dict) else None
            temporal = item.get("temporal_classification") if isinstance(item, dict) else None
            if (
                isinstance(item, dict)
                and isinstance(item.get("label"), str) and item["label"].strip()
                and role in DAG_VARIABLE_ROLES
                and temporal in ROLE_ALLOWED_TEMPORAL_CLASSES.get(role, ())
                and isinstance(item.get("pmid"), str) and item["pmid"] in known_citation_ids
                and isinstance(item.get("rationale"), str) and item["rationale"].strip()
            ):
                label = item["label"].strip()
                accepted_variables.append({
                    "label": label,
                    "role": role,
                    "temporal_classification": temporal,
                    "pmid": item["pmid"],
                    "rationale": item["rationale"].strip(),
                    "observed": item.get("observed", True) if isinstance(item.get("observed", True), bool) else True,
                    "certainty": _parsed_certainty(item),
                    "kg_evidence_id": _kg_reinforcement(label, (treatment, outcome), kg_edges, item["pmid"]),
                })
                known_labels.add(label)
            else:
                rejected += 1

        accepted_relations: list[dict] = []
        raw_relations = payload.get("covariate_relations", [])
        if isinstance(raw_relations, list):
            for item in raw_relations:
                if (
                    isinstance(item, dict)
                    and isinstance(item.get("source"), str) and item["source"].strip() in known_labels
                    and isinstance(item.get("target"), str) and item["target"].strip() in known_labels
                    and item["source"].strip() != item["target"].strip()
                    and isinstance(item.get("pmid"), str) and item["pmid"] in known_citation_ids
                    and isinstance(item.get("rationale"), str) and item["rationale"].strip()
                ):
                    accepted_relations.append({
                        "source": item["source"].strip(),
                        "target": item["target"].strip(),
                        "pmid": item["pmid"],
                        "rationale": item["rationale"].strip(),
                        "certainty": _parsed_certainty(item),
                    })
                else:
                    rejected += 1

        unresolved: list[dict] = []
        raw_unresolved = payload.get("unresolved", [])
        if isinstance(raw_unresolved, list):
            for item in raw_unresolved:
                if (
                    isinstance(item, dict)
                    and isinstance(item.get("label"), str) and item["label"].strip()
                    and isinstance(item.get("reason"), str) and item["reason"].strip()
                ):
                    unresolved.append({"label": item["label"].strip(), "reason": item["reason"].strip()})

        return accepted_variables, accepted_relations, rejected, unresolved

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
        if not isinstance(study_id, str) or not study_id.strip():
            raise ValueError("LLM produced an invalid study_id")
        normalized_study_id = _STUDY_ID_INVALID_CHARS.sub("_", study_id.strip().lower())
        if not _STUDY_ID.fullmatch(normalized_study_id):
            raise ValueError("LLM produced an invalid study_id")
        protocol["study_id"] = normalized_study_id
        assumptions = protocol.get("assumptions_requiring_review", [])
        if not isinstance(assumptions, list) or not all(isinstance(item, str) for item in assumptions):
            raise ValueError("assumptions_requiring_review must be a list of strings")
        protocol["literature_search"] = self._validate_literature_search(
            protocol.get("literature_search")
        )
        protocol["concept_sets"] = self._pending_concept_sets(protocol)
        protocol["query_concept_sets"] = self._pending_query_concept_sets()
        protocol.setdefault("design_constraints", [])
        protocol["design_constraints"] = list(protocol["design_constraints"]) + [
            "LLM-generated draft requires human protocol approval",
            "OMOP availability and concept mappings must be verified before execution",
        ]
        return protocol

    @staticmethod
    def _pending_concept_sets(protocol: dict) -> dict:
        """Scaffold for the human concept-mapping review (see
        `causal.agents.data_science.DataScienceAgent._phenotype_from_concept_set`).
        `role` can be pre-filled from the target-trial role this dict key
        already names, but `domain` cannot: which OMOP domain (condition,
        drug, procedure, ...) a clinical concept lives in is exactly the
        clinical-mapping judgment call a human still has to make, so it is
        left blank and `DataScienceAgent` refuses to plan around it until
        concept-mapping review fills it in.
        """
        target = protocol.get("target_trial", {})
        return {
            role: {
                "status": "pending_omop_mapping",
                "concept_ids": [],
                "label": target.get(role, ""),
                "role": role,
                "domain": "",
            }
            for role in ("population", "treatment", "comparator", "outcome")
        }

    @staticmethod
    def _pending_query_concept_sets() -> dict:
        """Scaffold for the per-question feasibility/cohort SQL (see
        `causal.queries.feasibility`). `domain` and `concept_ids` are filled
        in during human concept-mapping review, same gate as `concept_sets`;
        an empty domain blocks query generation with a clear error rather
        than guessing an OMOP table.
        """
        return {
            role: {"domain": "", "concept_ids": []}
            for role in ("population", "treatment", "comparator", "outcome")
        }
