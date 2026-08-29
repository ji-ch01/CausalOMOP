"""Route one clinical question through the causal-question gate.

This is orchestration step 1 of docs/causal_roadmap.md ("classify and clarify
the causal question"): classify the question, then either

- answer a descriptive question via FastOMOP (`causal.fastomop_bridge`),
- return a clarifying question for the reviewer to answer, or
- draft a target trial + candidate DAG and open a governed run
  (`WorkflowManager.start_generated`).

`causal.agents.question_gate` classifies; `causal.agents.user_question`
drafts. This module is the glue that decides which of the three happens, so
that routing lives in exactly one place instead of being duplicated between
`causal.cli` and the Streamlit conversation panel.

It performs the same side effects those callers already performed -- one or
more LLM calls, an optional literature search, and (on the causal path) a new
run directory -- but returns a single typed `IntakeOutcome` instead of
printing or calling `st.*`. It never decides a gate: `start_generated`
opens the run with every gate still `pending`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from causal.agents.question_gate import (
    DEFAULT_GATE_PROMPT_PATH,
    QuestionGateResult,
    build_gate_prompt,
    parse_gate_response,
)
from causal.agents.user_question import LLMOutputCallback, UserQuestionHypothesisService
from causal.fastomop_bridge import ask_fastomop

if TYPE_CHECKING:
    from causal.config import BigQueryDatasets
    from causal.knowledge_graph import OptimusKGClient
    from causal.literature import LiteratureOrchestrator
    from causal.llm import ResolvedLLM
    from causal.orchestrator.state import RunState
    from causal.orchestrator.workflow import WorkflowManager

#: A clarification exchange the reviewer has already answered: the question
#: the assistant asked, and the reviewer's reply.
ClarificationRound = tuple[str, str]

#: One earlier answered descriptive question and FastOMOP's answer, replayed
#: as context so a terse follow-up ("and how many were women?") is
#: self-contained.
DescriptiveExchange = tuple[str, str]

#: How many times the reviewer may answer a clarifying question before the
#: panel stops re-gating and asks them to rephrase (or force the causal path).
MAX_CLARIFICATION_ROUNDS = 4

#: How many prior descriptive exchanges to replay to the gate and FastOMOP
#: for a follow-up. Bounded -- the point is "who is *them*", not a full
#: session history.
MAX_DESCRIPTIVE_CONTEXT = 3


@dataclass(frozen=True)
class DraftOptions:
    """Knobs the "New study" form already exposes, in one object."""

    mode: str = "exploratory"
    skip_question_gate: bool = False
    enable_critique: bool = True
    max_critique_rounds: int | None = None
    tournament_size: int | None = None


@dataclass(frozen=True)
class DescriptiveAnswer:
    """The gate judged the question descriptive; FastOMOP answered it inline.
    Never routed back into `causal.llm` (see `causal.fastomop_bridge`)."""

    reasoning: str
    answer: str
    sql: str | None


@dataclass(frozen=True)
class ClarificationNeeded:
    """The gate wants the question sharpened before any drafting happens."""

    reasoning: str
    clarifying_question: str


@dataclass(frozen=True)
class StudyDrafted:
    """A governed run was created from an LLM draft; every gate is pending."""

    state: "RunState"
    provider: str
    model: str


IntakeOutcome = DescriptiveAnswer | ClarificationNeeded | StudyDrafted


@dataclass
class QuestionIntake:
    """Reusable question -> gate -> {FastOMOP | clarify | draft} routing.

    The heavy collaborators are injected so both the CLI and the Streamlit
    panel can supply their own already-built literature/KG clients and so
    tests can pass fakes. `service` defaults to the real
    `UserQuestionHypothesisService` (which reads `causal/prompts/*`).
    """

    manager: "WorkflowManager"
    resolved_llm: "ResolvedLLM"
    literature: "LiteratureOrchestrator | None" = None
    knowledge_graph: "OptimusKGClient | None" = None
    data_source: str | None = None
    datasets: "BigQueryDatasets | None" = None
    gate_prompt_path: Path = DEFAULT_GATE_PROMPT_PATH
    service: UserQuestionHypothesisService = field(default_factory=UserQuestionHypothesisService)

    def run(
        self,
        question: str,
        options: DraftOptions | None = None,
        *,
        descriptive_context: Sequence[DescriptiveExchange] = (),
        on_llm_output: LLMOutputCallback | None = None,
        on_stage: Callable[[str], None] | None = None,
        on_gate_result: Callable[[QuestionGateResult], None] | None = None,
    ) -> IntakeOutcome:
        """Classify `question` and act on the classification.

        `descriptive_context` is recent answered FastOMOP exchanges from the
        same session, most recent last. When present, they are folded in
        (`combine_descriptive_followup`) so both the gate and FastOMOP see a
        terse follow-up ("and how many were women?") in context. It is never
        passed to the causal drafting call -- a target trial is drafted from
        the clean question only.

        `on_gate_result` fires once with the parsed gate verdict (for every
        category) before the outcome is produced, so a caller can show the
        classification in its own log/thread. Raises `ValueError` for an
        empty question or a malformed gate response,
        `causal.fastomop_bridge.FastOMOPError` if a descriptive question
        cannot be answered, and propagates provider/network errors from the
        drafting call. The caller owns how any of those is shown.
        """
        options = options or DraftOptions()
        question = question.strip()
        if not question:
            raise ValueError("Clinical question is required")

        contextual_question = combine_descriptive_followup(question, descriptive_context)

        def _stage(message: str) -> None:
            if on_stage is not None:
                on_stage(message)

        if not options.skip_question_gate:
            _stage("Classifying the question (causal / descriptive / needs clarification)...")
            gate_result = parse_gate_response(
                self.resolved_llm.call(build_gate_prompt(self.gate_prompt_path, contextual_question))
            )
            if on_gate_result is not None:
                on_gate_result(gate_result)
            if gate_result.category == "descriptive":
                _stage("Descriptive question -- routing to FastOMOP...")
                answer = ask_fastomop(contextual_question, datasets=self.datasets)
                return DescriptiveAnswer(
                    reasoning=gate_result.reasoning,
                    answer=answer.answer,
                    sql=answer.sql_query,
                )
            if gate_result.category == "needs_clarification":
                return ClarificationNeeded(
                    reasoning=gate_result.reasoning,
                    clarifying_question=gate_result.clarifying_question,
                )
            _stage(f"Causal question -- {gate_result.reasoning}")

        generate_kwargs: dict[str, object] = {}
        if options.max_critique_rounds is not None:
            generate_kwargs["max_critique_rounds"] = options.max_critique_rounds
        if options.tournament_size is not None:
            generate_kwargs["tournament_size"] = options.tournament_size

        _stage("Drafting the target trial and candidate DAG...")
        generated = self.service.generate(
            question,
            self.resolved_llm.call,
            data_source=self.data_source,
            literature=self.literature,
            knowledge_graph=self.knowledge_graph,
            on_llm_output=on_llm_output,
            enable_critique=options.enable_critique,
            **generate_kwargs,
        )
        state = self.manager.start_generated(
            generated, options.mode, f"{self.resolved_llm.provider}:{self.resolved_llm.model}"
        )
        return StudyDrafted(state=state, provider=self.resolved_llm.provider, model=self.resolved_llm.model)


def combine_clarifications(original_question: str, rounds: Sequence[ClarificationRound]) -> str:
    """Fold answered clarification rounds back into one question string.

    The drafting LLM only ever sees the resulting string, so each round
    keeps the assistant's own question alongside the reviewer's answer for
    context. The gate is then re-run on this combined string every round
    (it may still ask for more, or now settle on causal/descriptive).
    """
    combined = original_question.strip()
    for index, (clarifying_question, answer) in enumerate(rounds, start=1):
        combined += (
            f"\n\nClarification {index}"
            f"\nQ: {clarifying_question.strip()}"
            f"\nA: {answer.strip()}"
        )
    return combined


def combine_descriptive_followup(
    question: str, prior_exchanges: Sequence[DescriptiveExchange]
) -> str:
    """Prefix recent answered FastOMOP exchanges so a follow-up is
    self-contained.

    `prior_exchanges` is `(question, answer)` pairs, oldest first; only the
    last `MAX_DESCRIPTIVE_CONTEXT` are kept. Empty or all-blank input returns
    the question stripped and unchanged, so a first question never carries a
    context header. The result is used for the gate and the FastOMOP call
    only -- never the causal drafting path.
    """
    usable = [
        (q.strip(), a.strip())
        for q, a in prior_exchanges
        if q and q.strip() and a and a.strip()
    ][-MAX_DESCRIPTIVE_CONTEXT:]
    if not usable:
        return question.strip()
    lines = ["Earlier descriptive questions I asked in this session, with FastOMOP's answers:"]
    for index, (prior_question, prior_answer) in enumerate(usable, start=1):
        lines.append(f"Q{index}: {prior_question}")
        lines.append(f"A{index}: {prior_answer}")
    lines.append("")
    lines.append(f"Answer only this follow-up question, resolving any references to the above: {question.strip()}")
    return "\n".join(lines)
