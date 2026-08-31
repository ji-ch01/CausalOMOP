"""OpenRouter adapter for the chat model behind the governed causal workflow.

The adapter exposes the small callable contract the agents' LLM bridge expects
(``str -> str``, returning strict JSON). Any OpenRouter chat model can be
selected through ``CAUSALOMOP_LLM_MODEL`` as a ``vendor/model`` slug --
``openai/gpt-5`` (default), ``openai/o3``, ``anthropic/claude-opus-5``,
``google/gemini-2.5-pro`` -- without a code change. Clinical questions are sent
to the model, but database rows and patient-level data must never be included
in this call.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from causal.llm.base import load_dotenv_if_available, strip_json_code_fence


# OpenRouter model ids are always ``vendor/model`` and may carry a ``:variant``
# suffix (e.g. ``deepseek/deepseek-r1:free``).
_MODEL_SLUG = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._:-]+$")
_REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high"})


def _float_env(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class OpenRouterGPTConfig:
    """Runtime settings for the OpenAI-compatible OpenRouter endpoint."""

    model: str = "openai/gpt-5"
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_seconds: float = 180.0
    # A drafting call carries ~15+ retrieved abstracts, and on a reasoning
    # model (gpt-5, o3, ...) most of this budget can go to an internal
    # reasoning pass before any JSON is emitted. 2500, then 6000, then 12000
    # were each observed truncating a stage mid-response and silently falling
    # back to an empty candidate_dag; 20000 gives current reasoning models
    # room. Override per model with CAUSALOMOP_LLM_MAX_TOKENS.
    max_completion_tokens: int = 20000
    # Optional OpenRouter reasoning-effort hint (minimal|low|medium|high).
    # Left unset, the model uses its own default. Lowering it stops a
    # reasoning model from spending the whole token budget on analysis;
    # set via CAUSALOMOP_LLM_REASONING_EFFORT.
    reasoning_effort: str | None = None

    @classmethod
    def from_env(cls, model: str | None = None) -> "OpenRouterGPTConfig":
        load_dotenv_if_available()
        effort = (os.getenv("CAUSALOMOP_LLM_REASONING_EFFORT") or "").strip().lower()
        return cls(
            model=model or os.getenv("CAUSALOMOP_LLM_MODEL", cls.model),
            base_url=os.getenv(
                "OPENROUTER_BASE_URL",
                os.getenv("OPENAI_BASE_URL", cls.base_url),
            ).rstrip("/"),
            timeout_seconds=_float_env("CAUSALOMOP_LLM_TIMEOUT", cls.timeout_seconds),
            max_completion_tokens=_int_env(
                "CAUSALOMOP_LLM_MAX_TOKENS", cls.max_completion_tokens
            ),
            reasoning_effort=effort or None,
        ).validated()

    def validated(self) -> "OpenRouterGPTConfig":
        if not _MODEL_SLUG.match(self.model or ""):
            raise ValueError(
                "CAUSALOMOP_LLM_MODEL must be an OpenRouter 'vendor/model' slug, "
                "e.g. openai/gpt-5 or anthropic/claude-opus-5"
            )
        if not self.base_url.startswith("https://"):
            raise ValueError("The LLM base URL must use HTTPS")
        if self.max_completion_tokens <= 0:
            raise ValueError("CAUSALOMOP_LLM_MAX_TOKENS must be a positive integer")
        if (
            self.reasoning_effort is not None
            and self.reasoning_effort not in _REASONING_EFFORTS
        ):
            raise ValueError(
                "CAUSALOMOP_LLM_REASONING_EFFORT must be one of: "
                + ", ".join(sorted(_REASONING_EFFORTS))
            )
        return self


def build_request_kwargs(config: OpenRouterGPTConfig, prompt: str) -> dict:
    """The exact keyword arguments passed to ``chat.completions.create``.

    Split out so the request shape (JSON mode, token ceiling, optional
    reasoning-effort hint) is unit-testable without a live client.
    """
    kwargs: dict = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": config.max_completion_tokens,
    }
    if config.reasoning_effort is not None:
        kwargs["extra_body"] = {"reasoning": {"effort": config.reasoning_effort}}
    return kwargs


class OpenRouterGPTClient:
    """Callable OpenAI SDK wrapper that requests a strict JSON response."""

    def __init__(self, config: OpenRouterGPTConfig | None = None) -> None:
        self.config = (config or OpenRouterGPTConfig.from_env()).validated()
        load_dotenv_if_available()
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY (or OPENAI_API_KEY) in .env")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the LLM dependencies with: pip install -r requirements-llm.txt"
            ) from exc
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
            max_retries=2,
            default_headers={"X-OpenRouter-Title": "CausalOMOP"},
        )

    def __call__(self, prompt: str) -> str:
        completion = self._client.chat.completions.create(
            **build_request_kwargs(self.config, prompt)
        )
        content = completion.choices[0].message.content if completion.choices else None
        if not content:
            raise RuntimeError("The model returned no JSON content")
        return strip_json_code_fence(content)
