"""OpenRouter adapter for real OpenAI GPT models.

The adapter intentionally exposes the small callable contract expected by the
Hypothesis Agent bridge. Clinical questions are sent to the model, but database
rows and patient-level data must never be included in this call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from causal.llm.base import load_dotenv_if_available


@dataclass(frozen=True)
class OpenRouterGPTConfig:
    """Runtime settings for the OpenAI-compatible OpenRouter endpoint."""

    model: str = "openai/gpt-oss-120b"
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_seconds: float = 90.0
    max_completion_tokens: int = 2500

    @classmethod
    def from_env(cls, model: str | None = None) -> "OpenRouterGPTConfig":
        load_dotenv_if_available()
        return cls(
            model=model or os.getenv("CAUSALOMOP_LLM_MODEL", cls.model),
            base_url=os.getenv(
                "OPENROUTER_BASE_URL",
                os.getenv("OPENAI_BASE_URL", cls.base_url),
            ).rstrip("/"),
        ).validated()

    def validated(self) -> "OpenRouterGPTConfig":
        if not self.model.startswith("openai/gpt-"):
            raise ValueError("CausalOMOP currently requires an OpenAI GPT model on OpenRouter")
        if not self.base_url.startswith("https://"):
            raise ValueError("The LLM base URL must use HTTPS")
        return self


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
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_completion_tokens=self.config.max_completion_tokens,
        )
        content = completion.choices[0].message.content if completion.choices else None
        if not content:
            raise RuntimeError("The GPT model returned no JSON content")
        return content
