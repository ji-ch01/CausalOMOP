"""Adapter for an on-premise, OpenAI-compatible model server (e.g. Ollama, vLLM).

This is the required provider when `CAUSALOMOP_ENV=prod`, so that clinical
questions never leave the hospital network. It has not been exercised against
real on-premise infrastructure yet (none was available during development) —
validate it against the actual SNUH deployment before relying on it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from causal.llm.base import load_dotenv_if_available


@dataclass(frozen=True)
class LocalLLMConfig:
    """Runtime settings for a local OpenAI-compatible chat completion endpoint."""

    model: str = "local-model"
    base_url: str = "http://localhost:11434/v1"
    timeout_seconds: float = 120.0
    # Kept in sync with OpenRouterGPTConfig.max_completion_tokens — see that
    # field's comment for why 2500 was raised, then 6000.
    max_completion_tokens: int = 12000

    @classmethod
    def from_env(cls) -> "LocalLLMConfig":
        load_dotenv_if_available()
        return cls(
            model=os.getenv("LOCAL_LLM_MODEL", cls.model),
            base_url=os.getenv("LOCAL_LLM_BASE_URL", cls.base_url).rstrip("/"),
        ).validated()

    def validated(self) -> "LocalLLMConfig":
        if not self.model.strip():
            raise ValueError("LOCAL_LLM_MODEL is required")
        if not (self.base_url.startswith("http://") or self.base_url.startswith("https://")):
            raise ValueError("LOCAL_LLM_BASE_URL must be an http(s) URL")
        return self


class LocalLLMClient:
    """Callable OpenAI SDK wrapper pointed at a local, on-premise model server."""

    def __init__(self, config: LocalLLMConfig | None = None) -> None:
        self.config = (config or LocalLLMConfig.from_env()).validated()
        load_dotenv_if_available()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the LLM dependencies with: pip install -r requirements-llm.txt"
            ) from exc
        self._client = OpenAI(
            api_key=os.getenv("LOCAL_LLM_API_KEY", "not-required"),
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
            max_retries=2,
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
            raise RuntimeError("The local model returned no JSON content")
        return content
