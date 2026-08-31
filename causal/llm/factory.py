"""Environment-aware selection between the OpenRouter and local LLM providers.

`CAUSALOMOP_ENV=dev` (the default) allows the OpenRouter provider for
development, where no on-premise model is available yet; the specific model is
`CAUSALOMOP_LLM_MODEL` (any `vendor/model` slug). `CAUSALOMOP_ENV=prod`
requires the local, on-premise provider and refuses to start if a cloud
provider is still configured — clinical questions must never reach a cloud
API once this runs against real SNUH data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from causal.llm.base import LLMCallable, load_dotenv_if_available
from causal.llm.local import LocalLLMClient, LocalLLMConfig
from causal.llm.openrouter import OpenRouterGPTClient, OpenRouterGPTConfig


Provider = Literal["openrouter", "local"]

_CLOUD_KEY_ENV_VARS = ("OPENROUTER_API_KEY", "OPENAI_API_KEY")


@dataclass(frozen=True)
class ResolvedLLM:
    """A ready-to-call LLM together with the provenance the workflow records."""

    call: LLMCallable
    provider: Provider
    model: str


def resolve_provider(env: str | None = None) -> Provider:
    """Pick a provider name without instantiating any client or network call."""
    load_dotenv_if_available()
    resolved_env = (env or os.getenv("CAUSALOMOP_ENV", "dev")).strip().lower()
    if resolved_env not in {"dev", "prod"}:
        raise ValueError("CAUSALOMOP_ENV must be 'dev' or 'prod'")
    if resolved_env == "prod":
        configured = [name for name in _CLOUD_KEY_ENV_VARS if os.getenv(name)]
        if configured:
            raise RuntimeError(
                "CAUSALOMOP_ENV=prod forbids a cloud LLM provider, but found "
                f"{', '.join(configured)} set. Remove it and configure LOCAL_LLM_* "
                "for the on-premise provider instead."
            )
        return "local"
    return "openrouter"


def build_hypothesis_llm(env: str | None = None, model: str | None = None) -> ResolvedLLM:
    """Instantiate the provider the Hypothesis Agent's LLM bridge should call."""
    provider = resolve_provider(env)
    if provider == "local":
        config = LocalLLMConfig.from_env()
        return ResolvedLLM(call=LocalLLMClient(config), provider="local", model=config.model)
    config = OpenRouterGPTConfig.from_env(model)
    return ResolvedLLM(call=OpenRouterGPTClient(config), provider="openrouter", model=config.model)
