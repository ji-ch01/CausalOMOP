"""Shared types and helpers for LLM provider adapters."""

from __future__ import annotations

import re
from typing import Callable


LLMCallable = Callable[[str], str]


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


_JSON_CODE_FENCE = re.compile(
    r"^\s*```[ \t]*(?:json|JSON)?[ \t]*\n?(.*?)\n?\s*```\s*$",
    re.DOTALL,
)


def strip_json_code_fence(text: str) -> str:
    """Best-effort unwrap of a ```json ... ``` block around a JSON payload.

    OpenAI models honour ``response_format`` and return a bare object, but some
    other OpenRouter models still wrap it in a Markdown fence even when JSON
    mode is requested. Every downstream parser calls ``json.loads`` directly,
    so the adapters normalise the response here. A bare payload is returned
    unchanged.
    """
    match = _JSON_CODE_FENCE.match(text)
    return match.group(1).strip() if match else text
