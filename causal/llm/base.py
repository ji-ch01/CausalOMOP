"""Shared types and helpers for LLM provider adapters."""

from __future__ import annotations

from typing import Callable


LLMCallable = Callable[[str], str]


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()
