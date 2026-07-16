"""Small helpers for reproducible agent artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any


def artifact_metadata(agent_name: str, agent_version: str, inputs: Any) -> dict[str, str]:
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {
        "agent_name": agent_name,
        "agent_version": agent_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
