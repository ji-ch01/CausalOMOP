"""Deterministic faithfulness verification of a parsed results-narrative LLM
response against the same JSON artifacts it was shown.

Deliberately does NOT call a second LLM -- every check here is a plain
string/number comparison over `NarrativeInput.to_dict()` (the exact JSON the
model saw) and the parsed `NarrativeResult`. This is the enforcement half of
`causal.agents.results_narrative`'s prompt rules: the prompt *asks* the model
not to invent numbers, get the direction wrong, misread the confidence
interval, drop a diagnostic warning, or contradict a stated assumption --
this module checks whether it actually complied, so approval never has to
take the model's own compliance on faith.

A failed check never blocks the causal workflow itself (statistical
estimation and the `results` gate are entirely untouched by this module);
it only blocks `WorkflowManager.review_results_narrative`'s automatic
approval path, per `docs/causal_roadmap.md`'s LLM-boundary rule that an
advisory LLM step can never approve anything.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from causal.agents.results_narrative import ALLOWED_SOURCE_FIELD_ROOTS, Claim, NarrativeInput, NarrativeResult

_STATUS_RANK = {"passed": 0, "warning": 1, "failed": 2}


def _worse(a: str, b: str) -> str:
    return a if _STATUS_RANK[a] >= _STATUS_RANK[b] else b


@dataclass(frozen=True)
class FaithfulnessCheck:
    subject: str
    claim_type: str
    text: str
    status: str
    reason: str
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FaithfulnessReport:
    status: str
    checks: tuple[FaithfulnessCheck, ...]
    checked_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "checks": [check.to_dict() for check in self.checks],
            "checked_at_utc": self.checked_at_utc,
        }


def verify_narrative(narrative: NarrativeResult, narrative_input: NarrativeInput) -> FaithfulnessReport:
    """Run every deterministic check against `narrative_input.to_dict()` --
    the same dict `build_narrative_prompt` serialized into the prompt, so
    "present in the artifacts" always means "present in what the model was
    actually shown", never the wider run directory."""
    data = narrative_input.to_dict()
    allowed_numbers = _collect_number_tokens(data)

    estimation = data.get("estimation") or {}
    estimand = estimation.get("estimand") or {}
    point_estimate = estimand.get("point_estimate")
    ci_lower = estimand.get("ci_lower")
    ci_upper = estimand.get("ci_upper")
    effect_measure = estimand.get("effect_measure") or ""
    diagnostics = data.get("diagnostics") or {}
    assumptions = data.get("assumptions_and_limitations") or ()

    checks: list[FaithfulnessCheck] = [_check_narrative_text_numbers(narrative.narrative_text, allowed_numbers)]
    checks.extend(
        _check_claim(index, claim, data, allowed_numbers, point_estimate, ci_lower, ci_upper, effect_measure)
        for index, claim in enumerate(narrative.claims)
    )
    checks.append(_check_diagnostic_warning_coverage(narrative, diagnostics))
    checks.append(_check_assumptions_coverage(narrative, assumptions))

    overall = "passed"
    for check in checks:
        overall = _worse(overall, check.status)
    return FaithfulnessReport(
        status=overall, checks=tuple(checks), checked_at_utc=datetime.now(timezone.utc).isoformat()
    )


# --- number-literal check ---------------------------------------------------

_NUMBER_RE = re.compile(r"-?\d+\.\d+|-?\d+")


def _iter_leaves(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_leaves(item)
    else:
        yield value


def _collect_number_tokens(data: Any) -> set[str]:
    tokens = set(_NUMBER_RE.findall(json.dumps(data, sort_keys=True, ensure_ascii=False)))
    # A fractional field in [0, 1] (e.g. estimand.ci_level = 0.95) is
    # conventionally spoken of as a percentage ("95% CI"). Without this, any
    # narrative phrasing the fixed 0.95 CI level as "95%" would be flagged
    # as an invented number even though it is a faithful restatement, not a
    # fabrication -- so both spellings are accepted.
    for value in _iter_leaves(data):
        if isinstance(value, float) and 0.0 <= value <= 1.0:
            tokens.add(str(round(value * 100)))
    return tokens


def _check_numbers(text: str, allowed_numbers: set[str]) -> tuple[str, str]:
    found = _NUMBER_RE.findall(text)
    invented = sorted({token for token in found if token not in allowed_numbers})
    if invented:
        return "failed", f"number(s) not found verbatim in the source artifacts: {', '.join(invented)}"
    return "passed", ""


def _check_narrative_text_numbers(text: str, allowed_numbers: set[str]) -> FaithfulnessCheck:
    status, reason = _check_numbers(text, allowed_numbers)
    return FaithfulnessCheck(
        subject="narrative_text",
        claim_type="",
        text=text,
        status=status,
        reason=reason or "Every number in the narrative text appears verbatim in the source artifacts.",
        evidence="",
    )


# --- source_field allow-list / resolution -----------------------------------

_MISSING = object()
_SEGMENT_RE = re.compile(r"^([^\[\]]*)((?:\[\d+\])*)$")


def _root_of(field: str) -> str:
    return field.split(".", 1)[0].split("[", 1)[0]


def _resolve_path(data: Any, path: str) -> Any:
    current = data
    for part in path.split("."):
        match = _SEGMENT_RE.match(part)
        if not match:
            return _MISSING
        key, index_group = match.group(1), match.group(2)
        if key:
            if not isinstance(current, dict) or key not in current:
                return _MISSING
            current = current[key]
        for index_str in re.findall(r"\[(\d+)\]", index_group):
            index = int(index_str)
            if not isinstance(current, (list, tuple)) or index >= len(current):
                return _MISSING
            current = current[index]
    return current


# --- direction / uncertainty keyword heuristics -----------------------------
#
# These are deliberately simple English keyword lookups, not NLP --
# claim_type is already model-declared and structured, so this only has to
# catch a claim whose free-text direction/uncertainty wording contradicts
# the number it is about. A claim using no recognized keyword, or both
# directions at once, is graded "warning" (could not verify) rather than
# "failed" -- silence about a heuristic's own blind spot, not a false
# accusation of an error, matches this module's job of enforcement without
# a second LLM to fall back on for genuinely ambiguous phrasing. The prompt
# (`causal/prompts/results_narrative.md`) requires English output, so this
# list does not need to cover other languages.

_INCREASE_WORDS = (
    "increase", "higher", "greater", "elevated", "more likely", "raised", "worse", "harmful",
)
_DECREASE_WORDS = (
    "decrease", "lower", "reduced", "reduction", "less likely", "declined", "protective", "beneficial",
    "improved",
)
_NULL_EFFECT_WORDS = ("no difference", "no effect", "similar", "comparable", "negligible")

_CROSSES_NULL_WORDS = (
    "crosses the null", "crosses zero", "not statistically significant", "not significant",
    "includes the null", "includes zero", "consistent with no effect", "imprecise", "uncertain",
)
_EXCLUDES_NULL_WORDS = (
    "does not cross", "excludes the null", "excludes zero", "statistically significant",
)


def _check_direction(text: str, point_estimate: float) -> tuple[str, str]:
    lowered = text.lower()
    has_increase = any(word in lowered for word in _INCREASE_WORDS)
    has_decrease = any(word in lowered for word in _DECREASE_WORDS)
    has_null = any(word in lowered for word in _NULL_EFFECT_WORDS)
    if has_increase and has_decrease:
        return "warning", "claim text mixes increase- and decrease-direction language; direction could not be verified"
    if not has_increase and not has_decrease and not has_null:
        return "warning", "claim text does not use a recognized direction keyword; direction could not be verified"
    if has_increase and not point_estimate > 0:
        return "failed", f"claim states an increase but the point estimate is {point_estimate!r} (not positive)"
    if has_decrease and not point_estimate < 0:
        return "failed", f"claim states a decrease but the point estimate is {point_estimate!r} (not negative)"
    return "passed", ""


def _check_uncertainty(text: str, ci_lower: float, ci_upper: float, effect_measure: str) -> tuple[str, str]:
    null_value = 1.0 if "ratio" in effect_measure.lower() else 0.0
    crosses_null = ci_lower <= null_value <= ci_upper
    lowered = text.lower()
    says_crosses = any(word in lowered for word in _CROSSES_NULL_WORDS)
    says_excludes = any(word in lowered for word in _EXCLUDES_NULL_WORDS)
    if says_crosses and says_excludes:
        return "warning", "claim text mixes both crossing-null and excluding-null language; uncertainty could not be verified"
    if not says_crosses and not says_excludes:
        return "warning", "claim text does not use a recognized uncertainty keyword; could not be verified"
    if crosses_null and says_excludes:
        return (
            "failed",
            f"claim implies the effect excludes the null, but the CI [{ci_lower!r}, {ci_upper!r}] "
            f"includes {null_value!r}",
        )
    if not crosses_null and says_crosses:
        return (
            "failed",
            f"claim implies the CI crosses the null, but [{ci_lower!r}, {ci_upper!r}] does not include "
            f"{null_value!r}",
        )
    return "passed", ""


def _check_claim(
    index: int,
    claim: Claim,
    data: dict[str, Any],
    allowed_numbers: set[str],
    point_estimate: float | None,
    ci_lower: float | None,
    ci_upper: float | None,
    effect_measure: str,
) -> FaithfulnessCheck:
    reasons: list[str] = []
    evidence: list[str] = []
    status = "passed"

    if not claim.source_fields:
        status = _worse(status, "failed")
        reasons.append("no source_fields were provided")
    for field in claim.source_fields:
        if _root_of(field) not in ALLOWED_SOURCE_FIELD_ROOTS:
            status = _worse(status, "failed")
            reasons.append(f"source_field '{field}' is outside the allow-listed artifact roots")
            continue
        value = _resolve_path(data, field)
        if value is _MISSING:
            status = _worse(status, "failed")
            reasons.append(f"source_field '{field}' does not resolve inside the artifacts shown to the model")
        else:
            evidence.append(f"{field} = {value!r}")

    number_status, number_reason = _check_numbers(claim.text, allowed_numbers)
    status = _worse(status, number_status)
    if number_reason:
        reasons.append(number_reason)

    if claim.claim_type == "effect_direction" and isinstance(point_estimate, (int, float)):
        direction_status, direction_reason = _check_direction(claim.text, float(point_estimate))
        status = _worse(status, direction_status)
        if direction_reason:
            reasons.append(direction_reason)
    elif (
        claim.claim_type == "uncertainty"
        and isinstance(ci_lower, (int, float))
        and isinstance(ci_upper, (int, float))
    ):
        uncertainty_status, uncertainty_reason = _check_uncertainty(
            claim.text, float(ci_lower), float(ci_upper), effect_measure
        )
        status = _worse(status, uncertainty_status)
        if uncertainty_reason:
            reasons.append(uncertainty_reason)

    return FaithfulnessCheck(
        subject=f"claim_{index}",
        claim_type=claim.claim_type,
        text=claim.text,
        status=status,
        reason="; ".join(reasons) if reasons else "All deterministic checks passed.",
        evidence="; ".join(evidence),
    )


# --- diagnostics / assumptions coverage -------------------------------------

_IMBALANCE_KEYWORDS = ("balance", "imbalanc", "smd", "covariate", "confound")


def _check_diagnostic_warning_coverage(narrative: NarrativeResult, diagnostics: dict[str, Any]) -> FaithfulnessCheck:
    n_flagged = diagnostics.get("n_covariates_flagged") or 0
    if not n_flagged:
        return FaithfulnessCheck(
            subject="diagnostic_warning_coverage", claim_type="", text="",
            status="passed", reason="No flagged covariates to report.", evidence="",
        )
    has_typed_claim = any(claim.claim_type == "diagnostic_warning" for claim in narrative.claims)
    combined = (narrative.narrative_text + " " + " ".join(claim.text for claim in narrative.claims)).lower()
    mentions_imbalance = any(keyword in combined for keyword in _IMBALANCE_KEYWORDS)
    evidence = f"diagnostics.n_covariates_flagged = {n_flagged!r}"
    if has_typed_claim or mentions_imbalance:
        return FaithfulnessCheck(
            subject="diagnostic_warning_coverage", claim_type="", text="",
            status="passed", reason=f"{n_flagged} flagged covariate(s) are acknowledged.", evidence=evidence,
        )
    return FaithfulnessCheck(
        subject="diagnostic_warning_coverage", claim_type="", text="",
        status="failed",
        reason=(
            f"diagnostics flagged {n_flagged} covariate(s) for imbalance, but neither the narrative nor "
            "any claim mentions it"
        ),
        evidence=evidence,
    )


_ASSUMPTION_CONTRADICTION_KEYWORDS = (
    "no confounding", "unconfounded", "proves causation", "confirms causation", "eliminates all bias",
    "definitively causal",
)


def _check_assumptions_coverage(narrative: NarrativeResult, assumptions: tuple[str, ...]) -> FaithfulnessCheck:
    combined = (narrative.narrative_text + " " + " ".join(claim.text for claim in narrative.claims)).lower()
    contradiction = next((keyword for keyword in _ASSUMPTION_CONTRADICTION_KEYWORDS if keyword in combined), None)
    if contradiction:
        return FaithfulnessCheck(
            subject="assumptions_coverage", claim_type="", text="",
            status="failed",
            reason=f"narrative/claims contain language contradicting a stated assumption or limitation ('{contradiction}')",
            evidence="assumptions_and_limitations is non-empty" if assumptions else "",
        )
    if not assumptions:
        return FaithfulnessCheck(
            subject="assumptions_coverage", claim_type="", text="",
            status="passed", reason="No assumptions/limitations recorded.", evidence="",
        )
    has_typed_claim = any(claim.claim_type == "assumption_or_limitation" for claim in narrative.claims)
    if has_typed_claim:
        return FaithfulnessCheck(
            subject="assumptions_coverage", claim_type="", text="",
            status="passed", reason="At least one assumption_or_limitation claim is present.",
            evidence=f"{len(assumptions)} assumption(s)/limitation(s) recorded",
        )
    return FaithfulnessCheck(
        subject="assumptions_coverage", claim_type="", text="",
        status="warning",
        reason="No assumption_or_limitation claim was provided even though assumptions/limitations are recorded.",
        evidence=f"{len(assumptions)} assumption(s)/limitation(s) recorded",
    )
