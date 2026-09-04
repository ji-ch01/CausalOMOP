"""Deterministic, communication-ready rounding + derived-quantity layer for
the results-narrative step.

`causal.agents.results_narrative` feeds the results-narrative LLM the raw
`estimation.json` numbers, which carry full floating-point precision
(`point_estimate = 0.0418732...`). The prompt used to forbid the model from
rounding them, because `causal.agents.results_faithfulness` rejects any
number in the narrative that is not a verbatim token of the JSON it was
shown -- so the narrative was faithful but unreadable, and offered a
clinician none of the plain restatements (events per 1000, relative risk
reduction, number needed to treat) they actually reason with.

This module closes that gap deterministically. It rounds every reported
quantity to a sensible communication precision and derives the plain
clinical restatements as explicit fields the model copies rather than
computes. Nothing here estimates anything: it is arithmetic on numbers
already in `estimation.json` / `diagnostics.json` / `sensitivity_analysis.
json`. Every number it emits -- in a scalar field or inside a
`plain_statement` string -- therefore becomes an allowed verbatim token for
`verify_narrative`, so the narrative can quote a rounded value and still
pass faithfulness verification.

The raw high-precision fields stay in `estimation.json` for audit; the
`narrative_display` block this builds is only what the narrative step and
`build_causal_report` quote for humans.
"""

from __future__ import annotations

import math
from typing import Any

# Precision, per scale. Risk differences are reported to a clinician as
# whole events per 1000 patients (the headline) and to 4 decimals on the
# probability scale (for a reader who wants the raw contrast); risk ratios
# and diagnostics to 2 decimals, matching how they are conventionally
# written up.
_RISK_DIFFERENCE_DECIMALS = 4
_PERCENTAGE_POINT_DECIMALS = 1
_RATIO_DECIMALS = 2
_DIAGNOSTIC_DECIMALS = 2
_EVENTS_PER = 1000

ROUNDING_NOTE = (
    "Every number in this narrative_display block has already been rounded "
    "for communication, and every plain_statement has already been written "
    "by deterministic code from the unrounded values in estimation.json. "
    "Quote these values and these sentences as they stand. Do not round, "
    "recompute, rescale, combine, or derive any number of your own -- if a "
    "figure you want is not in this block, describe it in words instead."
)

EFFECT_SCALE_EXPLAINER = (
    "The primary estimate is a risk difference: the absolute change in the "
    "probability of the outcome attributable to treatment versus the "
    "comparator, under the study's assumptions. It is reported below as "
    f"events per {_EVENTS_PER} patients and as percentage points. The risk "
    "ratio is the same effect on the relative scale (a ratio of one means no "
    "effect; below one is lower risk with treatment, above one is higher)."
)


def display_or_dash(value: Any) -> Any:
    """Render a `narrative_display` scalar for a report bullet: a number
    as-is (already rounded by `build_narrative_display`), and `None` or a
    non-finite float as an em-dash placeholder."""
    if value is None:
        return "-"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "-"
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not (
        isinstance(value, float) and (math.isnan(value) or math.isinf(value))
    )


def _round(value: Any, digits: int) -> float | None:
    if not _is_number(value):
        return None
    return round(float(value), digits)


def _per_n(value: Any, n: int = _EVENTS_PER) -> int | None:
    if not _is_number(value):
        return None
    return int(round(float(value) * n))


def _abs(value: int | None) -> int | None:
    return None if value is None else abs(value)


def _direction_word(point: float) -> str:
    if point > 0:
        return "more"
    if point < 0:
        return "fewer"
    return "no change"


def _higher_lower(point: float) -> str:
    """Phrase the direction with a keyword `results_faithfulness._check_
    direction` recognises (`higher` / `lower`), so an `effect_direction`
    claim that echoes the plain_statement verifies cleanly."""
    if point > 0:
        return "a higher risk of the outcome"
    if point < 0:
        return "a lower risk of the outcome"
    return "no clear difference in the risk of the outcome"


def _phrase_events_interval(lower_per_n: int, upper_per_n: int) -> str:
    """`lower_per_n <= upper_per_n` always (the CI is on the risk-difference
    scale). Render the pair the way it is spoken: both negative -> "X to Y
    fewer", both positive -> "X to Y more", straddling zero -> "X fewer to Y
    more"."""
    if lower_per_n < 0 < upper_per_n:
        return f"{abs(lower_per_n)} fewer to {upper_per_n} more"
    if lower_per_n >= 0 and upper_per_n >= 0:
        return f"{lower_per_n} to {upper_per_n} more" if lower_per_n != upper_per_n else f"{lower_per_n} more"
    # both <= 0
    return (
        f"{abs(upper_per_n)} to {abs(lower_per_n)} fewer"
        if lower_per_n != upper_per_n
        else f"{abs(lower_per_n)} fewer"
    )


def _absolute_effect(estimand: dict[str, Any]) -> dict[str, Any]:
    point = estimand.get("point_estimate")
    lower = estimand.get("ci_lower")
    upper = estimand.get("ci_upper")
    ci_level = estimand.get("ci_level", 0.95)
    ci_level_pct = int(round(float(ci_level) * 100)) if _is_number(ci_level) else 95

    if not _is_number(point):
        return {"available": False, "reason": "No point estimate is present in estimation.json."}

    point = float(point)
    per_n_point = _per_n(point)
    per_n_lower = _per_n(lower)
    per_n_upper = _per_n(upper)
    crosses = _is_number(lower) and _is_number(upper) and float(lower) <= 0.0 <= float(upper)

    block: dict[str, Any] = {
        "available": True,
        "scale": "risk difference (absolute change in outcome probability)",
        "events_per_n_patients": _EVENTS_PER,
        "risk_difference": _round(point, _RISK_DIFFERENCE_DECIMALS),
        "ci_lower": _round(lower, _RISK_DIFFERENCE_DECIMALS),
        "ci_upper": _round(upper, _RISK_DIFFERENCE_DECIMALS),
        "ci_level_pct": ci_level_pct,
        "percentage_points_point": _round(point * 100, _PERCENTAGE_POINT_DECIMALS),
        "percentage_points_ci_lower": _round(lower * 100, _PERCENTAGE_POINT_DECIMALS) if _is_number(lower) else None,
        "percentage_points_ci_upper": _round(upper * 100, _PERCENTAGE_POINT_DECIMALS) if _is_number(upper) else None,
        "events_per_n_point": per_n_point,
        "events_per_n_ci_lower": per_n_lower,
        "events_per_n_ci_upper": per_n_upper,
        "direction": _direction_word(point),
        "interval_includes_no_effect": crosses,
    }

    magnitude = f"about {abs(per_n_point)} {_direction_word(point)} outcome events per {_EVENTS_PER} patients treated"
    if per_n_point == 0:
        magnitude = f"almost no change in outcome events per {_EVENTS_PER} patients treated"

    if per_n_lower is not None and per_n_upper is not None:
        interval = (
            f"the {ci_level_pct}% compatibility interval runs from "
            f"{_phrase_events_interval(per_n_lower, per_n_upper)} per {_EVENTS_PER}"
        )
        if crosses:
            block["plain_statement"] = (
                f"Taken at face value, treatment was associated with {_higher_lower(point)} "
                f"({magnitude}), but {interval} -- it includes the possibility of no effect, "
                "so this study on its own cannot distinguish a real effect of this size from none."
            )
        else:
            block["plain_statement"] = (
                f"Treatment was associated with {_higher_lower(point)} ({magnitude}); "
                f"{interval}, which does not include no effect."
            )
    else:
        block["plain_statement"] = (
            f"Treatment was associated with {_higher_lower(point)} ({magnitude}). "
            "No compatibility interval was available for this estimate."
        )
    return block


def _relative_effect(estimand: dict[str, Any]) -> dict[str, Any]:
    rr = estimand.get("risk_ratio_point_estimate")
    rr_lower = estimand.get("risk_ratio_ci_lower")
    rr_upper = estimand.get("risk_ratio_ci_upper")

    if not _is_number(rr):
        return {
            "available": False,
            "reason": (
                "The risk ratio could not be computed for this cohort (a standardized "
                "arm risk was zero), so only the absolute risk-difference scale is available."
            ),
        }

    rr = float(rr)
    relative_change_pct = int(round((rr - 1.0) * 100))
    block: dict[str, Any] = {
        "available": True,
        "scale": "risk ratio (relative change; a ratio of one means no effect)",
        "risk_ratio": _round(rr, _RATIO_DECIMALS),
        "ci_lower": _round(rr_lower, _RATIO_DECIMALS),
        "ci_upper": _round(rr_upper, _RATIO_DECIMALS),
        "relative_change_pct": relative_change_pct,
    }
    if relative_change_pct > 0:
        change = f"a {relative_change_pct}% higher relative risk of the outcome"
    elif relative_change_pct < 0:
        change = f"a {abs(relative_change_pct)}% lower relative risk of the outcome"
    else:
        change = "essentially no change in the relative risk of the outcome"

    if _is_number(rr_lower) and _is_number(rr_upper):
        block["plain_statement"] = (
            f"On the relative scale this is a risk ratio of {_round(rr, _RATIO_DECIMALS)} "
            f"({change}), with a compatibility interval of "
            f"{_round(rr_lower, _RATIO_DECIMALS)} to {_round(rr_upper, _RATIO_DECIMALS)}."
        )
    else:
        block["plain_statement"] = (
            f"On the relative scale this is a risk ratio of {_round(rr, _RATIO_DECIMALS)} "
            f"({change}); no interval was available for the risk ratio."
        )
    return block


def _number_needed_to_treat(estimand: dict[str, Any]) -> dict[str, Any]:
    point = estimand.get("point_estimate")
    lower = estimand.get("ci_lower")
    upper = estimand.get("ci_upper")

    if not _is_number(point) or float(point) == 0.0:
        return {
            "reported": False,
            "reason": "The risk difference is zero or unavailable, so a number-needed-to-treat is undefined.",
        }
    crosses = _is_number(lower) and _is_number(upper) and float(lower) <= 0.0 <= float(upper)
    if crosses:
        return {
            "reported": False,
            "reason": (
                "The compatibility interval includes no effect, so a single number-needed-to-treat "
                "would misrepresent the uncertainty and is not reported."
            ),
        }
    nnt = int(math.ceil(1.0 / abs(float(point))))
    if float(point) < 0:
        statement = (
            f"About 1 fewer outcome event for every {nnt} patients treated instead of given the "
            "comparator, if the point estimate is taken at face value."
        )
    else:
        statement = (
            f"About 1 additional outcome event for every {nnt} patients treated instead of given "
            "the comparator, if the point estimate is taken at face value."
        )
    return {"reported": True, "patients_per_event": nnt, "plain_statement": statement}


def _model_fit(estimand: dict[str, Any]) -> dict[str, Any]:
    auc = _round(estimand.get("outcome_model_auc"), _DIAGNOSTIC_DECIMALS)
    brier = _round(estimand.get("outcome_model_brier"), _DIAGNOSTIC_DECIMALS)
    if auc is None and brier is None:
        return {"available": False}
    parts = []
    if auc is not None:
        parts.append(
            f"an outcome-model AUC of {auc} (higher means the model better separated who did and "
            "did not have the outcome; around a half would be no better than chance)"
        )
    if brier is not None:
        parts.append(f"a Brier score of {brier} (lower means better-calibrated predicted risks)")
    return {
        "available": True,
        "outcome_model_auc": auc,
        "outcome_model_brier": brier,
        "plain_statement": (
            "The outcome model used for standardization had " + " and ".join(parts) + "."
        ),
    }


def _covariate_balance(diagnostics: dict[str, Any]) -> dict[str, Any]:
    n_flagged = diagnostics.get("n_covariates_flagged") or 0
    balance = diagnostics.get("balance") or ()
    threshold = _round(diagnostics.get("smd_flag_threshold", 0.1), _DIAGNOSTIC_DECIMALS)
    smds = [abs(float(entry["smd"])) for entry in balance if _is_number(entry.get("smd"))]
    largest = _round(max(smds), _DIAGNOSTIC_DECIMALS) if smds else None
    block: dict[str, Any] = {
        "n_covariates_flagged": int(n_flagged),
        "n_covariates_checked": len(balance),
        "smd_flag_threshold": threshold,
        "largest_absolute_smd": largest,
    }
    if n_flagged:
        block["plain_statement"] = (
            f"Between the treatment groups in the analytic cohort (crude, before outcome-model "
            f"adjustment), {int(n_flagged)} of {len(balance)} covariates were imbalanced "
            f"(standardized mean difference above {threshold}"
            + (f", largest {largest}" if largest is not None else "")
            + "); residual confounding from those covariates cannot be excluded."
        )
    else:
        block["plain_statement"] = (
            f"All {len(balance)} covariates were balanced between the treatment groups in the "
            f"analytic cohort (crude, before outcome-model adjustment; standardized mean "
            f"difference at or below {threshold})."
        )
    return block


def _sensitivity(sensitivity: dict[str, Any]) -> dict[str, Any]:
    e_point = _round(sensitivity.get("e_value_point"), _RATIO_DECIMALS)
    e_ci = _round(sensitivity.get("e_value_ci"), _RATIO_DECIMALS)
    if e_point is None:
        return {"available": False}
    block: dict[str, Any] = {
        "available": True,
        "e_value_point": e_point,
        "e_value_ci": e_ci,
    }
    statement = (
        f"An unmeasured confounder would need to be associated with both treatment and outcome "
        f"by a risk ratio of at least {e_point} each, beyond the measured covariates, to explain "
        "away the point estimate"
    )
    if e_ci is not None:
        statement += f", or at least {e_ci} to move the confidence interval to no effect"
    block["plain_statement"] = statement + "."
    return block


def build_narrative_display(
    estimation: dict[str, Any] | None,
    diagnostics: dict[str, Any] | None = None,
    sensitivity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the `narrative_display` block from already-computed artifacts.

    `estimation` is the parsed `estimation.json`; `diagnostics` the parsed
    `diagnostics.json`; `sensitivity` the parsed `sensitivity_analysis.json`
    (any may be `None`/empty). Pure and deterministic -- same inputs, same
    block -- so `build_causal_report` and `_results_narrative_input` can each
    call it and stay in lockstep.
    """
    estimand = (estimation or {}).get("estimand") or {}
    block: dict[str, Any] = {
        "rounding_note": ROUNDING_NOTE,
        "effect_scale_explainer": EFFECT_SCALE_EXPLAINER,
        "absolute_effect": _absolute_effect(estimand),
        "relative_effect": _relative_effect(estimand),
        "number_needed_to_treat": _number_needed_to_treat(estimand),
        "model_fit": _model_fit(estimand),
    }
    if diagnostics:
        block["covariate_balance"] = _covariate_balance(diagnostics)
    if sensitivity:
        block["unmeasured_confounding_evalue"] = _sensitivity(sensitivity)
    return block
