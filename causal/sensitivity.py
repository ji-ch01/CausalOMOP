"""E-value quantitative sensitivity analysis for unmeasured confounding.

VanderWeele TJ, Ding P. Sensitivity Analysis in Observational Research:
Introducing the E-Value. Ann Intern Med. 2017;167(4):268-274.

Plain top-level module (not under `causal/agents/`): pure math with no
`run()`/LLM/gate surface, same tier as `causal.nuisance`. Works identically
for a `causal.agents.estimation.GComputationEstimate` or a
`causal.agents.tmle.TMLEEstimate` -- both report a risk_ratio_point_estimate/
risk_ratio_ci_lower/risk_ratio_ci_upper alongside their primary risk-
difference estimate specifically so this module has a risk ratio to work
with, per the E-value's own risk-ratio-scale formula.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

_CITATION = (
    "VanderWeele TJ, Ding P. Sensitivity Analysis in Observational Research: "
    "Introducing the E-Value. Ann Intern Med. 2017;167(4):268-274."
)

_INTERPRETATION = (
    "The E-value is the minimum strength of association, on the risk-ratio scale, that an "
    "unmeasured confounder would need to have with both the treatment and the outcome -- above "
    "and beyond the measured adjustment set -- to fully explain away the observed association. "
    "It is not a probability that unmeasured confounding exists, and a large E-value does not "
    "rule unmeasured confounding out; it only bounds how strong such a confounder would need to "
    "be. The E-value for the confidence interval uses whichever bound is closer to the null (a "
    "risk ratio of 1); if the interval already includes the null, that E-value is 1 -- the "
    "observed association could already be due to chance alone, so no confounder is needed to "
    "explain it away."
)


@dataclass(frozen=True)
class EValueResult:
    risk_ratio: float
    e_value_point: float
    e_value_ci: float
    citation: str
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _e_value_from_risk_ratio_at_least_one(risk_ratio: float) -> float:
    """VanderWeele & Ding 2017's E-value formula, defined for `risk_ratio >=
    1` (a risk ratio below 1 is inverted by the caller first -- the formula
    is symmetric under that inversion, `e_value(RR) == e_value(1/RR)`, which
    is how a protective (RR<1) association is handled)."""
    if risk_ratio < 1.0:
        raise ValueError("_e_value_from_risk_ratio_at_least_one requires risk_ratio >= 1")
    return risk_ratio + math.sqrt(risk_ratio * (risk_ratio - 1.0))


def e_value(risk_ratio_point: float, risk_ratio_ci_lower: float, risk_ratio_ci_upper: float) -> EValueResult:
    """Compute the E-value for a risk-ratio point estimate and its
    confidence interval (VanderWeele & Ding 2017's own recommended
    reporting convention: both the point estimate and the CI bound closest
    to the null).

    Raises `ValueError` for a non-positive, non-finite (NaN -- a legitimate
    value for `GComputationEstimate.risk_ratio_point_estimate`/`TMLEEstimate.
    risk_ratio_point_estimate` when an arm's standardized risk was exactly
    0, not just a hypothetical input) risk ratio, or an inverted CI. Does
    not itself require a specific CI level -- the caller's `ci_level`
    (typically 0.95) is the level this E-value describes explaining away.
    """
    inputs = (risk_ratio_point, risk_ratio_ci_lower, risk_ratio_ci_upper)
    if any(math.isnan(value) for value in inputs):
        raise ValueError("risk ratios must not be NaN")
    if any(value <= 0 for value in inputs):
        raise ValueError("risk ratios must be positive")
    if risk_ratio_ci_lower > risk_ratio_ci_upper:
        raise ValueError("risk_ratio_ci_lower must not exceed risk_ratio_ci_upper")

    if risk_ratio_point >= 1.0:
        e_point = _e_value_from_risk_ratio_at_least_one(risk_ratio_point)
        e_ci = 1.0 if risk_ratio_ci_lower <= 1.0 else _e_value_from_risk_ratio_at_least_one(risk_ratio_ci_lower)
    else:
        e_point = _e_value_from_risk_ratio_at_least_one(1.0 / risk_ratio_point)
        e_ci = (
            1.0 if risk_ratio_ci_upper >= 1.0
            else _e_value_from_risk_ratio_at_least_one(1.0 / risk_ratio_ci_upper)
        )

    return EValueResult(
        risk_ratio=float(risk_ratio_point),
        e_value_point=float(e_point),
        e_value_ci=float(e_ci),
        citation=_CITATION,
        interpretation=_INTERPRETATION,
    )
