"""Jev answers -> side-aligned factors -> per-question thresholds and an explicit weighted score.

Each question isolates one factor (spec §5). Code combines them with weights written in
strategy.md; the score is then mapped to a win probability by the calibrator (spec §6).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from jevbot.jev.questions import SETUP_QUALITY_LEVELS


class JevParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # per-question thresholds: ALL must pass
    min_p_regime: float = Field(ge=0, le=1)
    min_p_direction: float = Field(ge=0, le=1)
    min_pressure: float = Field(ge=0, le=1)
    min_setup_quality: float = Field(ge=0, le=SETUP_QUALITY_LEVELS - 1)  # in score units, 0..4
    max_p_risk_extreme: float = Field(ge=0, le=1)
    # explicit weights for edge_score
    w_regime: float
    w_direction: float
    w_pressure: float
    w_quality: float
    w_risk_normal: float
    intercept: float
    # calibrated win probability below this -> size 0
    p_cutoff: float = Field(gt=0, lt=1)


def aligned_factors(answers: dict, side: int) -> dict[str, float]:
    """Probabilities expressed as 'in favour of this trade's side'."""
    long_ = side == 1
    reg = answers["regime"]["probabilities"]
    dirp = answers["direction"]["probabilities"]
    risk = answers["risk_state"]["probabilities"]
    pressure = float(answers["buy_pressure_real"]["noul"])
    return {
        "regime": float(reg["trending_up"] if long_ else reg["trending_down"]),
        "direction": float(dirp["up"] if long_ else dirp["down"]),
        "pressure": pressure if long_ else 1.0 - pressure,
        "quality": float(answers["setup_quality"]["score"]) / (SETUP_QUALITY_LEVELS - 1),
        "risk_normal": float(risk["normal"]),
        "risk_extreme": float(risk["extreme"]),
    }


def passes_thresholds(f: dict[str, float], p: JevParams) -> tuple[bool, list[str]]:
    fails = []
    if f["regime"] < p.min_p_regime:
        fails.append("regime")
    if f["direction"] < p.min_p_direction:
        fails.append("direction")
    if f["pressure"] < p.min_pressure:
        fails.append("pressure")
    if f["quality"] * (SETUP_QUALITY_LEVELS - 1) < p.min_setup_quality:
        fails.append("quality")
    if f["risk_extreme"] > p.max_p_risk_extreme:
        fails.append("risk_extreme")
    return (not fails), fails


def edge_score(f: dict[str, float], p: JevParams) -> float:
    return (
        p.intercept
        + p.w_regime * f["regime"]
        + p.w_direction * f["direction"]
        + p.w_pressure * f["pressure"]
        + p.w_quality * f["quality"]
        + p.w_risk_normal * f["risk_normal"]
    )
