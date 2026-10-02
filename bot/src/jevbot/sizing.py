"""Position sizing (spec §7): capped quarter Kelly on the calibrated win probability."""

from __future__ import annotations

from jevbot.config import RiskLimits


def kelly_fraction(p: float, b: float) -> float:
    """Kelly fraction of equity to risk on a bet that wins b per 1 risked with probability p."""
    return (p * b - (1.0 - p)) / b


def risk_pct_for(
    p_win: float, b: float, confidence: float, p_cutoff: float, calibrated: bool, limits: RiskLimits
) -> float:
    """% of equity to lose at the stop. 0 below the cutoff or without edge.

    Until calibration passes on our own fills, every trade uses the fixed small size.
    Confidence in [0, 1] scales Kelly by 0.5..1.0, so higher confidence never shrinks a position.
    """
    if not p_win >= p_cutoff:
        return 0.0
    f = kelly_fraction(p_win, b)
    if f <= 0:
        return 0.0
    if not calibrated:
        return limits.fixed_risk_pct_until_calibrated
    c = 0.5 + 0.5 * min(1.0, max(0.0, confidence))
    return min(100.0 * limits.kelly_fraction * f * c, limits.max_risk_per_trade_pct)
