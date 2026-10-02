"""THE decision function (architecture §1). Backtest, paper and live all call it.

Pure: no I/O, no clock, no model calls. Jev's answers arrive as numbers inside a JevResult;
decide() turns them into an Intent. The risk guard checks the resulting order separately, and
nothing in here can loosen it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from jevbot.config import RiskLimits
from jevbot.jev.client import JevFailure, JevResult
from jevbot.sizing import risk_pct_for
from jevbot.strategy.calibration import Calibrator
from jevbot.strategy.combine import aligned_factors, edge_score, passes_thresholds
from jevbot.strategy.params import Strategy


@dataclass(frozen=True)
class Intent:
    enter: bool
    side: int = 0
    risk_pct: float = 0.0
    p_win: float = float("nan")
    edge: float = float("nan")
    confidence: float = float("nan")
    reasons: list[str] = field(default_factory=list)
    factors: dict = field(default_factory=dict)


def _no(side: int, *reasons: str, **kw) -> Intent:
    return Intent(enter=False, side=side, reasons=list(reasons), **kw)


def decide(
    candidate_side: int,
    jev: JevResult | JevFailure | None,
    strategy: Strategy,
    calibrator: Calibrator | None,
    calibrated: bool,
    limits: RiskLimits,
) -> Intent:
    if candidate_side == 0:
        return _no(0, "no_candidate")
    if jev is None:
        return _no(candidate_side, "jev_missing")
    if isinstance(jev, JevFailure):
        return _no(candidate_side, f"jev_{jev.reason}")
    if not jev.trusted:
        return _no(candidate_side, f"jev_untrusted:{jev.untrusted_reason}")
    if strategy.jev is None:
        return _no(candidate_side, "no_jev_params")

    f = aligned_factors(jev.answers, candidate_side)
    ok, fails = passes_thresholds(f, strategy.jev)
    if not ok:
        return _no(candidate_side, "threshold:" + ",".join(fails), factors=f)
    edge = edge_score(f, strategy.jev)
    if calibrator is None:
        return _no(candidate_side, "no_calibrator", factors=f, edge=edge)
    p = float(calibrator.predict(np.array([edge]))[0])
    conf = float(jev.answers["setup_quality"].get("confidence", 0.0))
    if p < strategy.jev.p_cutoff:
        return _no(candidate_side, "below_cutoff", factors=f, edge=edge, p_win=p, confidence=conf)
    risk = risk_pct_for(p, strategy.rules.take_profit_r, conf, strategy.jev.p_cutoff, calibrated, limits)
    if risk <= 0:
        return _no(candidate_side, "no_edge", factors=f, edge=edge, p_win=p, confidence=conf)
    return Intent(True, candidate_side, risk, p, edge, conf, [], f)
