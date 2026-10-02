from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jevbot.config import load_risk_limits
from jevbot.decide import decide
from jevbot.jev.client import JevFailure, JevResult
from jevbot.sizing import kelly_fraction, risk_pct_for
from jevbot.strategy.calibration import Calibrator
from jevbot.strategy.combine import JevParams
from jevbot.strategy.params import RuleParams, Strategy

ROOT = Path(__file__).resolve().parents[1]
LIMITS, _ = load_risk_limits(ROOT / "config" / "risk.yaml")

RULES = RuleParams(
    pullback_min_atr=1.0,
    pullback_max_atr=2.5,
    rsi_long_max=40,
    stop_atr_k=1.5,
    take_profit_r=1.5,
    time_stop_bars=32,
    direction="both",
)
JEVP = JevParams(
    min_p_regime=0.5,
    min_p_direction=0.4,
    min_pressure=0.5,
    min_setup_quality=2.0,
    max_p_risk_extreme=0.2,
    w_regime=1.0,
    w_direction=1.0,
    w_pressure=0.5,
    w_quality=1.0,
    w_risk_normal=0.5,
    intercept=-2.0,
    p_cutoff=0.55,
)
STRAT = Strategy(rules=RULES, jev=JEVP, body="", sha256="x")
# identity-ish calibrator: p_win = clip(0.5 + edge/2)
CAL = Calibrator("isotonic", {"x": [-1.0, 1.0], "y": [0.0, 1.0]})

ANS = {
    "regime": {
        "probabilities": {"trending_up": 0.7, "trending_down": 0.05, "ranging": 0.2, "high_vol_chop": 0.05}
    },
    "direction": {"probabilities": {"up": 0.55, "down": 0.15, "flat": 0.3}},
    "buy_pressure_real": {"noul": 0.64},
    "setup_quality": {"score": 2.9, "confidence": 0.7},
    "risk_state": {"probabilities": {"normal": 0.85, "elevated": 0.13, "extreme": 0.02}},
}
GOOD = JevResult(ANS, "jev-1.13.0", 90.0, 1000, 0.00004, True)
EDGE = -2.0 + 0.7 + 0.55 + 0.5 * 0.64 + 2.9 / 4 + 0.5 * 0.85  # = 0.72


def _decide(side=1, jev=GOOD, calibrated=False, cal=CAL, strat=STRAT):
    return decide(side, jev, strat, cal, calibrated=calibrated, limits=LIMITS)


def test_enter_long_with_fixed_risk_until_calibrated():
    i = _decide()
    assert i.enter and i.side == 1
    assert i.p_win == pytest.approx(0.5 + EDGE / 2)
    assert i.risk_pct == LIMITS.fixed_risk_pct_until_calibrated
    assert i.reasons == []


def test_no_candidate_no_trade():
    assert not _decide(side=0).enter


@pytest.mark.parametrize("reason", ["timeout", "error", "malformed", "auth"])
def test_any_jev_failure_means_no_trade(reason):
    i = _decide(jev=JevFailure(reason, "x", 1500.0))
    assert not i.enter and i.reasons == [f"jev_{reason}"]


def test_untrusted_model_means_no_trade():
    i = _decide(jev=JevResult(ANS, "jev-9", 90.0, 1000, 0.0, False, "model_mismatch"))
    assert not i.enter and i.reasons == ["jev_untrusted:model_mismatch"]


def test_threshold_failure_lists_questions():
    i = _decide(side=-1)
    assert not i.enter and i.reasons[0].startswith("threshold:")


def test_below_cutoff_means_no_trade():
    low = Calibrator("isotonic", {"x": [-1.0, 1.0], "y": [0.0, 0.6]})  # p_win = 0.3 * (edge + 1) ~ 0.52
    i = _decide(cal=low)
    assert not i.enter and i.reasons == ["below_cutoff"]


def test_no_calibrator_no_trade():
    i = _decide(cal=None)
    assert not i.enter and i.reasons == ["no_calibrator"]


def test_strategy_without_jev_section_cannot_trade_live():
    i = _decide(strat=Strategy(rules=RULES, jev=None, body="", sha256="x"))
    assert not i.enter and i.reasons == ["no_jev_params"]


def test_kelly_sizing_once_calibrated():
    i = _decide(calibrated=True)
    p, b = 0.5 + EDGE / 2, RULES.take_profit_r
    c = 0.5 + 0.5 * 0.7
    expected = min(100 * LIMITS.kelly_fraction * kelly_fraction(p, b) * c, LIMITS.max_risk_per_trade_pct)
    assert i.enter and i.risk_pct == pytest.approx(expected)


# --- sizing ----------------------------------------------------------------------------------


def test_kelly_known_values():
    assert kelly_fraction(0.6, 1.0) == pytest.approx(0.2)
    assert kelly_fraction(0.5, 1.0) == 0.0
    assert kelly_fraction(0.3, 1.0) < 0


@settings(max_examples=2000, deadline=None)
@given(
    p=st.floats(0.0, 1.0),
    b=st.floats(0.1, 10.0),
    conf=st.floats(0.0, 1.0),
    calibrated=st.booleans(),
    cutoff=st.floats(0.01, 0.99),
)
def test_risk_never_exceeds_cap_and_is_zero_below_cutoff(p, b, conf, calibrated, cutoff):
    r = risk_pct_for(p, b, conf, cutoff, calibrated, LIMITS)
    assert 0.0 <= r <= LIMITS.max_risk_per_trade_pct
    if p < cutoff or kelly_fraction(p, b) <= 0:
        assert r == 0.0
    if not calibrated and r > 0:
        assert r == LIMITS.fixed_risk_pct_until_calibrated


def test_more_confidence_never_means_smaller_size():
    lo = risk_pct_for(0.62, 1.5, 0.2, 0.55, True, LIMITS)
    hi = risk_pct_for(0.62, 1.5, 0.9, 0.55, True, LIMITS)
    assert hi >= lo > 0


def test_decide_is_pure_same_inputs_same_output():
    a = _decide(calibrated=True)
    b = _decide(calibrated=True)
    assert a == b
    assert np.isfinite(a.p_win)
