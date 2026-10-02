import json

import numpy as np
import pytest

from jevbot.strategy.calibration import (
    Calibrator,
    brier,
    ece,
    fit_calibrator,
    reliability,
)
from jevbot.strategy.combine import JevParams, aligned_factors, edge_score, passes_thresholds

ANS = {
    "regime": {
        "probabilities": {"trending_up": 0.7, "trending_down": 0.05, "ranging": 0.2, "high_vol_chop": 0.05}
    },
    "direction": {"probabilities": {"up": 0.55, "down": 0.15, "flat": 0.3}},
    "buy_pressure_real": {"noul": 0.64},
    "setup_quality": {"score": 2.9, "confidence": 0.7},
    "risk_state": {"probabilities": {"normal": 0.85, "elevated": 0.13, "extreme": 0.02}},
}

PARAMS = JevParams(
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


def test_factors_are_aligned_with_the_trade_side():
    long_ = aligned_factors(ANS, side=1)
    short = aligned_factors(ANS, side=-1)
    assert long_ == pytest.approx(
        {
            "regime": 0.7,
            "direction": 0.55,
            "pressure": 0.64,
            "quality": 2.9 / 4,
            "risk_normal": 0.85,
            "risk_extreme": 0.02,
        }
    )
    assert short["regime"] == 0.05 and short["direction"] == 0.15
    assert short["pressure"] == pytest.approx(0.36)


def test_every_question_threshold_must_pass():
    ok, fails = passes_thresholds(aligned_factors(ANS, 1), PARAMS)
    assert ok and fails == []
    ok, fails = passes_thresholds(aligned_factors(ANS, -1), PARAMS)
    assert not ok and {"regime", "direction", "pressure"} <= set(fails)


def test_extreme_risk_blocks():
    a = json.loads(json.dumps(ANS))
    a["risk_state"]["probabilities"] = {"normal": 0.3, "elevated": 0.3, "extreme": 0.4}
    ok, fails = passes_thresholds(aligned_factors(a, 1), PARAMS)
    assert not ok and fails == ["risk_extreme"]


def test_edge_score_is_explicit_weighted_sum():
    f = aligned_factors(ANS, 1)
    expected = -2.0 + 0.7 + 0.55 + 0.5 * 0.64 + 2.9 / 4 + 0.5 * 0.85
    assert edge_score(f, PARAMS) == pytest.approx(expected)


# --- calibration -----------------------------------------------------------------------------


def test_brier_known_values():
    assert brier(np.array([1.0, 0.0]), np.array([1, 0])) == 0.0
    assert brier(np.array([0.5, 0.5]), np.array([1, 0])) == pytest.approx(0.25)


def test_ece_zero_for_perfect_calibration_and_positive_otherwise():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, p.size) < p).astype(int)
    assert ece(p, y) < 0.01
    assert ece(np.clip(p + 0.2, 0, 1), y) > 0.1


def test_reliability_bins_cover_all_samples():
    p = np.linspace(0.01, 0.99, 100)
    y = (p > 0.5).astype(int)
    r = reliability(p, y, bins=10)
    assert r["n"].sum() == 100 and len(r) == 10
    assert set(r.columns) == {"bin_lo", "bin_hi", "n", "mean_p", "frac_pos"}


def test_isotonic_calibrator_is_monotone_and_roundtrips(tmp_path):
    rng = np.random.default_rng(1)
    s = rng.normal(0, 1, 5000)
    y = (rng.uniform(0, 1, s.size) < 1 / (1 + np.exp(-2 * s))).astype(int)
    cal = fit_calibrator(s, y)
    assert cal.method == "isotonic"
    grid_ = np.linspace(-3, 3, 200)
    out = cal.predict(grid_)
    assert np.all(np.diff(out) >= -1e-12) and out.min() >= 0 and out.max() <= 1
    assert ece(cal.predict(s), y) < 0.03
    path = tmp_path / "calibrator.json"
    cal.save(path)
    assert np.allclose(Calibrator.load(path).predict(grid_), out)


def test_small_samples_use_platt():
    rng = np.random.default_rng(2)
    s = rng.normal(0, 1, 120)
    y = (rng.uniform(0, 1, s.size) < 1 / (1 + np.exp(-s))).astype(int)
    cal = fit_calibrator(s, y)
    assert cal.method == "platt"
    assert 0 < cal.predict(np.array([0.0]))[0] < 1


def test_calibrator_refuses_single_class_data():
    with pytest.raises(ValueError):
        fit_calibrator(np.arange(300.0), np.ones(300, dtype=int))
