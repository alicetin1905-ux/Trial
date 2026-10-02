import math

import numpy as np
import pandas as pd
import pytest
from conftest import BAR_MS, T0, make_bars15

from jevbot.state.features import adx, atr, ema, rsi
from jevbot.state.snapshot import FEATURES, WARMUP_BARS, compute_features, snapshot_at

H_MS = 3600 * 1000


# --- indicator known values ------------------------------------------------------------------


def test_rsi_all_gains_is_100_all_losses_is_0():
    up = pd.Series(np.arange(1.0, 60.0))
    assert rsi(up, 14).iloc[-1] == pytest.approx(100.0)
    assert rsi(up[::-1].reset_index(drop=True), 14).iloc[-1] == pytest.approx(0.0)


def test_rsi_alternating_equal_moves_is_50():
    s = pd.Series([100.0 + (i % 2) for i in range(200)])
    assert rsi(s, 14).iloc[-1] == pytest.approx(50.0, abs=2.0)


def test_atr_of_constant_range_bars_is_the_range():
    n = 100
    df = pd.DataFrame({"high": [11.0] * n, "low": [9.0] * n, "close": [10.0] * n})
    assert atr(df, 14).iloc[-1] == pytest.approx(2.0)


def test_ema_of_constant_is_constant_and_first_value_seeds():
    s = pd.Series([5.0] * 30)
    assert ema(s, 10).iloc[-1] == pytest.approx(5.0)


def test_adx_strong_on_steady_trend_weak_on_chop():
    n = 200
    up = pd.DataFrame({"high": np.arange(n) + 1.0, "low": np.arange(n) - 1.0, "close": np.arange(n) + 0.5})
    chop_c = 100 + np.where(np.arange(n) % 2, 1.0, -1.0)
    chop = pd.DataFrame({"high": chop_c + 1, "low": chop_c - 1, "close": chop_c})
    assert adx(up, 14).iloc[-1] > 50
    assert adx(chop, 14).iloc[-1] < 20


# --- snapshot schema -------------------------------------------------------------------------


def test_snapshot_has_exactly_the_declared_numeric_features(bars120d):
    t = int(bars120d["available_at"].iloc[-1])
    snap = snapshot_at(bars120d, t)
    assert list(snap) == list(FEATURES)
    assert all(isinstance(v, float) and math.isfinite(v) for v in snap.values())


def test_snapshot_never_exposes_prices_times_or_symbols(bars120d):
    t = int(bars120d["available_at"].iloc[-1])
    snap = snapshot_at(bars120d, t)
    banned = ("price", "time", "date", "symbol", "btc", "usd", "open", "close", "high", "low", "vwap")
    for k in snap:
        assert not any(b in k.lower().split("_") for b in banned), k
    price_level = bars120d["close"].iloc[-1]
    assert all(abs(v) < price_level / 10 for v in snap.values())  # no raw price-scale value leaks


def test_snapshot_values_are_rounded_for_compact_stable_state(bars120d):
    t = int(bars120d["available_at"].iloc[-1])
    snap = snapshot_at(bars120d, t)
    assert all(round(v, 4) == v for v in snap.values())


def test_flow_imbalance_known_value():
    b = make_bars15(WARMUP_BARS + 10, seed=3)
    b.loc[b.index[-1], ["buy_volume", "sell_volume"]] = [3.0, 1.0]
    f = compute_features(b)
    assert f["flow_imb_15m"].iloc[-1] == pytest.approx(0.5)


def test_hours_to_funding():
    b = make_bars15(WARMUP_BARS + 40, seed=4)
    f = compute_features(b).set_index("t")
    t_0745 = next(int(t) for t in f.index if (t % (8 * H_MS)) == 7 * H_MS + 45 * 60 * 1000)
    assert f.loc[t_0745, "hours_to_funding"] == pytest.approx(0.25)


def test_features_before_warmup_are_not_emitted():
    b = make_bars15(WARMUP_BARS - 1, seed=5)
    assert compute_features(b).empty


# --- leakage ---------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [WARMUP_BARS + 5, WARMUP_BARS + 500, -200, -1])
def test_truncation_invariance(bars120d, k):
    """Features at t are identical whether the data stops at t or continues for weeks."""
    full = compute_features(bars120d).set_index("t")
    t = int(bars120d["available_at"].iloc[k])
    cut = compute_features(bars120d[bars120d["available_at"] <= t]).set_index("t")
    pd.testing.assert_series_equal(full.loc[t], cut.loc[t], check_names=False)


def test_perturbing_the_future_never_changes_the_past(bars120d):
    k = len(bars120d) - 300
    t = int(bars120d["available_at"].iloc[k])
    future = bars120d.copy()
    future.loc[future.index[k + 1 :], ["open", "high", "low", "close"]] *= 3.0
    future.loc[future.index[k + 1 :], "buy_volume"] *= 50
    a = compute_features(bars120d).set_index("t").loc[t]
    b = compute_features(future).set_index("t").loc[t]
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_perturbing_the_current_bar_does_change_features(bars120d):
    k = len(bars120d) - 300
    t = int(bars120d["available_at"].iloc[k])
    now = bars120d.copy()
    now.loc[now.index[k], ["high", "close"]] *= 1.05
    a = compute_features(bars120d).set_index("t").loc[t]
    b = compute_features(now).set_index("t").loc[t]
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_live_window_matches_full_history(bars120d):
    """The live engine keeps only WARMUP_BARS of history; it must agree with the backtest."""
    t = int(bars120d["available_at"].iloc[-1])
    window = bars120d.iloc[-WARMUP_BARS:]
    full = compute_features(bars120d).set_index("t").loc[t]
    live = snapshot_at(window, t)
    np.testing.assert_allclose([live[k] for k in FEATURES], full[list(FEATURES)].to_numpy(), atol=1e-4)


def test_snapshot_at_rejects_a_time_without_a_closed_bar(bars120d):
    with pytest.raises(KeyError):
        snapshot_at(bars120d, T0 + 7 * BAR_MS + 1)
