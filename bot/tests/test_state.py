import math

import numpy as np
import pandas as pd
import pytest
from conftest import H1_MS, T0, make_bars1h

from jevbot.data.bars import resample
from jevbot.state.features import adx, atr, ema, rsi
from jevbot.state.snapshot import FEATURES, WARMUP_BARS, compute_features, snapshot_at

H_MS = 3600 * 1000
DAY_MS = 24 * H_MS


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


def test_snapshot_has_exactly_the_declared_numeric_features(bars1h_200d):
    t = int(bars1h_200d["available_at"].iloc[-1])
    snap = snapshot_at(bars1h_200d, t)
    assert list(snap) == list(FEATURES)
    assert all(isinstance(v, float) and math.isfinite(v) for v in snap.values())


def test_snapshot_never_exposes_prices_times_or_symbols(bars1h_200d):
    t = int(bars1h_200d["available_at"].iloc[-1])
    snap = snapshot_at(bars1h_200d, t)
    banned = ("price", "time", "date", "symbol", "btc", "usd", "open", "close", "high", "low", "vwap")
    for k in snap:
        assert not any(b in k.lower().split("_") for b in banned), k
    price_level = bars1h_200d["close"].iloc[-1]
    assert all(abs(v) < price_level / 10 for v in snap.values())


def test_snapshot_values_are_rounded_for_compact_stable_state(bars1h_200d):
    t = int(bars1h_200d["available_at"].iloc[-1])
    snap = snapshot_at(bars1h_200d, t)
    assert all(round(v, 4) == v for v in snap.values())


def test_flow_imbalance_known_value():
    b = make_bars1h(WARMUP_BARS + 10, seed=3)
    b.loc[b.index[-1], ["buy_volume", "sell_volume"]] = [3.0, 1.0]
    f = compute_features(b)
    assert f["flow_imb_1h"].iloc[-1] == pytest.approx(0.5)


def test_hours_to_funding():
    b = make_bars1h(WARMUP_BARS + 40, seed=4)
    f = compute_features(b).set_index("t")
    t_07 = next(int(t) for t in f.index if (t % (8 * H_MS)) == 7 * H_MS)
    assert f.loc[t_07, "hours_to_funding"] == pytest.approx(1.0)


def test_pullback_in_1h_atr_from_24h_high():
    b = make_bars1h(WARMUP_BARS + 50, seed=9)
    f = compute_features(b).set_index("t")
    t = int(b["available_at"].iloc[-1])
    a1h = atr(b, 14).iloc[-1]
    assert f.loc[t, "pullback_24h_atr"] == pytest.approx(
        round((b["high"].iloc[-24:].max() - b["close"].iloc[-1]) / a1h, 4), abs=1e-4
    )
    assert f.loc[t, "bounce_24h_atr"] == pytest.approx(
        round((b["close"].iloc[-1] - b["low"].iloc[-24:].min()) / a1h, 4), abs=1e-4
    )


def test_daily_trend_uses_completed_days_and_a_50_day_sma():
    b = make_bars1h(WARMUP_BARS + 30, seed=10)
    f = compute_features(b).set_index("t")
    t = int(b["available_at"].iloc[-1])
    d = resample(b, 1440, base_minutes=60)
    d = d[d["available_at"] <= t]
    sma = d["close"].rolling(50).mean()
    tr = pd.concat(
        [d["high"] - d["low"], (d["high"] - d["close"].shift()).abs(), (d["low"] - d["close"].shift()).abs()],
        axis=1,
    ).max(axis=1)
    a1d = tr.rolling(14).mean()
    assert f.loc[t, "dist_sma50_1d_atr"] == pytest.approx(
        round((d["close"].iloc[-1] - sma.iloc[-1]) / a1d.iloc[-1], 4), abs=1e-4
    )


def test_features_before_warmup_are_not_emitted():
    b = make_bars1h(WARMUP_BARS - 1, seed=5)
    assert compute_features(b).empty


# --- leakage ---------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [WARMUP_BARS + 5, WARMUP_BARS + 500, -200, -1])
def test_truncation_invariance(bars1h_200d, k):
    """Features at t are identical whether the data stops at t or continues for weeks."""
    full = compute_features(bars1h_200d).set_index("t")
    t = int(bars1h_200d["available_at"].iloc[k])
    cut = compute_features(bars1h_200d[bars1h_200d["available_at"] <= t]).set_index("t")
    pd.testing.assert_series_equal(full.loc[t], cut.loc[t], check_names=False)


def test_perturbing_the_future_never_changes_the_past(bars1h_200d):
    k = len(bars1h_200d) - 300
    t = int(bars1h_200d["available_at"].iloc[k])
    future = bars1h_200d.copy()
    future.loc[future.index[k + 1 :], ["open", "high", "low", "close"]] *= 3.0
    future.loc[future.index[k + 1 :], "buy_volume"] *= 50
    a = compute_features(bars1h_200d).set_index("t").loc[t]
    b = compute_features(future).set_index("t").loc[t]
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_perturbing_the_current_bar_does_change_features(bars1h_200d):
    k = len(bars1h_200d) - 300
    t = int(bars1h_200d["available_at"].iloc[k])
    now = bars1h_200d.copy()
    now.loc[now.index[k], ["high", "close"]] *= 1.05
    a = compute_features(bars1h_200d).set_index("t").loc[t]
    b = compute_features(now).set_index("t").loc[t]
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_live_window_matches_full_history(bars1h_200d):
    """The live engine keeps only WARMUP_BARS of history; it must agree with the backtest."""
    t = int(bars1h_200d["available_at"].iloc[-1])
    window = bars1h_200d.iloc[-WARMUP_BARS:]
    full = compute_features(bars1h_200d).set_index("t").loc[t]
    live = snapshot_at(window, t)
    np.testing.assert_allclose([live[k] for k in FEATURES], full[list(FEATURES)].to_numpy(), atol=2e-4)


def test_snapshot_at_rejects_a_time_without_a_closed_bar(bars1h_200d):
    with pytest.raises(KeyError):
        snapshot_at(bars1h_200d, T0 + 7 * H1_MS + 1)
