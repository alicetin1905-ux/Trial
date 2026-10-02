"""The state engine (spec §3): the compact, numeric, normalised snapshot that Jev sees.

Entry timeframe: 1h bars (built from the 15m history). Trend context: 4h and 1d bars.

The snapshot never contains a price, a timestamp or a symbol, only ratios, z-scores and
percentages, so Jev cannot recognise a period it may have seen in training.

`compute_features` runs vectorised over a whole history (backtest). `snapshot_at` uses the last
WARMUP_BARS bars only (live). Tests show both agree, and that no feature reads the future.

The daily trend uses a 50-day SMA and a 14-day simple-average ATR (finite memory), not an EMA:
an EMA's seed would make a live window disagree with the full history unless the live engine
kept more than a year of bars. The 4h EMA's seed weight is below 1e-9 after the warm-up.

v1 deliberately leaves out the funding rate: there is no history for it in the cloud build, and
Jev must never see a live input the backtest did not have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jevbot.data.bars import as_of, resample
from jevbot.state.features import adx, atr, ema, rsi, true_range

BAR_MS = 60 * 60 * 1000
BASE_MINUTES = 60
FUNDING_MS = 8 * 3600 * 1000
BARS_PER_YEAR = 365 * 24
VOL_WINDOW = 168  # 7 days of 1h bars
SWING_WINDOW = 24  # 24 hours of 1h bars
WARMUP_BARS = 120 * 24  # 120 days: daily SMA50 + ATR14 need 64 days; 4h EMA50 seed weight < 1e-9
ROUND_DP = 4

FEATURES = (
    "ret_1h_z",
    "ret_4h_z",
    "ret_24h_z",
    "ret_72h_z",
    "rv_24h_ann_pct",
    "rv_7d_ann_pct",
    "rv_ratio_24h_7d",
    "ema50_slope_4h_atr",
    "dist_ema50_4h_atr",
    "sma50_slope_1d_atr",
    "dist_sma50_1d_atr",
    "adx14_4h",
    "pullback_24h_atr",
    "bounce_24h_atr",
    "rsi14_1h",
    "flow_imb_1h",
    "flow_imb_4h",
    "trade_count_z_1h",
    "hours_to_funding",
)


def _imbalance(buy: pd.Series, sell: pd.Series) -> pd.Series:
    tot = buy + sell
    return ((buy - sell) / tot).where(tot > 0, 0.0)


def _four_hour(b1h: pd.DataFrame) -> pd.DataFrame:
    h = resample(b1h, 240, base_minutes=BASE_MINUTES)
    a = atr(h, 14)
    e = ema(h["close"], 50)
    return pd.DataFrame(
        {
            "t": h["available_at"].astype(np.int64),
            "ema50_slope_4h_atr": (e - e.shift(3)) / a,
            "dist_ema50_4h_atr": (h["close"] - e) / a,
            "adx14_4h": adx(h, 14),
        }
    )


def _daily(b1h: pd.DataFrame) -> pd.DataFrame:
    d = resample(b1h, 1440, base_minutes=BASE_MINUTES)
    a = true_range(d).rolling(14).mean()
    s = d["close"].rolling(50).mean()
    return pd.DataFrame(
        {
            "t": d["available_at"].astype(np.int64),
            "sma50_slope_1d_atr": (s - s.shift(3)) / a,
            "dist_sma50_1d_atr": (d["close"] - s) / a,
        }
    )


def compute_features(bars1h: pd.DataFrame) -> pd.DataFrame:
    """One row per closed 1h bar, keyed by decision time t (= that bar's available_at).

    Rows before WARMUP_BARS of history, or with any undefined feature, are not emitted.
    """
    b = bars1h.sort_values("open_time").reset_index(drop=True)
    if len(b) < WARMUP_BARS:
        return pd.DataFrame(columns=["t", *FEATURES])
    c = b["close"]
    lr = np.log(c).diff()
    sigma = lr.rolling(VOL_WINDOW).std()
    a1h = atr(b, 14)
    f = pd.DataFrame({"t": b["available_at"].astype(np.int64)})

    for name, k in (("1h", 1), ("4h", 4), ("24h", 24), ("72h", 72)):
        f[f"ret_{name}_z"] = np.log(c / c.shift(k)) / (sigma * np.sqrt(k))
    rv24 = lr.rolling(24).std() * np.sqrt(BARS_PER_YEAR) * 100
    f["rv_24h_ann_pct"] = rv24
    f["rv_7d_ann_pct"] = sigma * np.sqrt(BARS_PER_YEAR) * 100
    f["rv_ratio_24h_7d"] = rv24 / f["rv_7d_ann_pct"]
    f["pullback_24h_atr"] = (b["high"].rolling(SWING_WINDOW).max() - c) / a1h
    f["bounce_24h_atr"] = (c - b["low"].rolling(SWING_WINDOW).min()) / a1h
    f["rsi14_1h"] = rsi(c, 14)
    f["flow_imb_1h"] = _imbalance(b["buy_volume"], b["sell_volume"])
    f["flow_imb_4h"] = _imbalance(b["buy_volume"].rolling(4).sum(), b["sell_volume"].rolling(4).sum())
    tc = b["trade_count"].astype(float)
    f["trade_count_z_1h"] = (tc - tc.rolling(VOL_WINDOW).mean()) / tc.rolling(VOL_WINDOW).std()
    f["hours_to_funding"] = ((f["t"] // FUNDING_MS + 1) * FUNDING_MS - f["t"]) / 3_600_000

    # Higher timeframes: attach the latest 4h / 1d bar that had closed by t (as-of join).
    for tf in (_four_hour(b), _daily(b)):
        f = pd.merge_asof(f.sort_values("t"), tf.sort_values("t"), on="t", direction="backward")

    f = f[["t", *FEATURES]]
    f = f.iloc[WARMUP_BARS - 1 :]
    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    # Rounded HERE (not only in snapshot_at), so the backtest and live compare the exact same
    # numbers against rule thresholds and send Jev the exact same state.
    f[list(FEATURES)] = f[list(FEATURES)].round(ROUND_DP)
    return f.reset_index(drop=True)


def snapshot_at(bars1h: pd.DataFrame, t: int) -> dict[str, float]:
    """The live snapshot at decision time t, using only the last WARMUP_BARS closed 1h bars."""
    seen = as_of(bars1h, t).sort_values("open_time").iloc[-WARMUP_BARS:]
    if seen.empty or int(seen["available_at"].iloc[-1]) != t:
        raise KeyError(f"no 1h bar closed exactly at {t}")
    feats = compute_features(seen)
    row = feats[feats["t"] == t]
    if row.empty:
        raise KeyError(f"features undefined at {t} (not enough history)")
    return {k: float(row[k].iloc[0]) for k in FEATURES}
