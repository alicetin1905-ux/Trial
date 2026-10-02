"""The state engine (spec §3): the compact, numeric, normalised snapshot that Jev sees.

The snapshot never contains a price, a timestamp or a symbol, only ratios, z-scores and
percentages, so Jev cannot recognise a period it may have seen in training.

`compute_features` runs vectorised over a whole history (backtest). `snapshot_at` uses the last
WARMUP_BARS bars only (live). Tests show both agree, and that no feature reads the future.

v1 deliberately leaves out the funding rate: there is no history for it in the cloud build, and
Jev must never see a live input the backtest did not have. It returns as a gated change once the
Mac Mini has real funding history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jevbot.data.bars import as_of, resample
from jevbot.state.features import adx, atr, ema, rsi

BAR_MS = 15 * 60 * 1000
FUNDING_MS = 8 * 3600 * 1000
BARS_PER_YEAR = 365 * 96
VOL_WINDOW = 288  # 3 days of 15m bars
SWING_WINDOW = 24  # 6 hours of 15m bars
# EMA(50) on 4h needs ~500 4h bars before the seed's weight is negligible (<1e-8): 90 days.
WARMUP_BARS = 90 * 96
ROUND_DP = 4

FEATURES = (
    "ret_15m_z",
    "ret_1h_z",
    "ret_4h_z",
    "ret_24h_z",
    "rv_4h_ann_pct",
    "rv_3d_ann_pct",
    "rv_ratio_4h_3d",
    "ema50_slope_1h_atr",
    "ema50_slope_4h_atr",
    "dist_ema50_1h_atr",
    "dist_ema50_4h_atr",
    "adx14_1h",
    "pullback_6h_atr",
    "bounce_6h_atr",
    "pullback_6h_atr1h",
    "bounce_6h_atr1h",
    "rsi14_15m",
    "flow_imb_15m",
    "flow_imb_1h",
    "trade_count_z_15m",
    "hours_to_funding",
)


def _imbalance(buy: pd.Series, sell: pd.Series) -> pd.Series:
    tot = buy + sell
    return ((buy - sell) / tot).where(tot > 0, 0.0)


def _trend_features(bars15: pd.DataFrame, minutes: int, tag: str) -> pd.DataFrame:
    h = resample(bars15, minutes)
    a = atr(h, 14)
    e = ema(h["close"], 50)
    out = pd.DataFrame(
        {
            "available_at": h["available_at"],
            f"ema50_slope_{tag}_atr": (e - e.shift(3)) / a,
            f"dist_ema50_{tag}_atr": (h["close"] - e) / a,
        }
    )
    if tag == "1h":
        out["adx14_1h"] = adx(h, 14)
        out["_atr_1h"] = a  # temporary, price units: used for the 1h-ATR pullback, then dropped
    return out


def compute_features(bars15: pd.DataFrame) -> pd.DataFrame:
    """One row per closed 15m bar, keyed by decision time t (= that bar's available_at).

    Rows before WARMUP_BARS of history, or with any undefined feature, are not emitted.
    """
    b = bars15.sort_values("open_time").reset_index(drop=True)
    if len(b) < WARMUP_BARS:
        return pd.DataFrame(columns=["t", *FEATURES])
    c = b["close"]
    lr = np.log(c).diff()
    sigma = lr.rolling(VOL_WINDOW).std()
    a15 = atr(b, 14)
    f = pd.DataFrame({"t": b["available_at"].astype(np.int64)})

    for name, k in (("15m", 1), ("1h", 4), ("4h", 16), ("24h", 96)):
        f[f"ret_{name}_z"] = np.log(c / c.shift(k)) / (sigma * np.sqrt(k))
    rv4h = lr.rolling(16).std() * np.sqrt(BARS_PER_YEAR) * 100
    f["rv_4h_ann_pct"] = rv4h
    f["rv_3d_ann_pct"] = sigma * np.sqrt(BARS_PER_YEAR) * 100
    f["rv_ratio_4h_3d"] = rv4h / f["rv_3d_ann_pct"]
    swing_hi = b["high"].rolling(SWING_WINDOW).max()
    swing_lo = b["low"].rolling(SWING_WINDOW).min()
    f["pullback_6h_atr"] = (swing_hi - c) / a15
    f["bounce_6h_atr"] = (c - swing_lo) / a15
    f["_hi_minus_c"] = swing_hi - c  # temporary, price units: never sent to Jev
    f["_c_minus_lo"] = c - swing_lo
    f["rsi14_15m"] = rsi(c, 14)
    f["flow_imb_15m"] = _imbalance(b["buy_volume"], b["sell_volume"])
    f["flow_imb_1h"] = _imbalance(b["buy_volume"].rolling(4).sum(), b["sell_volume"].rolling(4).sum())
    tc = b["trade_count"].astype(float)
    f["trade_count_z_15m"] = (tc - tc.rolling(VOL_WINDOW).mean()) / tc.rolling(VOL_WINDOW).std()
    f["hours_to_funding"] = ((f["t"] // FUNDING_MS + 1) * FUNDING_MS - f["t"]) / 3_600_000

    # Higher timeframes: attach the latest 1h/4h bar that had closed by t (as-of join).
    for minutes, tag in ((60, "1h"), (240, "4h")):
        tf = _trend_features(b, minutes, tag).rename(columns={"available_at": "t"}).astype({"t": np.int64})
        f = pd.merge_asof(f.sort_values("t"), tf.sort_values("t"), on="t", direction="backward")

    # The rules measure the pullback in 1h ATRs (spec §4); the 15m-ATR versions stay for Jev.
    f["pullback_6h_atr1h"] = f["_hi_minus_c"] / f["_atr_1h"]
    f["bounce_6h_atr1h"] = f["_c_minus_lo"] / f["_atr_1h"]
    f = f[["t", *FEATURES]]
    f = f.iloc[WARMUP_BARS - 1 :]
    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    # Rounded HERE (not only in snapshot_at), so the backtest and live compare the exact same
    # numbers against rule thresholds and send Jev the exact same state.
    f[list(FEATURES)] = f[list(FEATURES)].round(ROUND_DP)
    return f.reset_index(drop=True)


def snapshot_at(bars15: pd.DataFrame, t: int) -> dict[str, float]:
    """The live snapshot at decision time t, using only the last WARMUP_BARS closed bars."""
    seen = as_of(bars15, t).sort_values("open_time").iloc[-WARMUP_BARS:]
    if seen.empty or int(seen["available_at"].iloc[-1]) != t:
        raise KeyError(f"no 15m bar closed exactly at {t}")
    feats = compute_features(seen)
    row = feats[feats["t"] == t]
    if row.empty:
        raise KeyError(f"features undefined at {t} (not enough history)")
    return {k: float(row[k].iloc[0]) for k in FEATURES}
