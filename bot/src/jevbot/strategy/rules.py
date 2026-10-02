"""Deterministic trend + pullback rules (spec §4). They read the same rounded features Jev sees."""

from __future__ import annotations

import numpy as np
import pandas as pd

from jevbot.strategy.params import RuleParams


def trend_dir(f: pd.DataFrame) -> pd.Series:
    """+1 up / -1 down when the 4h EMA(50) slope and the side of the daily SMA(50) agree, else 0."""
    s4h = np.sign(f["ema50_slope_4h_atr"])
    s1d = np.sign(f["dist_sma50_1d_atr"])
    return pd.Series(np.where((s4h == s1d) & (s4h != 0), s4h, 0).astype(int), index=f.index)


def candidates(f: pd.DataFrame, rules: RuleParams) -> pd.Series:
    """+1 long candidate, -1 short candidate, 0 none. Jev only scores candidates (spec §5)."""
    trend = trend_dir(f)
    lo, hi = rules.pullback_min_atr, rules.pullback_max_atr
    # Pullback of 1-2.5 ATR(1h) from the 24h high (spec §4), RSI(14) on 1h.
    long_ = (trend == 1) & f["pullback_24h_atr"].between(lo, hi) & (f["rsi14_1h"] < rules.rsi_long_max)
    short = (trend == -1) & f["bounce_24h_atr"].between(lo, hi) & (f["rsi14_1h"] > 100 - rules.rsi_long_max)
    if rules.direction == "long":
        short = short & False
    elif rules.direction == "short":
        long_ = long_ & False
    return pd.Series(np.where(long_, 1, np.where(short, -1, 0)), index=f.index)


def exit_levels(entry: float, side: int, atr: float, rules: RuleParams) -> tuple[float, float]:
    """(stop, take_profit) prices. The stop is k x ATR away; take profit is R x the stop distance."""
    if not atr > 0 or side not in (1, -1):
        raise ValueError(f"bad exit inputs: atr={atr} side={side}")
    d = rules.stop_atr_k * atr
    return entry - side * d, entry + side * rules.take_profit_r * d


def trend_flipped(side: int, trend: int) -> bool:
    """Invalidation: exit when the 4h trend (sign of the 4h EMA slope) turns against the position."""
    return trend == -side
