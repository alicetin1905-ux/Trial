"""Bar resampling and point-in-time access.

Every bar row carries `available_at` (its close time). Anything that builds a decision at time t
must go through `as_of`, so it can only see bars that had closed by t (spec §3 leakage rule).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE_MS = 15 * 60 * 1000


def resample(bars15: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate 15m bars into `minutes` bars aligned to UTC multiples.

    Only complete groups are emitted: a group missing any 15m bar (a gap, or the still-forming
    trailing period) is dropped, so a partial higher-timeframe bar can never leak into a decision.
    """
    span = minutes * 60 * 1000
    if span % BASE_MS:
        raise ValueError(f"{minutes}m is not a multiple of 15m")
    need = span // BASE_MS
    b = bars15.sort_values("open_time")
    key = (b["open_time"] // span) * span
    g = b.groupby(key, sort=True)
    out = pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
            "volume": g["volume"].sum(),
            "buy_volume": g["buy_volume"].sum(),
            "sell_volume": g["sell_volume"].sum(),
            "quote_volume": g["quote_volume"].sum(),
            "trade_count": g["trade_count"].sum(),
            "n": g["open"].size(),
        }
    )
    out = out[out["n"] == need].drop(columns="n")
    out.index.name = "open_time"
    out = out.reset_index()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["vwap"] = np.where(out["volume"] > 0, out["quote_volume"] / out["volume"], np.nan)
    out["available_at"] = out["open_time"] + span
    return out[list(bars15.columns)].reset_index(drop=True)


def as_of(bars: pd.DataFrame, t: int) -> pd.DataFrame:
    """Rows whose bar had closed by time t (available_at <= t)."""
    return bars[bars["available_at"] <= t]


def find_gaps(bars: pd.DataFrame, step_ms: int) -> list[tuple[int, int]]:
    """Missing [start, end) ranges in a bar series."""
    t = np.sort(bars["open_time"].to_numpy())
    jumps = np.nonzero(np.diff(t) != step_ms)[0]
    return [(int(t[i] + step_ms), int(t[i + 1])) for i in jumps]
