"""Funding: real history from the Bybit v5 API (Mac Mini) or a conservative stand-in.

The Bybit API is geo-blocked from the cloud build environment, so backtests built there charge
CONSERVATIVE_RATE on every settlement held, to longs AND shorts. In reality shorts usually
receive funding when it is positive, so this only makes results worse, never better. The
backtest report states which funding source was used.
"""

from __future__ import annotations

import pandas as pd

INTERVAL_MS = 8 * 60 * 60 * 1000  # BTCUSDT settles at 00:00, 08:00, 16:00 UTC
CONSERVATIVE_RATE = 0.0001  # Bybit's base rate (0.01% / 8h), charged as a pure cost


def settlement_times(start: int, end: int) -> list[int]:
    """Settlements in (start, end]: a position open over the instant pays/receives funding."""
    first = (start // INTERVAL_MS + 1) * INTERVAL_MS
    return list(range(first, end + 1, INTERVAL_MS))


def conservative_funding_cost(side: int, notional: float, start: int, end: int) -> float:
    """Funding cost (positive = paid) for holding `notional` over (start, end], either side."""
    del side  # deliberately ignored: both sides pay in the conservative model
    return len(settlement_times(start, end)) * abs(notional) * CONSERVATIVE_RATE


def parse_funding_history(payload: dict) -> pd.DataFrame:
    """Parse GET /v5/market/funding/history into ascending (time, rate)."""
    if payload.get("retCode") != 0:
        raise RuntimeError(f"Bybit error {payload.get('retCode')}: {payload.get('retMsg')}")
    rows = payload["result"]["list"]
    df = pd.DataFrame(
        {
            "time": [int(r["fundingRateTimestamp"]) for r in rows],
            "rate": [float(r["fundingRate"]) for r in rows],
        }
    )
    return df.sort_values("time").reset_index(drop=True)
