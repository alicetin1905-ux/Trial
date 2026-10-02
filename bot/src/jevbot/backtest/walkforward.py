"""Anchored walk-forward over the design period only (spec §9).

Every test fold picks its config using only data before the fold starts. The out-of-sample
series is the concatenation of the test folds. This module refuses holdout data outright.
"""

from __future__ import annotations

import pandas as pd

from jevbot.backtest.metrics import sharpe_daily

DESIGN_START = "2021-01-01"
DESIGN_END = "2025-12-31"
FOLD_TESTS = [
    ("2022-01-01", "2022-06-30"),
    ("2022-07-01", "2022-12-31"),
    ("2023-01-01", "2023-06-30"),
    ("2023-07-01", "2023-12-31"),
    ("2024-01-01", "2024-06-30"),
    ("2024-07-01", "2024-12-31"),
    ("2025-01-01", "2025-06-30"),
    ("2025-07-01", "2025-12-31"),
]


class HoldoutLeak(RuntimeError):
    pass


def folds() -> list[tuple[str, str, str, str]]:
    out = []
    for test_start, test_end in FOLD_TESTS:
        train_end = (pd.Timestamp(test_start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        out.append((DESIGN_START, train_end, test_start, test_end))
    return out


def _check(series: pd.Series) -> None:
    idx = (
        pd.DatetimeIndex(series.index).tz_localize(None)
        if getattr(series.index, "tz", None)
        else series.index
    )
    if len(series) and pd.Timestamp(idx.max()) > pd.Timestamp(DESIGN_END):
        raise HoldoutLeak(f"walk-forward got data up to {idx.max()}, beyond design end {DESIGN_END}")


def _naive(s: pd.Series) -> pd.Series:
    if getattr(s.index, "tz", None) is not None:
        s = s.copy()
        s.index = s.index.tz_localize(None)
    return s


def select_and_stitch(daily_returns: dict[str, pd.Series]) -> tuple[list[str], pd.Series]:
    """Per fold: pick the config with the best in-sample daily Sharpe, take its test-fold returns."""
    series = {k: _naive(v) for k, v in daily_returns.items()}
    for s in series.values():
        _check(s)
    picks, parts = [], []
    for train_start, train_end, test_start, test_end in folds():
        best = max(series, key=lambda k: sharpe_daily(series[k].loc[train_start:train_end].to_numpy()))
        picks.append(best)
        parts.append(series[best].loc[test_start:test_end])
    return picks, pd.concat(parts)
