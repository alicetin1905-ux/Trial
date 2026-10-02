"""Causal indicator functions. Each output at row i depends only on rows <= i."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's moving average."""
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1)
    return tr.max(axis=1)


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    return rma(true_range(df), n)


def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    gain = rma(d.clip(lower=0.0), n)
    loss = rma((-d).clip(lower=0.0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 - 100.0 / (1.0 + gain / loss)
    out = out.where(loss > 0, 100.0)
    return out.where((gain > 0) | (loss > 0), 50.0)


def adx(df: pd.DataFrame, n: int) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = rma(true_range(df), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * rma(plus_dm, n) / tr
        minus_di = 100.0 * rma(minus_dm, n) / tr
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return rma(dx.fillna(0.0), n)
