"""Performance metrics. All computed by the harness, never by a model (spec §9)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def sharpe_daily(r: np.ndarray) -> float:
    r = np.asarray(r, float)
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    return 0.0 if sd == 0 else float(r.mean() / sd)


def sharpe_annual(r: np.ndarray) -> float:
    return sharpe_daily(r) * math.sqrt(365)


def max_drawdown(closes: np.ndarray, mins: np.ndarray | None = None) -> float:
    """Worst peak-to-trough fall. With `mins`, intraday troughs count against the prior peak."""
    closes = np.asarray(closes, float)
    peak = np.maximum.accumulate(closes)
    dd = 1.0 - closes / peak
    if mins is not None:
        prior_peak = np.concatenate([[closes[0]], peak[:-1]])
        dd = np.maximum(dd, 1.0 - np.asarray(mins, float) / prior_peak)
    return float(dd.max()) if len(dd) else 0.0


def hit_rate(trade_returns: np.ndarray) -> float:
    x = np.asarray(trade_returns, float)
    return float((x > 0).mean()) if len(x) else 0.0


def t_stat(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def probabilistic_sharpe(sr: float, sr0: float, n: int, skew: float, kurt: float) -> float:
    """P(true SR > sr0), Bailey & López de Prado. SRs are per-period (daily), kurt is raw (normal = 3)."""
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr))
    return float(norm.cdf((sr - sr0) * math.sqrt(n - 1) / denom))


def deflated_sharpe(sr: float, n: int, skew: float, kurt: float, trial_srs: Sequence[float]) -> float:
    """PSR against the SR expected from the best of N trials of pure noise."""
    trials = np.asarray(trial_srs, float)
    if len(trials) < 2:
        sr0 = 0.0
    else:
        k = len(trials)
        sr0 = math.sqrt(trials.var(ddof=1)) * (
            (1 - EULER_GAMMA) * norm.ppf(1 - 1.0 / k) + EULER_GAMMA * norm.ppf(1 - 1.0 / (k * math.e))
        )
    return probabilistic_sharpe(sr, sr0, n, skew, kurt)
