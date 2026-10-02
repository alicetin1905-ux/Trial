"""Calibration (spec §6): Brier, ECE, reliability curves, and the score -> P(win) map.

Isotonic (pool-adjacent-violators) with enough data, Platt (logistic) for small samples. Fitted
on in-sample walk-forward data only and frozen to calibrator.json for out-of-sample and live use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

MIN_ISOTONIC = 500


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append(
            {
                "bin_lo": edges[b],
                "bin_hi": edges[b + 1],
                "n": int(m.sum()),
                "mean_p": float(p[m].mean()) if m.any() else np.nan,
                "frac_pos": float(y[m].mean()) if m.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    r = reliability(p, y, bins)
    r = r[r["n"] > 0]
    return float((r["n"] / r["n"].sum() * (r["mean_p"] - r["frac_pos"]).abs()).sum())


def _pav(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Isotonic regression by pool-adjacent-violators. Returns block centres and values."""
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order].astype(float)
    sums, wts, xsum = [], [], []
    for xi, yi in zip(xs, ys, strict=True):
        sums.append(yi)
        wts.append(1.0)
        xsum.append(xi)
        while len(sums) > 1 and sums[-2] / wts[-2] > sums[-1] / wts[-1]:
            s, w, xx = sums.pop(), wts.pop(), xsum.pop()
            sums[-1] += s
            wts[-1] += w
            xsum[-1] += xx
    w = np.array(wts)
    return np.array(xsum) / w, np.array(sums) / w


@dataclass
class Calibrator:
    method: str  # isotonic | platt
    params: dict

    def predict(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, float)
        if self.method == "isotonic":
            return np.clip(np.interp(s, self.params["x"], self.params["y"]), 0.0, 1.0)
        a, b = self.params["a"], self.params["b"]
        return 1.0 / (1.0 + np.exp(-(a * s + b)))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"method": self.method, "params": self.params}, indent=1))

    @classmethod
    def load(cls, path: Path) -> Calibrator:
        d = json.loads(Path(path).read_text())
        return cls(d["method"], d["params"])


def fit_calibrator(score: np.ndarray, won: np.ndarray) -> Calibrator:
    s, y = np.asarray(score, float), np.asarray(won, int)
    if len(np.unique(y)) < 2:
        raise ValueError("calibration needs both wins and losses")
    if len(s) >= MIN_ISOTONIC:
        x, v = _pav(s, y)
        return Calibrator("isotonic", {"x": x.tolist(), "y": v.tolist()})

    def nll(ab):
        z = ab[0] * s + ab[1]
        return float(np.sum(np.logaddexp(0, z) - y * z) + 1e-3 * ab[0] ** 2)

    res = minimize(nll, x0=np.array([1.0, 0.0]), method="L-BFGS-B")
    return Calibrator("platt", {"a": float(res.x[0]), "b": float(res.x[1])})
