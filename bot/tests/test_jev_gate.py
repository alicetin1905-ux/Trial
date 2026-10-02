import numpy as np
import pandas as pd
import pytest

from jevbot.backtest.harness import ConfigRun
from jevbot.backtest.jev_gate import answer_stats, direction_calibration, estimate, stitch
from jevbot.backtest.jev_run import FoldPick
from jevbot.jev.client import JEV_USD_PER_INPUT_TOKEN, JevFailure, JevResult

H = 3_600_000


def _res(p_up, p_dn, trusted=True, model="jev-1", cached=False):
    flat = round(1 - p_up - p_dn, 6)
    a = {"direction": {"probabilities": {"up": p_up, "down": p_dn, "flat": flat}}}
    return JevResult(a, model, 100.0, 1000, 1000 * JEV_USD_PER_INPUT_TOKEN, trusted,
                     None if trusted else "model_mismatch", cached)  # fmt: skip


def test_estimate_scales_with_calls():
    e1 = estimate(1000, {"a": 1.0}, {"q": "x" * 350}, rate_per_s=18)
    e2 = estimate(2000, {"a": 1.0}, {"q": "x" * 350}, rate_per_s=18)
    assert e2["cost_usd_est"] >= e1["cost_usd_est"] and e2["minutes_est"] == pytest.approx(
        2 * e1["minutes_est"], rel=0.1
    )


def test_answer_stats_counts_everything():
    ans = {
        (1, 1): _res(0.5, 0.2),
        (2, 1): _res(0.5, 0.2, trusted=False, model="jev-2"),
        (3, 0): JevFailure("timeout", "", 1500),
        (4, 0): _res(0.5, 0.2, cached=True),
    }
    s = answer_stats(ans)
    assert s["calls"] == 4 and s["ok_trusted"] == 2 and s["failures"]["timeout"] == 1
    assert s["untrusted"]["model_mismatch"] == 1 and s["models"]["jev-2"] == 1 and s["cached"] == 1


def test_stitch_keeps_a_fold_with_no_model_as_flat_days():
    days = pd.date_range("2022-01-01", "2022-06-30", freq="D")
    part = ConfigRun(pd.Series(0.01, index=days), pd.Series(0.0, index=days),
                     pd.DataFrame({"ret": [0.01], "entry_day": [days[0]]}))  # fmt: skip
    picks = [
        FoldPick(("2022-01-01", "2022-06-30"), "a", 0.1, part, None, None),
        FoldPick(("2022-07-01", "2022-12-31"), None, -np.inf, None, None, None),
    ]
    daily, mins, trades = stitch(picks)
    assert len(daily) == len(days) + 184 and daily.loc["2022-07-01":].eq(0).all()
    assert len(trades) == 1


def test_direction_calibration_rewards_information():
    n = 400
    t = np.arange(n, dtype=np.int64) * H + H
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    bars = pd.DataFrame({"available_at": t, "close": close})
    feats = pd.DataFrame({"t": t, "rv_7d_ann_pct": 0.01 * np.sqrt(365 * 24) * 100})
    future = np.log(np.roll(close, -12) / close)
    thr = 0.5 * 0.01 * np.sqrt(24)  # the question's "half a typical day's move"
    ans_good, ans_flat = {}, {}
    for i in range(n - 12):
        up = 0.8 if future[i] > thr else 0.05  # informative AND uses the question's definition
        ans_good[(int(t[i]), 0)] = _res(up, 0.1)
        ans_flat[(int(t[i]), 0)] = _res(0.3, 0.3)
    g = direction_calibration(ans_good, feats, bars)
    f = direction_calibration(ans_flat, feats, bars)
    assert g["n"] == n - 12
    assert g["brier_up"] < g["brier_up_climatology"] and g["brier_up"] < f["brier_up"]
