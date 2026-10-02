import numpy as np
import pandas as pd
import pytest

from jevbot.backtest.harness import ConfigRun, invalidate_after_gaps, oos_evaluate
from jevbot.backtest.metrics import max_drawdown, sharpe_annual
from jevbot.backtest.trials import TrialLog

BAR = 15 * 60 * 1000


def test_invalidate_after_gaps_drops_rows_whose_lookback_crosses_a_gap():
    t = np.arange(100, dtype=np.int64) * BAR
    feats = pd.DataFrame({"t": t, "x": 1.0})
    gaps = [(20 * BAR, 30 * BAR)]  # bars 20..29 missing; bar 30 opens at 30*BAR, closes (t) at 31*BAR
    out = invalidate_after_gaps(feats, gaps, warmup_bars=10, bar_ms=BAR)
    # rows with t in [gap_end, gap_end + warmup) are gone
    assert not ((out.t >= 30 * BAR) & (out.t < 40 * BAR)).any()
    assert (out.t == 40 * BAR).any() and (out.t == 19 * BAR).any()


def _run(rets, trade_rets, days, trade_days):
    daily_ret = pd.Series(rets, index=days)
    return ConfigRun(
        daily_ret=daily_ret,
        daily_min_ret=daily_ret.clip(upper=0.0),
        trades=pd.DataFrame({"entry_day": pd.DatetimeIndex(trade_days), "ret": trade_rets}),
    )


def test_oos_evaluate_stitches_picks_and_logs_every_config_as_a_trial(tmp_path):
    days = pd.date_range("2021-01-01", "2025-12-31", freq="D")
    rng = np.random.default_rng(0)
    runs = {}
    for k, mu in (("a", 0.002), ("b", 0.0), ("c", -0.001)):
        r = rng.normal(mu, 0.01, len(days))
        tdays = days[::3]
        runs[k] = _run(r, rng.normal(mu, 0.01, len(tdays)), days, tdays)
    log = TrialLog(tmp_path / "trials.jsonl")
    m = oos_evaluate(runs, log, label="test")
    assert len(log.trial_sharpes()) == 3
    assert m["picks"][0] == "a"
    oos = pd.concat(
        [runs[p].daily_ret.loc[s:e] for p, (s, e) in zip(m["picks"], m["fold_tests"], strict=True)]
    )
    assert m["sharpe"] == pytest.approx(sharpe_annual(oos.to_numpy()))
    closes = (1 + oos).cumprod().to_numpy()
    assert m["max_dd"] >= max_drawdown(closes) - 1e-12
    assert m["n_trades"] == sum(
        int(((runs[p].trades.entry_day >= s) & (runs[p].trades.entry_day <= e)).sum())
        for p, (s, e) in zip(m["picks"], m["fold_tests"], strict=True)
    )
    assert 0.0 <= m["deflated_sharpe"] <= 1.0
