import json

import numpy as np
import pandas as pd
import pytest

from jevbot.backtest.gates import (
    DESIGN_END,
    HOLDOUT_START,
    GateThresholds,
    HoldoutAlreadyUsed,
    evaluate_gates,
    holdout_once,
)
from jevbot.backtest.trials import TrialLog
from jevbot.backtest.walkforward import HoldoutLeak, folds, select_and_stitch


def test_folds_are_anchored_six_month_tests_2022_to_2025():
    f = folds()
    assert len(f) == 8
    assert f[0] == ("2021-01-01", "2021-12-31", "2022-01-01", "2022-06-30")
    assert f[-1] == ("2021-01-01", "2025-06-30", "2025-07-01", "2025-12-31")
    assert all(train_start == "2021-01-01" for train_start, *_ in f)


def test_design_period_ends_before_holdout():
    assert DESIGN_END < HOLDOUT_START == "2026-01-01"


def _daily(values: dict[str, list[float]], days: pd.DatetimeIndex) -> dict[str, pd.Series]:
    return {k: pd.Series(v, index=days) for k, v in values.items()}


def test_selection_uses_only_in_sample_data():
    days = pd.date_range("2021-01-01", "2025-12-31", freq="D")
    n = len(days)
    in_2021 = days < "2022-01-01"
    # A: great in 2021, terrible afterwards. B: mediocre always.
    a = np.where(in_2021, 0.01, -0.01) + np.random.default_rng(0).normal(0, 1e-4, n)
    b = np.full(n, 0.0005) + np.random.default_rng(1).normal(0, 1e-4, n)
    picks, oos = select_and_stitch(_daily({"A": list(a), "B": list(b)}, days))
    assert picks[0] == "A"  # first fold only knows 2021 -> picks A, and must eat its OOS loss
    assert oos.loc["2022-01-01":"2022-06-30"].mean() < 0


def test_walkforward_refuses_holdout_data():
    days = pd.date_range("2021-01-01", "2026-02-01", freq="D")
    with pytest.raises(HoldoutLeak):
        select_and_stitch({"A": pd.Series(0.0, index=days)})


def test_gates_pass_and_fail_reasons():
    th = GateThresholds()
    good = dict(sharpe=1.8, max_dd=0.10, hit_rate=0.58, t_stat=2.4, n_trades=150, deflated_sharpe=0.97)
    ok, reasons = evaluate_gates(good, th)
    assert ok and reasons == []
    bad = good | {"hit_rate": 0.50, "n_trades": 80}
    ok, reasons = evaluate_gates(bad, th)
    assert not ok
    assert any("hit_rate" in r for r in reasons) and any("n_trades" in r for r in reasons)


def test_gate_thresholds_match_spec():
    th = GateThresholds()
    assert (th.sharpe, th.max_dd, th.hit_rate, th.t_stat, th.min_trades, th.deflated_sharpe) == (
        1.5,
        0.15,
        0.55,
        2.0,
        100,
        0.95,
    )


def test_holdout_can_be_used_once_per_candidate(tmp_path):
    log = TrialLog(tmp_path / "trials.jsonl")
    calls = []
    holdout_once(log, "cand-1", lambda: calls.append(1) or {"sharpe": 1.0})
    with pytest.raises(HoldoutAlreadyUsed):
        holdout_once(log, "cand-1", lambda: calls.append(1) or {"sharpe": 9.9})
    assert calls == [1]


def test_trial_log_is_append_only(tmp_path):
    p = tmp_path / "trials.jsonl"
    log = TrialLog(p)
    log.append({"kind": "config", "id": "a", "sharpe_daily": 0.05})
    first = p.read_text()
    log.append({"kind": "config", "id": "b", "sharpe_daily": 0.07})
    assert p.read_text().startswith(first)
    assert [r["id"] for r in log.read()] == ["a", "b"]
    assert all("ts" in json.loads(line) for line in p.read_text().splitlines())
    assert log.trial_sharpes() == [0.05, 0.07]
