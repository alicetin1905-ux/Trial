import numpy as np
import pytest

from jevbot.backtest.metrics import (
    deflated_sharpe,
    hit_rate,
    max_drawdown,
    probabilistic_sharpe,
    sharpe_annual,
    t_stat,
)


def test_sharpe_annual_matches_hand_calc():
    r = np.array([0.01, -0.005, 0.002, 0.004])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(365)
    assert sharpe_annual(r) == pytest.approx(expected)


def test_sharpe_of_flat_series_is_zero():
    assert sharpe_annual(np.zeros(10)) == 0.0


def test_max_drawdown_on_closes():
    assert max_drawdown(np.array([100, 120, 90, 130, 65.0])) == pytest.approx(0.5)


def test_max_drawdown_uses_intraday_minimums_when_given():
    closes = np.array([100, 110, 105.0])
    mins = np.array([100, 88, 104.0])  # dipped to 88 intraday after a 110 close? peak so far is 100
    assert max_drawdown(closes, mins) == pytest.approx(0.12)


def test_hit_rate_and_t_stat():
    x = np.array([0.01, -0.01, 0.02, 0.03])
    assert hit_rate(x) == 0.75
    assert t_stat(x) == pytest.approx(x.mean() / (x.std(ddof=1) / 2))
    assert t_stat(np.array([0.01])) == 0.0


def test_probabilistic_sharpe_is_half_at_the_benchmark():
    assert probabilistic_sharpe(0.1, 0.1, n=500, skew=0.0, kurt=3.0) == pytest.approx(0.5)


def test_deflated_sharpe_falls_as_trials_grow():
    rng = np.random.default_rng(0)
    trial_srs = rng.normal(0.0, 0.05, 200)
    d1 = deflated_sharpe(0.12, n=1000, skew=0.0, kurt=3.0, trial_srs=trial_srs[:2])
    d200 = deflated_sharpe(0.12, n=1000, skew=0.0, kurt=3.0, trial_srs=trial_srs)
    assert d1 > d200


def test_deflated_sharpe_with_one_trial_is_psr_vs_zero():
    assert deflated_sharpe(0.1, n=500, skew=0.0, kurt=3.0, trial_srs=[0.1]) == pytest.approx(
        probabilistic_sharpe(0.1, 0.0, n=500, skew=0.0, kurt=3.0)
    )
