import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import make_bars15

import jevbot.backtest.jev_run as jr
from jevbot.backtest.engine import CostModel, EngineState
from jevbot.backtest.harness import prepare
from jevbot.config import load_risk_limits
from jevbot.jev.client import JevFailure
from jevbot.jev.fake import FakeJev
from jevbot.strategy.calibration import Calibrator
from jevbot.strategy.params import RuleParams, Strategy, grid

ROOT = Path(__file__).resolve().parents[1]
LIMITS, _ = load_risk_limits(ROOT / "config" / "risk.yaml")
START_2020_09 = 1598918400000


@pytest.fixture(scope="module")
def world():
    """~5.3 years of synthetic 15m bars (2020-09 -> 2025-12), features, ATR and fake-Jev answers."""
    bars = make_bars15(int(5.33 * 365 * 96), seed=11, start=START_2020_09, vol=0.003)
    feats, atr15 = prepare(bars, [])
    configs = [grid()[0], grid()[-1]]
    calls = jr.plan_calls(bars, feats, configs, sample_frac=0.01)
    fake = FakeJev()

    async def ask_many(items):
        return [await fake.ask(s, side) for s, side in items]

    answers = asyncio.run(jr.fetch_answers(ask_many, feats, calls))
    return bars, feats, atr15, configs, calls, answers


def test_plan_calls_cover_every_candidate_and_a_sample(world):
    bars, feats, _, configs, calls, _ = world
    valid, aligned = jr.aligned_inputs(bars, feats)
    t = bars["available_at"].to_numpy(np.int64)
    callset = set(calls)
    for p in configs:
        sig = jr.signals_for(aligned, valid, p)
        assert all((int(t[i]), int(sig[i])) in callset for i in np.nonzero(sig)[0])
    assert any(side == 0 for _, side in calls)
    assert calls == sorted(set(calls))


def test_fit_model_needs_enough_samples():
    s = pd.DataFrame({"entry_time": range(10), "won": [0, 1] * 5, **{k: [0.5] * 10 for k in jr.FACTORS}})
    assert jr.fit_model(s, grid()[0]) is None


def test_fitted_weights_find_a_planted_signal():
    rng = np.random.default_rng(3)
    n = 2000
    f = {k: rng.uniform(0, 1, n) for k in jr.FACTORS}
    won = (rng.uniform(0, 1, n) < 0.2 + 0.6 * f["regime"]).astype(int)
    s = pd.DataFrame({"entry_time": range(n), "won": won, **f})
    jp, cal = jr.fit_model(s, grid()[0])
    assert jp.w_regime > 1.0 and abs(jp.w_pressure) < jp.w_regime / 3
    assert jp.min_p_regime == jr.FIXED_THRESHOLDS["min_p_regime"]  # thresholds are fixed, not fitted
    assert jp.p_cutoff >= 0.5


def _strategy(p_cutoff=0.5):
    from jevbot.strategy.combine import JevParams

    jp = JevParams(**jr.FIXED_THRESHOLDS, w_regime=1, w_direction=1, w_pressure=1, w_quality=1,
                   w_risk_normal=1, intercept=-1, p_cutoff=p_cutoff)  # fmt: skip
    return Strategy(rules=grid()[0], jev=jp, body="", sha256="x")


def test_filter_vetoes_on_jev_failure_and_counts_it():
    bars = make_bars15(10)
    t0 = int(bars.available_at[0])
    log = jr.FilterLog(vetoes={})
    cal = Calibrator("isotonic", {"x": [-5.0, 5.0], "y": [0.99, 0.99]})
    f = jr.make_filter(
        bars, {(t0, 1): JevFailure("timeout", "", 1500)}, _strategy(), cal, LIMITS, np.full(10, 50.0), log
    )
    st = EngineState(10_000.0, 10_000.0, 10_000.0, t0 + 4 * 3_600_000, 30_000.0)
    assert f(0, 1, st) == 0.0 and log.vetoes == {"jev_timeout": 1}


def test_filter_halts_for_good_after_a_drawdown_kill(world):
    bars, feats, atr15, configs, calls, answers = world
    from jevbot.decide import decide

    cal = Calibrator("isotonic", {"x": [-50.0, 50.0], "y": [0.99, 0.99]})
    strat = _strategy()
    good = next(
        (t, s)
        for (t, s), a in answers.items()
        if s != 0 and decide(s, a, strat, cal, True, LIMITS).enter  # passes decide(): reaches the guard
    )
    i = int(np.nonzero(bars["available_at"].to_numpy() == good[0])[0][0])
    log = jr.FilterLog(vetoes={})
    f = jr.make_filter(bars, answers, strat, cal, LIMITS, atr15, log)
    t = good[0] - (good[0] % (8 * 3_600_000)) + 4 * 3_600_000  # far from funding
    crashed = EngineState(8_900.0, 10_000.0, 8_900.0, t, float(bars.close[i]))
    assert f(i, good[1], crashed) == 0.0 and log.killed_at == t
    fine = EngineState(10_000.0, 10_000.0, 10_000.0, t, float(bars.close[i]))
    assert f(i, good[1], fine) == 0.0  # still halted


def test_walk_forward_fits_each_fold_only_on_earlier_trades(world, monkeypatch):
    bars, feats, atr15, configs, _, answers = world
    seen = []
    real = jr.fit_model

    def spy(samples, rules):
        seen.append(int(samples.entry_time.max()) if len(samples) else -1)
        return real(samples, rules)

    monkeypatch.setattr(jr, "fit_model", spy)
    picks = jr.walk_forward_jev(bars, feats, atr15, configs, answers, LIMITS, CostModel())
    assert len(picks) == 8
    fold_starts = [jr._ms(p.test[0]) for p in picks]
    # fit_model is called len(configs) times per fold, in fold order
    for k, start in enumerate(fold_starts):
        for v in seen[k * len(configs) : (k + 1) * len(configs)]:
            assert v < start
    for p in picks:
        if p.part is not None:
            assert p.part.daily_ret.index.min() >= pd.Timestamp(p.test[0])
            assert p.part.daily_ret.index.max() <= pd.Timestamp(p.test[1])


def test_rule_params_type_is_used(world):
    assert all(isinstance(p, RuleParams) for p in world[3])
