"""M7: walk-forward backtest of rules + Jev + decide() + risk guard (spec §5-§9).

For every test fold and every rule config:
  1. take the rules' in-sample trades (entries before the fold) and Jev's answers at their signals
  2. fit explicit weights (L2 logistic on the side-aligned factors) and a calibrator, in-sample only
  3. run the full period with decide() + the guard as the entry filter; Sharpe over the training
     window picks the config, and its test-fold slice goes into the out-of-sample record
Thresholds per question are fixed a priori (FIXED_THRESHOLDS) and not tuned.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from jevbot.backtest.engine import BAR_MS, CostModel, EngineState, run
from jevbot.backtest.harness import EQUITY0, ConfigRun, to_config_run
from jevbot.backtest.metrics import sharpe_daily
from jevbot.backtest.walkforward import folds
from jevbot.config import RiskLimits
from jevbot.decide import decide
from jevbot.jev.client import JevFailure, JevResult
from jevbot.risk.guard import Account, Market, Order, check
from jevbot.state.snapshot import FEATURES
from jevbot.strategy.calibration import Calibrator, fit_calibrator
from jevbot.strategy.combine import JevParams, aligned_factors
from jevbot.strategy.params import RuleParams, Strategy, params_hash
from jevbot.strategy.rules import candidates

FACTORS = ("regime", "direction", "pressure", "quality", "risk_normal")
FIXED_THRESHOLDS = dict(
    min_p_regime=0.35,
    min_p_direction=0.30,
    min_pressure=0.30,
    min_setup_quality=1.5,
    max_p_risk_extreme=0.25,
)
MIN_FIT_TRADES = 60

Answers = dict[tuple[int, int], JevResult | JevFailure]
Asker = Callable[[list[tuple[dict, int]]], "asyncio.Future"]


# --- 1. which (t, side) pairs need a Jev answer ------------------------------------------------


def aligned_inputs(bars: pd.DataFrame, feats: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    t = bars["available_at"].to_numpy(np.int64)
    aligned = feats.set_index("t").reindex(t)
    return aligned.notna().all(axis=1).to_numpy(), aligned


def signals_for(aligned: pd.DataFrame, valid: np.ndarray, rules: RuleParams) -> np.ndarray:
    sig = np.zeros(len(aligned), dtype=int)
    sig[valid] = candidates(aligned[valid].reset_index(), rules).to_numpy()
    return sig


def plan_calls(
    bars: pd.DataFrame,
    feats: pd.DataFrame,
    configs: list[RuleParams],
    sample_frac: float = 0.10,
    seed: int = 0,
) -> list[tuple[int, int]]:
    valid, aligned = aligned_inputs(bars, feats)
    t = bars["available_at"].to_numpy(np.int64)
    calls: set[tuple[int, int]] = set()
    for p in configs:
        sig = signals_for(aligned, valid, p)
        calls.update((int(t[i]), int(sig[i])) for i in np.nonzero(sig)[0])
    rng = np.random.default_rng(seed)
    sample = np.nonzero(valid & (rng.uniform(size=len(t)) < sample_frac))[0]
    calls.update((int(t[i]), 0) for i in sample)
    return sorted(calls)


def snapshot_of(feats_by_t: pd.DataFrame, t: int) -> dict[str, float]:
    row = feats_by_t.loc[t]
    return {k: float(row[k]) for k in FEATURES}


async def fetch_answers(
    ask_many, feats: pd.DataFrame, calls: list[tuple[int, int]], chunk: int = 2000
) -> Answers:
    by_t = feats.set_index("t")
    out: Answers = {}
    for k in range(0, len(calls), chunk):
        part = calls[k : k + chunk]
        res = await ask_many([(snapshot_of(by_t, t), side) for t, side in part])
        out.update(zip(part, res, strict=True))
    return out


# --- 2. fitting weights and the calibrator, in-sample ----------------------------------------


def trade_samples(res_trades: pd.DataFrame, t_of_bar: np.ndarray, answers: Answers) -> pd.DataFrame:
    rows = []
    for tr in res_trades.itertuples():
        a = answers.get((int(t_of_bar[tr.signal_i]), int(tr.side)))
        if not isinstance(a, JevResult) or not a.trusted:
            continue
        f = aligned_factors(a.answers, int(tr.side))
        rows.append({"entry_time": tr.entry_time, "won": int(tr.pnl > 0), **{k: f[k] for k in FACTORS}})
    return pd.DataFrame(rows, columns=["entry_time", "won", *FACTORS])


def fit_jev_params(samples: pd.DataFrame, take_profit_r: float) -> JevParams:
    X = samples[list(FACTORS)].to_numpy(float)
    y = samples["won"].to_numpy(int)

    def nll(w):
        z = w[0] + X @ w[1:]
        return float(np.sum(np.logaddexp(0, z) - y * z) + 0.5 * np.sum(w[1:] ** 2))

    w = minimize(nll, np.zeros(1 + len(FACTORS)), method="L-BFGS-B").x
    # Kelly needs p > 1/(1+R) for any edge; demand a little margin above that, and above a coin flip.
    p_cutoff = float(min(0.95, max(0.5, 1.0 / (1.0 + take_profit_r) + 0.05)))
    return JevParams(
        **FIXED_THRESHOLDS,
        w_regime=float(w[1]),
        w_direction=float(w[2]),
        w_pressure=float(w[3]),
        w_quality=float(w[4]),
        w_risk_normal=float(w[5]),
        intercept=float(w[0]),
        p_cutoff=p_cutoff,
    )


def fit_model(samples: pd.DataFrame, rules: RuleParams) -> tuple[JevParams, Calibrator] | None:
    if len(samples) < MIN_FIT_TRADES or samples["won"].nunique() < 2:
        return None
    jp = fit_jev_params(samples, rules.take_profit_r)
    X = samples[list(FACTORS)].to_numpy(float)
    edge = jp.intercept + X @ np.array(
        [jp.w_regime, jp.w_direction, jp.w_pressure, jp.w_quality, jp.w_risk_normal]
    )
    return jp, fit_calibrator(edge, samples["won"].to_numpy(int))


# --- 3. the entry filter: decide() + guard ----------------------------------------------------


@dataclass
class FilterLog:
    vetoes: dict
    approvals_needed: int = 0
    killed_at: int | None = None


def make_filter(
    bars: pd.DataFrame,
    answers: Answers,
    strategy: Strategy,
    calibrator: Calibrator,
    limits: RiskLimits,
    atr15: np.ndarray,
    log: FilterLog,
):
    t_of_bar = bars["available_at"].to_numpy(np.int64)
    halted = {"v": False}

    def f(i: int, side: int, st: EngineState) -> float:
        if halted["v"]:
            return 0.0
        intent = decide(side, answers.get((int(t_of_bar[i]), side)), strategy, calibrator, True, limits)
        if not intent.enter:
            key = intent.reasons[0].split(":")[0]
            log.vetoes[key] = log.vetoes.get(key, 0) + 1
            return 0.0
        stop_d = strategy.rules.stop_atr_k * atr15[i]
        units = min(
            st.equity * intent.risk_pct / 100 / stop_d, limits.max_notional_x_equity * st.equity / st.close
        )
        g = check(
            Order(side, units, st.close, st.close - side * stop_d),
            Account(st.equity, st.peak_equity, st.day_start_equity, 0, 0, False),
            Market(st.now_ms, st.now_ms, BAR_MS, st.close, None),
            limits,
            backtest=True,
        )
        if g.kill:
            halted["v"] = True
            log.killed_at = int(st.now_ms)
        if not g.ok:
            for r in g.reasons:
                log.vetoes["guard:" + r] = log.vetoes.get("guard:" + r, 0) + 1
            return 0.0
        if g.requires_approval:
            log.approvals_needed += 1  # counted; assumed approved in the backtest (reported)
        return intent.risk_pct

    return f


# --- 4. walk-forward ------------------------------------------------------------------------


@dataclass
class FoldPick:
    test: tuple[str, str]
    config: str | None
    is_sharpe: float
    part: ConfigRun | None
    log: FilterLog | None
    model: tuple[JevParams, Calibrator] | None


def _ms(day: str) -> int:
    return int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)


def walk_forward_jev(
    bars: pd.DataFrame,
    feats: pd.DataFrame,
    atr15: np.ndarray,
    configs: list[RuleParams],
    answers: Answers,
    limits: RiskLimits,
    costs: CostModel,
) -> list[FoldPick]:
    valid, aligned = aligned_inputs(bars, feats)
    t_of_bar = bars["available_at"].to_numpy(np.int64)
    slope = np.sign(aligned["ema50_slope_1h_atr"].fillna(0.0).to_numpy())
    sigs = {params_hash(p): signals_for(aligned, valid, p) for p in configs}
    rules_runs = {
        params_hash(p): run(bars, sigs[params_hash(p)], atr15, slope, p, costs, equity0=EQUITY0)
        for p in configs
    }
    samples = {k: trade_samples(r.trades, t_of_bar, answers) for k, r in rules_runs.items()}

    ot = bars["open_time"].to_numpy(np.int64)

    def window(start: str, end: str, p: RuleParams, model) -> tuple[ConfigRun, FilterLog]:
        """A fresh account over [start, end]: drawdown/halt state never leaks across windows."""
        lo, hi = np.searchsorted(ot, _ms(start)), np.searchsorted(ot, _ms(end) + 86_400_000)
        b = bars.iloc[lo:hi].reset_index(drop=True)
        jp, cal = model
        strat = Strategy(rules=p, jev=jp, body="", sha256=params_hash(p))
        flog = FilterLog(vetoes={})
        filt = make_filter(b, answers, strat, cal, limits, atr15[lo:hi], flog)
        res = run(b, sigs[params_hash(p)][lo:hi], atr15[lo:hi], slope[lo:hi], p, costs, equity0=EQUITY0,
                  entry_filter=filt)  # fmt: skip
        return to_config_run(res), flog

    picks: list[FoldPick] = []
    for train_start, train_end, test_start, test_end in folds():
        best = FoldPick((test_start, test_end), None, -np.inf, None, None, None)
        for p in configs:
            k = params_hash(p)
            s = samples[k]
            model = fit_model(s[s.entry_time < _ms(test_start)], p)
            if model is None:
                continue
            is_run, _ = window(train_start, train_end, p, model)
            is_sr = sharpe_daily(is_run.daily_ret.to_numpy())
            if is_sr > best.is_sharpe:
                best = FoldPick((test_start, test_end), k, is_sr, None, None, model)
        if best.config is not None:
            p = next(c for c in configs if params_hash(c) == best.config)
            best.part, best.log = window(test_start, test_end, p, best.model)
        picks.append(best)
    return picks
