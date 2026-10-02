"""The backtest harness: grid -> walk-forward -> out-of-sample gate metrics -> report.

    python -m jevbot.backtest.harness baseline      # rules only (M4)

The same functions run the Jev-filtered candidate in M7 through the engine's entry_filter hook.
The holdout is never loaded here. Only gates.holdout_once evaluates it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from jevbot.backtest.engine import CostModel, Result, run_rules
from jevbot.backtest.gates import GateThresholds, evaluate_gates
from jevbot.backtest.metrics import (
    deflated_sharpe,
    hit_rate,
    max_drawdown,
    sharpe_annual,
    sharpe_daily,
    t_stat,
)
from jevbot.backtest.trials import TrialLog
from jevbot.backtest.walkforward import DESIGN_END, DESIGN_START, folds, select_and_stitch
from jevbot.data.bars import find_gaps, resample
from jevbot.data.bars1h_file import load_1h
from jevbot.data.history import load_bars15
from jevbot.state.features import atr
from jevbot.state.snapshot import BAR_MS as SNAP_BAR_MS
from jevbot.state.snapshot import WARMUP_BARS, compute_features
from jevbot.strategy.params import RuleParams, grid, params_hash

EQUITY0 = 10_000.0
HISTORY_1H = Path(__file__).resolve().parents[3] / "history" / "BTCUSDT_1h.parquet"


@dataclass
class ConfigRun:
    daily_ret: pd.Series  # indexed by UTC day (naive)
    daily_min_ret: pd.Series  # worst intraday mark-to-market vs the previous close
    trades: pd.DataFrame  # needs entry_day, ret


def invalidate_after_gaps(
    feats: pd.DataFrame, gaps: list[tuple[int, int]], warmup_bars: int, bar_ms: int
) -> pd.DataFrame:
    """Drop feature rows whose lookback window crosses a data gap."""
    bad = np.zeros(len(feats), dtype=bool)
    t = feats["t"].to_numpy(np.int64)
    for _, gap_end in gaps:
        bad |= (t >= gap_end) & (t < gap_end + warmup_bars * bar_ms)
    return feats[~bad].reset_index(drop=True)


def to_config_run(res: Result, equity0: float = EQUITY0) -> ConfigRun:
    d = res.daily.copy()
    d["day"] = pd.to_datetime(d["day"]).dt.tz_localize(None)
    prev = d["equity_close"].shift(1).fillna(equity0)
    tr = res.trades.copy()
    tr["entry_day"] = pd.to_datetime(tr["entry_time"].astype("int64"), unit="ms").dt.floor("D")
    return ConfigRun(
        daily_ret=pd.Series((d["equity_close"] / prev - 1).to_numpy(), index=d["day"]),
        daily_min_ret=pd.Series((d["equity_min"] / prev - 1).to_numpy(), index=d["day"]),
        trades=tr,
    )


def oos_evaluate(runs: dict[str, ConfigRun], log: TrialLog, label: str) -> dict:
    for cid, r in runs.items():
        log.append(
            {
                "kind": "config",
                "label": label,
                "id": cid,
                "sharpe_daily": sharpe_daily(r.daily_ret.to_numpy()),
            }
        )
    picks, oos = select_and_stitch({k: r.daily_ret for k, r in runs.items()})
    tests = [(s, e) for _, _, s, e in folds()]
    min_parts, trade_parts = [], []
    for p, (s, e) in zip(picks, tests, strict=True):
        min_parts.append(runs[p].daily_min_ret.loc[s:e])
        tr = runs[p].trades
        trade_parts.append(tr[(tr.entry_day >= s) & (tr.entry_day <= e)])
    m = metrics_from_parts(oos, pd.concat(min_parts), trade_parts, log.trial_sharpes(), label)
    return m | {"picks": picks, "fold_tests": tests}


def metrics_from_parts(
    oos: pd.Series, oos_min: pd.Series, trade_parts: list[pd.DataFrame], trial_srs: list[float], label: str
) -> dict:
    """Gate metrics from stitched out-of-sample daily returns, intraday minimums and trades."""
    oos_trades = pd.concat(trade_parts) if trade_parts else pd.DataFrame(columns=["ret"])
    r = oos.to_numpy()
    closes = np.cumprod(1 + r)
    prev = np.concatenate([[1.0], closes[:-1]])
    mins = prev * (1 + oos_min.to_numpy())
    sr_d = sharpe_daily(r)
    m = {
        "label": label,
        "sharpe": sharpe_annual(r),
        "sharpe_daily": sr_d,
        "max_dd": max_drawdown(closes, mins) if len(closes) else 0.0,
        "hit_rate": hit_rate(oos_trades["ret"].to_numpy()),
        "t_stat": t_stat(oos_trades["ret"].to_numpy()),
        "n_trades": len(oos_trades),
        "total_return": float(closes[-1] - 1) if len(closes) else 0.0,
        "skew": float(skew(r)) if len(r) > 2 else 0.0,
        "kurt": float(kurtosis(r, fisher=False)) if len(r) > 3 else 3.0,
        "n_trials": len(trial_srs),
    }
    m["deflated_sharpe"] = (
        deflated_sharpe(sr_d, len(r), m["skew"], m["kurt"], trial_srs) if len(r) > 1 else 0.0
    )
    m["avg_trade_ret"] = float(oos_trades["ret"].mean()) if len(oos_trades) else 0.0
    return m


# --- data prep -------------------------------------------------------------------------------


def load_design(
    data_dir: Path, symbol: str = "BTCUSDT", start: str = "2020-09-01"
) -> tuple[pd.DataFrame, list]:
    """1h bars for warm-up + design period. Never reads past DESIGN_END.

    Built from the 15m history in data_dir when it exists, else read from the committed,
    checksummed history/BTCUSDT_1h.parquet.
    """
    if (Path(data_dir) / symbol).exists():
        bars15, _ = load_bars15(data_dir, symbol, start, DESIGN_END)
        bars = resample(bars15, 60)
    else:
        bars = load_1h(HISTORY_1H, start=start, end=DESIGN_END)
    return bars, find_gaps(bars, SNAP_BAR_MS)


def prepare(bars: pd.DataFrame, gaps: list) -> tuple[pd.DataFrame, np.ndarray]:
    feats = compute_features(bars)
    feats = invalidate_after_gaps(feats, gaps, WARMUP_BARS, SNAP_BAR_MS)
    design_start_ms = int(pd.Timestamp(DESIGN_START, tz="UTC").timestamp() * 1000)
    feats = feats[feats["t"] >= design_start_ms].reset_index(drop=True)
    return feats, atr(bars, 14).to_numpy()


def run_grid(bars, feats, atr15, costs: CostModel, configs: list[RuleParams], entry_filter=None) -> dict:
    out = {}
    for p in configs:
        res = run_rules(
            bars, feats, p, costs, atr15, equity0=EQUITY0, risk_pct=1.0, entry_filter=entry_filter
        )
        out[params_hash(p)] = to_config_run(res)
    return out


def write_report(
    path: Path, title: str, m: dict, stress: dict, configs: dict[str, RuleParams], notes: list[str]
):
    ok, fails = evaluate_gates(m, GateThresholds())
    lines = [
        f"# {title}",
        "",
        f"Design period {DESIGN_START} → {DESIGN_END}, anchored walk-forward, 8 out-of-sample half-years "
        "(2022-H1 → 2025-H2). Holdout (2026) untouched. Costs: taker 0.055%/side, slippage 2 bps + 5% of "
        "ATR/price per side, conservative funding (0.01%/8h charged to both sides). Risk 1% of equity per "
        "trade at the stop, notional capped at 2× equity.",
        "",
        "| Gate | Out-of-sample | Required | Pass |",
        "|---|---|---|---|",
        f"| Sharpe (annualised) | {m['sharpe']:.2f} | > 1.5 | {'✅' if m['sharpe'] > 1.5 else '❌'} |",
        f"| Max drawdown | {m['max_dd']:.1%} | < 15% | {'✅' if m['max_dd'] < 0.15 else '❌'} |",
        f"| Hit rate | {m['hit_rate']:.1%} | > 55% | {'✅' if m['hit_rate'] > 0.55 else '❌'} |",
        f"| t-stat (mean trade) | {m['t_stat']:.2f} | > 2.0 | {'✅' if m['t_stat'] > 2.0 else '❌'} |",
        f"| Trades | {m['n_trades']} | ≥ 100 | {'✅' if m['n_trades'] >= 100 else '❌'} |",
        f"| Deflated Sharpe ({m['n_trials']} trials) | {m['deflated_sharpe']:.2f} | > 0.95 | "
        f"{'✅' if m['deflated_sharpe'] > 0.95 else '❌'} |",
        "",
        f"**Verdict: {'PASS' if ok else 'FAIL'}**" + ("" if ok else f" ({'; '.join(fails)})"),
        "",
        f"Total out-of-sample return: {m['total_return']:.1%}. "
        f"Average trade: {m['avg_trade_ret']:.3%} of equity.",
        "",
        "## Cost stress (2× fees and slippage, same picks)",
        "",
        f"Sharpe {stress['sharpe']:.2f}, max drawdown {stress['max_dd']:.1%}, "
        f"hit rate {stress['hit_rate']:.1%}, "
        f"t-stat {stress['t_stat']:.2f}, total return {stress['total_return']:.1%}.",
        "",
        "## Config picked per fold (chosen on in-sample data only)",
        "",
        "| Test fold | Config | RSI max | Stop k×ATR | TP R | Time stop (bars) |",
        "|---|---|---|---|---|---|",
    ]
    for (s, e), p in zip(m["fold_tests"], m["picks"], strict=True):
        c = configs[p]
        lines.append(
            f"| {s} → {e} | `{p}` | {c.rsi_long_max:g} | {c.stop_atr_k:g} | "
            f"{c.take_profit_r:g} | {c.time_stop_bars} |"
        )
    lines += ["", "## Notes", ""] + [f"- {n}" for n in notes] + [""]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))
    return ok


def stress_from_picks(
    bars, feats, atr15, picks: list[str], configs: dict[str, RuleParams], log_label: str
) -> dict:
    stressed = CostModel(multiplier=2.0)
    # Stress reruns the already-picked configs: no new selection, so they are not new trials.
    runs = run_grid(bars, feats, atr15, stressed, [configs[p] for p in dict.fromkeys(picks)])
    tests = [(s, e) for _, _, s, e in folds()]
    r = pd.concat([runs[p].daily_ret.loc[s:e] for p, (s, e) in zip(picks, tests, strict=True)]).to_numpy()
    trades = pd.concat(
        [
            runs[p].trades[(runs[p].trades.entry_day >= s) & (runs[p].trades.entry_day <= e)]
            for p, (s, e) in zip(picks, tests, strict=True)
        ]
    )
    closes = np.cumprod(1 + r)
    return {
        "label": log_label,
        "sharpe": sharpe_annual(r),
        "max_dd": max_drawdown(closes),
        "hit_rate": hit_rate(trades["ret"].to_numpy()),
        "t_stat": t_stat(trades["ret"].to_numpy()),
        "total_return": float(closes[-1] - 1),
    }


def baseline(
    data_dir: Path, reports_dir: Path, trials_path: Path, label: str = "baseline_rules_only"
) -> dict:
    bars, gaps = load_design(data_dir)
    feats, atr15 = prepare(bars, gaps)
    configs = {params_hash(p): p for p in grid()}
    runs = run_grid(bars, feats, atr15, CostModel(), list(configs.values()))
    log = TrialLog(trials_path)
    m = oos_evaluate(runs, log, label=label)
    stress = stress_from_picks(bars, feats, atr15, m["picks"], configs, "baseline_stress_2x")
    notes = [
        f"Bars: {len(bars):,} 15m bars, {len(gaps)} data gap(s); "
        "decisions whose lookback crosses a gap are skipped.",
        "No Jev and no risk guard in this run: the daily-loss and drawdown kill rules arrive with M6.",
        "Funding history is not reachable from the cloud build (Bybit geo-block); the conservative model "
        "makes results worse, never better.",
    ]
    ok = write_report(
        reports_dir / f"{label}.md", f"Baseline: rules only (no Jev), {label}", m, stress, configs, notes
    )
    (reports_dir / f"{label}.json").write_text(json.dumps(m | {"stress": stress, "pass": ok}, indent=2))
    return m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["baseline"])
    p.add_argument("--data-dir", default="data/bars15")
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--trials", default="reports/trials.jsonl")
    p.add_argument("--label", default="baseline_rules_only")
    a = p.parse_args()
    m = baseline(Path(a.data_dir), Path(a.reports_dir), Path(a.trials), a.label)
    print(json.dumps({k: v for k, v in m.items() if k not in ("fold_tests",)}, indent=2, default=str))


if __name__ == "__main__":
    main()
