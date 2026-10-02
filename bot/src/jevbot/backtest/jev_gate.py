"""M7: the gated backtest of rules + Jev (spec §9). One command:

    python -m jevbot.backtest.jev_gate --plan-only     # count Jev calls and estimate cost, spend nothing
    python -m jevbot.backtest.jev_gate                 # real Jev (needs TYPESAFE_API_KEY), cached
    python -m jevbot.backtest.jev_gate --fake          # pipeline dry run with FakeJev (meaningless numbers)

Out-of-sample metrics come from the anchored walk-forward in jev_run. The candidate must clear
every gate AND beat the rules-only baseline. Only then is the holdout (2026) evaluated, once.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from jevbot.backtest.engine import CostModel
from jevbot.backtest.gates import GateThresholds, evaluate_gates
from jevbot.backtest.harness import load_design as _load_design
from jevbot.backtest.harness import metrics_from_parts, prepare
from jevbot.backtest.jev_run import FoldPick, fetch_answers, plan_calls, walk_forward_jev
from jevbot.backtest.trials import TrialLog
from jevbot.config import Secrets, load_risk_limits, load_settings
from jevbot.jev.client import JEV_USD_PER_INPUT_TOKEN, JevCache, JevClient, JevFailure, JevResult
from jevbot.jev.fake import FakeJev
from jevbot.jev.questions import build_state, load_questions
from jevbot.strategy.calibration import brier, ece
from jevbot.strategy.params import grid

ROOT = Path(__file__).resolve().parents[3]
CHARS_PER_TOKEN = 3.5  # rough, for the estimate only; the real count comes back in `usage`


# --- pure helpers (tested) ------------------------------------------------------------------


def estimate(n_calls: int, example_state: dict, wire: dict, rate_per_s: float) -> dict:
    chars = len(json.dumps({"state": example_state, "questions": wire}))
    tokens = chars / CHARS_PER_TOKEN
    return {
        "calls": n_calls,
        "tokens_per_call_est": int(tokens),
        "cost_usd_est": round(n_calls * tokens * JEV_USD_PER_INPUT_TOKEN, 2),
        "minutes_est": round(n_calls / rate_per_s / 60, 1),
    }


def answer_stats(answers: dict) -> dict:
    res = [a for a in answers.values() if isinstance(a, JevResult)]
    fails = [a for a in answers.values() if isinstance(a, JevFailure)]
    return {
        "calls": len(answers),
        "ok_trusted": sum(a.trusted for a in res),
        "untrusted": Counter(a.untrusted_reason for a in res if not a.trusted),
        "failures": Counter(a.reason for a in fails),
        "models": Counter(a.model for a in res),
        "cost_usd": round(sum(a.cost_usd for a in res), 4),
        "cached": sum(a.cached for a in res),
        "mean_latency_ms": round(float(np.mean([a.latency_ms for a in res if not a.cached] or [0.0])), 1),
    }


def stitch(picks: list[FoldPick]) -> tuple[pd.Series, pd.Series, list[pd.DataFrame]]:
    daily, mins, trades = [], [], []
    for p in picks:
        if p.part is None:  # no config could be fitted for this fold: it sits out (flat), honestly
            days = pd.date_range(p.test[0], p.test[1], freq="D")
            daily.append(pd.Series(0.0, index=days))
            mins.append(pd.Series(0.0, index=days))
            continue
        daily.append(p.part.daily_ret)
        mins.append(p.part.daily_min_ret)
        trades.append(p.part.trades)
    return pd.concat(daily), pd.concat(mins), trades


def direction_calibration(
    answers: dict, feats: pd.DataFrame, bars: pd.DataFrame, horizon_h: int = 12
) -> dict:
    """Score Jev's `direction` probabilities against what happened over the next 12 hours.

    'Up'/'down' = a 12h move beyond half a typical day's move (0.5 x 1h sigma x sqrt(24)), matching
    the question's wording. Uses every trusted answer, candidates and the random sample alike.
    """
    close = pd.Series(bars["close"].to_numpy(), index=bars["available_at"].to_numpy(np.int64))
    by_t = feats.set_index("t")
    p_up, p_dn, y_up, y_dn = [], [], [], []
    for (t, _side), a in answers.items():
        if not isinstance(a, JevResult) or not a.trusted:
            continue
        t_end = t + horizon_h * 3_600_000
        if t_end not in close.index or t not in by_t.index:
            continue
        sigma_1h = by_t.loc[t, "rv_7d_ann_pct"] / 100 / math.sqrt(365 * 24)
        thr = 0.5 * sigma_1h * math.sqrt(24)
        r = math.log(close[t_end] / close[t])
        pr = a.answers["direction"]["probabilities"]
        p_up.append(pr["up"])
        p_dn.append(pr["down"])
        y_up.append(int(r > thr))
        y_dn.append(int(r < -thr))
    if not p_up:
        return {"n": 0}
    pu, pd_, yu, yd = map(np.asarray, (p_up, p_dn, y_up, y_dn))
    return {
        "n": len(pu),
        "brier_up": brier(pu, yu),
        "ece_up": ece(pu, yu),
        "base_rate_up": float(yu.mean()),
        "brier_up_climatology": brier(np.full_like(pu, yu.mean()), yu),
        "brier_down": brier(pd_, yd),
        "ece_down": ece(pd_, yd),
        "base_rate_down": float(yd.mean()),
        "brier_down_climatology": brier(np.full_like(pd_, yd.mean()), yd),
    }


# --- the run --------------------------------------------------------------------------------


async def _probe_model(client: JevClient, snap: dict) -> str:
    r = await client.ask(snap, 0)
    if isinstance(r, JevFailure):
        raise SystemExit(f"Jev probe call failed: {r.reason} {r.detail}")
    return r.model


def write_report(path: Path, m: dict, ok: bool, fails: list[str], rules_sharpe: float, stats: dict,
                 cal: dict, picks: list[FoldPick], est: dict, fake: bool, pinned: str) -> None:  # fmt: skip
    vetoes: Counter = Counter()
    approvals = 0
    for p in picks:
        if p.log:
            vetoes.update(p.log.vetoes)
            approvals += p.log.approvals_needed
    beats = m["sharpe"] > rules_sharpe
    title = "Gated backtest: rules + Jev" + (" — FAKE JEV DRY RUN, NUMBERS ARE MEANINGLESS" if fake else "")
    L = [
        f"# {title}",
        "",
        f"Jev model: `{pinned}`. 1h entries, anchored walk-forward 2022-H1 → 2025-H2, weights and calibrator fitted "
        "on earlier trades only, decide() + full risk guard in every trade, fresh account per window. "
        "Holdout (2026) untouched unless every gate passes.",
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
        f"| Beats rules only (Sharpe {rules_sharpe:.2f}) | {m['sharpe']:.2f} | higher | {'✅' if beats else '❌'} |",
        "",
        f"**Verdict: {'PASS' if ok and beats else 'FAIL'}**"
        + (
            "" if ok and beats else f" ({'; '.join(fails + ([] if beats else ['does not beat rules only']))})"
        ),
        "",
        f"Total out-of-sample return {m['total_return']:.1%}, average trade {m['avg_trade_ret']:.3%} of equity. "
        f"Orders above the $25k approval line: {approvals} (assumed approved in the backtest).",
        "",
        "## Jev calls",
        "",
        f"Planned {est['calls']} calls (estimate ${est['cost_usd_est']}). Trusted answers {stats['ok_trusted']}, "
        f"untrusted {dict(stats['untrusted'])}, failures {dict(stats['failures'])}, models {dict(stats['models'])}, "
        f"cost ${stats['cost_usd']}, {stats['cached']} from cache, mean latency {stats['mean_latency_ms']} ms.",
        "",
        "## Why candidates did not trade (out-of-sample windows)",
        "",
        "| Reason | Count |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in vetoes.most_common()],
        "",
        "## Calibration of the `direction` question (12h outcome)",
        "",
    ]
    if cal.get("n"):
        L += [
            f"{cal['n']} answers. P(up): Brier {cal['brier_up']:.4f} vs {cal['brier_up_climatology']:.4f} for "
            f"always guessing the base rate {cal['base_rate_up']:.1%}; ECE {cal['ece_up']:.3f}. "
            f"P(down): Brier {cal['brier_down']:.4f} vs {cal['brier_down_climatology']:.4f} "
            f"(base rate {cal['base_rate_down']:.1%}); ECE {cal['ece_down']:.3f}. "
            "Lower Brier than the base-rate guess means Jev adds information.",
        ]
    L += ["", "## Config and model per fold", "", "| Test fold | Config | In-sample Sharpe (daily) | p_cutoff |",
          "|---|---|---|---|"]  # fmt: skip
    for p in picks:
        cut = f"{p.model[0].p_cutoff:.2f}" if p.model else "-"
        L.append(f"| {p.test[0]} → {p.test[1]} | `{p.config}` | {p.is_sharpe:.3f} | {cut} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")


async def run(args) -> int:
    settings = load_settings(ROOT / "config" / "settings.yaml")
    limits, _ = load_risk_limits(ROOT / "config" / "risk.yaml")
    q = load_questions(ROOT / "active" / "questions.yaml")
    bars, gaps = _load_design(ROOT / "data" / "bars15")
    feats, atr15 = prepare(bars, gaps)
    configs = grid()
    calls = plan_calls(bars, feats, configs, sample_frac=0.10)
    by_t = feats.set_index("t")
    example = build_state({k: float(v) for k, v in by_t.iloc[0].items()}, 1, q.legend)
    est = estimate(len(calls), example, q.wire(), args.rate)
    print(json.dumps({"plan": est}, indent=1))
    if args.plan_only:
        return 0

    if args.fake:
        fake = FakeJev()

        async def ask_many(items):
            return [await fake.ask(s, side) for s, side in items]

        pinned = fake.model
        trials = TrialLog(ROOT / "reports" / "trials_fake.jsonl")
        report = ROOT / "reports" / "jev_gate_FAKE.md"
    else:
        secrets = Secrets(_env_file=ROOT / ".env")
        secrets.require("TYPESAFE_API_KEY")
        probe = JevClient(q, "live", settings.jev.model, "", secrets.TYPESAFE_API_KEY.get_secret_value())
        first = {k: float(v) for k, v in by_t.iloc[0].items()}
        pinned = settings.jev.pinned_model or await _probe_model(probe, first)
        print(f"Jev model pinned for this run: {pinned}")
        client = JevClient(
            q, "backtest", settings.jev.model, pinned, secrets.TYPESAFE_API_KEY.get_secret_value(),
            cache=JevCache(ROOT / "data" / "jev_cache.sqlite"),
        )  # fmt: skip

        async def ask_many(items):
            return await client.ask_many(items, concurrency=args.concurrency, rate_per_s=args.rate)

        trials = TrialLog(ROOT / "reports" / "trials.jsonl")
        report = ROOT / "reports" / "jev_gate_v1.md"

    answers = await fetch_answers(ask_many, feats, calls)
    stats = answer_stats(answers)
    print(json.dumps({"answers": {k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()}}))
    bad = sum(stats["failures"].values()) + sum(stats["untrusted"].values())
    if bad > 0.01 * len(answers):
        print(f"ABORT: {bad} of {len(answers)} answers failed or untrusted (> 1%). Nothing evaluated.")
        return 2

    picks = walk_forward_jev(bars, feats, atr15, configs, answers, limits, CostModel())
    last = picks[-1].is_sharpes or {}
    for cid, sr in last.items():
        trials.append({"kind": "config", "label": "jev_v1" + ("_fake" if args.fake else ""), "id": cid,
                       "sharpe_daily": sr})  # fmt: skip
    daily, mins, trade_parts = stitch(picks)
    m = metrics_from_parts(daily, mins, trade_parts, trials.trial_sharpes(), "jev_v1")
    ok, fails = evaluate_gates(m, GateThresholds())
    rules = json.loads((ROOT / "reports" / "baseline_rules_only_v3_1h.json").read_text())
    cal = direction_calibration(answers, feats, bars)
    write_report(report, m, ok, fails, rules["sharpe"], stats, cal, picks, est, args.fake, pinned)
    (report.with_suffix(".json")).write_text(json.dumps(m | {"pass": ok and m["sharpe"] > rules["sharpe"],
                                                             "calibration": cal}, indent=1, default=str))  # fmt: skip
    print(f"report: {report}")
    if ok and m["sharpe"] > rules["sharpe"] and not args.fake:
        print("ALL GATES PASSED. The holdout has NOT been touched. Run the holdout step only after review.")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--fake", action="store_true")
    p.add_argument("--rate", type=float, default=18.0, help="Jev requests per second (limit is 20)")
    p.add_argument("--concurrency", type=int, default=16)
    a = p.parse_args()
    sys.exit(asyncio.run(run(a)))


if __name__ == "__main__":
    main()
