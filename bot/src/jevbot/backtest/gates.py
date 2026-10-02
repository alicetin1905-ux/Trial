"""Acceptance gates and the holdout (spec §9). The ONLY module allowed to touch holdout data."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from jevbot.backtest.trials import TrialLog
from jevbot.backtest.walkforward import DESIGN_END

HOLDOUT_START = "2026-01-01"
HOLDOUT_END = "2026-09-30"


class HoldoutAlreadyUsed(RuntimeError):
    pass


@dataclass(frozen=True)
class GateThresholds:
    sharpe: float = 1.5  # annualised, daily returns, after costs
    max_dd: float = 0.15
    hit_rate: float = 0.55
    t_stat: float = 2.0
    min_trades: int = 100
    deflated_sharpe: float = 0.95


def evaluate_gates(m: dict, th: GateThresholds) -> tuple[bool, list[str]]:
    fails = []
    if not m["sharpe"] > th.sharpe:
        fails.append(f"sharpe {m['sharpe']:.2f} <= {th.sharpe}")
    if not m["max_dd"] < th.max_dd:
        fails.append(f"max_dd {m['max_dd']:.1%} >= {th.max_dd:.0%}")
    if not m["hit_rate"] > th.hit_rate:
        fails.append(f"hit_rate {m['hit_rate']:.1%} <= {th.hit_rate:.0%}")
    if not m["t_stat"] > th.t_stat:
        fails.append(f"t_stat {m['t_stat']:.2f} <= {th.t_stat}")
    if not m["n_trades"] >= th.min_trades:
        fails.append(f"n_trades {m['n_trades']} < {th.min_trades}")
    if not m["deflated_sharpe"] > th.deflated_sharpe:
        fails.append(f"deflated_sharpe {m['deflated_sharpe']:.2f} <= {th.deflated_sharpe}")
    return (not fails), fails


def holdout_once(log: TrialLog, candidate_id: str, evaluate: Callable[[], dict]) -> dict:
    """Evaluate a candidate on the holdout exactly once, ever. The use is logged before evaluating."""
    if any(r.get("kind") == "holdout" and r.get("candidate") == candidate_id for r in log.read()):
        raise HoldoutAlreadyUsed(candidate_id)
    log.append({"kind": "holdout", "candidate": candidate_id, "status": "started"})
    result = evaluate()
    log.append({"kind": "holdout_result", "candidate": candidate_id, "result": result})
    return result


assert DESIGN_END < HOLDOUT_START
