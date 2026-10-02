"""Bar-by-bar backtest engine (spec §9, architecture §3 `backtest/`).

Order of events in each bar i:
  1. a pending exit (from bar i-1's close) fills at open[i]
  2. a pending entry (signal at bar i-1's close) fills at open[i]
  3. intrabar stop / take profit: gap through the stop exits at the open; if both are touched,
     the stop wins
  4. at close[i]: mark to market; time stop or 1h-trend flip -> exit at open[i+1]; if the next
     bar is missing (data gap) -> exit now at close[i]
  5. if flat: a signal at close[i] -> entry at open[i+1] (only if bar i+1 directly follows)

Costs: taker fee on both legs; slippage of base + k x ATR/price per side (against us on every
fill, take profit included, because Bybit TP/SL trigger market orders); funding per settlement
held in (entry, exit] using the conservative model (data.funding).

`entry_filter(i, side) -> risk_pct` is the hook where M7 plugs in Jev + decide() + the risk
guard. It returns the % of equity to risk; 0 means no trade.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from jevbot.data.funding import conservative_funding_cost
from jevbot.strategy.params import RuleParams
from jevbot.strategy.rules import exit_levels

BAR_MS = 15 * 60 * 1000

EntryFilter = Callable[[int, int], float]


@dataclass(frozen=True)
class CostModel:
    fee_rate: float = 0.00055  # Bybit taker, per side
    base_slip: float = 0.0002  # 2 bps per side
    vol_slip_k: float = 0.05  # + 5% of ATR/price per side
    multiplier: float = 1.0  # 2.0 = the cost stress test

    def slip(self, atr: float, price: float) -> float:
        return self.multiplier * (self.base_slip + self.vol_slip_k * atr / price)

    @property
    def fee(self) -> float:
        return self.multiplier * self.fee_rate


@dataclass
class Result:
    trades: pd.DataFrame
    bar_equity: np.ndarray
    daily: pd.DataFrame
    equity_end: float


TRADE_COLUMNS = [
    "side",
    "signal_i",
    "entry_time",
    "exit_time",
    "entry_px",
    "exit_px",
    "sl_px",
    "tp_px",
    "units",
    "risk_pct",
    "fees",
    "funding",
    "pnl",
    "ret",
    "reason",
]


def run(
    bars: pd.DataFrame,
    signal: np.ndarray,
    atr: np.ndarray,
    slope_sign_1h: np.ndarray,
    rules: RuleParams,
    costs: CostModel,
    equity0: float = 10_000.0,
    risk_pct: float = 1.0,
    max_notional_x: float = 2.0,
    entry_filter: EntryFilter | None = None,
) -> Result:
    ot = bars["open_time"].to_numpy(np.int64)
    o, h, lo, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(bars)
    contiguous_next = np.zeros(n, dtype=bool)
    contiguous_next[:-1] = np.diff(ot) == BAR_MS

    equity = equity0
    bar_equity = np.empty(n)
    trades: list[dict] = []
    pos: dict | None = None
    pending_entry: tuple[int, int, float] | None = None  # (side, signal_i, risk_pct)
    pending_exit: str | None = None

    def close_pos(px: float, t: int, reason: str) -> None:
        nonlocal pos, equity
        assert pos is not None
        u, s = pos["units"], pos["side"]
        exit_fee = costs.fee * u * px
        funding = conservative_funding_cost(s, u * pos["entry_px"], pos["entry_time"], t)
        pnl = s * u * (px - pos["entry_px"]) - pos["entry_fee"] - exit_fee - funding
        trades.append(
            pos
            | {
                "exit_time": t,
                "exit_px": px,
                "fees": pos["entry_fee"] + exit_fee,
                "funding": funding,
                "pnl": pnl,
                "ret": pnl / equity,
                "reason": reason,
            }
        )
        equity += pnl
        pos = None

    for i in range(n):
        if pos is not None and pending_exit is not None:
            slip = costs.slip(pos["atr"], o[i])
            close_pos(o[i] * (1 - pos["side"] * slip), int(ot[i]), pending_exit)
        pending_exit = None

        if pending_entry is not None:
            side, si, rp = pending_entry
            pending_entry = None
            entry_px = o[i] * (1 + side * costs.slip(atr[si], o[i]))
            sl, tp = exit_levels(entry_px, side, atr[si], rules)
            units = min(equity * rp / 100.0 / abs(entry_px - sl), max_notional_x * equity / entry_px)
            pos = {
                "side": side,
                "signal_i": si,
                "entry_time": int(ot[i]),
                "entry_px": entry_px,
                "sl_px": sl,
                "tp_px": tp,
                "units": units,
                "risk_pct": rp,
                "atr": atr[si],
                "entry_fee": costs.fee * units * entry_px,
                "held": 0,
            }

        if pos is not None:
            s, sl, tp = pos["side"], pos["sl_px"], pos["tp_px"]
            slip = costs.slip(pos["atr"], o[i])
            bar_close_t = int(ot[i]) + BAR_MS
            if (s == 1 and o[i] <= sl) or (s == -1 and o[i] >= sl):
                close_pos(o[i] * (1 - s * slip), int(ot[i]), "sl")
            elif (s == 1 and lo[i] <= sl) or (s == -1 and h[i] >= sl):
                close_pos(sl * (1 - s * slip), bar_close_t, "sl")
            elif (s == 1 and h[i] >= tp) or (s == -1 and lo[i] <= tp):
                close_pos(tp * (1 - s * slip), bar_close_t, "tp")

        if pos is not None:
            pos["held"] += 1
            if not contiguous_next[i]:
                slip = costs.slip(pos["atr"], c[i])
                close_pos(c[i] * (1 - pos["side"] * slip), int(ot[i]) + BAR_MS, "data_gap")
            elif pos["held"] >= rules.time_stop_bars:
                pending_exit = "time"
            elif slope_sign_1h[i] == -pos["side"]:
                pending_exit = "invalidated"

        unreal = (
            0.0 if pos is None else pos["side"] * pos["units"] * (c[i] - pos["entry_px"]) - pos["entry_fee"]
        )
        bar_equity[i] = equity + unreal

        if pos is None and pending_entry is None and signal[i] != 0 and contiguous_next[i] and atr[i] > 0:
            rp = risk_pct if entry_filter is None else entry_filter(i, int(signal[i]))
            if rp > 0:
                pending_entry = (int(signal[i]), i, rp)

    if pos is not None:  # still open at the end of the data: close at the last close
        close_pos(c[-1] * (1 - pos["side"] * costs.slip(pos["atr"], c[-1])), int(ot[-1]) + BAR_MS, "end")
        bar_equity[-1] = equity

    tr = pd.DataFrame(trades, columns=TRADE_COLUMNS) if trades else pd.DataFrame(columns=TRADE_COLUMNS)
    day = pd.to_datetime(ot, unit="ms", utc=True).floor("D")
    eq = pd.Series(bar_equity, index=day)
    daily = (
        pd.DataFrame({"equity_close": eq.groupby(level=0).last(), "equity_min": eq.groupby(level=0).min()})
        .rename_axis("day")
        .reset_index()
    )
    return Result(trades=tr, bar_equity=bar_equity, daily=daily, equity_end=equity)


def run_rules(
    bars15: pd.DataFrame,
    feats: pd.DataFrame,
    rules: RuleParams,
    costs: CostModel,
    atr15: np.ndarray,
    **kw,
) -> Result:
    """Align features (keyed by decision time t) to bars and run the rules.

    Bars with no valid features (warm-up, or a lookback that crosses a data gap) get no signal.
    """
    from jevbot.strategy.rules import candidates

    t = bars15["available_at"].to_numpy(np.int64)
    f = feats.set_index("t")
    aligned = f.reindex(t)
    valid = aligned.notna().all(axis=1).to_numpy()
    sig = np.zeros(len(t), dtype=int)
    sig[valid] = candidates(aligned[valid].reset_index(), rules).to_numpy()
    slope = np.sign(aligned["ema50_slope_1h_atr"].fillna(0.0).to_numpy())
    return run(bars15, sig, atr15, slope, rules, costs, **kw)
