"""Hard pre-trade checks (spec §8). Run before EVERY order. No model output reaches this module.

check() returns every failed rule, not only the first, so the journal shows the whole picture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jevbot.config import RiskLimits

FUNDING_MS = 8 * 3_600_000
EPS = 1e-9


@dataclass(frozen=True)
class Order:
    side: int  # +1 long, -1 short
    units: float
    ref_px: float  # expected fill price (market) or limit price
    stop_px: float


@dataclass(frozen=True)
class Account:
    equity: float
    peak_equity: float
    day_start_equity: float
    open_positions: int
    order_attempts_last_min: int
    halted: bool


@dataclass(frozen=True)
class Market:
    now_ms: int
    last_bar_close_ms: int
    bar_ms: int
    mid: float
    spread_bps: float | None


@dataclass
class GuardResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    requires_approval: bool = False
    kill: bool = False


def check(o: Order, a: Account, m: Market, L: RiskLimits, backtest: bool = False) -> GuardResult:
    r: list[str] = []
    kill = False

    if a.halted:
        r.append("halted")
    if a.open_positions >= L.max_open_positions:
        r.append("max_open_positions")
    if not (o.units > 0 and o.ref_px > 0 and a.equity > 0) or o.side not in (1, -1):
        r.append("bad_size")
    if (o.stop_px - o.ref_px) * o.side >= 0:
        r.append("stop_wrong_side")

    risk = o.units * abs(o.ref_px - o.stop_px)
    if risk > a.equity * L.max_risk_per_trade_pct / 100 * (1 + EPS):
        r.append("max_risk_per_trade")
    cap_x = min(L.max_notional_x_equity, L.max_leverage)
    if o.units * o.ref_px > a.equity * cap_x * (1 + EPS):
        r.append("max_notional")

    if (
        a.day_start_equity <= 0
        or (a.day_start_equity - a.equity) / a.day_start_equity >= L.daily_loss_limit_pct / 100
    ):
        r.append("daily_loss_limit")
    peak = max(a.peak_equity, a.equity)
    if peak <= 0 or (peak - a.equity) / peak >= L.max_drawdown_pct / 100:
        r.append("max_drawdown")
        kill = True
    if a.order_attempts_last_min >= L.max_order_attempts_per_min:
        r.append("max_order_attempts")

    age = m.now_ms - m.last_bar_close_ms
    if age < 0 or age > L.stale_data_multiple * m.bar_ms:
        r.append("stale_data")
    if m.spread_bps is None:
        if not backtest:
            r.append("spread_unknown")
    elif m.spread_bps > L.max_spread_bps:
        r.append("spread")
    since = m.now_ms % FUNDING_MS
    until = FUNDING_MS - since
    if min(since, until) < L.funding_blackout_min * 60_000:
        r.append("funding_blackout")
    if not m.mid > 0 or abs(o.ref_px / m.mid - 1) > L.price_band_pct / 100:
        r.append("price_band")

    ok = not r
    return GuardResult(
        ok=ok,
        reasons=r,
        requires_approval=ok and o.units * o.ref_px > L.manual_approval_notional_usd,
        kill=kill,
    )
