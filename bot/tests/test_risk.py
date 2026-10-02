import json
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jevbot.config import load_risk_limits
from jevbot.risk.guard import Account, Market, Order, check
from jevbot.risk.killswitch import ErrorTracker, KillSwitch

ROOT = Path(__file__).resolve().parents[1]
LIMITS, _ = load_risk_limits(ROOT / "config" / "risk.yaml")
MIN = 60_000
H = 3_600_000
NOW = 1_700_000_000_000 - (1_700_000_000_000 % (8 * H)) + 4 * H  # 4h after a settlement: no blackout


def _acct(**kw) -> Account:
    base = dict(
        equity=10_000.0,
        peak_equity=10_000.0,
        day_start_equity=10_000.0,
        open_positions=0,
        order_attempts_last_min=0,
        halted=False,
    )
    return Account(**(base | kw))


def _mkt(**kw) -> Market:
    base = dict(now_ms=NOW, last_bar_close_ms=NOW - 5_000, bar_ms=15 * MIN, mid=100.0, spread_bps=1.0)
    return Market(**(base | kw))


def _order(**kw) -> Order:
    # risk at stop: 20 units x $1 = $20 = 0.2% of equity; notional $2,000 (under the $2,500 approval line)
    base = dict(side=1, units=20.0, ref_px=100.0, stop_px=99.0)
    return Order(**(base | kw))


def test_clean_order_passes():
    r = check(_order(), _acct(), _mkt(), LIMITS)
    assert r.ok and r.reasons == [] and not r.requires_approval and not r.kill


@pytest.mark.parametrize(
    "order_kw, acct_kw, mkt_kw, reason",
    [
        ({}, {"halted": True}, {}, "halted"),
        ({}, {"open_positions": 1}, {}, "max_open_positions"),
        ({"units": 150.0}, {}, {}, "max_risk_per_trade"),  # $150 at stop = 1.5%
        ({"units": 250.0, "stop_px": 99.9}, {}, {}, "max_notional"),  # $25k = 2.5x
        ({}, {"equity": 9_650.0, "peak_equity": 10_000.0}, {}, "daily_loss_limit"),
        ({}, {"order_attempts_last_min": 3}, {}, "max_order_attempts"),
        ({}, {}, {"last_bar_close_ms": NOW - 31 * MIN}, "stale_data"),
        ({}, {}, {"spread_bps": 3.5}, "spread"),
        ({}, {}, {"spread_bps": None}, "spread_unknown"),
        (
            {},
            {},
            {"now_ms": NOW + 4 * H - 5 * MIN, "last_bar_close_ms": NOW + 4 * H - 6 * MIN},
            "funding_blackout",
        ),
        ({}, {}, {"now_ms": NOW - 4 * H + 3 * MIN, "last_bar_close_ms": NOW - 4 * H}, "funding_blackout"),
        ({"ref_px": 102.0, "stop_px": 101.0}, {}, {}, "price_band"),
        ({"stop_px": 101.0}, {}, {}, "stop_wrong_side"),
        ({"units": 0.0}, {}, {}, "bad_size"),
    ],
)
def test_each_limit_vetoes(order_kw, acct_kw, mkt_kw, reason):
    r = check(_order(**order_kw), _acct(**acct_kw), _mkt(**mkt_kw), LIMITS)
    assert not r.ok and reason in r.reasons, r.reasons


def test_drawdown_breach_vetoes_and_requests_kill():
    r = check(_order(), _acct(equity=8_990.0, peak_equity=10_000.0, day_start_equity=9_000.0), _mkt(), LIMITS)
    assert not r.ok and "max_drawdown" in r.reasons and r.kill


def test_spread_may_be_unmodelled_in_backtest_only():
    r = check(_order(), _acct(), _mkt(spread_bps=None), LIMITS, backtest=True)
    assert r.ok


def test_large_order_needs_manual_approval_but_is_not_vetoed():
    big = 100_000.0
    r = check(
        _order(units=260.0, stop_px=99.0),
        _acct(equity=big, peak_equity=big, day_start_equity=big),
        _mkt(),
        LIMITS,
    )
    assert r.ok and r.requires_approval  # $26,000 notional > $25,000


def test_typical_paper_trade_does_not_need_approval():
    # $10k account, 0.25% risk at a 0.4% stop = ~$6.25k notional: runs without a tap
    r = check(_order(units=62.5, stop_px=99.6), _acct(), _mkt(), LIMITS)
    assert r.ok and not r.requires_approval


# --- property tests: the guard never passes an order that breaks a limit ----------------------

orders = st.builds(
    Order,
    side=st.sampled_from([1, -1]),
    units=st.floats(0.0, 2_000.0),
    ref_px=st.floats(50.0, 150.0),
    stop_px=st.floats(40.0, 160.0),
)
accounts = st.builds(
    Account,
    equity=st.floats(1_000.0, 50_000.0),
    peak_equity=st.floats(1_000.0, 60_000.0),
    day_start_equity=st.floats(1_000.0, 60_000.0),
    open_positions=st.integers(0, 2),
    order_attempts_last_min=st.integers(0, 5),
    halted=st.booleans(),
)
markets = st.builds(
    Market,
    now_ms=st.integers(NOW - 8 * H, NOW + 8 * H),
    last_bar_close_ms=st.integers(NOW - 9 * H, NOW + 8 * H),
    bar_ms=st.just(15 * MIN),
    mid=st.floats(50.0, 150.0),
    spread_bps=st.one_of(st.none(), st.floats(0.0, 10.0)),
)


@settings(max_examples=3000, deadline=None)
@given(orders, accounts, markets)
def test_guard_never_passes_a_breach(o, a, m):
    a = replace(a, peak_equity=max(a.peak_equity, a.equity))
    r = check(o, a, m, LIMITS)
    if not r.ok:
        return
    L = LIMITS
    risk = o.units * abs(o.ref_px - o.stop_px)
    assert not a.halted
    assert a.open_positions < L.max_open_positions
    assert o.units > 0
    assert risk <= a.equity * L.max_risk_per_trade_pct / 100 * (1 + 1e-9)
    assert o.units * o.ref_px <= a.equity * min(L.max_notional_x_equity, L.max_leverage) * (1 + 1e-9)
    assert (a.day_start_equity - a.equity) / a.day_start_equity < L.daily_loss_limit_pct / 100
    assert (a.peak_equity - a.equity) / a.peak_equity < L.max_drawdown_pct / 100
    assert a.order_attempts_last_min < L.max_order_attempts_per_min
    assert m.spread_bps is not None and m.spread_bps <= L.max_spread_bps
    assert 0 <= m.now_ms - m.last_bar_close_ms <= L.stale_data_multiple * m.bar_ms
    assert abs(o.ref_px / m.mid - 1) <= L.price_band_pct / 100
    assert (o.stop_px - o.ref_px) * o.side < 0


# --- kill switch -----------------------------------------------------------------------------


class FakeBroker:
    def __init__(self, position=(1, 3.0), fail=False):
        self.pos = position
        self.fail = fail
        self.calls = []

    def cancel_all(self):
        self.calls.append("cancel_all")
        if self.fail:
            raise RuntimeError("exchange down")

    def position(self):
        return self.pos

    def close_reduce_only(self, side, units):
        self.calls.append(("close", side, units))
        self.pos = (0, 0.0)


def test_kill_cancels_flattens_reduce_only_and_halts(tmp_path):
    b, alerts = FakeBroker(), []
    k = KillSwitch(b, tmp_path / "HALTED", alerts.append)
    k.fire("max_drawdown", now_ms=NOW)
    assert b.calls == ["cancel_all", ("close", 1, 3.0)]
    assert k.is_halted()
    assert json.loads((tmp_path / "HALTED").read_text())["reason"] == "max_drawdown"
    assert alerts and "KILL" in alerts[0]


def test_kill_is_idempotent(tmp_path):
    b = FakeBroker()
    k = KillSwitch(b, tmp_path / "HALTED", lambda m: None)
    k.fire("manual", now_ms=NOW)
    k.fire("manual", now_ms=NOW)
    assert b.calls.count(("close", 1, 3.0)) == 1


def test_kill_still_halts_and_alerts_when_the_exchange_fails(tmp_path):
    b, alerts = FakeBroker(fail=True), []
    k = KillSwitch(b, tmp_path / "HALTED", alerts.append)
    k.fire("order_errors", now_ms=NOW)
    assert k.is_halted()
    assert any("FAILED" in a for a in alerts)


def test_only_a_human_reset_clears_the_halt(tmp_path):
    k = KillSwitch(FakeBroker(), tmp_path / "HALTED", lambda m: None)
    k.fire("manual", now_ms=NOW)
    with pytest.raises(PermissionError):
        k.reset(confirmed_by="")
    k.reset(confirmed_by="telegram:owner")
    assert not k.is_halted()


def test_three_errors_in_ten_minutes_fire_the_kill(tmp_path):
    fired = []
    t = ErrorTracker(LIMITS, on_trip=fired.append)
    t.record(NOW)
    t.record(NOW + 4 * MIN)
    assert not fired
    t.record(NOW + 9 * MIN)
    assert fired == ["order_errors"]


def test_errors_spread_over_more_than_the_window_do_not_fire():
    fired = []
    t = ErrorTracker(LIMITS, on_trip=fired.append)
    for i in range(3):
        t.record(NOW + i * 6 * MIN)
    assert not fired
