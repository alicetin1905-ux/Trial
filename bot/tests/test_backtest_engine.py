import numpy as np
import pandas as pd
import pytest

from jevbot.backtest.engine import CostModel, run
from jevbot.strategy.params import RuleParams

BAR = 15 * 60 * 1000
H = 3600 * 1000
T1 = 1609462800000  # 2021-01-01 01:00 UTC: no funding settlement for the next 7 hours

RULES = RuleParams(
    pullback_min_atr=1.0,
    pullback_max_atr=2.5,
    rsi_long_max=40,
    stop_atr_k=1.0,
    take_profit_r=2.0,
    time_stop_bars=4,
    direction="both",
)
NO_COST = CostModel(fee_rate=0.0, base_slip=0.0, vol_slip_k=0.0)


def _bars(ohlc, start=T1):
    n = len(ohlc)
    ot = start + np.arange(n, dtype=np.int64) * BAR
    o, h, lo, c = (np.array(x, dtype=float) for x in zip(*ohlc, strict=True))
    return pd.DataFrame(
        {"open_time": ot, "open": o, "high": h, "low": lo, "close": c, "available_at": ot + BAR}
    )


def _run(bars, signal_at=0, side=1, atr=1.0, slope=None, costs=NO_COST, rules=RULES, **kw):
    n = len(bars)
    sig = np.zeros(n, dtype=int)
    if signal_at is not None:
        sig[signal_at] = side
    slope = np.full(n, side) if slope is None else np.asarray(slope)
    return run(bars, sig, np.full(n, atr), slope, rules, costs, equity0=10_000.0, risk_pct=1.0, **kw)


FLAT = (100.0, 100.5, 99.5, 100.0)


def test_long_enters_next_open_and_exits_at_take_profit():
    # signal at bar 0 close; entry at bar 1 open = 100; SL 99, TP 102; TP touched in bar 3
    r = _run(_bars([FLAT, FLAT, FLAT, (100.0, 102.5, 99.6, 102.0), FLAT]))
    t = r.trades.iloc[0]
    assert (t.entry_px, t.exit_px, t.reason) == (100.0, 102.0, "tp")
    assert t.entry_time == T1 + BAR
    # risk 1% of 10k = $100 at a $1 stop -> 100 units, but capped at 2x equity notional = 200 units? no:
    # 100 units x $100 = $10k notional = 1x equity, under the 2x cap
    assert t.units == pytest.approx(100.0)
    assert t.pnl == pytest.approx(200.0)
    assert r.equity_end == pytest.approx(10_200.0)


def test_stop_wins_when_both_levels_touched_in_one_bar():
    r = _run(_bars([FLAT, FLAT, (100.0, 103.0, 98.0, 100.0), FLAT]))
    assert r.trades.iloc[0].reason == "sl" and r.trades.iloc[0].exit_px == 99.0


def test_gap_through_stop_exits_at_the_worse_open():
    r = _run(_bars([FLAT, FLAT, (97.0, 97.5, 96.5, 97.0), FLAT]))
    t = r.trades.iloc[0]
    assert (t.reason, t.exit_px) == ("sl", 97.0)


def test_short_mirrors_long():
    r = _run(_bars([FLAT, FLAT, (100.0, 100.4, 97.5, 98.0), FLAT]), side=-1)
    t = r.trades.iloc[0]
    assert (t.side, t.reason, t.exit_px) == (-1, "tp", 98.0)
    assert t.pnl == pytest.approx(200.0)


def test_time_stop_exits_at_next_open_after_n_bars():
    r = _run(_bars([FLAT] * 8))
    t = r.trades.iloc[0]
    assert t.reason == "time"
    assert t.exit_time == T1 + 5 * BAR  # entered bar 1, held bars 1-4, exit at bar 5 open


def test_trend_flip_exits_at_next_open():
    slope = [1, 1, -1, 1, 1, 1]
    r = _run(_bars([FLAT] * 6), slope=slope)
    t = r.trades.iloc[0]
    assert (t.reason, t.exit_time) == ("invalidated", T1 + 3 * BAR)


def test_notional_is_capped_at_2x_equity():
    # tight stop: 1% risk at a $0.10 stop = 1,000 units = $100k notional -> capped to $20k = 200 units
    r = _run(_bars([FLAT] * 8), atr=0.1)
    assert r.trades.iloc[0].units == pytest.approx(200.0)


def test_fees_and_slippage_reduce_pnl():
    costs = CostModel(fee_rate=0.00055, base_slip=0.0002, vol_slip_k=0.0)
    r = _run(_bars([FLAT, FLAT, FLAT, (100.0, 102.5, 99.6, 102.0), FLAT]), costs=costs)
    t = r.trades.iloc[0]
    assert t.entry_px == pytest.approx(100.0 * 1.0002)
    assert t.exit_px == pytest.approx(t.tp_px * (1 - 0.0002))
    gross = t.units * (t.exit_px - t.entry_px)
    fees = 0.00055 * t.units * (t.entry_px + t.exit_px)
    assert t.pnl == pytest.approx(gross - fees)
    assert t.fees == pytest.approx(fees)


def test_cost_multiplier_doubles_costs():
    c1 = CostModel(fee_rate=0.00055, base_slip=0.0002, vol_slip_k=0.0)
    c2 = CostModel(fee_rate=0.00055, base_slip=0.0002, vol_slip_k=0.0, multiplier=2.0)
    b = _bars([FLAT, FLAT, FLAT, (100.0, 102.5, 99.6, 102.0), FLAT])
    assert _run(b, costs=c2).trades.iloc[0].fees > 1.9 * _run(b, costs=c1).trades.iloc[0].fees


def test_conservative_funding_charged_when_crossing_a_settlement():
    start = 1609485300000  # 2021-01-01 07:15 UTC; entry 07:30, exit 08:45 -> pays the 08:00 settlement
    r = _run(_bars([FLAT] * 8, start=start))
    t = r.trades.iloc[0]
    assert t.funding == pytest.approx(t.units * t.entry_px * 0.0001)


def test_data_gap_forces_exit_at_last_close_before_gap():
    b = _bars([FLAT] * 6)
    b.loc[3:, ["open_time", "available_at"]] += 10 * BAR  # gap after bar 2
    r = _run(b)
    t = r.trades.iloc[0]
    assert (t.reason, t.exit_time) == ("data_gap", b.available_at[2])


def test_no_entry_into_a_gap():
    b = _bars([FLAT] * 6)
    b.loc[1:, ["open_time", "available_at"]] += 10 * BAR
    assert _run(b).trades.empty


def test_entry_filter_can_veto_and_set_risk():
    b = _bars([FLAT] * 8)
    assert _run(b, entry_filter=lambda i, side, st: 0.0).trades.empty
    half = _run(b, entry_filter=lambda i, side, st: 0.5).trades.iloc[0]
    assert half.units == pytest.approx(50.0)


def test_mark_to_market_equity_and_daily_rows():
    r = _run(_bars([FLAT, FLAT, (100.0, 100.4, 99.6, 99.8), FLAT, FLAT, FLAT, FLAT]))
    assert len(r.bar_equity) == 7
    assert r.bar_equity[2] == pytest.approx(10_000 - 100 * 0.2)
    assert list(r.daily.columns) == ["day", "equity_close", "equity_min"]


def test_entry_filter_sees_equity_peak_and_day_start():
    seen = []

    def spy(i, side, st):
        seen.append(st)
        return 1.0

    # trade 1 loses at the stop in bar 2; a second signal at bar 4 sees the lower equity
    b = _bars([FLAT, FLAT, (100.0, 100.4, 98.5, 99.0), FLAT, FLAT, FLAT, FLAT, FLAT])
    sig = np.zeros(len(b), dtype=int)
    sig[0] = sig[4] = 1
    run(b, sig, np.full(len(b), 1.0), np.ones(len(b)), RULES, NO_COST, equity0=10_000.0, entry_filter=spy)
    assert seen[0].equity == 10_000.0 and seen[0].peak_equity == 10_000.0
    assert seen[1].equity == pytest.approx(9_900.0)
    assert seen[1].peak_equity == 10_000.0 and seen[1].day_start_equity == 10_000.0
    assert seen[1].now_ms == b.available_at[4] and seen[1].close == 100.0


def test_engine_runs_on_1h_bars():
    H1 = 3600 * 1000
    b = _bars([FLAT] * 8)
    b["open_time"] = T1 + np.arange(8, dtype=np.int64) * H1
    b["available_at"] = b["open_time"] + H1
    n = len(b)
    sig = np.zeros(n, dtype=int)
    sig[0] = 1
    r = run(b, sig, np.full(n, 1.0), np.ones(n), RULES, NO_COST, equity0=10_000.0, bar_ms=H1)
    t = r.trades.iloc[0]
    assert t.reason == "time" and t.entry_time == T1 + H1 and t.exit_time == T1 + 5 * H1
