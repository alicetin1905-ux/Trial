import gzip

import numpy as np
import pytest

from jevbot.data.trade_archive import BAR_MS, DAY_MS, aggregate_day

DAY = "2022-06-13"
D0 = 1655078400  # 2022-06-13 00:00:00 UTC, in seconds

HEADER = "timestamp,symbol,side,size,price,tickDirection,trdMatchID,grossValue,homeNotional,foreignNotional"


def _row(ts, side, size, price):
    return f"{ts},BTCUSDT,{side},{size},{price},ZeroPlusTick,id,0,{size},{size * price}"


# Bar 0 [00:00, 00:15): four trades, two at the same whole second (file order decides open).
# Bar 1 [00:15, 00:30): no trades.
# Bar 2 [00:30, 00:45): one sell.
TRADES = [
    (D0 + 0, "Buy", 1.0, 100.0),  # open (first in file order at the tied second)
    (D0 + 0, "Sell", 2.0, 101.0),
    (D0 + 60.5, "Buy", 1.0, 104.0),  # high
    (D0 + 899.9, "Sell", 1.0, 99.0),  # low + close of bar 0
    (D0 + 1800, "Sell", 3.0, 98.0),  # bar 2, exactly on its boundary
]


def _gz(rows) -> bytes:
    lines = [HEADER] + [_row(*r) for r in rows]
    return gzip.compress(("\n".join(lines) + "\n").encode())


def test_full_day_has_96_bars_with_available_at_at_bar_close():
    bars, meta = aggregate_day(_gz(TRADES), DAY, prev_close=None)
    assert len(bars) == 96
    assert (np.diff(bars["open_time"]) == BAR_MS).all()
    assert bars["open_time"].iloc[0] == D0 * 1000
    assert (bars["available_at"] == bars["open_time"] + BAR_MS).all()
    assert meta["trades"] == 5 and meta["dropped_out_of_day"] == 0


def test_bar_ohlcv_and_flow():
    bars, _ = aggregate_day(_gz(TRADES), DAY, prev_close=None)
    b0 = bars.iloc[0]
    assert (b0.open, b0.high, b0.low, b0.close) == (100.0, 104.0, 99.0, 99.0)
    assert b0.volume == 5.0
    assert b0.buy_volume == 2.0 and b0.sell_volume == 3.0
    assert b0.trade_count == 4
    assert b0.quote_volume == pytest.approx(100 + 202 + 104 + 99)
    assert b0.vwap == pytest.approx((100 + 202 + 104 + 99) / 5.0)


def test_trade_exactly_on_boundary_belongs_to_the_later_bar():
    bars, _ = aggregate_day(_gz(TRADES), DAY, prev_close=None)
    b2 = bars.iloc[2]
    assert b2.trade_count == 1 and b2.open == 98.0 and b2.sell_volume == 3.0


def test_empty_bar_is_flat_at_previous_close_with_zero_volume():
    bars, _ = aggregate_day(_gz(TRADES), DAY, prev_close=None)
    b1 = bars.iloc[1]
    assert (b1.open, b1.high, b1.low, b1.close) == (99.0,) * 4
    assert b1.volume == 0 and b1.trade_count == 0
    assert np.isnan(b1.vwap)


def test_descending_file_gives_identical_bars():
    asc, _ = aggregate_day(_gz(TRADES), DAY, prev_close=None)
    desc, _ = aggregate_day(_gz(list(reversed(TRADES))), DAY, prev_close=None)
    assert asc.equals(desc)


def test_leading_empty_bars_use_prev_close_from_previous_day():
    late = [(D0 + 3600, "Buy", 1.0, 200.0)]
    bars, _ = aggregate_day(_gz(late), DAY, prev_close=150.0)
    assert bars.iloc[0].close == 150.0 and bars.iloc[0].trade_count == 0
    assert bars.iloc[4].open == 200.0


def test_leading_empty_bars_without_prev_close_are_nan():
    late = [(D0 + 3600, "Buy", 1.0, 200.0)]
    bars, _ = aggregate_day(_gz(late), DAY, prev_close=None)
    assert np.isnan(bars.iloc[0].close)


def test_trades_outside_the_day_are_dropped_and_counted():
    rows = TRADES + [(D0 - 1, "Buy", 1.0, 1.0), (D0 + DAY_MS / 1000, "Buy", 1.0, 1.0)]
    bars, meta = aggregate_day(_gz(rows), DAY, prev_close=None)
    assert meta["dropped_out_of_day"] == 2
    assert bars["low"].min() > 1.0


def test_newer_files_with_extra_columns_parse():
    lines = [HEADER + ",RPI"] + [_row(*r) + ",0" for r in TRADES]
    raw = gzip.compress(("\n".join(lines) + "\n").encode())
    bars, _ = aggregate_day(raw, DAY, prev_close=None)
    assert bars.iloc[0].volume == 5.0


def test_unknown_side_value_fails_loudly():
    with pytest.raises(ValueError, match="side"):
        aggregate_day(_gz([(D0, "Hold", 1.0, 1.0)]), DAY, prev_close=None)
