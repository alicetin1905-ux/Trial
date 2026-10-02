import numpy as np
import pandas as pd
import pytest

from jevbot.data.bars import as_of, find_gaps, resample

BAR_MS = 15 * 60 * 1000
T0 = 1655078400000  # 2022-06-13 00:00 UTC, ms


def _bars15(n: int, start: int = T0) -> pd.DataFrame:
    i = np.arange(n)
    open_time = start + i * BAR_MS
    return pd.DataFrame(
        {
            "open_time": open_time,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.5 + i,
            "volume": 1.0 + 0 * i,
            "buy_volume": 0.75 + 0 * i,
            "sell_volume": 0.25 + 0 * i,
            "quote_volume": 100.0 + i,
            "trade_count": 10 + 0 * i,
            "vwap": 100.0 + i,
            "available_at": open_time + BAR_MS,
        }
    )


def test_resample_to_1h_aggregates_ohlcv_and_flow():
    h = resample(_bars15(8), 60)
    assert len(h) == 2
    first = h.iloc[0]
    assert first.open_time == T0
    assert (first.open, first.high, first.low, first.close) == (100.0, 104.0, 99.0, 103.5)
    assert first.volume == 4.0 and first.buy_volume == 3.0 and first.sell_volume == 1.0
    assert first.trade_count == 40
    assert first.available_at == T0 + 60 * 60 * 1000


def test_resample_drops_incomplete_trailing_group():
    h = resample(_bars15(7), 60)  # second hour only has 3 of 4 bars
    assert len(h) == 1


def test_resample_drops_group_with_missing_bar_inside():
    b = _bars15(8).drop(index=5).reset_index(drop=True)
    h = resample(b, 60)
    assert list(h.open_time) == [T0]


def test_resample_aligns_to_utc_multiples_not_to_first_bar():
    b = _bars15(8, start=T0 + BAR_MS)  # starts at 00:15
    h = resample(b, 60)
    assert list(h.open_time) == [T0 + 60 * 60 * 1000]  # 01:00-02:00 is the only full hour


def test_resample_vwap_is_quote_over_volume():
    h = resample(_bars15(4), 60)
    assert h.iloc[0].vwap == pytest.approx(sum(100.0 + i for i in range(4)) / 4.0)


def test_resample_4h():
    h = resample(_bars15(32), 240)
    assert len(h) == 2 and h.iloc[1].open_time == T0 + 4 * 3600 * 1000


def test_resample_rejects_non_multiple():
    with pytest.raises(ValueError):
        resample(_bars15(8), 50)


def test_as_of_only_returns_rows_available_at_or_before_t():
    b = _bars15(8)
    t = T0 + 3 * BAR_MS  # bars 0,1,2 are closed by then; bar 3 is still open
    seen = as_of(b, t)
    assert list(seen.open_time) == [T0, T0 + BAR_MS, T0 + 2 * BAR_MS]


def test_as_of_on_resampled_frame_hides_the_unfinished_hour():
    h = resample(_bars15(8), 60)
    assert len(as_of(h, T0 + 60 * 60 * 1000 - 1)) == 0
    assert len(as_of(h, T0 + 60 * 60 * 1000)) == 1


def test_find_gaps_reports_missing_ranges():
    b = _bars15(10).drop(index=[3, 4]).reset_index(drop=True)
    gaps = find_gaps(b, BAR_MS)
    assert gaps == [(T0 + 3 * BAR_MS, T0 + 5 * BAR_MS)]


def test_resample_from_1h_base_to_4h_and_1d():
    H = 3600 * 1000
    n = 48
    ot = T0 + np.arange(n, dtype=np.int64) * H
    b = pd.DataFrame(
        {
            "open_time": ot,
            "open": 1.0 + np.arange(n),
            "high": 2.0 + np.arange(n),
            "low": 0.0 + np.arange(n),
            "close": 1.5 + np.arange(n),
            "volume": 1.0,
            "buy_volume": 0.5,
            "sell_volume": 0.5,
            "quote_volume": 1.0,
            "trade_count": 1,
            "vwap": 1.0,
            "available_at": ot + H,
        }
    )
    h4 = resample(b, 240, base_minutes=60)
    assert len(h4) == 12 and h4.iloc[0].close == 4.5 and h4.iloc[0].volume == 4.0
    d = resample(b, 1440, base_minutes=60)
    assert len(d) == 2 and d.iloc[1].available_at == T0 + 48 * H
