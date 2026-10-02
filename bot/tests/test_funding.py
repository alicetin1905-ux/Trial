import pytest

from jevbot.data.funding import (
    CONSERVATIVE_RATE,
    conservative_funding_cost,
    parse_funding_history,
    settlement_times,
)

H = 3600 * 1000
T0 = 1655078400000  # 2022-06-13 00:00 UTC


def test_settlements_every_8h_at_00_08_16_utc():
    ts = settlement_times(T0 - 1, T0 + 24 * H)
    assert ts == [T0, T0 + 8 * H, T0 + 16 * H, T0 + 24 * H]


def test_settlement_window_is_half_open_on_the_left():
    assert settlement_times(T0, T0 + 8 * H) == [T0 + 8 * H]


def test_conservative_cost_is_charged_to_longs_and_shorts():
    for side in (1, -1):
        cost = conservative_funding_cost(side=side, notional=10_000.0, start=T0 - 1, end=T0 + 9 * H)
        assert cost == pytest.approx(2 * 10_000.0 * CONSERVATIVE_RATE)


def test_conservative_rate_is_the_bybit_default_or_higher():
    assert CONSERVATIVE_RATE >= 0.0001


def test_parse_bybit_v5_funding_history_sorted_ascending():
    payload = {
        "retCode": 0,
        "retMsg": "OK",
        "result": {
            "category": "linear",
            "list": [
                {"symbol": "BTCUSDT", "fundingRate": "-0.00005", "fundingRateTimestamp": str(T0 + 8 * H)},
                {"symbol": "BTCUSDT", "fundingRate": "0.0001", "fundingRateTimestamp": str(T0)},
            ],
        },
    }
    df = parse_funding_history(payload)
    assert list(df.time) == [T0, T0 + 8 * H]
    assert list(df.rate) == [0.0001, -0.00005]


def test_parse_rejects_api_error():
    with pytest.raises(RuntimeError, match="10001"):
        parse_funding_history({"retCode": 10001, "retMsg": "params error", "result": {}})
