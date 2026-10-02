import gzip
import json

import numpy as np
import pytest

from jevbot.data.history import MissingDay, build, days_between, load_bars15
from jevbot.data.trade_archive import BAR_MS, BARS_PER_DAY, day_start_ms

HEADER = "timestamp,symbol,side,size,price,tickDirection,trdMatchID,grossValue,homeNotional,foreignNotional"


def _gz(trades):
    lines = [HEADER] + [f"{ts},BTCUSDT,{s},{q},{p},PlusTick,id,0,{q},{q * p}" for ts, s, q, p in trades]
    return gzip.compress(("\n".join(lines) + "\n").encode())


def _day_s(day):
    return day_start_ms(day) // 1000


class FakeArchive:
    def __init__(self, files):
        self.files = files
        self.calls = []

    def __call__(self, symbol, day):
        self.calls.append(day)
        if day not in self.files:
            raise MissingDay(day)
        return self.files[day]


@pytest.fixture
def archive():
    d1, d2 = "2022-06-13", "2022-06-14"
    return FakeArchive(
        {
            d1: _gz([(_day_s(d1) + 10, "Buy", 1.0, 100.0), (_day_s(d1) + 86000, "Sell", 1.0, 110.0)]),
            # day 2 starts trading only at 02:00, so its first 8 bars are empty
            d2: _gz([(_day_s(d2) + 7200, "Buy", 2.0, 120.0)]),
        }
    )


def test_days_between_is_inclusive():
    assert days_between("2022-06-13", "2022-06-15") == ["2022-06-13", "2022-06-14", "2022-06-15"]


def test_build_writes_one_parquet_per_day_and_a_manifest(tmp_path, archive):
    build("BTCUSDT", "2022-06-13", "2022-06-14", tmp_path, fetch=archive, workers=1)
    assert sorted(p.name for p in (tmp_path / "BTCUSDT").glob("*.parquet")) == [
        "2022-06-13.parquet",
        "2022-06-14.parquet",
    ]
    manifest = [
        json.loads(line) for line in (tmp_path / "BTCUSDT" / "manifest.jsonl").read_text().splitlines()
    ]
    assert {m["day"] for m in manifest} == {"2022-06-13", "2022-06-14"}
    assert all(len(m["sha256"]) == 64 for m in manifest)


def test_build_is_resumable_and_skips_finished_days(tmp_path, archive):
    build("BTCUSDT", "2022-06-13", "2022-06-14", tmp_path, fetch=archive, workers=1)
    archive.calls.clear()
    build("BTCUSDT", "2022-06-13", "2022-06-14", tmp_path, fetch=archive, workers=1)
    assert archive.calls == []


def test_missing_day_is_recorded_not_fatal(tmp_path, archive):
    report = build("BTCUSDT", "2022-06-13", "2022-06-15", tmp_path, fetch=archive, workers=1)
    assert report["missing"] == ["2022-06-15"]
    assert report["built"] == 2


def test_load_stitches_empty_day_start_to_previous_close(tmp_path, archive):
    build("BTCUSDT", "2022-06-13", "2022-06-14", tmp_path, fetch=archive, workers=1)
    bars, gaps = load_bars15(tmp_path, "BTCUSDT", "2022-06-13", "2022-06-14")
    assert len(bars) == 2 * BARS_PER_DAY and gaps == []
    day2 = bars.iloc[BARS_PER_DAY:]
    assert (day2.iloc[:8][["open", "high", "low", "close"]] == 110.0).all().all()
    assert day2.iloc[8].open == 120.0
    assert not bars[["open", "high", "low", "close"]].iloc[1:].isna().any().any()
    assert (np.diff(bars["open_time"]) == BAR_MS).all()


def test_load_reports_gap_for_missing_day(tmp_path):
    d1, d3 = "2022-06-13", "2022-06-15"
    fa = FakeArchive(
        {d1: _gz([(_day_s(d1) + 10, "Buy", 1.0, 100.0)]), d3: _gz([(_day_s(d3) + 10, "Buy", 1.0, 90.0)])}
    )
    build("BTCUSDT", d1, d3, tmp_path, fetch=fa, workers=1)
    _, gaps = load_bars15(tmp_path, "BTCUSDT", d1, d3)
    assert gaps == [(day_start_ms("2022-06-14"), day_start_ms("2022-06-15"))]


def test_corrupt_download_is_reported_failed_and_leaves_no_parquet(tmp_path):
    def broken(symbol, day):
        return b"not gzip"

    report = build("BTCUSDT", "2022-06-13", "2022-06-13", tmp_path, fetch=broken, workers=1)
    assert report["failed"] == ["2022-06-13"] and report["built"] == 0
    assert not list((tmp_path / "BTCUSDT").glob("*.parquet"))
