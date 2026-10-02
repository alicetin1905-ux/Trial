import pytest
from conftest import make_bars1h

from jevbot.data.bars1h_file import ChecksumMismatch, load_1h, save_1h


def test_roundtrip_and_checksum(tmp_path):
    b = make_bars1h(500, seed=1)
    p = tmp_path / "BTCUSDT_1h.parquet"
    save_1h(b, p)
    assert (tmp_path / "BTCUSDT_1h.parquet.sha256").exists()
    out = load_1h(p)
    assert out.equals(b.reset_index(drop=True))


def test_tampered_file_is_refused(tmp_path):
    b = make_bars1h(500, seed=1)
    p = tmp_path / "BTCUSDT_1h.parquet"
    save_1h(b, p)
    save_1h(make_bars1h(500, seed=2), tmp_path / "other.parquet")
    p.write_bytes((tmp_path / "other.parquet").read_bytes())  # contents swapped, old checksum kept
    with pytest.raises(ChecksumMismatch):
        load_1h(p)


def test_range_filter(tmp_path):
    b = make_bars1h(48, seed=1)  # 2021-01-01 .. 2021-01-02
    p = tmp_path / "x.parquet"
    save_1h(b, p)
    out = load_1h(p, end="2021-01-01")
    assert len(out) == 24
