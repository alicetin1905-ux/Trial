"""A committed, checksummed copy of the 1h bar history (history/BTCUSDT_1h.parquet).

A fresh session can run backtests straight away instead of rebuilding ~125 GB of raw trade
archives. It is derived from data/bars15 (see reports/data_history.md) and verified on every load.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from jevbot.data.trade_archive import day_start_ms

DAY_MS = 86_400_000


class ChecksumMismatch(RuntimeError):
    pass


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save_1h(bars1h: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bars1h.reset_index(drop=True).to_parquet(path, index=False, compression="zstd")
    Path(f"{path}.sha256").write_text(_sha(path) + "\n")


def load_1h(path: Path, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Load and verify. `end` is an inclusive day: bars that open on or before it are returned."""
    path = Path(path)
    want = Path(f"{path}.sha256").read_text().strip()
    if _sha(path) != want:
        raise ChecksumMismatch(f"{path} does not match its .sha256")
    b = pd.read_parquet(path)
    if start is not None:
        b = b[b["open_time"] >= day_start_ms(start)]
    if end is not None:
        b = b[b["open_time"] < day_start_ms(end) + DAY_MS]
    return b.reset_index(drop=True)
