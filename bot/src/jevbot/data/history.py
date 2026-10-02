"""Build and load the 15m bar history from Bybit's public trade archive.

    python -m jevbot.data.history --start 2021-01-01 --end 2026-09-30

Each day's raw file is streamed, aggregated and discarded, so only the bars are kept (a few MB
in total). The build can be resumed: finished days are skipped. A day that is missing from the
archive, or whose download is corrupt, is reported and never written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd

from jevbot.data.bars import find_gaps
from jevbot.data.trade_archive import BAR_MS, aggregate_day

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://public.bybit.com/trading/{symbol}/{symbol}{day}.csv.gz"

Fetch = Callable[[str, str], bytes]


class MissingDay(Exception):
    """The archive has no file for this day."""


def days_between(start: str, end: str) -> list[str]:
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    return [(d0 + timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]


def http_fetch(symbol: str, day: str, attempts: int = 4) -> bytes:
    url = ARCHIVE_URL.format(symbol=symbol, day=day)
    for i in range(attempts):
        try:
            r = httpx.get(url, timeout=300.0)
            if r.status_code == 404:
                raise MissingDay(day)
            r.raise_for_status()
            return r.content
        except (httpx.TransportError, httpx.HTTPStatusError):
            if i == attempts - 1:
                raise
            time.sleep(2 ** (i + 1))
    raise AssertionError("unreachable")


def _process(symbol: str, day: str, out: Path, fetch: Fetch) -> dict:
    raw = fetch(symbol, day)
    bars, meta = aggregate_day(raw, day, prev_close=None)  # day starts are stitched in load_bars15
    tmp = out / f".{day}.parquet.tmp"
    bars.to_parquet(tmp, index=False)
    os.replace(tmp, out / f"{day}.parquet")  # atomic: a crash never leaves a half-written day
    return meta | {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def build(
    symbol: str, start: str, end: str, data_dir: Path, fetch: Fetch = http_fetch, workers: int = 4
) -> dict:
    out = Path(data_dir) / symbol
    out.mkdir(parents=True, exist_ok=True)
    todo = [d for d in days_between(start, end) if not (out / f"{d}.parquet").exists()]
    report: dict = {
        "built": 0,
        "skipped": len(days_between(start, end)) - len(todo),
        "missing": [],
        "failed": [],
    }
    with ThreadPoolExecutor(max_workers=workers) as pool, open(out / "manifest.jsonl", "a") as manifest:
        futures = {pool.submit(_process, symbol, d, out, fetch): d for d in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            day = futures[fut]
            try:
                meta = fut.result()
            except MissingDay:
                report["missing"].append(day)
                continue
            except Exception as e:  # corrupt or failed download: report it, keep going
                log.error("day %s failed: %s", day, e)
                report["failed"].append(day)
                continue
            manifest.write(json.dumps(meta) + "\n")
            manifest.flush()
            report["built"] += 1
            if n % 50 == 0:
                log.info("%d/%d days processed", n, len(todo))
    report["missing"].sort()
    report["failed"].sort()
    return report


def load_bars15(
    data_dir: Path, symbol: str, start: str, end: str
) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """Load the continuous 15m series for [start, end] and report any gaps (missing days)."""
    out = Path(data_dir) / symbol
    files = [out / f"{d}.parquet" for d in days_between(start, end)]
    frames = [pd.read_parquet(f) for f in files if f.exists()]
    if not frames:
        raise FileNotFoundError(f"no bars for {symbol} {start}..{end} in {out}")
    bars = pd.concat(frames, ignore_index=True).sort_values("open_time").reset_index(drop=True)
    # Empty bars at the start of a day were built without yesterday's close: carry it over now.
    carried = bars["close"].ffill()
    empty = bars["trade_count"] == 0
    for col in ("open", "high", "low", "close"):
        bars.loc[empty, col] = carried[empty]
    return bars, find_gaps(bars, BAR_MS)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--data-dir", default="data/bars15")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = build(a.symbol, a.start, a.end, Path(a.data_dir), workers=a.workers)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
