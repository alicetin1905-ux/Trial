"""Bybit public trade archive (public.bybit.com/trading/<SYMBOL>/<SYMBOL><YYYY-MM-DD>.csv.gz) -> 15m bars.

The same aggregation feeds both history and the live feed (architecture §4), so backtest and
live order-flow features are computed identically.

Archive quirks handled here:
- Row order varies by year: some files are newest-first, others oldest-first.
- Older files have whole-second timestamps, so many trades tie. Ties are broken by the file's own
  order, read in chronological direction, which decides each bar's open and close.
- Newer files add columns (e.g. RPI). Columns are read by name.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

BAR_MS = 15 * 60 * 1000
DAY_MS = 24 * 60 * 60 * 1000
BARS_PER_DAY = DAY_MS // BAR_MS

BAR_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "buy_volume",
    "sell_volume",
    "quote_volume",
    "trade_count",
    "vwap",
    "available_at",
]


def day_start_ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)


def _read_trades(raw_gz: bytes) -> pa.Table:
    return pacsv.read_csv(
        pa.input_stream(pa.BufferReader(raw_gz), compression="gzip"),
        read_options=pacsv.ReadOptions(),
        parse_options=pacsv.ParseOptions(),
        convert_options=pacsv.ConvertOptions(
            include_columns=["timestamp", "side", "size", "price"],
            column_types={"timestamp": pa.float64(), "size": pa.float64(), "price": pa.float64()},
        ),
    )


def aggregate_day(raw_gz: bytes, day: str, prev_close: float | None) -> tuple[pd.DataFrame, dict]:
    """Aggregate one day's gzipped trade CSV into exactly 96 15m bars.

    Bars with no trades are flat at the previous close with zero volume. Before the day's first
    trade, "previous close" is `prev_close` (yesterday's last close), or NaN if unknown.
    """
    start = day_start_ms(day)
    t = _read_trades(raw_gz)
    ts_s = t.column("timestamp").to_numpy()
    size = t.column("size").to_numpy()
    price = t.column("price").to_numpy()

    side = t.column("side")
    is_buy = pc.equal(side, "Buy").to_numpy(zero_copy_only=False)
    is_sell = pc.equal(side, "Sell").to_numpy(zero_copy_only=False)
    bad = ~(is_buy | is_sell)
    if bad.any():
        raise ValueError(f"unexpected side value(s): {sorted(set(side.filter(pa.array(bad)).to_pylist()))}")

    # Chronological order with file order as the tie-break, read in the file's direction.
    n = len(ts_s)
    descending = n > 1 and ts_s[0] > ts_s[-1]
    file_pos = np.arange(n)[::-1] if descending else np.arange(n)
    order = np.lexsort((file_pos, ts_s))
    ts_s, is_buy, size, price = ts_s[order], is_buy[order], size[order], price[order]

    ts_ms = np.floor(ts_s * 1000.0).astype(np.int64)
    in_day = (ts_ms >= start) & (ts_ms < start + DAY_MS)
    dropped = int((~in_day).sum())
    ts_ms, is_buy, size, price = ts_ms[in_day], is_buy[in_day], size[in_day], price[in_day]

    idx = (ts_ms - start) // BAR_MS
    df = pd.DataFrame(
        {
            "bar": idx,
            "price": price,
            "size": size,
            "buy": np.where(is_buy, size, 0.0),
            "sell": np.where(is_buy, 0.0, size),
            "quote": price * size,
        }
    )
    g = df.groupby("bar", sort=True)
    agg = pd.DataFrame(
        {
            "open": g["price"].first(),
            "high": g["price"].max(),
            "low": g["price"].min(),
            "close": g["price"].last(),
            "volume": g["size"].sum(),
            "buy_volume": g["buy"].sum(),
            "sell_volume": g["sell"].sum(),
            "quote_volume": g["quote"].sum(),
            "trade_count": g["price"].size(),
        }
    ).reindex(range(BARS_PER_DAY))

    empty = agg["trade_count"].isna()
    for col in ("volume", "buy_volume", "sell_volume", "quote_volume", "trade_count"):
        agg[col] = agg[col].fillna(0.0)
    agg["trade_count"] = agg["trade_count"].astype(np.int64)

    close = agg["close"].copy()
    if prev_close is not None and np.isnan(close.iloc[0]):
        close.iloc[0] = prev_close
    close = close.ffill()
    for col in ("open", "high", "low", "close"):
        agg.loc[empty, col] = close[empty]

    with np.errstate(invalid="ignore", divide="ignore"):
        agg["vwap"] = np.where(agg["volume"] > 0, agg["quote_volume"] / agg["volume"], np.nan)
    agg["open_time"] = start + np.arange(BARS_PER_DAY, dtype=np.int64) * BAR_MS
    agg["available_at"] = agg["open_time"] + BAR_MS
    bars = agg.reset_index(drop=True)[BAR_COLUMNS]

    meta = {
        "day": day,
        "trades": int(in_day.sum()),
        "dropped_out_of_day": dropped,
        "descending_file": bool(descending),
        "empty_bars": int(empty.sum()),
    }
    return bars, meta
