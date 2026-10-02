import numpy as np
import pandas as pd
import pytest

BAR_MS = 15 * 60 * 1000
T0 = 1609459200000  # 2021-01-01 00:00 UTC


def make_bars15(
    n: int, seed: int = 0, start: int = T0, drift: float = 0.0, vol: float = 0.002
) -> pd.DataFrame:
    """Synthetic 15m bars shaped exactly like the history loader's output."""
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    close = 30_000.0 * np.exp(np.cumsum(r))
    open_ = np.concatenate([[30_000.0], close[:-1]])
    spread = np.abs(rng.normal(0, vol, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.uniform(50, 150, n)
    buy_share = np.clip(rng.normal(0.5, 0.1, n), 0.05, 0.95)
    open_time = start + np.arange(n, dtype=np.int64) * BAR_MS
    quote = volume * (high + low) / 2
    return pd.DataFrame(
        {
            "open_time": open_time,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "buy_volume": volume * buy_share,
            "sell_volume": volume * (1 - buy_share),
            "quote_volume": quote,
            "trade_count": rng.integers(100, 1000, n),
            "vwap": quote / volume,
            "available_at": open_time + BAR_MS,
        }
    )


@pytest.fixture
def bars120d():
    return make_bars15(120 * 96, seed=1)
