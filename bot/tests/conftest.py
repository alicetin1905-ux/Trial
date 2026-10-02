import numpy as np
import pandas as pd
import pytest

BAR_MS = 15 * 60 * 1000
T0 = 1609459200000  # 2021-01-01 00:00 UTC


def make_bars15(
    n: int, seed: int = 0, start: int = T0, drift: float = 0.0, vol: float = 0.002, bar_ms: int = BAR_MS
) -> pd.DataFrame:
    """Synthetic bars shaped exactly like the history loader's output (15m by default)."""
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    close = 30_000.0 * np.exp(np.cumsum(r))
    open_ = np.concatenate([[30_000.0], close[:-1]])
    spread = np.abs(rng.normal(0, vol, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.uniform(50, 150, n)
    buy_share = np.clip(rng.normal(0.5, 0.1, n), 0.05, 0.95)
    open_time = start + np.arange(n, dtype=np.int64) * bar_ms
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
            "available_at": open_time + bar_ms,
        }
    )


@pytest.fixture
def bars120d():
    return make_bars15(120 * 96, seed=1)


H1_MS = 60 * 60 * 1000


def make_bars1h(
    n: int, seed: int = 0, start: int = T0, drift: float = 0.0, vol: float = 0.004
) -> pd.DataFrame:
    return make_bars15(n, seed=seed, start=start, drift=drift, vol=vol, bar_ms=H1_MS)


@pytest.fixture
def bars1h_200d():
    return make_bars1h(200 * 24, seed=1)
