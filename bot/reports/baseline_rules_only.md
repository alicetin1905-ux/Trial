# Baseline: rules only (no Jev)

Design period 2021-01-01 → 2025-12-31, anchored walk-forward, 8 out-of-sample half-years (2022-H1 → 2025-H2). Holdout (2026) untouched. Costs: taker 0.055%/side, slippage 2 bps + 5% of ATR/price per side, conservative funding (0.01%/8h charged to both sides). Risk 1% of equity per trade at the stop, notional capped at 2× equity.

| Gate | Out-of-sample | Required | Pass |
|---|---|---|---|
| Sharpe (annualised) | -1.39 | > 1.5 | ❌ |
| Max drawdown | 10.3% | < 15% | ✅ |
| Hit rate | 22.2% | > 55% | ❌ |
| t-stat (mean trade) | -3.48 | > 2.0 | ❌ |
| Trades | 18 | ≥ 100 | ❌ |
| Deflated Sharpe (54 trials) | 0.00 | > 0.95 | ❌ |

**Verdict: FAIL** (sharpe -1.39 <= 1.5; hit_rate 22.2% <= 55%; t_stat -3.48 <= 2.0; n_trades 18 < 100; deflated_sharpe 0.00 <= 0.95)

Total out-of-sample return: -10.3%. Average trade: -0.602% of equity.

## Cost stress (2× fees and slippage, same picks)

Sharpe -1.68, max drawdown 14.9%, hit rate 16.7%, t-stat -5.26, total return -14.9%.

## Config picked per fold (chosen on in-sample data only)

| Test fold | Config | RSI max | Stop k×ATR | TP R | Time stop (bars) |
|---|---|---|---|---|---|
| 2022-01-01 → 2022-06-30 | `aa5bf61806b39cd5` | 40 | 1.5 | 1 | 32 |
| 2022-07-01 → 2022-12-31 | `aa5bf61806b39cd5` | 40 | 1.5 | 1 | 32 |
| 2023-01-01 → 2023-06-30 | `7722ec33518ab830` | 35 | 1 | 1 | 16 |
| 2023-07-01 → 2023-12-31 | `7722ec33518ab830` | 35 | 1 | 1 | 16 |
| 2024-01-01 → 2024-06-30 | `7722ec33518ab830` | 35 | 1 | 1 | 16 |
| 2024-07-01 → 2024-12-31 | `7722ec33518ab830` | 35 | 1 | 1 | 16 |
| 2025-01-01 → 2025-06-30 | `ee362b22f6b11c1a` | 40 | 1.5 | 2 | 32 |
| 2025-07-01 → 2025-12-31 | `ee362b22f6b11c1a` | 40 | 1.5 | 2 | 32 |

## Notes

- Bars: 187,008 15m bars, 0 data gap(s); decisions whose lookback crosses a gap are skipped.
- No Jev and no risk guard in this run: the daily-loss and drawdown kill rules arrive with M6.
- Funding history is not reachable from the cloud build (Bybit geo-block); the conservative model makes results worse, never better.
