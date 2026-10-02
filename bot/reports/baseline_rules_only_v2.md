# Baseline: rules only (no Jev), baseline_rules_only_v2

Design period 2021-01-01 → 2025-12-31, anchored walk-forward, 8 out-of-sample half-years (2022-H1 → 2025-H2). Holdout (2026) untouched. Costs: taker 0.055%/side, slippage 2 bps + 5% of ATR/price per side, conservative funding (0.01%/8h charged to both sides). Risk 1% of equity per trade at the stop, notional capped at 2× equity.

| Gate | Out-of-sample | Required | Pass |
|---|---|---|---|
| Sharpe (annualised) | -3.47 | > 1.5 | ❌ |
| Max drawdown | 87.8% | < 15% | ❌ |
| Hit rate | 35.4% | > 55% | ❌ |
| t-stat (mean trade) | -7.59 | > 2.0 | ❌ |
| Trades | 848 | ≥ 100 | ✅ |
| Deflated Sharpe (108 trials) | 0.00 | > 0.95 | ❌ |

**Verdict: FAIL** (sharpe -3.47 <= 1.5; max_dd 87.8% >= 15%; hit_rate 35.4% <= 55%; t_stat -7.59 <= 2.0; deflated_sharpe 0.00 <= 0.95)

Total out-of-sample return: -87.8%. Average trade: -0.244% of equity.

## Cost stress (2× fees and slippage, same picks)

Sharpe -6.61, max drawdown 99.1%, hit rate 25.6%, t-stat -16.92, total return -99.1%.

## Config picked per fold (chosen on in-sample data only)

| Test fold | Config | RSI max | Stop k×ATR | TP R | Time stop (bars) |
|---|---|---|---|---|---|
| 2022-01-01 → 2022-06-30 | `32abff7d692a62bf` | 35 | 2 | 1.5 | 16 |
| 2022-07-01 → 2022-12-31 | `7deb00585820dd42` | 35 | 2 | 1.5 | 32 |
| 2023-01-01 → 2023-06-30 | `cb11a40971179a06` | 35 | 2 | 1 | 32 |
| 2023-07-01 → 2023-12-31 | `a96249b8b2429d68` | 35 | 2 | 2 | 32 |
| 2024-01-01 → 2024-06-30 | `a96249b8b2429d68` | 35 | 2 | 2 | 32 |
| 2024-07-01 → 2024-12-31 | `a96249b8b2429d68` | 35 | 2 | 2 | 32 |
| 2025-01-01 → 2025-06-30 | `a96249b8b2429d68` | 35 | 2 | 2 | 32 |
| 2025-07-01 → 2025-12-31 | `a96249b8b2429d68` | 35 | 2 | 2 | 32 |

## Notes

- Bars: 187,008 15m bars, 0 data gap(s); decisions whose lookback crosses a gap are skipped.
- No Jev and no risk guard in this run: the daily-loss and drawdown kill rules arrive with M6.
- Funding history is not reachable from the cloud build (Bybit geo-block); the conservative model makes results worse, never better.

## Diagnosis: costs vs edge (same picks; diagnostic only, not used for selection, no new trials)

| Cost model | Trades | Hit rate | Avg trade (% of equity) | Sharpe | Total return |
|---|---|---|---|---|---|
| Zero costs | 837 | 46.8% | +0.040% | +0.60 | +34.3% |
| Taker fees only (0.055%/side) | 837 | 41.2% | −0.136% | −1.99 | −69.2% |
| Fees + slippage (the real test) | 848 | 35.4% | −0.244% | −3.47 | −87.8% |

Median position notional is 1.56× equity, so fees and slippage cost ~0.17% of equity per trade.
That is about 4× the gross edge per trade. Even with zero costs this would fail every gate except
trade count. A Jev filter would have to pick trades averaging ~5× the current gross edge to break even.
