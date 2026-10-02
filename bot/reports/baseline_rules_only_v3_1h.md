# Baseline: rules only (no Jev), baseline_rules_only_v3_1h

Design period 2021-01-01 → 2025-12-31, anchored walk-forward, 8 out-of-sample half-years (2022-H1 → 2025-H2). Holdout (2026) untouched. Costs: taker 0.055%/side, slippage 2 bps + 5% of ATR/price per side, conservative funding (0.01%/8h charged to both sides). Risk 1% of equity per trade at the stop, notional capped at 2× equity.

| Gate | Out-of-sample | Required | Pass |
|---|---|---|---|
| Sharpe (annualised) | -1.81 | > 1.5 | ❌ |
| Max drawdown | 63.6% | < 15% | ❌ |
| Hit rate | 39.4% | > 55% | ❌ |
| t-stat (mean trade) | -3.66 | > 2.0 | ❌ |
| Trades | 584 | ≥ 100 | ✅ |
| Deflated Sharpe (162 trials) | 0.00 | > 0.95 | ❌ |

**Verdict: FAIL** (sharpe -1.81 <= 1.5; max_dd 63.6% >= 15%; hit_rate 39.4% <= 55%; t_stat -3.66 <= 2.0; deflated_sharpe 0.00 <= 0.95)

Total out-of-sample return: -62.1%. Average trade: -0.160% of equity.

## Cost stress (2× fees and slippage, same picks)

Sharpe -3.63, max drawdown 87.0%, hit rate 35.8%, t-stat -7.79, total return -86.8%.

## Config picked per fold (chosen on in-sample data only)

| Test fold | Config | RSI max | Stop k×ATR | TP R | Time stop (bars) |
|---|---|---|---|---|---|
| 2022-01-01 → 2022-06-30 | `1cb9dd2b851701f3` | 51 | 2 | 1.5 | 24 |
| 2022-07-01 → 2022-12-31 | `1cb9dd2b851701f3` | 51 | 2 | 1.5 | 24 |
| 2023-01-01 → 2023-06-30 | `1cb9dd2b851701f3` | 51 | 2 | 1.5 | 24 |
| 2023-07-01 → 2023-12-31 | `1cb9dd2b851701f3` | 51 | 2 | 1.5 | 24 |
| 2024-01-01 → 2024-06-30 | `1cb9dd2b851701f3` | 51 | 2 | 1.5 | 24 |
| 2024-07-01 → 2024-12-31 | `1cb9dd2b851701f3` | 51 | 2 | 1.5 | 24 |
| 2025-01-01 → 2025-06-30 | `1cb9dd2b851701f3` | 51 | 2 | 1.5 | 24 |
| 2025-07-01 → 2025-12-31 | `1cb9dd2b851701f3` | 51 | 2 | 1.5 | 24 |

## Notes

- Bars: 46,752 15m bars, 0 data gap(s); decisions whose lookback crosses a gap are skipped.
- No Jev and no risk guard in this run: the daily-loss and drawdown kill rules arrive with M6.
- Funding history is not reachable from the cloud build (Bybit geo-block); the conservative model makes results worse, never better.

## Diagnosis: costs vs edge (same picks; diagnostic only, not used for selection, no new trials)

| Cost model | Trades | Hit rate | Avg trade (% of equity) | Sharpe | Total return |
|---|---|---|---|---|---|
| Zero costs | 584 | 43.8% | +0.017% | +0.20 | +6.9% |
| Taker fees only (0.055%/side) | 584 | 42.1% | -0.081% | -0.91 | -39.6% |
| Fees + slippage (the real test) | 584 | 39.4% | -0.160% | -1.81 | -62.1% |

On 1h bars the median position is 0.7× equity, so taker fees cost ~0.08% of equity per trade
(half the 15m figure). But the gross edge shrank too: +0.017% per trade, Sharpe +0.20 even with
zero costs. Long and short sides are both roughly flat before costs. Conclusion: trend + pullback
on BTCUSDT, as specified, had no usable gross edge in 2022–2025 on either 15m or 1h bars.
