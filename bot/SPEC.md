# Spec: Opus + Jev trading bot (phase 1 of 6, awaiting approval)

Status: **DRAFT, needs your approval before architecture starts.**
Harness: the same six phases AgenKit uses (brainstorm, architecture, plan, test-first build, review, ship), run by hand without the paid AgenKit licence. Work stops for your approval after this spec, after the architecture and after the plan.

## 0. Decisions so far

| Blank in the prompt | Decision |
|---|---|
| Venue | Bybit testnet for orders, Bybit mainnet public data for signals (see §10) |
| Asset | BTCUSDT linear perpetual |
| Strategy idea | Trend + pullback, with Jev filtering regime and setup quality |
| Deploy target | Mac Mini, launchd `KeepAlive`, sleep disabled |
| Manual approval threshold | **Proposed $2,500 notional.** You need to confirm this (§18) |
| Jev access | Your own TypeSafe key, read from `TYPESAFE_API_KEY`, with the model version pinned |

## 1. Goal and non-goals

Goal: a bot that paper-trades BTCUSDT 24/7. Jev scores each candle, deterministic code decides and enforces every limit, and Opus 5.5 improves the strategy offline every night. It ships with a live dashboard, Telegram alerts, a daily report, `strategy.md`, and a go-live checklist that blocks real money until every answer is clean.

Non-goals for v1: more than one asset, more than one open position, leverage above 2x, market making, trading the news.

## 2. Three layers that never overlap

| Layer | Runs | Can do | Can never do |
|---|---|---|---|
| **Opus 5.5** (slow brain) | Nightly, offline | Read logs, propose edits to `strategy.md` and `questions.yaml`, write code in a branch, review | Grade its own output, touch `risk.yaml`, place orders, change anything that is live |
| **Jev** (fast reflex) | Each closed 1h candle | Return calibrated probabilities for fixed typed questions about one snapshot | Return text, see anything except the snapshot, decide size, veto or order |
| **Deterministic code** | Always | Own market state, thresholds, sizing, risk vetoes, orders, kill switch | Be changed at runtime by any model output |

How this is enforced: the nightly loop can only write to `staging/`. A test fails the build if a candidate diff touches `risk.yaml`, `risk/` or `execution/`. Jev's answers reach the decision function as floats and nothing else.

## 3. State engine (the only thing Jev sees)

On each closed candle, build one compact, numeric, normalised snapshot. It contains **no prices, dates or symbol names**, so Jev can't recognise a known historical period from its training data.

Inputs are limited to data that can be **rebuilt historically with exact timestamps**, so the backtest and live trading see the same thing:

| Feature | Source | History available |
|---|---|---|
| Returns over 1h, 4h, 24h, 72h (in units of current volatility) | klines | yes |
| Realised volatility, 24h and 7d, plus their ratio | klines | yes |
| Trend: EMA slope on 4h and 1d, distance from EMA in ATRs, ADX | klines | yes |
| Pullback depth (from swing high/low, in ATRs), RSI(14) on 1h | klines | yes |
| Order flow: signed taker volume imbalance over 1h and 4h, trade-count z-score | Bybit public trade archive (`public.bybit.com/trading`) | yes, daily files |
| Funding rate, hours to next funding | funding history API | yes |
| Spread, top-10 book imbalance | live websocket only | **no** |

Spread and book imbalance have no reliable free history, so **Jev does not see them in v1**. They are logged live from day 1 and used only in a deterministic veto: no entry when the spread is above a set number of bps. Once enough live history has built up, adding them to the snapshot becomes a strategy change that has to clear the gates again.

Leakage rule: every input has `available_at <= decision_time`. The candle that just closed is the newest bar allowed. Each daily trade archive is only used after its file timestamp. A unit test shifts every series by +1 bar and checks that the backtest result changes. If it doesn't change, something is reading the future.

## 4. Strategy hypothesis: trend + pullback (to be backtested, not assumed)

- **Trend filter:** 4h EMA(50) slope sign and 1d close relative to EMA(50) agree.
- **Pullback entry (1h):** in an uptrend, price pulls back between 1 and 2.5 ATR(1h) from the 24h swing high and RSI(14) drops below a threshold. The mirror rule applies for shorts. A deterministic **candidate** flag is set only when this holds.
- **Jev filter:** on candidate candles, the questions in §5 have to clear the thresholds in `strategy.md`.
- **Exit:** stop at entry minus k×ATR, take profit at R×stop distance, time stop after N hours, and an invalidation exit if the 4h trend flips.
- **Parameter grid is fixed in advance and kept small:** about 3×3×3×2 = 54 combinations. Every combination tried is counted as a trial (§9).

Honest prior: trend strategies usually win 35–45% of trades. Entering on pullbacks pushes the win rate up, but **the hit rate above 55% and Sharpe above 1.5 gates together are a high bar. This hypothesis may well fail.** If it does, the harness says so, and nothing goes to paper trading.

## 5. Jev questions (one call per candle, each question isolates one factor)

```yaml
regime:      choice  {trending_up, trending_down, ranging, high_vol_chop}
direction:   choice  {up, down, flat}   # next 12h, relative to current volatility
buy_pressure_real: noul                 # is the taker imbalance persistent, not one print?
setup_quality: score [0..4]             # rubric: clean pullback in trend ... broken structure
risk_state:  choice  {normal, elevated, extreme}
```

Live: Jev is called on **every** closed 1h candle (for the dashboard and calibration data). Decisions only use candidate candles. Backtest: same calls on the same snapshots. A whole-history call costs a few dollars at the published $0.042 per 1M input tokens, and takes about an hour at the 1,200 requests/min limit.

No answer within 1.5 s, an error, or a malformed answer all mean **no trade**. It is never retried into a late entry.

## 6. From Jev answers to a trade probability

Jev is calibrated on its own answers ("is the regime trending?"). That doesn't make it calibrated on "will this trade hit TP before SL?", and Kelly needs the second number. So:

1. `edge_score = Σ wᵢ · featureᵢ(Jev answers)`, with explicit weights from `strategy.md`.
2. `p_win = calibrator(edge_score)`. The calibrator is isotonic or Platt, fitted on in-sample walk-forward folds only and frozen for out-of-sample.
3. A trade fires only if **every** per-question threshold passes **and** `p_win ≥ p_cutoff`.

Calibration is measured per question (Brier score and reliability curve, 10 bins) and for `p_win`, first on backtest out-of-sample, then on **our own testnet/shadow fills**. If a curve bends (ECE above 0.05), a recalibration map is applied in code and versioned.

## 7. Sizing

- `b = R` (TP/SL ratio). `f* = (p·b − (1−p)) / b`, the Kelly fraction.
- `f = clamp(0.25 · f* · c, 0, f_max)`, where `c ∈ [0.5, 1.0]` scales with Jev's confidence on `setup_quality`.
- The position is set so that losing at the stop costs `f · equity`. `f_max` = 1% of equity at risk. Below `p_cutoff`, size is 0.
- **Sizing on Jev uses a fixed minimum size until calibration passes on our own fills** (≥ 100 resolved candidate decisions, ECE < 0.05). Before that, every trade uses a fixed 0.25% risk.

## 8. Hard risk rules (in code, checked before every order, no model can change them)

| Rule | Proposed default |
|---|---|
| Max risk per trade | 1% of equity at the stop |
| Max position notional | 2× equity (leverage cap 2x, isolated margin) |
| Max open positions | 1 |
| Daily loss limit | 3% of start-of-day equity → flat, then no new entries until 00:00 UTC |
| Max drawdown | 10% from peak equity → **kill switch** |
| Manual approval | Order notional above **$2,500** → Telegram approve/deny, auto-deny after 5 min |
| Stale data | Last candle or websocket heartbeat older than 2× expected → no entries |
| Spread veto | Spread above 3 bps → no entry |
| Funding veto | No entry within 10 min of funding settlement |
| Order sanity | Reduce-only on exits, price band ±1% from mid, max 3 order attempts per minute |

**Kill switch:** cancel all open orders, close the position with reduce-only market orders, write `HALTED` to disk, alert Telegram, and refuse every order until you reset it by hand. It can be triggered by the drawdown rule, the `/kill` command, the dashboard button, the `HALT` file, or 3 order errors in 10 minutes. A test fires it against testnet.

**Keys:** Bybit API key has trade permission only, withdrawals off, and is IP-whitelisted to the Mac Mini. All secrets live in `.env` (git-ignored), are loaded once, and are masked in logs. A test greps the logs for secret values. The bot never asks for or stores passwords or 2FA codes.

**Untrusted input:** Telegram commands are only accepted from your chat ID. Headlines and feeds aren't part of v1 at all. Any text that reaches Opus at night (logs, fills) is wrapped and labelled as data.

## 9. Backtest and gates (the harness grades; Opus never grades its own work)

- **Data:** Bybit mainnet BTCUSDT, 2021-01-01 → 2026-09-30. That covers the 2021 bull run, the 2022 crash, the 2023 range, the 2024 ETF rally and 2025–26.
- **Costs:** taker fee 0.055% per side, slippage of 2 bps per side plus a volatility-scaled component, and actual historical funding. Stress test: costs doubled.
- **Walk-forward:** anchored, with 6-month test folds. Out-of-sample = all test folds joined together. **Final holdout = 2026-01-01 → 2026-09-30, untouched during design and evaluated once per candidate.**
- **Gates (all out-of-sample, after costs):** Sharpe > 1.5 (annualised, on daily returns), max drawdown < 15%, hit rate > 55%, t-stat of mean trade return > 2.0, **and** at least 100 out-of-sample trades, **and** Jev + rules beats rules alone (otherwise Jev isn't adding anything).
- **Overfitting control:** every configuration ever tested goes into `trials.jsonl`. The Sharpe gate is also checked as a **deflated Sharpe** (adjusted for the number of trials), and that must be > 0.95.
- **Output:** a harness-written `reports/backtest_<id>.md`, plus `strategy.md` when a candidate passes. `strategy.md` contains entry, exit, stop, take profit, timeframe, thresholds, weights and the exact invalidation condition.

## 10. Paper mode

Bybit testnet has its own thin, synthetic order book, so testnet fills say nothing about real slippage. The design is:

- **Signals** come from mainnet public market data, so they match the backtest.
- **Orders** go to testnet, to prove the order, cancel, reduce-only and kill-switch plumbing works.
- **Shadow fills** are simulated against the mainnet book at decision time. Shadow P&L is the number compared to the backtest.

To move paper → live, you need **all** of: ≥ 60 days and ≥ 50 trades on paper; paper Sharpe, hit rate and average trade inside the backtest's 90% bootstrap band; slippage on our fills within 1.5× the modelled slippage; calibration passing on our own fills; the kill switch having fired once in testing; and every answer in §17 clean. Even then, the bot doesn't switch itself to live. That's a separate config flag that you flip.

## 11. Nightly self-improvement (Opus 5.5)

At 02:00 local time, a job sends Opus the day's decisions, fills, misses and calibration stats, labelled as data. Opus does root-cause analysis on each loss and writes **one new candidate rule per loss** (for example, "flatten before scheduled macro"), as an edit to `staging/strategy.md` and/or `staging/questions.yaml`. The harness then reruns the full backtest and gates. The candidate trial counts toward the deflated Sharpe, and the holdout is spent only once per candidate.

- Fails → it's logged with the reason and nothing ships.
- Passes → it's queued, and you get a Telegram `/promote <id>` request. **Proposed:** promotion needs your tap, and only one change per week can be promoted. That stops nightly overfitting to recent noise (§18).

Opus runs through the Anthropic API using `claude-opus-5-5`, with tools limited to reading logs and writing to `staging/`.

## 12. Dashboard

A local FastAPI page with server-sent events. It shows each Jev call (question → probabilities, confidence, latency, cost), whether the candle was a candidate, the decision with the vetoes that fired, size, order, fill, and the result when the trade closes. It also shows the equity curve, drawdown and limits headroom, calibration curves per question, the kill-switch button, and HALTED status. It's served on the Mac Mini behind basic auth, and you reach it remotely through Tailscale. **It is not exposed to the public internet.**

## 13. Telegram alerts (BotFather bot)

Events: fill, order error, veto that blocked a candidate, manual-approval request, kill switch fired, process restart, stale data, nightly candidate passed or failed, and the daily report. Commands (your chat ID only): `/status`, `/kill`, `/approve <id>`, `/deny <id>`, `/promote <id>`.

## 14. Daily report (00:05 UTC, Telegram + `reports/daily_YYYY-MM-DD.md`)

Trades, P&L, win rate, largest loss, average Jev latency and cost per decision, Brier score and ECE per question and for `p_win`, and shadow-vs-backtest drift.

## 15. Deployment (Mac Mini)

Python 3.12 in a venv. `launchd` agents for the bot (`KeepAlive`), the dashboard, and the nightly job (`StartCalendarInterval`). `pmset` set so the machine never sleeps. Logs rotate. If the process restarts, the bot rebuilds its position state from the exchange before it trades. I'll write a step-by-step walkthrough. This cloud session is temporary, so it can't host the bot itself.

## 16. Stack

Python 3.12, `pybit` (Bybit v5), `typesafe-sdk` (Jev), `anthropic`, pandas/numpy, FastAPI + SSE, SQLite (decision, fill and calibration log), pytest. Each module ships with a failing test first.

## 17. Final check (the go-live gate, written out in `GO_LIVE.md`)

Does paper match the backtest? Did the kill switch fire in testing? Is any hard limit delegated to a model instead of code? Is Jev's confidence calibrated on our own fills? What market regime would break this? Then a section titled **"WHAT COULD BLOW UP THIS ACCOUNT?"** The bot refuses to go live until every answer is clean.

Already on that list: exchange outage while in a position, a gap through the stop, a Jev outage or model version change (we pin the model version and alert on change), testnet behaving differently from mainnet, too few out-of-sample trades, and the nightly loop overfitting.

## 18. Questions you need to answer before I start the architecture

1. **Manual approval threshold:** is $2,500 notional OK?
2. **Risk defaults in §8:** OK as written, especially the 10% live drawdown kill and the 2x leverage cap?
3. **Promotion:** do nightly changes that pass the gates need your `/promote` tap and stay capped at one per week (my recommendation), or should they promote automatically?
4. ~~**Keys:** TypeSafe key or a gateway?~~ **Answered: you have a TypeSafe key.** It goes in `.env` as `TYPESAFE_API_KEY`, never in chat.
5. **Timeframe:** is 1h entries with a 4h/1d trend filter OK, or do you want 15m?
