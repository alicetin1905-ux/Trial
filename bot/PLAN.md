# Plan (phase 3 of 6, approved)

Status: **APPROVED 2026-10-02.** Build in progress: M0 done. It implements the approved `SPEC.md` and `ARCHITECTURE.md`.

## Ground rules for every task

1. **Failing test first:** write the test, run it, and record that it fails. Then implement until it passes.
2. **Review against the spec:** at the end of each milestone, review the diff for correctness and against the relevant spec sections, and list findings and fixes in the milestone's commit message.
3. **Rollback:** each task is one commit on `claude/opus-jev-trading-bot`, so it can be undone with a plain `git revert`. Strategy versions roll back separately with `jevbot promote --to <id>`. Nothing reaches `main` without a PR you approve.
4. **Fast checks before each push:** `ruff check`, `ruff format --check`, `pytest -m "not integration"`.

## Where each part can run

| Needs | Here (cloud session) | Mac Mini |
|---|---|---|
| History from Bybit's public archive (`public.bybit.com`) | ✅ reachable, about 70 MB/s | ✅ |
| Bybit API: REST, WS, demo-account orders, funding history | ❌ **geo-blocked** (this server is in the US) | ✅ (confirmed: you already trade on Bybit demo) |
| Jev | ✅ once `TYPESAFE_API_KEY` is added to the cloud environment settings | ✅ via `.env` |
| Telegram, Opus | ✅ | ✅ |

So milestones M0–M7 (everything up to and including the gated backtest) can run here. M8 onward is built and unit-tested here, and the integration tests run on the Mac Mini.

## Milestones

### M0 Scaffold (about 0.5 day)
`pyproject.toml` (Python ≥ 3.11, ruff, pytest, hypothesis), `.gitignore` (excludes `.env`, `data/`, `*.db`), `.env.example`, `config/risk.yaml`, `config/settings.yaml`.
Tests first:
- `RiskLimits` is frozen.
- An unknown or missing key in `risk.yaml` → startup error.
- The sha256 of `risk.yaml` is exposed.
- The secret-redaction filter hides values in logs.
- `settings.mode` only accepts `paper`.

### M1 Data (about 1 day)
- `trade_archive.py`: stream each daily `.csv.gz` → 15m bars with OHLCV, buy/sell taker volume, trade count, VWAP. The raw file is discarded and only the bars are kept as parquet. That's about 125 GB downloaded once (2021-01 → 2026-09) and a few MB stored.
- `bars.py`: resample 15m → 1h/4h. Each row gets `available_at` = bar close.
- **Funding:** from the API on the Mac Mini. Here, the API is blocked, so the backtest charges a **conservative** funding cost: it always pays the historical median absolute rate on every 8h settlement held. The gap is recorded in the backtest report.
- Tests first: aggregation on a hand-written fixture CSV with known bars, bar boundaries in UTC, gaps, and `available_at`.

### M2 State engine (about 1 day)
`features.py`, `snapshot.py`, as specified in spec §3, with 15m/1h/4h features.
Tests first:
- Known-value features.
- The snapshot contains no price, timestamp or symbol fields.
- **Leakage tests:** shift-by-one changes the result; a decision is identical whether the data is truncated at `t` or at `t+30d`.

### M3 Strategy rules (about 0.5 day)
`params.py` (front-matter of `strategy.md` → validated params) and `rules.py` (trend filter, pullback candidate, exits, invalidation).
Tests first: hand-built bar sequences that must, or must not, produce a candidate, plus each exit type.

### M4 Backtest harness, rules only (about 1.5 days)
`fills.py`: next-open fill, intrabar SL/TP with the stop winning ties, 0.055% taker fee, slippage of 2 bps plus a volatility component, funding. Also `metrics.py`, `walkforward.py`, `gates.py`, `trials.py`, and the deflated Sharpe.
Tests first: metrics against hand-computed series, the fill model on constructed candles, the holdout only readable from `gates.py`, and `trials.jsonl` being append-only.
**Output:** a **rules-only baseline** report on real data. This is the first real number, and the bar Jev has to beat.

### M5 Jev layer (about 1 day)
`questions.py`, `client.py` (live profile: 1.5 s timeout, no retries, model-pin check; backtest profile: retries, concurrency cap, disk cache), `fake.py`, `combine.py`, `calibration.py`.
Tests first: request shape from `questions.yaml`, timeout → no answer, model mismatch → blocked, cache hit with no network, Brier/ECE against known values, and the isotonic fit being monotonic.

### M6 Decision, sizing, risk (about 1 day)
`decide.py`, `sizing.py`, `risk/guard.py`, `risk/killswitch.py`.
Tests first:
- **Property tests:** the guard never passes an intent that breaks any limit; size is ≤ `f_max` and 0 below the cutoff; size stays fixed until calibrated.
- One unit test per limit.
- The kill switch against a fake broker: cancels all, reduce-only close, `HALTED` blocks every later order.

### M7 Gated backtest with Jev (about 1 day, plus about 1–3 h of Jev calls and about $1–4 in Jev cost)
Run walk-forward with the parameter grid (54 configurations), calibrate on in-sample folds, evaluate the out-of-sample folds, then run **one** holdout evaluation.
Writes `reports/backtest_<id>.md`. **If it passes, it also writes `active/strategy.md`.**
**⏸ CHECKPOINT: I'll report the numbers to you before building any live components.** If the hypothesis fails the gates, you decide: refine it (every change counts as a trial), try a different idea, or stop. I won't adjust anything to force a pass.

### M8 Execution (about 1.5 days; integration tests run on the Mac Mini)
`broker.py`, `bybit.py` (pybit v5, `demo=True`, position-level SL/TP), `shadow.py` (walk the mainnet book), `reconcile.py`, and the startup check that the API key can't withdraw.
Tests first, with recorded fixtures:
- Order-building correctness.
- Reduce-only on exits.
- Price band ±1%.
- Reconcile treating the exchange as correct.
- Shadow fills across book levels.
Marked `integration`: demo-account round trip and the kill switch flattening a real demo position.

### M9 Live engine and journal (about 1 day)
`engine.py`, `journal.py`, `live_feed.py` (WS aggregation shared with M1).
Tests first:
- **Equivalence test:** recorded candles + fake Jev → identical intents from the backtest engine and the live engine.
- Stale data → no entries.
- Restart → reconcile before trading.

### M10 Ops: Telegram, kill path, approvals, watchdog (about 1 day)
`/status`, `/kill`, `/approve`, `/deny`, `/promote`, `/ackmodel`; chat-ID allowlist; manual approval above $25,000 with auto-deny after 5 min; heartbeat watchdog.
Tests first: messages from other chat IDs are ignored, approval timeout → deny, a missing heartbeat → alert and then kill.

### M11 Dashboard and daily report (about 1 day)
FastAPI + SSE, read-only; the kill button writes a HALT request; basic auth; binds to 127.0.0.1.
Daily report: trades, P&L, win rate, largest loss, Jev latency and cost per decision, Brier/ECE, drift against the backtest, Opus spend month to date.
Tests first: auth required, report numbers computed from a fixture journal.

### M12 Weekly Opus loop (about 1 day)
`improve/improve.py` (Tool Runner, `claude-opus-5-5`, effort `high`, caching, refusal fallback on) and `improve/tools.py`. The **$5/month hard cap** is metered per turn. Skipped when no trades closed that week.
Tests first, against a mocked Anthropic client:
- The cap aborts the run.
- Any write outside `staging/` is rejected.
- A candidate touching risk settings is rejected.
- `run_walkforward` never returns holdout numbers.
- Promotion is limited to one per week.

### M13 Ship (about 1 day)
`deploy/launchd/*.plist`, `install.sh`, and `deploy/README.md`: a Mac Mini walkthrough covering `pmset`, Tailscale, `.env`, BotFather, and Bybit key settings (trade only, no withdrawals, IP whitelist).
`golive.py` → `GO_LIVE.md`, ending with "WHAT COULD BLOW UP THIS ACCOUNT?". It exits non-zero until every item is clean.
Final review of the whole diff, then a PR, opened only once you ask for it.

**Total: about 13 working days of build.** Then at least 60 days of paper trading before the go-live check can even pass (spec §10).

## What I need from you

1. ~~Is Bybit allowed where you are?~~ **Answered: yes. You trade on Bybit Demo now.** Venue switched from testnet to Demo Trading. **Before M8:** create a sub-account for the bot, turn on Demo Trading for it, and create a trade-only demo API key there. Don't share the demo account you trade by hand.
2. **Add `TYPESAFE_API_KEY` to this cloud environment's settings** (environment menu in the session title bar → Edit) before M7. It's picked up in a new session. M0–M6 don't need it.
3. **Approve this plan.** Then I start M0.
