# Architecture (phase 2 of 6, awaiting approval)

Status: **DRAFT, needs your approval before the plan.** It implements the approved `SPEC.md`.

## 1. The one rule the whole design hangs on

There is **one pure decision function**, `decide()`, and backtest, paper and live all call it. It takes a snapshot, Jev's answers, the strategy parameters, the calibrator and the account state, and returns an `Intent`, which can be "no trade". Nothing in `decide()` does I/O, reads a clock or talks to a model. Because backtest and live run the same code on the same snapshot, the only differences between them are data and fills, and both of those are measured (§7).

```
            ┌────── weekly (Sun 02:00, offline) ──────────┐
            │  Opus 5.5 ──writes──▶ staging/             │
            │                 harness: backtest + gates  │
            │                 ──▶ queue ──/promote──▶ active/
            └────────────────────────────────────────────┘
                                   │ (strategy.md, questions.yaml, calibrator.json — read-only at runtime)
every closed 15m candle            ▼
 Bybit mainnet WS ─▶ bars ─▶ snapshot() ─▶ Jev (1 call, ≤1.5 s) ─▶ decide() ─▶ Intent
   (kline, trades,          pure, t≤T only     answers as floats        pure
    orderbook, funding)                                                   │
                                                                          ▼
                                      risk.guard(Intent, account, limits) ── veto ──▶ journal + alert
                                                       │ pass
                                       notional > $2,500? ── yes ──▶ Telegram approve/deny (5 min)
                                                       │
                                     broker: testnet orders  +  shadow fills vs mainnet book
                                                       │
                                               journal (SQLite) ──▶ dashboard (SSE), daily report, calibration
```

## 2. Processes on the Mac Mini (launchd)

| Process | launchd | Job | Talks to exchange? |
|---|---|---|---|
| `jevbot-engine` | `KeepAlive` | Market data → snapshot → Jev → decide → guard → orders. The only process that **opens** positions. | yes |
| `jevbot-ops` | `KeepAlive` | Polls Telegram (your chat ID only) and runs the **independent kill path**. Watchdog: if the engine's heartbeat is older than 2 minutes while a position is open, it alerts you, and after 10 minutes it kills. | close/cancel only |
| `jevbot-dashboard` | `KeepAlive` | FastAPI + SSE, read-only on SQLite. Its kill button writes a `HALT` request that ops carries out. | no |
| `jevbot-weekly` | `StartCalendarInterval` Sun 02:00 | Opus improvement loop + backtest harness. Skipped if no trades closed that week or the monthly cap is used up. | no |
| `jevbot-report` | `StartCalendarInterval` 00:05 UTC | Daily report → Telegram + `reports/` | no |

Why the kill path is a separate process: if the engine hangs, crashes or deadlocks, `/kill` still works. Killing is idempotent (cancel all orders, reduce-only close, write `HALTED`), so it's safe even if the engine and ops both trigger it at once. The engine checks `HALTED` before **every** order.

**Stops live on the exchange.** Each entry is placed with Bybit position-level SL/TP attached, so a bot crash never leaves a position without a stop. On restart, the engine **reconciles** before doing anything: it reads the exchange position, open orders and wallet. If they disagree with the journal, the exchange is treated as correct, you get an alert, and new entries stay blocked until they match.

## 3. Package layout

```
bot/
  SPEC.md  ARCHITECTURE.md  PLAN.md  GO_LIVE.md
  pyproject.toml  .env.example  .gitignore
  config/
    risk.yaml            # hard limits. Loaded once into a frozen dataclass; its sha256 is journaled.
    settings.yaml        # symbol, timeframe, model pins, paths, mode (paper only, see §8)
  active/                # promoted strategy: strategy.md (+ YAML front-matter), questions.yaml, calibrator.json
  staging/               # the ONLY directory the weekly Opus loop can write
  src/jevbot/
    config.py            # pydantic-settings; secrets as SecretStr; log redaction filter
    clock.py             # injectable clock (real / simulated) — nothing else calls time.time()
    data/
      bybit_rest.py      # klines, funding history, instrument info (mainnet, public)
      trade_archive.py   # public.bybit.com daily trades → 15m signed-flow bars
      live_feed.py       # WS: kline.15, publicTrade, orderbook.50, tickers; same 15m aggregation as archive
      bars.py            # resampling 15m→1h/4h, as-of joins; every row carries available_at
      cache.py           # parquet cache of history
    state/
      features.py        # pure feature functions
      snapshot.py        # snapshot(bars, t) -> Snapshot (numeric, normalised, no price/date/symbol)
    jev/
      questions.py       # questions.yaml -> typesafe_sdk Choice/Score/Noul
      client.py          # AsyncTypeSafeClient wrapper: timeout 1.5 s, retries OFF (live), model pin check,
                         #   latency + token cost metering, response cache keyed by sha256(snapshot+questions+model)
      fake.py            # deterministic fake Jev for tests and dry runs
    strategy/
      params.py          # parse strategy.md front-matter -> StrategyParams (validated)
      rules.py           # trend filter, pullback candidate, exits, invalidation
      combine.py         # edge_score = Σ w·f(answers)
      calibration.py     # isotonic/Platt fit, Brier, ECE, reliability bins, recalibration map
    decide.py            # THE pure decision function
    sizing.py            # quarter-Kelly × confidence, f_max, fixed-size mode until calibrated
    risk/
      limits.py          # RiskLimits (frozen) from risk.yaml
      guard.py           # check(intent, account, market) -> Pass | Veto[reasons]   (every order)
      killswitch.py      # flatten + cancel + HALTED; triggers: drawdown, /kill, button, HALT file, 3 errors/10 min
    execution/
      broker.py          # Broker protocol
      bybit.py           # pybit v5 HTTP + private WS (testnet now; mainnet only behind live unlock)
      shadow.py          # simulated fill against the mainnet book at decision time
      reconcile.py
    engine.py            # live loop: on candle close -> ... (diagram above)
    journal.py           # SQLite (WAL): jev_calls, decisions, vetoes, orders, fills, positions, equity, events
    alerts/telegram.py   # send + command poller (chat-id allowlist, commands only, no free text parsed)
    dashboard/app.py, static/
    report.py            # daily report
    backtest/
      engine.py          # event loop over historical bars calling snapshot()+jev+decide()+guard()
      fills.py           # next-open fill, intrabar SL/TP (stop wins ties), fees, slippage, funding
      metrics.py         # Sharpe (daily), MDD, hit rate, t-stat, deflated Sharpe, bootstrap bands
      walkforward.py     # anchored folds; holdout handled by gates.py only
      gates.py           # pass/fail + report; the ONLY place that touches the holdout
      trials.py          # append-only trials.jsonl
    improve/
      improve.py         # Opus loop (Tool Runner)
      tools.py           # the tools Opus gets (§6)
    golive.py            # final check -> GO_LIVE.md, exits non-zero unless all clean
  deploy/
    launchd/*.plist  install.sh  README.md   # Mac Mini walkthrough
  tests/
```

## 4. Data and leakage

- All timestamps are integer milliseconds in UTC. Each derived row carries `available_at`, and `snapshot(t)` only reads rows where `available_at ≤ t`.
- A 15m bar at `[t−15m, t)` becomes available at `t` in live trading, once Bybit marks the kline `confirm=true`. Backtest uses the same rule.
- Live flow bars are aggregated from the `publicTrade` WS with **the same function** that aggregates the daily archive files. A test records a live window and checks it matches the archive for the same window within tolerance.
- **Leakage tests:** shift-by-one (the result must change), truncation (a decision at t is identical whether the data ends at t or at t+30 days), and a "no-future-columns" scan over the snapshot.

## 5. Jev integration

- One `system_one` call per closed candle, carrying the 5 questions from `questions.yaml` and the snapshot as a JSON `state`.
- **Live:** timeout 1.5 s, **SDK retries disabled**. Timeout, error, malformed answer, or `response.model ≠ pinned model` → no trade, journal entry, alert. A model change also blocks entries until you acknowledge it with `/ackmodel`. That stops a silent Jev upgrade from changing a calibrated system.
- **Backtest:** async with concurrency capped below 1,200 requests/min, retries on, and a response cache on disk. Reruns cost nothing and give identical results. A cache entry is only reused if the model name matches.
- Each call is metered: latency, input tokens, cost. These feed the dashboard and the daily report.

## 6. Weekly Opus loop (offline only)

Python `anthropic` SDK, **Tool Runner**, `claude-opus-5-5`, adaptive thinking, `effort: "high"`, streaming. Server-side refusal fallback (`fallbacks: "default"`) is **turned on**, so a refusal falls back to another model instead of failing the run. Say if you want that off.

Tools Opus gets (and nothing else):

| Tool | Does |
|---|---|
| `get_session_summary(date)` | Day's decisions, fills, misses, vetoes, calibration, all labelled as data |
| `get_trade(id)` | Full record of one decision: snapshot, answers, intent, fills |
| `read_active(file)` | Read `strategy.md` / `questions.yaml` from `active/` |
| `write_candidate(strategy_md, questions_yaml, rationale, new_rules[])` | Validates the schema and writes `staging/<id>/`. **Rejected if it changes anything risk-related.** |
| `run_walkforward(candidate_id)` | Harness runs the in-sample walk-forward and returns metrics. **No holdout numbers.** Counts as a trial. |

Up to 3 candidates per run. After Opus finishes, `gates.py` evaluates the **final** candidate on the holdout once, including deflated Sharpe over every trial so far. Pass → queued, and Telegram `/promote <id>`, limited to one per week. Promotion copies `staging/<id>` → `active/` and records the git commit. **Rollback:** `jevbot promote --to <previous id>`. Every decision row records the hash of the strategy version that made it.

**Cost control (you set the cap):** a **hard $5/month cap**, at most $1.25 per run. Spend is metered from each turn's `usage` at $4 per 1M input, $0.20 per 1M cached input and $20 per 1M output, and the loop aborts before going over. The system prompt and tools are cached, and runs are weekly and skipped when no trades closed. Expected cost is about $0–5 a month. Month-to-date spend appears in the daily report.

## 7. Paper mode (decided in spec §10)

`broker = Composite(testnet_orders, shadow_fills)`. Every intent goes to the Bybit testnet, to exercise order placement, attached SL/TP, cancels, reduce-only and the kill switch. In parallel, `shadow.py` simulates the fill by walking the **mainnet** `orderbook.50` snapshot taken at send time. The P&L compared to the backtest is the **shadow** P&L. Testnet fills are logged too, but only to check the plumbing.

## 8. Live mode is locked

`settings.mode` accepts only `paper`. Code paths that can reach a mainnet private endpoint are guarded by `golive.py --unlock`, which only succeeds when every item in `GO_LIVE.md` is clean. After that, you set `mode: live` by hand. No model and no automated process can write `settings.yaml`.

## 9. Secrets and security

- `.env` holds `TYPESAFE_API_KEY`, `ANTHROPIC_API_KEY`, `BYBIT_TESTNET_KEY`/`_SECRET`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` and `DASHBOARD_USER`/`_PASS`. It's git-ignored, set to `chmod 600`, and loaded as `SecretStr`.
- A logging filter redacts every secret value. A test fails if any secret appears in logs or the journal.
- On startup, the engine checks the key's permissions through the Bybit API (`/v5/user/query-api`) and **refuses to start** if withdrawals are enabled.
- The dashboard binds to `127.0.0.1` and is reached through Tailscale, with basic auth. Telegram only accepts commands from your chat ID, and only the fixed command set is parsed.

## 10. Testing strategy (test-first, per module)

- **Unit:** features, snapshot, rules, combine, calibration, sizing, guard (one test per limit), killswitch (fake broker), metrics (known answers), gates.
- **Property tests:** the guard never passes an intent that breaks any limit, for random intents and account states. Sizing is never above `f_max` and is 0 below the cutoff.
- **Leakage tests:** see §4.
- **Equivalence:** the backtest engine and the live engine produce identical intents when fed the same recorded candles and a fake Jev.
- **Integration (needs keys, run manually or on the Mac Mini):** testnet order round-trip, attached SL/TP, the kill switch flattening a real testnet position, reconcile after a forced restart.
- **Improvement-loop guardrail test:** a candidate that edits `risk.yaml` or adds a risk-related key is rejected.

## 11. What's deliberately simple in v1

One symbol and one position. SQLite rather than Postgres. Polling Telegram rather than webhooks. A static dashboard page without a framework. Each can be swapped out later without touching `decide()`.
