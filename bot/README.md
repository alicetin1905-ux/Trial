# jevbot

BTCUSDT perpetual bot: Opus 5.5 designs offline, Jev scores live setups, deterministic code decides
and enforces every limit. Read `SPEC.md`, `ARCHITECTURE.md` and `PLAN.md` first.

## Setup

```bash
cd bot
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q          # unit tests (integration tests need keys and run on the Mac Mini)
```

Secrets go in `bot/.env` (see `.env.example`) or the environment. Never in code, chat or logs.

## Data

`history/BTCUSDT_1h.parquet` (checksummed) holds the 1h bars for 2020-09-01 → 2026-09-30, so
backtests run without rebuilding. To rebuild from Bybit's public trade archive (~125 GB streamed,
about 45 minutes): `.venv/bin/python -m jevbot.data.history --start 2020-09-01 --end 2026-09-30`.

## Backtests

```bash
.venv/bin/python -m jevbot.backtest.harness baseline --label <name>   # rules only
.venv/bin/python -m jevbot.backtest.jev_gate --plan-only             # Jev: count calls, estimate cost
.venv/bin/python -m jevbot.backtest.jev_gate                         # Jev: real run (TYPESAFE_API_KEY)
```

Every configuration evaluated is appended to `reports/trials.jsonl` and counts against the
deflated Sharpe gate. The 2026 holdout is only read by `backtest/gates.py`, once per candidate.
