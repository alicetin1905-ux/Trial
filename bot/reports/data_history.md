# Data: BTCUSDT 15m bar history

Built 2026-10-02 from Bybit's public trade archive (`public.bybit.com/trading/BTCUSDT/`) with
`python -m jevbot.data.history`. Raw files were streamed and discarded; only the bars are kept
(`data/bars15/`, about 34 MB, git-ignored, rebuildable). A sha256 of every raw file is in `manifest.jsonl`.

| | |
|---|---|
| Range | 2020-09-01 00:00 → 2026-09-30 23:45 UTC (2,221 days) |
| 15m bars | 213,216 |
| Missing days / failed downloads | 0 / 0 |
| Gaps in the bar series | none |
| Bars with no trades | 2 (flat at the previous close, zero volume) |
| Trades outside their file's day | 0 |
| Newest-first files (re-sorted) | 462 (2020–2021) |

Uses: warm-up 2020-09 → 2020-12, design 2021-01 → 2025-12, holdout 2026-01 → 2026-09 (only
`backtest.gates` reads it).

Sanity check against known prices (yearly close / high / low):

| Year | Close | High | Low |
|---|---|---|---|
| 2021 | 46,200 | 69,138 | 27,923 |
| 2022 | 16,550 | 48,185 | 15,440 |
| 2023 | 42,325 | 44,777 | 16,511 |
| 2024 | 93,530 | 108,422 | 38,532 |
| 2025 | 87,595 | 126,150 | 74,456 |
| 2026 (to Sep) | 83,574 | 97,963 | 57,756 |
