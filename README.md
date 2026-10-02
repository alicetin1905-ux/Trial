# Trial

A single-file strategy backtester for Bybit perpetual futures. No build step, no backend, no API keys.

Open `index.html` in a browser, or host it on GitHub Pages.

## Features

- 32 strategies ported from the most popular TradingView community scripts, in four groups:
  - Trend following: Supertrend, UT Bot Alerts, Chandelier Exit, Chandelier Exit + ZLSMA, HalfTrend, Pivot Point SuperTrend, Range Filter, Twin Range Filter, SSL Channel, Hull Suite, OTT, MOST, AlphaTrend, Trend Magic, Coral Trend, MavilimW, Tillson T3, Gaussian Channel, Smoothed Heiken Ashi, Ehlers Instantaneous Trendline
  - Momentum: Squeeze Momentum, WaveTrend, QQE MOD, MACD Custom (ChrisMoody), Waddah Attar Explosion, Schaff Trend Cycle, Volume Flow Indicator, Williams Vix Fix
  - Bands and breakouts: Nadaraya-Watson Envelope (non-repainting), Trendlines with Breaks, Support and Resistance Levels with Breaks
  - Machine learning: Lorentzian Classification
- Each strategy uses the original script's default settings, except Chandelier Exit, which starts at 4 / 2. The script author is shown in the strategy name.
- Symbols: BTC, ETH, SOL, XRP, DOGE, BNB, LINK, AVAX (USDT perps)
- Timeframes: 15m, 30m, 1H, 2H, 4H, 6H, 12H, 1D, with up to 20,000 candles
- Editable settings for every strategy
- Trading rules: long/short/both, capital, position size, leverage, fee per side, slippage, stop loss, take profit
- Equity curve against buy and hold, drawdown, win rate, profit factor, Sharpe ratio, fees paid, full trade list
- "Compare all strategies" ranks every strategy on the same candles and rules

## How the test works

- Signals are read at candle close and filled at the next candle open.
- Stop loss and take profit are checked inside the candle. If both are touched, the stop wins.
- After a stop or take profit, the strategy stays out until its signal changes.
- Fees are charged on entry and exit on the full position value.
- With leverage above 1, liquidation is approximated at 90% of the margin distance.
- Funding payments are not included.

## Data

Candles come from the public Bybit v5 kline endpoint, loaded in the browser. Settings are saved in `localStorage`.

## Disclaimer

For research only. Past results do not predict future results, and a strategy that wins on one symbol and timeframe is often just luck. Not financial advice.
