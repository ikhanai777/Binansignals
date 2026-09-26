# Walk-forward backtest (out-of-sample only) — 2026-09-26 01:30 UTC

4h signals, daily trend filter, BTC regime filter for alts, fees + slippage + real funding included. R = multiples of the initial risk; % columns assume 1% risk per trade compounded.

| Pair | Symbol | From | Trades | Win % | Avg R | PF | Total R | Max DD (R) | Return @1% | Max DD @1% | Validated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | BTCUSDT | 2023-09-27 | 64 | 47 | +0.09 | 1.17 | +5.8 | 9.7 | +5.2% | 9.5% | yes |
| ETHUSDT | ETHUSDT | 2023-09-27 | 70 | 37 | +0.12 | 1.24 | +8.3 | 6.8 | +7.7% | 6.7% | yes |
| XRPUSDT | XRPUSDT | 2023-09-27 | 40 | 45 | +0.17 | 1.36 | +6.9 | 4.3 | +6.7% | 4.3% | yes |
| LTCUSDT | LTCUSDT | 2023-09-27 | 43 | 40 | -0.09 | 0.85 | -3.8 | 9.4 | -4.0% | 9.1% | no |
| DOGEUSDT | DOGEUSDT | 2023-09-27 | 63 | 49 | +0.08 | 1.17 | +5.2 | 10.2 | +4.8% | 9.8% | yes |
| DOTUSDT | DOTUSDT | 2023-09-27 | 51 | 39 | -0.06 | 0.89 | -3.1 | 7.4 | -3.4% | 7.4% | no |
| AVAXUSDT | AVAXUSDT | 2023-09-27 | 54 | 54 | +0.04 | 1.09 | +2.4 | 7.8 | +2.1% | 7.7% | no |
| BNBUSDT | BNBUSDT | 2023-09-27 | 67 | 52 | +0.12 | 1.24 | +8.0 | 8.4 | +7.8% | 8.1% | yes |
| SOLUSDT | SOLUSDT | 2023-09-27 | 39 | 36 | -0.06 | 0.91 | -2.3 | 11.9 | -2.6% | 11.4% | no |
| MATICUSDT | POLUSDT | 2024-09-13 | 30 | 57 | +0.03 | 1.06 | +0.8 | 4.2 | +0.7% | 4.2% | no |
| LINKUSDT | LINKUSDT | 2023-09-27 | 58 | 52 | +0.08 | 1.17 | +4.7 | 5.7 | +4.4% | 5.6% | yes |
| ADAUSDT | ADAUSDT | 2023-09-27 | 51 | 43 | +0.02 | 1.03 | +0.9 | 6.8 | +0.4% | 6.7% | no |

**All pairs combined** (every OOS trade, 1% risk each, chronological): 630 trades, win rate 46%, avg +0.054R, PF 1.10, total +33.9R, max DD 34.4R, t-stat of mean R 1.02.

Honest reading: the out-of-sample edge is positive but NOT statistically significant (t = 1.02 < 2), so it could still be luck. Profits come from a minority of large trend trades, so expect long flat or losing stretches. Pairs that are not validated deserve extra caution, and correlated positions across pairs make portfolio drawdowns deeper than single-pair ones.

_Not financial advice. Signals are generated mechanically from live Binance data; the backtest statistics are historical and out-of-sample, and they do not guarantee future results. Leveraged futures can lose more than the stake._
