# Binansignals

Signals for 12 Binance USD-M perpetual futures pairs, built only from **real, live data**. The trade rules are
**validated out-of-sample** before a pair is marked tradable.

| # | Pair | Traded symbol | Notes |
|---|---|---|---|
| 1 | BTCUSDT | BTCUSDT | Bitcoin |
| 2 | ETHUSDT | ETHUSDT | Ethereum |
| 3 | XRPUSDT | XRPUSDT | XRP |
| 4 | LTCUSDT | LTCUSDT | Litecoin |
| 5 | DOGEUSDT | DOGEUSDT | Dogecoin |
| 6 | DOTUSDT | DOTUSDT | Polkadot |
| 7 | AVAXUSDT | AVAXUSDT | Avalanche |
| 8 | BNBUSDT | BNBUSDT | BNB |
| 9 | SOLUSDT | SOLUSDT | Solana |
| 10 | MATICUSDT | **POLUSDT** | Polygon. MATICUSDT was delisted after the MATIC→POL migration; POLUSDT is the live contract |
| 11 | LINKUSDT | LINKUSDT | Chainlink |
| 12 | ADAUSDT | ADAUSDT | Cardano |

## Quick start

```bash
pip install -r requirements.txt
python -m binansignals signals --equity 1000 --risk 0.01    # live signals -> reports/signals.md + .json
python -m binansignals backtest                              # walk-forward report -> reports/backtest.md
python -m pytest -q                                          # offline tests on recorded real data
```

No API key is needed because only public market-data endpoints are used. If `fapi.binance.com` is geo-blocked
on your network (HTTP 451), the client falls back to `www.binance.com/fapi`, which serves the same data.

## What goes into a signal

| Layer | Source (all live) | Role |
|---|---|---|
| Directional bias | Daily EMA20/50 stack + slope, from 4h klines | No bias → no trade. The system is built for trending markets |
| BTC regime | BTC daily bias | Alts are never traded against BTC's daily trend |
| Entry rule (4h) | Pullback-to-EMA21 continuation **or** Donchian 20/55 breakout | The part that is backtested and walk-forward validated |
| Order-flow confirmation | Binance taker-buy volume per bar (bar delta / CVD) | Part of the backtested entry rules |
| Footprint | `aggTrades` → bid/ask volume per price level, diagonal 3:1 stacked imbalances, absorption | Live confluence |
| Volume profile | 72h of 1m klines split by aggressor → POC, value area, HVNs | Live context / targets |
| Support / resistance | 4h + daily swing pivots, clustered by ATR, recency-weighted | Checks there is room to 2R |
| Positioning | Funding, open-interest change vs price, retail and top-trader long/short ratios | Crowding filter |
| Order book | 500-level depth: ±1% imbalance, largest walls | Context |
| News | CoinDesk, Cointelegraph, Decrypt and The Block RSS (48h), Fear & Greed, Binance delisting notices | Risk filter only; a delisting blocks the pair |

### Statuses
* **ACTIVE**: the validated rule fired on one of the last two closed 4h bars and price is still within 0.5R of
  the signal close. Enter at market with the given stop.
* **PENDING**: bias is present. The report says exactly which 4h close triggers the entry (pullback zone or
  breakout level) and gives a pre-computed stop and targets.
* **NO TRADE**: either there is no directional bias, or no rule variant had a positive edge on the latest
  180 days. Stay flat.

**Grade** combines the 0–100 confluence score with out-of-sample validation. Only pairs with ≥25 OOS trades and
profit factor ≥ 1.15 can be graded A or B. The confluence factors are live context and are **not** backtested.

### Trade management (the same logic as the backtest)
Stop-market at the stop. At TP1 (1R or 1.5R) the rule either closes 50% or, for variants without a partial, only
arms the next step. In both cases the stop then moves to entry and the rest trails 2.5–3.5× ATR(4h) behind the
best price. There is a 15-day time stop. Size = equity × risk ÷ stop distance. Leverage is only what that
notional needs, capped at 10×.

## Backtest methodology

* About 3 years of real 4h Binance futures history. POLUSDT only has data since its Sept-2024 listing.
* Costs: 0.05% taker and 0.02% maker fees, plus 0.02% slippage on every market or stop fill. Real historical
  funding is charged or credited at every 8h settlement.
* Conservative fills: entry at the next bar's open. Gaps through the stop fill at the open. If one bar touches
  both the stop and the target, the stop is assumed to fill first.
* **Walk-forward:** 48 rule variants. Every 60 days, the variant with the best t-stat of mean R over the previous
  180 days is traded blind for the next 60. If no variant was positive, the pair stays flat. Only these
  out-of-sample trades are reported.

See `reports/backtest.md` for the latest numbers. Honest summary from the run committed here: the combined
out-of-sample result across all 12 pairs is **positive but thin** (PF ≈ 1.10, t ≈ 1.0, which is not statistically
significant). Individual pairs range from PF 0.85 to 1.36. Several pairs are not validated, and correlated positions
deepen portfolio drawdowns. 1h versions of the same rules lost money after costs and were dropped. No strategy
here is "highly profitable" in a way a real backtest can prove, and anyone who shows you one without
out-of-sample, cost-inclusive results is showing you a curve fit.

> Not financial advice. Historical and out-of-sample results do not guarantee future returns. Leveraged futures
> can lose more than the margin posted.

## Layout

```
binansignals/
  client.py      Binance USD-M REST client (pagination, caching, host fallback, rate-limit aware)
  indicators.py  EMA/ATR/RSI/ADX, causal higher-timeframe resampling and as-of joins
  strategy.py    feature preparation + pullback / breakout entry rules
  backtest.py    trade simulator (fees, slippage, funding) + walk-forward optimiser
  levels.py      S/R clustering, volume profile
  orderflow.py   footprint from aggTrades, order-book imbalance, derivatives positioning
  news.py        RSS headlines, sentiment lexicon, Fear & Greed, Binance delistings
  signals.py     live engine: plan (ACTIVE/PENDING/NO TRADE), confluence score, sizing
  report.py      Markdown / JSON output
reports/         latest generated signals and backtest (snapshot; regenerate before trading)
tests/           offline tests on recorded real Binance data
```
