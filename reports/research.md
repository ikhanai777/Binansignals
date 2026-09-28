# Can the win rate be raised? Research log (2026-09-26)

**Question:** find strategies that raise the win rate substantially while staying profitable after costs.

**Protocol** (so the answer can't be tuned into the data):
* Real Binance USD-M history for all 12 pairs, Sep 2023 → Sep 2026: 1h and 4h klines with taker-buy
  volume, plus real funding.
* All design choices were made on the **development period (before 2025-10-01)**. The **holdout
  (2025-10-01 → 2026-09-25)** was evaluated **once**, after the rules were frozen.
* Costs: 0.05% taker, 0.02% maker, 0.02% slippage per market/stop fill, real funding. Entry at the next
  bar's open. If a bar touches both the stop and the target, the stop fills first.
* "Win" = trade closed with a positive net return.

Reproduce: `python -m research.run rules | ml | carry`.

## 1. Win rate is a dial, not an edge

With a take-profit closer than the stop, 70–75% win rates are trivial to get, and they still lose after fees.
Rule families tested on the dev period, pooled over 12 pairs:

| Idea | Example config | Trades | Win rate | PF | t |
|---|---|---|---|---|---|
| RSI(2) dip-buy in daily uptrend (1h) | TP 1 ATR / SL 3 ATR | 2,904 | **74.5%** | 1.01 | 0.2 |
| Bollinger −2.2σ dip (1h) | TP 0.75 / SL 2 ATR | 2,234 | **71.2%** | 0.88 | −2.4 |
| Stop-run sweep of 72h low (1h, no filter) | TP 1 / SL 3 ATR | 3,790 | **72.6%** | 0.85 | −3.9 |
| Taker-delta absorption bar (1h) | TP 1 / SL 3 ATR | 256 | **74.2%** | 1.09 | 0.6 |
| EMA20 pullback, limit entry (4h) | TP 0.75 / SL 2 ATR | 1,648 | 68.9% | 0.91 | −1.6 |
| Sweep of 24-bar low, limit entry 1 ATR (4h) | TP 1 / SL 2 ATR | 202 | 68.3% | 1.26 | 1.5 |

The 1h rows come from the first exploratory scan, which used market entries. `research.run rules` reproduces the 4h scan.
None is statistically distinguishable from zero or better, and most are clearly negative. Crypto 1h
mean-reversion edges are a few basis points, below the ~0.11–0.14% round-trip cost.

The same applies to the existing trend system. Walk-forward OOS, all 12 pairs, $500 start at 1% risk:

| Exit | Win rate | PF | $500 → |
|---|---|---|---|
| Current (let winners run) | 45.9% | 1.10 | **$664** |
| Take 50% at +0.5R, rest to break-even | **67.5%** | 1.04 | $532 |
| Take 50% at +0.75R | 56.6% | 0.96 | $424 |

Raising the win rate this way lowered profits every time.

## 2. Machine-learning filter (walk-forward gradient boosting)

* Around 40 causal features per 4h bar: multi-horizon returns, volatility regime, RSI, EMA distances, taker
  delta / CVD, volume z, close location, Donchian distances, daily and 4h trend state, BTC returns and
  relative strength, funding level and z-score, hour and weekday.
* Label: 2 ATR target vs 2 ATR stop within 18 bars (3 days).
* The model is retrained every 30 days on data whose labels had already closed (purged). It trades when
  P(win) exceeds the threshold.

| Period | Threshold | Trades | Win rate | PF | t |
|---|---|---|---|---|---|
| Dev walk-forward (Jun 2024 – Sep 2025) | 0.60 | 1,098 | 55.1% | 1.28 | 3.6 |
| Dev walk-forward | 0.65 | 645 | **57.8%** | **1.45** | 4.1 |
| **Holdout (Oct 2025 – Sep 2026)** | 0.60 | 681 | **47.0%** | **0.87** | −1.7 |
| **Holdout** | 0.65 | 374 | **47.3%** | **0.87** | −1.2 |

The dev result looked strong: it was consistent across all 12 pairs and every quarter, even though the
model only ever saw past data. It **did not survive the holdout**. Losses were concentrated in Q4 2025–Q1 2026,
and later quarters were roughly flat. Choosing this configuration from about 20 tried options, plus a regime
change, was enough to erase it. **It is not used for live signals.** In the same holdout year, the existing
trend system made PF 1.16 on 316 trades.

Other barrier shapes (1/1.5, 1/2 ATR) produced 59–67% win rates with no edge after fees.

## 3. Funding carry: the high-win-rate strategy that held up

The position is long spot plus short the perp in equal size, collecting funding. This is not directional,
and it needs a spot leg. From real funding history, Oct 2023 → Sep 2026, net of an assumed 0.34% round-trip cost:

| Pair | APR | Months positive | Worst month | Holdout-year APR |
|---|---|---|---|---|
| LTC | 8.6% | **97%** | −0.01% | 3.1% |
| LINK | 8.5% | **97%** | −0.04% | 4.7% |
| DOGE | 8.1% | **94%** | −0.23% | 2.9% |
| BTC | 7.4% | **92%** | −0.18% | 3.4% |
| ETH | 7.5% | **92%** | −0.31% | 2.5% |
| ADA | 7.4% | 89% | −0.35% | 1.9% |
| XRP / SOL / AVAX / DOT / BNB / POL | −1.6% … 7.1% | 48–83% | down to −2.3% | −8.6% … 2.0% |

Month-level win rates of 89–97% are real, but yields are modest and fell in the last year. Returns are on the
hedged notional, so APR on total capital is lower once the perp margin buffer is counted. A switching filter
(only in the trade when trailing funding is positive) did worse because of the extra fees.
`python -m binansignals carry` shows the live figures.

## Conclusions

1. No directional strategy tested here raises the win rate substantially **and** stays profitable on unseen data.
   Every 65–75% win-rate variant paid for it with lower or negative expectancy.
2. The existing trend system (≈46% wins, PF ≈ 1.10, and PF 1.16 in the holdout year) remains the best
   directional option found. Its win rate is low because profits come from a few large trends.
3. For a genuinely high win rate, funding carry on BTC/ETH/LTC/DOGE/LINK is the honest answer. It is
   market-neutral, with single-digit APR.

_Not financial advice._

## 4. Session fade: "buy every dip, sell every rise" (added 2026-09-26)

Tested on real 15m data for all 12 pairs, Sep 2023 → Sep 2026 (`python -m research.session_fade`; full grid in
`reports/session_fade_grid.csv`). At each session's open, a resting buy is placed k × (yesterday's daily ATR)
below the open and a resting short the same distance above. The target is the session open. The stop is
s × ATR, or none (exit at session end). Limit fills must trade through the level. Maker/taker fees and slippage
are included. There are 128 variants: 4 sessions × dip size × stop × optional daily-trend filter.

$500 start, fixed slot of 1/12 of equity per pair (x leverage):

| Variant | Win rate | 1x | 3x | 5x | Last 12 months at 3x |
|---|---|---|---|---|---|
| Every small dip/rise (0.25 ATR), full day, no stop | 63% | $63 | $0 | $0 | $21 |
| Every small dip/rise, NY session, 0.5 ATR stop | 54% | $68 | $0 | $0 | $20 |
| Every small dip/rise, Asia session, 0.5 ATR stop | 52% | $211 | $26 | $2 | $209 |
| Big dips only (0.75 ATR), Asia, no stop | 57% | $660 | $1,065 (DD 36%) | $1,549 (DD 56%) | $934 |
| Big dips only, Asia, trend filter, no stop | 57% | $590 | $808 | $1,080 | $575 |
| Big dips only, London, trend filter, no stop | 59% | $659 | $1,083 | $1,655 | **$473** (lost in last 12 months) |

* Fading every small move loses in every session, with every stop setting, in both periods. It trades
  thousands of times, and fees plus trending sessions outweigh a 52–63% win rate.
* Stops: tighter stops cut the win rate and did not rescue any variant. No stop (session-end exit) was usually
  best but had single-trade losses of 6–15% (e.g. 2024-08-05).
* Only large dips in the Asia session were positive in both periods, and modestly. That variant was picked
  from 128 after seeing both periods, so expect less going forward. London worked in 2023–25 but lost in the
  last 12 months. New York lost throughout.

## 5. 5-minute scalping with high leverage (added 2026-09-26)

One year of real 5m data for all 12 pairs (`python -m research.scalp5m`). Five common scalps were tested:
EMA 9/21 cross with the 1h trend, VWAP stretch reversion, Bollinger + RSI(2) fade, a 12-bar breakout on
volume + taker delta, and 3-bar delta momentum. Each ran with five take-profit/stop shapes, for 25 variants and
3,000–244,000 trades each.

* Gross edge before costs: between −1.9 and +0.7 basis points per trade for every variant, which is noise.
  Realistic costs are 9–14 bp per round trip.
* Net per trade with realistic costs: −10.5 to −12.8 bp for all 25 variants. Even with optimistic all-maker
  fills at 0.02%, every variant loses 3.3–5.9 bp per trade. Win rates range from 31% to 72% depending on the
  target/stop shape, and none of them profit.
* $500, whole account as margin, one position at a time, best variant (Bollinger/RSI fade):

| Leverage | Peak reached | 90% of the account lost after |
|---|---|---|
| 1x | $504 | 64 days |
| 10x | $539 | 4 days |
| 25x | $600 | 1–2 days |
| 50x | $709–736 | < 1 day |
| 100x | $954–1,051 | < 1 day |

Leverage does not create an edge. It multiplies the per-trade fee loss, so the account dies faster. The
brief spikes toward $1,000 at 100x are the lure, not the outcome.

## 6. User-specified 5m strategy: EMA9/21 + VWAP + volume z + RSI, 2% SL / 2% TP, 5h window (added 2026-09-26)

Implemented as specified (`python -m research.user_strategy_5m`; the interpretation notes are in the file
header). Tested on one year of real 5m data for all 12 pairs, with entry at the next bar's open,
realistic fees and one position per pair.

| Variant | Trades | Win rate | Avg per trade | PF | $500 @1% risk | $500 @5x full margin | $500 @10x |
|---|---|---|---|---|---|---|---|
| As specified (cross within last 6 bars) | 2,032 | 44.2% | −0.22% | 0.72 | $51 | $0 | $0 |
| Strict (cross on the same bar) | 977 | 43.1% | −0.26% | 0.68 | $135 | $0 | $0 |
| Cross within 12 bars | 2,860 | 44.2% | −0.19% | 0.74 | $29 | $0 | $0 |
| + footprint absorption required | 171 | 55.0% | +0.17% | 1.29 | $573 | $567 | $281 |

* 95% of signals are longs. Longs lose (−0.24%/trade); the few shorts are slightly positive (+0.08%, 106 trades).
* 52% of trades hit the 5-hour timeout: a 2% target/stop is far beyond typical 5m moves, so the outcome is
  mostly where price drifts in 5 hours.
* Only 2 of 13 months had a positive average trade in the as-specified version.
* The absorption-filtered version is slightly positive but not significant (t = 1.44, 171 trades, 6 of 12
  months positive). Beyond 10x, leverage destroys it. At ≥ ~40x the isolated liquidation price sits inside
  the 2% stop, so every stop-out is a liquidation.

## 7. Gold: XAUUSDT / PAXGUSDT (added 2026-09-26)

`python -m research.gold`. Binance gold contracts are young: **XAUUSDT** (TradFi perp) has traded since
2025-12-11 and **PAXGUSDT** perp since 2025-03-27. For a longer test, PAXGUSDT **spot** 4h (a gold-backed
token, since 2020-08) is used as a gold proxy with futures costs.

* Volatility: gold moved 1.4–1.6% per day in this period (BTC 2.3%). Its median 5m range is 7–8 bp, *below*
  the ~11 bp round-trip cost, so 5m gold scalping is structurally worse than crypto scalping.
* Futures-only history is too short: 7–28 OOS trades, PF 0.58–2.31 depending on window length (noise).
* **Trend system on 5 years of gold (PAXG spot, Jun 2021 – Aug 2026, walk-forward OOS): 107 trades, win 46%,
  avg −0.17R, PF 0.73; $500 → $413 at 1% risk.** Positive only in 2025. Shorts were very poor (−0.50R avg);
  longs broke even.
* Buy-and-hold gold over the same period: $500 → $993 at 1x (max drawdown 29%), $1,631 at 2x (DD 52%),
  $2,210 at 3x (DD 70%). Yearly: 2022 −1%, 2023 +11%, 2024 +30%, 2025 +65%, 2026 YTD −1%.
* Conclusion: gold's big gains came from one long bull run that simply holding captured. The crypto trend
  rules did not add value on gold, and leverage mainly scaled the drawdowns.

## 8. Spot dip-buying with a $500 monthly budget vs plain DCA (added 2026-09-26)

`python -m research.dip_dca`. Real Binance spot daily closes, 0.1% fee, no leverage. $500 is added each
month; on every dip day, 10% of the available cash is spent. Dips were defined four ways: a daily drop of at
least 5% or 8%, or a close at least 10% or 20% below the 30-day high. The benchmark (DCA) invests the full
$500 on the 1st of each month. There were 3 baskets × 3 start dates × 4 dip rules = 36 comparisons.

| Start | Basket | DCA | Best dip rule | Worst dip rule |
|---|---|---|---|---|
| Jan 2020 | BTC | **+195%** ($40.5k → $119k) | +157% (dd10) | +79% (red8) |
| Jan 2020 | All 12 | **+250%** | +203% (dd10) | +119% (red8) |
| Nov 2021 (peak) | BTC | **+97%** | +86% (dd10) | +32% (red8) |
| Nov 2021 | All 12 | **+30%** | +28% (dd10) | +10% (red8) |
| Jan 2024 | BTC | **+13%** | +11% (dd20) | +3% (red8) |
| Jan 2024 | BTC + ETH | +8% | **+12% (dd20)** | +4% (red5) |
| Jan 2024 | All 12 | −5% | −8% (dd10) | −10% |

* Dip-buying beat DCA in only 1 of 36 comparisons (BTC + ETH from 2024 with dd20, +12% vs +8%). Waiting for
  dips leaves cash idle (up to 85% of the pot) while prices rise, and spending only 10% per dip deploys
  money too slowly.
* It does reduce the worst paper drop (e.g. 45% vs 70%), but only because less money is invested.
* Coin choice matters far more than timing. DCA from the Nov 2021 peak: SOL +131%, XRP +123%, BTC +97%,
  BNB +86%, LINK +31%, ETH +21%, LTC −6%, DOGE −8%, ADA −37%, AVAX −40%, DOT −66%.
* Note: POL only has spot history since 2024-09 (the MATIC era is not included).

## 9. Gold: $500/month on the 1st vs $20 on every dip, per year from 2020 (added 2026-09-26)

`python -m research.gold_dip_dca`. Real COMEX gold daily closes (Yahoo GC=F; within a 0.6% median of Binance
PAXG). 0.1% buy fee. Each year starts from zero with $500 added monthly. Unspent cash earns 0%. Full table:
`reports/gold_dip_vs_monthly.csv`.

| Year | Gold price | Monthly on the 1st | $20 every down day | $20 on ≥1% drops | $20 when ≥3% below 30d high |
|---|---|---|---|---|---|
| 2020 | +24.3% | **$6,522** (+8.7%) | $6,172 (35% invested) | $6,068 (13%) | $6,080 (34%) |
| 2021 | −6.1% | **$6,065** (+1.1%) | $6,050 | $6,015 | **$6,075** |
| 2022 | +1.4% | **$6,081** | $6,044 | $6,012 | $6,064 |
| 2023 | +12.2% | **$6,371** | $6,155 | $6,030 | $6,134 |
| 2024 | +27.4% | **$6,700** | $6,253 | $6,063 | $6,175 |
| 2025 | +62.6% | **$7,812** (+30%) | $6,571 | $6,194 | $6,404 |
| 2026 (to Sep) | −0.2% | $4,308 (−4.3%) | $4,426 | **$4,458** | $4,386 |
| **2020 → Sep 2026** | +183.5% | **$79,929** (+97%) | $55,497 (+37%) | $44,721 (+10%) | $54,634 (+35%) |

(Each year's contribution is $6,000; 2026 is $4,500. The total contributed is $40,500.)

* Monthly buying won 5 of 7 years and won the cumulative result by about $25k. $20 per dip only ever deployed
  12–40% of the budget. The rest sat in cash while gold rose 183%.
* Dip days did not buy cheaper gold. The average cost per ounce was $2,190 for monthly buying, vs $2,179,
  $2,308 and $2,233 for the three dip rules.
* The hybrid (buy $20 on dips, then invest the leftover at month end) ended at $79.5k, the same as plain
  monthly buying.
* Dip-buying only "won" in the two flat or down years (2021, 2026), because it held more cash.

## 10. Silver: $500/month on the 1st vs $20 on every dip, per year from 2020 (added 2026-09-26)

`python -m research.gold_dip_dca silver`. Real COMEX silver daily closes (Yahoo SI=F; within a 0.7% median
of Binance XAGUSDT where both exist). Same rules as section 9. Full table: `reports/silver_dip_vs_monthly.csv`.

| Year | Silver price | Monthly on the 1st | $20 every down day | $20 on ≥1% drops | $20 when ≥3% below 30d high |
|---|---|---|---|---|---|
| 2020 | +46.6% | **$8,162** (+36%) | $6,815 | $6,499 | $7,202 |
| 2021 | −14.5% | $5,485 (−9%) | $5,862 | **$5,931** | $5,821 |
| 2022 | +4.7% | **$6,661** (+11%) | $6,295 | $6,175 | $6,494 |
| 2023 | −0.9% | $6,102 | $6,082 | $6,029 | **$6,133** |
| 2024 | +21.9% | **$6,355** | $6,111 | $6,033 | $6,140 |
| 2025 | +136.8% | **$11,333** (+89%) | $7,740 | $6,883 | $7,986 |
| 2026 (to Sep) | −8.9% | $4,097 (−9%) | $4,349 | **$4,388** | $4,238 |
| **2020 → Sep 2026** | +257.6% | **$97,972** (+142%) | $63,562 (+57%) | $52,746 (+30%) | $73,520 (+82%) |

* Monthly buying won 4 of 7 years and the cumulative result by $24k–45k. Dip plans deployed only 17–66% of
  the budget.
* Average cost per ounce was about the same: $26.56 monthly vs $26.02–26.76 for the dip rules. Hybrids ended
  at about $97.3k, the same as monthly buying.
* Dip-buying won only the down years (2021, 2023, 2026), by holding more cash. Silver was about twice as
  volatile as gold, with bigger up years (+137% in 2025) and bigger down years (−14.5% in 2021).

## 11. BTC and ETH: $500/month on the 1st vs $20 on every dip, per year from 2020 (added 2026-09-26)

`python -m research.gold_dip_dca btc|eth`. Binance spot daily closes, same rules as sections 9–10. Tables:
`reports/btc_dip_vs_monthly.csv`, `reports/eth_dip_vs_monthly.csv`.

**BTC**

| Year | BTC price | Monthly on the 1st | $20 every down day | $20 on ≥1% drops | $20 when ≥3% below 30d high |
|---|---|---|---|---|---|
| 2020 | +301.7% | **$17,648** | $12,177 | $9,823 | $15,187 |
| 2021 | +57.6% | **$6,366** | $6,096 | $6,141 | $6,232 |
| 2022 | −65.3% | $3,755 (−37%) | $4,738 | **$5,144** | $4,072 |
| 2023 | +154.5% | **$9,454** | $7,931 | $6,827 | $8,341 |
| 2024 | +111.8% | **$9,250** | $7,710 | $6,957 | $8,459 |
| 2025 | −7.3% | $5,264 | $5,575 | **$5,787** | $5,507 |
| 2026 (to Sep) | −5.3% | **$5,310** | $4,998 | $4,792 | $5,234 |
| **2020 → Sep 2026** | +1,068% | **$119,470** (+195%) | $85,820 | $67,097 | $105,111 |

**ETH**

| Year | ETH price | Monthly on the 1st | $20 every down day | $20 on ≥1% drops | $20 when ≥3% below 30d high |
|---|---|---|---|---|---|
| 2020 | +463.1% | **$18,348** | $12,740 | $10,737 | $14,930 |
| 2021 | +404.3% | **$10,729** | $7,728 | $7,485 | $8,678 |
| 2022 | −68.2% | $3,896 (−35%) | $4,908 | **$5,118** | $4,245 |
| 2023 | +90.1% | **$7,910** | $7,049 | $6,516 | $7,503 |
| 2024 | +41.9% | **$6,878** | $6,471 | $6,280 | $6,868 |
| 2025 | −11.6% | **$6,400** | $6,297 | $6,191 | $6,276 |
| 2026 (to Sep) | −10.4% | **$5,737** | $5,232 | $4,983 | $5,530 |
| **2020 → Sep 2026** | +1,958% | **$110,095** (+172%) | $78,673 | $67,237 | $92,321 |

* Monthly buying won 5 of 7 years for BTC and 6 of 7 for ETH, and won the cumulative result by $14k–52k.
  Dip-buying won only the crash years (2022, and BTC 2025).
* Dip days did not buy cheaper coins. BTC average cost was $28,510 monthly vs $28,478–29,272 for the dip rules;
  ETH was $990 monthly vs $1,019–1,094. Hybrids (dips + leftover at month end) ended at $116–118k (BTC) and
  $104–105k (ETH), slightly below monthly buying.
* Risk: even plain monthly buying fell 70% (BTC) and 78% (ETH) from its peak in 2022, vs 50% for silver and
  23% for gold.

## 12. Spot: buy the month's dip, sell the month's peak ($500 start + $500/month) (added 2026-09-28)

`python -m research.monthly_swing`. Binance spot daily OHLC, Jan 2020 → Sep 2026 (81 months, $40,500 in),
0.1% fee per fill.

| Plan | BTC | ETH |
|---|---|---|
| Perfect hindsight: buy each month's exact low, sell the highest high after it | $39.7 **billion** | $5.0 **trillion** |
| **Buy on the 1st and hold** | **$120,971** | **$111,941** |
| Best realistic limit-order version (dip buy, sell target, unsold coins kept) | $87,668 (−3% / +10%) | $91,710 (−10% / +10%) |
| Realistic versions that are always flat at month end | $34,282 – $55,644 | $20,800 – $38,430 |

* The hindsight figure is impossible: the month's low and high are only known after the month ends. It shows
  why the idea feels so attractive.
* All 24 realistic variants per coin trailed buy-and-hold. The 12 "sold at month end" variants mostly ended
  **below the $40,500 put in**, despite 52–82% win rates: the few months that crashed after the buy (2022)
  cost more than the many small wins earned.
* "Kept until target" shows 100% win rates only because losing positions are never closed. It still ends
  $29k–63k behind buy-and-hold, because selling at +3–20% misses the large rallies.

## 13. Intraday trend-catching ("day scalping with the trend") (added 2026-09-28)

`python -m research.intraday_trend`. 3 years of real 15m data, 12 pairs. Every trade is closed by the end of
the UTC day, with taker fees and slippage included. The strategies are opening-range breakouts at the Asia
(00:00), London (08:00) and US (13:30 UTC) opens, with and without a daily-trend filter; a 20-bar Donchian
breakout with the 4h trend; and a VWAP cross with the daily trend. Each was run with 4 exits: day-end,
2 ATR trail, 3 ATR trail, and 2R target. That is 32 variants and 7,700–38,000 trades each. Full grid:
`reports/intraday_trend_grid.csv`.

* **0 of 32 variants were profitable in either the dev or the holdout period.** The average was −0.06 to
  −0.28R per trade, clearly significant (t between −2 and −27). Win rates were 20–41%.
* The trend effect itself is real but tiny. Before costs, the best variants (trend-filtered opening-range
  breakouts, Donchian with the 4h trend) earned **+0.04 to +0.05R** per trade. Round-trip costs were
  **0.11–0.26R**, because intraday stops are tight, so the fee is a big fraction of the risk.
* Best-looking variant: US-open breakout with the daily trend and a 2R target. It won 41% of trades at
  −0.045R in the holdout, still a loss.
* $500 at 1% risk per trade fell to ~$0–200 over 3 years for every variant, because 7–35 trades a day compound
  a small negative edge very quickly.
