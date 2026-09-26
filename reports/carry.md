# Funding carry monitor — 2026-09-26 02:43 UTC

Position: buy the coin on spot and short the same quantity of the USDT-M perpetual (1-2x, isolated, with a margin buffer). You collect funding while it is positive. Costs ~0.34% round trip.

| Pair | Funding now (/8h) | APR 30d | APR 90d | Positive settlements 90d | Days to cover costs | Historically steady |
|---|---|---|---|---|---|---|
| DOGEUSDT | +0.0100% | +7.4% | +6.4% | 85% | 19 | yes |
| LTCUSDT | +0.0100% | +7.3% | +5.6% | 86% | 22 | yes |
| BTCUSDT | +0.0014% | +6.5% | +6.6% | 99% | 19 | yes |
| LINKUSDT | +0.0100% | +6.2% | +6.6% | 90% | 19 | yes |
| AVAXUSDT | +0.0100% | +5.8% | +4.2% | 79% | 29 | no |
| ADAUSDT | +0.0100% | +5.6% | +3.9% | 73% | 32 | yes |
| ETHUSDT | +0.0099% | +4.7% | +4.7% | 92% | 26 | yes |
| DOTUSDT | +0.0100% | +4.2% | +3.6% | 71% | 35 | no |
| BNBUSDT | +0.0000% | +4.0% | +5.2% | 72% | 24 | no |
| XRPUSDT | +0.0100% | +4.0% | +3.3% | 73% | 38 | no |
| SOLUSDT | +0.0050% | +2.9% | +3.5% | 71% | 35 | no |
| MATICUSDT | +0.0050% | -1.4% | -1.1% | 54% | inf | no |

Only consider pairs that are historically steady AND show positive 30d and 90d APR. Exit if funding turns persistently negative. Risks: funding compression (yields fell to ~2.5–4.7% APR in the last 12 months), short-leg liquidation on violent rallies without enough margin, exchange/custody risk.

_Not financial advice._
