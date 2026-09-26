"""Live funding-carry monitor: long spot + short the perpetual, collecting funding.

This was the only high-win-rate approach that held up on real data (reports/research.md):
funding on these perps is positive most of the time, so the hedged position earns it with
little price exposure. It is not a directional signal and needs a spot leg as well.
"""
from __future__ import annotations

import time

import pandas as pd

from .client import BinanceFutures
from .config import PAIRS

ROUND_TRIP_COST = 2 * (0.001 + 0.0005 + 0.0002)   # spot + perp taker fees and slippage, in and out
# Pairs whose funding was positive in >= 89% of months over Oct 2023 - Sep 2026 (reports/research.md).
HISTORICALLY_STEADY = {"BTCUSDT", "ETHUSDT", "LTCUSDT", "DOGEUSDT", "LINKUSDT", "ADAUSDT"}


def carry_table(pairs=None) -> pd.DataFrame:
    c = BinanceFutures()
    now = int(time.time() * 1000)
    rows = []
    for p in PAIRS:
        if pairs and p["pair"] not in pairs and p["symbol"] not in pairs:
            continue
        f = c.funding_history(p["symbol"], now - 90 * 86_400_000).set_index("time")["rate"]
        prem = c.premium_index(p["symbol"])
        last30 = f[f.index >= f.index[-1] - pd.Timedelta(days=30)]
        rows.append({
            "pair": p["pair"], "symbol": p["symbol"],
            "current_%_8h": float(prem["lastFundingRate"]) * 100,
            "apr_30d_%": last30.sum() * 365 / 30 * 100, "apr_90d_%": f.sum() * 365 / 90 * 100,
            "positive_settlements_90d": (f > 0).mean(),
            "breakeven_days_90d": ROUND_TRIP_COST / (f.sum() / 90) if f.sum() > 0 else float("inf"),
            "steady": p["symbol"] in HISTORICALLY_STEADY,
        })
    return pd.DataFrame(rows).sort_values("apr_30d_%", ascending=False)


def carry_markdown(t: pd.DataFrame) -> str:
    L = [f"# Funding carry monitor — {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC", "",
         "Position: buy the coin on spot and short the same quantity of the USDT-M perpetual (1-2x, isolated, "
         "with a margin buffer). You collect funding while it is positive. Costs ~0.34% round trip.", "",
         "| Pair | Funding now (/8h) | APR 30d | APR 90d | Positive settlements 90d | Days to cover costs | Historically steady |",
         "|---|---|---|---|---|---|---|"]
    for r in t.itertuples():
        L.append(f"| {r.pair} | {r._3:+.4f}% | {r._4:+.1f}% | {r._5:+.1f}% | {r.positive_settlements_90d:.0%} | "
                 f"{r.breakeven_days_90d:.0f} | {'yes' if r.steady else 'no'} |")
    L += ["", "Only consider pairs that are historically steady AND show positive 30d and 90d APR. Exit if funding "
          "turns persistently negative. Risks: funding compression (yields fell to ~2.5–4.7% APR in the last "
          "12 months), short-leg liquidation on violent rallies without enough margin, exchange/custody risk.", "",
          "_Not financial advice._", ""]
    return "\n".join(L)
