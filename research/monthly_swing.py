"""Spot "buy the month's dip, sell the month's peak", $500 start + $500 every month, real Binance daily OHLC.

Versions:
  hindsight   buy at each month's lowest low and sell at the highest high AFTER that low in the same month.
              Not tradable (needs the future); shows the ceiling.
  limit d/u   on the 1st, a limit BUY at open*(1-d); if filled, a limit SELL at buy*(1+u).
              "flat": anything not sold by month end is sold at the month's last close (out every month)
              "hold": unsold coins are kept (and the sell order kept) until the target is hit; the new
              month's cash still follows the monthly buy rule.
  hold (DCA)  benchmark: all cash buys on the 1st and is never sold.
Fills: a limit fills if the day's low/high trades through it, at the limit price. 0.1% fee per fill.
Intraday order unknown: if a day touches both the buy and the sell level, the sell is only allowed from the
next day (conservative).

Run: python -m research.monthly_swing
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import requests

FEE = 0.001


def ohlc(symbol: str, start="2020-01-01") -> pd.DataFrame:
    rows, cur = [], int(pd.Timestamp(start, tz="UTC").value // 1_000_000)
    while True:
        r = requests.get("https://data-api.binance.vision/api/v3/klines", timeout=20,
                         params={"symbol": symbol, "interval": "1d", "startTime": cur, "limit": 1000}).json()
        if not r:
            break
        rows += r
        cur = r[-1][0] + 86_400_000
        if len(r) < 1000:
            break
    df = pd.DataFrame([(x[0], float(x[1]), float(x[2]), float(x[3]), float(x[4])) for x in rows],
                      columns=["t", "open", "high", "low", "close"])
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    return df.iloc[:-1].set_index("t")


def hindsight(df):
    cash, units = 0.0, 0.0
    for _, m in df.groupby(df.index.to_period("M")):
        cash += 500
        i_low = m["low"].to_numpy().argmin()
        buy = m["low"].iloc[i_low]
        after = m["high"].iloc[i_low:]
        sell = after.max()
        cash *= (1 - FEE) * (sell / buy) * (1 - FEE)
    return cash


def dca_hold(df):
    units = 0.0
    for _, m in df.groupby(df.index.to_period("M")):
        units += 500 * (1 - FEE) / m["open"].iloc[0]
    return units * df["close"].iloc[-1]


def limit_swing(df, d, u, mode):
    cash, units, entry, trades, wins = 0.0, 0.0, None, 0, 0
    months = list(df.groupby(df.index.to_period("M")))
    for k, (_, m) in enumerate(months):
        cash += 500
        buy_lvl = m["open"].iloc[0] * (1 - d)
        bought_today = False
        for t, row in m.iterrows():
            bought_today = False
            if cash > 1 and row["low"] <= buy_lvl:          # month's cash buys the dip
                px = min(buy_lvl, row["open"])
                new_units = cash * (1 - FEE) / px
                entry = px if units == 0 else (entry * units + px * new_units) / (units + new_units)
                units += new_units
                cash, bought_today = 0.0, True
            if units > 0 and not bought_today and row["high"] >= entry * (1 + u):
                px = max(entry * (1 + u), row["open"])
                cash += units * px * (1 - FEE)
                trades, wins = trades + 1, wins + 1
                units, entry = 0.0, None
        if mode == "flat" and units > 0:                      # forced exit at month end
            px = m["close"].iloc[-1]
            trades, wins = trades + 1, wins + (px > entry)
            cash += units * px * (1 - FEE)
            units, entry = 0.0, None
    final = cash + units * df["close"].iloc[-1]
    return final, trades, wins


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    for sym in ("BTCUSDT", "ETHUSDT"):
        df = ohlc(sym)
        n_months = df.index.to_period("M").nunique()
        contributed = 500 * n_months
        print(f"\n=== {sym}: {df.index[0].date()} -> {df.index[-1].date()}, {n_months} months, ${contributed:,} contributed ===")
        print(f"  Perfect hindsight (buy month's low, sell highest high after it): ${hindsight(df):,.0f}   <- impossible in practice")
        print(f"  Buy on the 1st and hold (DCA):                                     ${dca_hold(df):,.0f}")
        rows = []
        for d, u, mode in itertools.product([0.03, 0.05, 0.10], [0.03, 0.05, 0.10, 0.20], ["flat", "hold"]):
            f, n, w = limit_swing(df, d, u, mode)
            rows.append({"buy dip": f"-{d:.0%}", "sell at": f"+{u:.0%}", "unsold coins": "sold at month end" if mode == "flat" else "kept until target",
                         "final": f, "trades": n, "win rate": w / n if n else np.nan})
        r = pd.DataFrame(rows).sort_values("final", ascending=False)
        r["final"] = r["final"].map("${:,.0f}".format)
        r["win rate"] = r["win rate"].map("{:.0%}".format)
        print(r.to_string(index=False))
