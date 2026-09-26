"""Spot dip-buying with a monthly budget vs plain monthly DCA, on real Binance spot daily closes.

Plan tested (as asked): $500 is added to cash on the 1st of every month; on every "dip" day 10% of the cash
available is spent buying at that day's close (0.1% spot fee). Unspent cash stays in USDT (0% yield).
Dip definitions:
  red5 / red8    daily close <= -5% / -8% vs the previous close
  dd10 / dd20    close >= 10% / 20% below the 30-day high (at most one buy every 3 days)
Benchmark: DCA - all $500 invested on the 1st of each month.
Baskets split the monthly budget equally among coins trading at that time.

Run: python -m research.dip_dca
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

FEE = 0.001
BUDGET = 500.0
SPOT = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT", "LTC": "LTCUSDT", "DOGE": "DOGEUSDT",
        "DOT": "DOTUSDT", "AVAX": "AVAXUSDT", "BNB": "BNBUSDT", "SOL": "SOLUSDT", "POL": "POLUSDT",
        "LINK": "LINKUSDT", "ADA": "ADAUSDT"}
BASKETS = {"BTC only": ["BTC"], "BTC + ETH": ["BTC", "ETH"], "All 12 coins": list(SPOT)}
RULES = ["dca", "red5", "red8", "dd10", "dd20"]


def daily(symbol: str, start="2020-01-01") -> pd.Series:
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
    s = pd.Series([float(x[4]) for x in rows], index=pd.to_datetime([x[0] for x in rows], unit="ms", utc=True))
    return s.iloc[:-1]  # drop today's unfinished candle


def dip_days(px: pd.Series, rule: str) -> pd.Series:
    if rule.startswith("red"):
        return px.pct_change() <= -int(rule[3:]) / 100
    thr = int(rule[2:]) / 100
    raw = px <= (1 - thr) * px.rolling(30, min_periods=1).max()
    out, last = pd.Series(False, index=px.index), None
    for t in px.index[raw.to_numpy()]:
        if last is None or (t - last).days >= 3:
            out[t], last = True, t
    return out


def simulate(prices: dict[str, pd.Series], rule: str, start: str):
    idx = sorted(set().union(*[p.index for p in prices.values()]))
    idx = [t for t in idx if t >= pd.Timestamp(start, tz="UTC")]
    cash = {c: 0.0 for c in prices}
    units = {c: 0.0 for c in prices}
    dips = {c: dip_days(p, rule) if rule != "dca" else None for c, p in prices.items()}
    contributed, flows, values, month = 0.0, [], [], None
    for t in idx:
        live = [c for c, p in prices.items() if t in p.index]
        if (t.year, t.month) != month:
            month = (t.year, t.month)
            share = BUDGET / len(live)
            for c in live:
                cash[c] += share
            contributed += BUDGET
            flows.append((t, -BUDGET))
            if rule == "dca":
                for c in live:
                    units[c] += cash[c] * (1 - FEE) / prices[c][t]
                    cash[c] = 0.0
        if rule != "dca":
            for c in live:
                if dips[c].get(t, False) and cash[c] > 1:
                    spend = 0.10 * cash[c]
                    units[c] += spend * (1 - FEE) / prices[c][t]
                    cash[c] -= spend
        held = sum(units[c] * prices[c][t] for c in live if t in prices[c].index)
        values.append((t, held + sum(cash.values()), sum(cash.values())))
    v = pd.DataFrame(values, columns=["t", "value", "cash"]).set_index("t")
    final = v["value"].iloc[-1]
    flows.append((v.index[-1], final))
    return {"contributed": contributed, "final": final, "profit_pct": final / contributed - 1,
            "irr": _xirr(flows), "idle_cash_pct": v["cash"].iloc[-1] / final,
            "max_drop_pct": float(((v["value"].cummax() - v["value"]) / v["value"].cummax()).max()),
            "buys": int(sum(d.loc[d.index >= pd.Timestamp(start, tz="UTC")].sum() for d in dips.values()))
            if rule != "dca" else None}


def _xirr(flows):
    t0 = flows[0][0]
    yrs = np.array([(t - t0).days / 365.25 for t, _ in flows])
    cf = np.array([f for _, f in flows])
    lo, hi = -0.99, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        npv = (cf / (1 + mid) ** yrs).sum()
        lo, hi = (mid, hi) if npv > 0 else (lo, mid)
    return mid


if __name__ == "__main__":
    prices = {c: daily(s) for c, s in SPOT.items()}
    pd.set_option("display.width", 200)
    for start in ("2020-01-01", "2021-11-01", "2024-01-01"):
        rows = []
        for bname, coins in BASKETS.items():
            for rule in RULES:
                r = simulate({c: prices[c] for c in coins}, rule, start)
                rows.append({"basket": bname, "rule": rule, **r})
        df = pd.DataFrame(rows)
        print(f"\n=== Start {start} (monthly $500 until {max(p.index[-1] for p in prices.values()).date()}) ===")
        print(df.assign(contributed=df.contributed.map("${:,.0f}".format), final=df.final.map("${:,.0f}".format),
                        profit_pct=df.profit_pct.map("{:+.0%}".format), irr=df.irr.map("{:+.1%}".format),
                        idle_cash_pct=df.idle_cash_pct.map("{:.0%}".format),
                        max_drop_pct=df.max_drop_pct.map("{:.0%}".format)).to_string(index=False))
