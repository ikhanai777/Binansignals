"""Gold: $500/month invested on the 1st vs "$20 on every dip", per calendar year from 2020.

Prices: COMEX gold futures daily closes (Yahoo Finance GC=F), cross-checked against Binance PAXGUSDT spot
(gold-backed token) where both exist. A 0.1% buy fee (Binance spot) is applied to both plans.

Plans (each year starts from zero; $500 is added on the first trading day of every month):
  monthly  all $500 buys gold on the first trading day of the month
  dip $20  every dip day buys $20 of gold while cash lasts; unspent cash stays in cash (0% interest)
Dip definitions: any down day (close below the previous close), a down day of at least 1%, and a close at
least 3% below the 30-day high.

Run: python -m research.gold_dip_dca
"""
from __future__ import annotations

import pandas as pd
import requests

FEE = 0.001
MONTHLY = 500.0
DIP_BUY = 20.0
DIPS = {"any down day": "down", "down >= 1%": "down1", ">= 3% below 30d high": "dd3"}


def gold_daily() -> pd.Series:
    r = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/GC=F", timeout=30,
                     headers={"User-Agent": "Mozilla/5.0"},
                     params={"period1": 1577836800, "period2": int(pd.Timestamp.now().timestamp()), "interval": "1d"}).json()
    res = r["chart"]["result"][0]
    s = pd.Series(res["indicators"]["quote"][0]["close"], index=pd.to_datetime(res["timestamp"], unit="s")).dropna()
    s.index = s.index.normalize()
    return s[~s.index.duplicated(keep="last")]


def paxg_daily() -> pd.Series:
    r = requests.get("https://data-api.binance.vision/api/v3/klines", timeout=30,
                     params={"symbol": "PAXGUSDT", "interval": "1d", "startTime": 1598486400000, "limit": 1000}).json()
    rows = list(r)
    while len(r) == 1000:
        r = requests.get("https://data-api.binance.vision/api/v3/klines", timeout=30,
                         params={"symbol": "PAXGUSDT", "interval": "1d", "startTime": rows[-1][0] + 86_400_000,
                                 "limit": 1000}).json()
        rows += r
    return pd.Series([float(x[4]) for x in rows], index=pd.to_datetime([x[0] for x in rows], unit="ms"))


def is_dip(px: pd.Series, kind: str) -> pd.Series:
    if kind == "down":
        return px.diff() < 0
    if kind == "down1":
        return px.pct_change() <= -0.01
    return px <= 0.97 * px.rolling(30, min_periods=1).max()


def run_period(px: pd.Series, start, end, plan: str, dip_kind: str | None = None) -> dict:
    p = px[(px.index >= start) & (px.index < end)]
    dips = is_dip(px, dip_kind).reindex(p.index) if dip_kind else None
    cash = units = contributed = 0.0
    month = None
    for t, price in p.items():
        if (t.year, t.month) != month:
            month = (t.year, t.month)
            cash += MONTHLY
            contributed += MONTHLY
            if plan == "monthly":
                units += cash * (1 - FEE) / price
                cash = 0.0
        if plan == "dip" and dips[t] and cash >= DIP_BUY:
            units += DIP_BUY * (1 - FEE) / price
            cash -= DIP_BUY
    last = p.iloc[-1]
    gold_value = units * last
    return {"contributed": contributed, "gold_value": gold_value, "cash_left": cash,
            "total": gold_value + cash, "profit": gold_value + cash - contributed,
            "return": (gold_value + cash) / contributed - 1, "invested_share": 1 - cash / contributed}


def table(px: pd.Series) -> pd.DataFrame:
    rows = []
    periods = [(str(y), pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")) for y in range(2020, px.index[-1].year + 1)]
    periods.append((f"2020 -> {px.index[-1]:%Y-%m-%d} (cumulative)", pd.Timestamp("2020-01-01"), px.index[-1] + pd.Timedelta(days=1)))
    for name, a, b in periods:
        yr = px[(px.index >= a) & (px.index < b)]
        row = {"period": name, "gold price change": f"{yr.iloc[-1] / yr.iloc[0] - 1:+.1%}"}
        m = run_period(px, a, b, "monthly")
        row["contributed"] = m["contributed"]
        row["monthly: end value"] = m["total"]
        row["monthly: return"] = m["return"]
        for label, kind in DIPS.items():
            d = run_period(px, a, b, "dip", kind)
            row[f"$20/dip ({label}): end value"] = d["total"]
            row[f"$20/dip ({label}): % invested"] = d["invested_share"]
            row[f"$20/dip ({label}): return"] = d["return"]
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    gold = gold_daily()
    paxg = paxg_daily()
    both = pd.concat([gold, paxg.reindex(gold.index)], axis=1).dropna()
    print(f"Gold data {gold.index[0].date()} -> {gold.index[-1].date()} ({len(gold)} days); "
          f"median |COMEX - PAXG| gap {((both.iloc[:, 0] / both.iloc[:, 1] - 1).abs().median()):.2%} over {len(both)} overlapping days")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    t = table(gold)
    t.to_csv("reports/gold_dip_vs_monthly.csv", index=False)
    print(t.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
