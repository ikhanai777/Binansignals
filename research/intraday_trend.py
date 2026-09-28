"""Intraday trend-catching ("day scalping with the trend") on real 15m Binance futures data, 12 pairs, ~3 years.

Every position is closed by the end of the UTC day (23:45 bar close). Entries are stop/market orders at the
next bar's open (taker 0.05% + 0.02% slippage); exits are stops/trailing stops/day-end (taker + slippage).

Strategies:
  orb_<session>      opening-range breakout: the range of the first hour after the session open
                     (Asia 00:00, London 08:00, US 13:30 UTC); first close outside the range that day enters,
                     stop at the other side of the range; only the first breakout per day
  orb_<session>_tf   same, only in the direction of the daily trend (yesterday's EMA20 vs EMA50)
  donchian_tf        15m close breaks the 20-bar high/low in the direction of the 4h trend (EMA50 slope),
                     stop 1.5 ATR
  vwap_tf            close crosses the daily VWAP in the direction of the daily trend, stop 1 ATR beyond VWAP
Exits (each strategy is tested with each):
  eod                hold until the stop or the day's last bar
  trail2 / trail3    chandelier trailing stop 2 / 3 ATR(15m) behind the best price, else day end
  tp2r               take profit at 2R (limit, maker), else stop / day end
Sizing: 1% of equity risked per trade (stop distance), up to 12 pairs at once. DEV = before 2025-10-01,
HOLDOUT = after (untouched when the rules were written).

Run: python -m research.intraday_trend
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from numba import njit

from binansignals.client import BinanceFutures, drop_unclosed
from binansignals.config import MAKER_FEE, PAIRS, SLIPPAGE, TAKER_FEE
from binansignals.indicators import atr, ema, resample_ohlcv

from .lab import DEV_END

TAKER = TAKER_FEE + SLIPPAGE
SESSIONS = {"asia": (0, 0), "london": (8, 0), "us": (13, 30)}
EXITS = {"eod": (0, 0.0), "trail2": (1, 2.0), "trail3": (1, 3.0), "tp2r": (2, 2.0)}


def features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c = df["close"]
    f["atr"] = atr(df, 14)
    t = df["open_time"]
    f["day"] = t.dt.floor("1D")
    f["mins"] = t.dt.hour * 60 + t.dt.minute
    f["last_bar"] = (f["day"] != f["day"].shift(-1)).to_numpy()
    d1 = resample_ohlcv(df, "1D")
    d1["trend"] = np.sign(ema(d1["close"], 20) - ema(d1["close"], 50)).shift(1)   # known at today's open
    f["dtrend"] = f["day"].map(d1.set_index("open_time")["trend"]).fillna(0)
    h4 = resample_ohlcv(df, "4h")
    e50 = ema(h4["close"], 50)
    h4s = pd.DataFrame({"close_time": h4["close_time"], "t4": np.sign(e50 - e50.shift(3))})
    m = pd.merge_asof(df[["close_time"]], h4s, on="close_time", direction="backward")
    f["t4"] = m["t4"].fillna(0).to_numpy()
    tp = (df["high"] + df["low"] + c) / 3
    f["vwap"] = (tp * df["volume"]).groupby(f["day"]).cumsum() / df["volume"].groupby(f["day"]).cumsum()
    f["hi20"] = df["high"].rolling(20).max().shift(1)
    f["lo20"] = df["low"].rolling(20).min().shift(1)
    return f


def signals(df, f, name):
    c = df["close"]
    n = len(df)
    d = np.zeros(n, np.int64)
    stop = np.full(n, np.nan)
    if name.startswith("orb"):
        sess = name.split("_")[1]
        h, m = SESSIONS[sess]
        start = h * 60 + m
        in_or = (f["mins"] >= start) & (f["mins"] < start + 60)
        grp = f["day"]
        orh = df["high"].where(in_or).groupby(grp).transform("max")
        orl = df["low"].where(in_or).groupby(grp).transform("min")
        after = f["mins"] >= start + 60
        up = after & (c > orh)
        dn = after & (c < orl)
        if name.endswith("_tf"):
            up &= f["dtrend"] > 0
            dn &= f["dtrend"] < 0
        first = (up | dn) & ~((up | dn).groupby(grp).cumsum().shift(fill_value=0).gt(0) & (grp == grp.shift()))
        d = np.where(first & up, 1, np.where(first & dn, -1, 0))
        stop = np.where(d == 1, orl, np.where(d == -1, orh, np.nan))
    elif name == "donchian_tf":
        up = (c > f["hi20"]) & (f["t4"] > 0)
        dn = (c < f["lo20"]) & (f["t4"] < 0)
        d = np.where(up, 1, np.where(dn, -1, 0))
        stop = np.where(d == 1, c - 1.5 * f["atr"], np.where(d == -1, c + 1.5 * f["atr"], np.nan))
    elif name == "vwap_tf":
        above = c > f["vwap"]
        up = above & ~above.shift(fill_value=False) & (f["dtrend"] > 0)
        dn = ~above & above.shift(fill_value=False) & (f["dtrend"] < 0)
        d = np.where(up, 1, np.where(dn, -1, 0))
        stop = np.where(d == 1, f["vwap"] - f["atr"], np.where(d == -1, f["vwap"] + f["atr"], np.nan))
    # no new entries in the last hour of the UTC day
    late = f["mins"].to_numpy() >= 23 * 60
    d = np.where(late, 0, d)
    return d.astype(np.int64), np.asarray(stop, float)


@njit(cache=True)
def _sim(o, h, l, c, a, last_bar, sig, stp, mode, k, maker, taker):
    n = len(o)
    out_r = np.empty(n)
    out_ret = np.empty(n)
    out_i = np.empty(n, np.int64)
    m, busy = 0, -1
    for i in range(n - 1):
        d = sig[i]
        if d == 0 or i <= busy or last_bar[i] or not np.isfinite(stp[i]) or not np.isfinite(a[i]):
            continue
        e = i + 1
        en = o[e]
        stop = stp[i]
        risk = (en - stop) * d
        if risk <= 0.2 * a[i]:
            continue
        tgt = en + d * k * risk
        best = en
        px, fee, j = 0.0, taker, e
        while True:
            if (l[j] <= stop) if d == 1 else (h[j] >= stop):
                gap = (d == 1 and o[j] < stop) or (d == -1 and o[j] > stop)
                px, fee = (o[j] if gap else stop), taker
                break
            if mode == 2 and j > e and ((h[j] >= tgt) if d == 1 else (l[j] <= tgt)):
                px, fee = tgt, maker
                break
            if last_bar[j] or j == n - 1:
                px, fee = c[j], taker
                break
            if mode == 1:
                best = max(best, h[j]) if d == 1 else min(best, l[j])
                trail = best - d * k * a[j]
                stop = max(stop, trail) if d == 1 else min(stop, trail)
            j += 1
        ret = (px - en) * d / en - taker - fee
        out_ret[m] = ret
        out_r[m] = ret * en / risk
        out_i[m] = e
        m += 1
        busy = j
    return out_r[:m], out_ret[:m], out_i[:m]


def portfolio(tr: pd.DataFrame, risk: float, start=500.0) -> tuple[float, float]:
    """Compound daily: each trade risks `risk` of the equity at the start of its day."""
    day = tr.groupby(tr["time"].dt.floor("1D"))["r"].sum() * risk
    eq = start * np.cumprod(1 + day.clip(lower=-0.99).to_numpy())
    peak = np.maximum.accumulate(eq)
    return float(eq[-1]), float(((peak - eq) / peak).max())


def run():
    c = BinanceFutures()
    data = {p["symbol"]: drop_unclosed(c.cached_history(p["symbol"], "15m", 1095)) for p in PAIRS}
    feats = {s: features(df) for s, df in data.items()}
    strategies = [f"orb_{s}" for s in SESSIONS] + [f"orb_{s}_tf" for s in SESSIONS] + ["donchian_tf", "vwap_tf"]
    rows, keep = [], {}
    for strat, (ename, (mode, k)) in itertools.product(strategies, EXITS.items()):
        parts = []
        for s, df in data.items():
            f = feats[s]
            sig, stp = signals(df, f, strat)
            r, ret, idx = _sim(df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(),
                               df["close"].to_numpy(), f["atr"].to_numpy(), f["last_bar"].to_numpy(),
                               sig, stp, mode, k, MAKER_FEE, TAKER)
            parts.append(pd.DataFrame({"sym": s, "r": r, "ret": ret,
                                       "time": pd.to_datetime(df["open_time"].to_numpy()[idx], utc=True)}))
        tr = pd.concat(parts, ignore_index=True)
        keep[(strat, ename)] = tr
        row = {"strategy": strat, "exit": ename}
        for part, mask in (("dev", tr["time"] < DEV_END), ("holdout", tr["time"] >= DEV_END)):
            x = tr.loc[mask, "r"]
            loss = -x[x < 0].sum()
            row |= {f"{part}_n": len(x), f"{part}_win": (x > 0).mean(), f"{part}_avgR": x.mean(),
                    f"{part}_pf": x[x > 0].sum() / loss if loss else np.nan,
                    f"{part}_t": x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 else np.nan}
        row["$500@1%"], row["maxDD@1%"] = portfolio(tr, 0.01)
        rows.append(row)
    return pd.DataFrame(rows), keep


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    res, keep = run()
    res.to_csv("reports/intraday_trend_grid.csv", index=False)
    print(res.round(3).to_string(index=False))
    both = res[(res.dev_avgR > 0) & (res.holdout_avgR > 0)]
    print(f"\nVariants positive in BOTH dev and holdout: {len(both)} of {len(res)}")
    if len(both):
        print(both[["strategy", "exit", "dev_n", "dev_avgR", "dev_t", "holdout_n", "holdout_avgR", "holdout_t", "$500@1%"]].round(3).to_string(index=False))
