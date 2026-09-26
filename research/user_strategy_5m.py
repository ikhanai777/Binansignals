"""Backtest of the user-specified 5m EMA/VWAP/volume strategy on one year of real Binance 5m data.

Rules as specified (interpretations of ambiguous points are marked [I] and varied in VARIANTS):
 1. Volume z-score (vs. last 50 bars) >= 1.0 for longs, > 1.2 for shorts
 2. EMA9 crosses EMA21 in the trade direction
    [I] the cross must have happened within the last `cross_window` bars and EMA9 is still on that side
        (a cross and a pullback-to-EMA9 on the very same bar rarely coexist; window=1 = same bar)
 3. |close - EMA50| / EMA50 >= 0.5%
 4. |EMA9 slope over 5 bars| < 0.5%
 5. RSI(14): long 55-75, short 40-60
 6. Candle body >= 50% of the average body of the last 20 bars   [I] candle closes in trade direction
 7. Pullback-bounce: [I] low touched EMA9 within the last 3 bars (high for shorts) and close is back
    beyond EMA9 in the trade direction
 8. VWAP (daily reset): long above, short below
 9. Footprint absorption (optional bonus) [I] within the last 3 bars, a bar with volume z >= 1 whose taker
    delta opposed the trade direction (<= -10% of volume for longs) but closed in the trade half of its range
    -> tested as an optional extra filter
 Shorts additionally: EMA50 declining (over 5 bars). Longs: no EMA50 slope requirement ("relaxed").
Risk: SL 2%, TP 2% fixed, exit at the close after 5 hours (60 bars). Entry at the next bar's open.
One position per pair at a time. Costs: 0.05% taker entry + 0.02% slippage; TP 0.02% maker; stop/timeout
0.05% taker + 0.02% slippage. Stop checked before target when both are inside one bar.

Run: python -m research.user_strategy_5m
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from binansignals.client import BinanceFutures, drop_unclosed
from binansignals.config import PAIRS
from binansignals.indicators import ema, rsi

VARIANTS = {
    "as specified (cross within 6 bars)": dict(cross_window=6, absorption=False),
    "strict (cross on the same bar)": dict(cross_window=1, absorption=False),
    "cross within 12 bars": dict(cross_window=12, absorption=False),
    "with footprint absorption required": dict(cross_window=6, absorption=True),
}
# Requiring longs above / shorts below EMA50 (entries(..., side=True)) gives identical trades: the VWAP,
# EMA9>EMA21 and RSI conditions already imply it.
TP, SL, HOLD = 0.02, 0.02, 60


def features(df):
    c, o, v = df["close"], df["open"], df["volume"]
    f = pd.DataFrame(index=df.index)
    f["e9"], f["e21"], f["e50"] = ema(c, 9), ema(c, 21), ema(c, 50)
    f["vz"] = (v - v.rolling(50).mean()) / v.rolling(50).std()
    f["rsi"] = rsi(c, 14)
    body = (c - o).abs()
    f["body_ok"] = body >= 0.5 * body.rolling(20).mean().shift()
    f["d50"] = (c - f["e50"]).abs() / f["e50"]
    f["slope9"] = (f["e9"] / f["e9"].shift(5) - 1).abs()
    f["e50_down"] = f["e50"] < f["e50"].shift(5)
    day = df["open_time"].dt.floor("1D")
    tp = (df["high"] + df["low"] + c) / 3
    f["vwap"] = (tp * v).groupby(day).cumsum() / v.groupby(day).cumsum()
    above = (f["e9"] > f["e21"]).astype(int)
    f["up_cross"] = above.diff() == 1
    f["dn_cross"] = above.diff() == -1
    f["touch_l"] = (df["low"] <= f["e9"]).rolling(3).max() > 0
    f["touch_s"] = (df["high"] >= f["e9"]).rolling(3).max() > 0
    dr = (2 * df["taker_buy_volume"] - v) / v.replace(0, np.nan)
    clv = (c - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    f["absorb_l"] = ((f["vz"] >= 1) & (dr <= -0.1) & (clv >= 0.5)).rolling(3).max() > 0
    f["absorb_s"] = ((f["vz"] >= 1) & (dr >= 0.1) & (clv <= 0.5)).rolling(3).max() > 0
    return f


def entries(df, f, cross_window, absorption, side=False):
    c, o = df["close"], df["open"]
    recent_up = f["up_cross"].rolling(cross_window).max() > 0
    recent_dn = f["dn_cross"].rolling(cross_window).max() > 0
    common = (f["d50"] >= 0.005) & (f["slope9"] < 0.005) & f["body_ok"]
    L = (common & (f["vz"] >= 1.0) & recent_up & (f["e9"] > f["e21"]) & f["rsi"].between(55, 75)
         & (c > o) & f["touch_l"] & (c > f["e9"]) & (c > f["vwap"]))
    S = (common & (f["vz"] > 1.2) & recent_dn & (f["e9"] < f["e21"]) & f["rsi"].between(40, 60)
         & (c < o) & f["touch_s"] & (c < f["e9"]) & (c < f["vwap"]) & f["e50_down"])
    if side:
        L &= c > f["e50"]
        S &= c < f["e50"]
    if absorption:
        L &= f["absorb_l"]
        S &= f["absorb_s"]
    return np.where(L, 1, np.where(S, -1, 0)).astype(np.int64)


@njit(cache=True)
def _sim(o, h, l, c, sig, tp, sl, hold):
    n = len(o)
    ent = np.empty(n, np.int64)
    ext = np.empty(n, np.int64)
    ret = np.empty(n)
    why = np.empty(n, np.int64)   # 0 tp, 1 sl, 2 timeout
    m, busy = 0, -1
    for i in range(n - 1):
        d = sig[i]
        if d == 0 or i <= busy:
            continue
        e = i + 1
        en = o[e]
        stop, tgt = en * (1 - d * sl), en * (1 + d * tp)
        j = e
        px, w = 0.0, 2
        for j in range(e, min(n, e + hold)):
            if (l[j] <= stop) if d == 1 else (h[j] >= stop):
                gap = (d == 1 and o[j] < stop) or (d == -1 and o[j] > stop)
                px, w = (o[j] if gap else stop), 1
                break
            if (h[j] >= tgt) if d == 1 else (l[j] <= tgt):
                px, w = tgt, 0
                break
        else:
            j = min(n - 1, e + hold - 1)
            px, w = c[j], 2
        gross = (px - en) * d / en
        fees = 0.0007 + (0.0002 if w == 0 else 0.0007)
        ent[m], ext[m], ret[m], why[m] = e, j, gross - fees, w
        m += 1
        busy = j
    return ent[:m], ext[:m], ret[:m], why[:m]


def equity(tr, lev, start=500.0, mmr=0.005):
    """Whole account as margin at `lev`, one position at a time across all pairs. With a 2% stop,
    isolated liquidation (move of 1/lev - 0.5%) comes BEFORE the stop once lev >= ~40x."""
    eq, busy, liq_dist, liqs = start, pd.Timestamp(0, tz="UTC"), 1 / lev - mmr, 0
    for r in tr.sort_values("entry").itertuples():
        if r.entry <= busy:
            continue
        busy = r.exit
        if liq_dist <= SL and r.why == 1:          # stop sits beyond the liquidation price
            pnl, liqs = -1.0, liqs + 1
        else:
            pnl = max(lev * r.ret, -1.0)
        eq *= 1 + pnl
        if eq < 0.01:
            break
    return eq, liqs


def run():
    c = BinanceFutures()
    data = {p["symbol"]: drop_unclosed(c.cached_history(p["symbol"], "5m", 365)) for p in PAIRS}
    feats = {s: features(df) for s, df in data.items()}
    out = {}
    for name, kw in VARIANTS.items():
        parts = []
        for s, df in data.items():
            sig = entries(df, feats[s], **kw)
            e, x, r, w = _sim(df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(),
                              df["close"].to_numpy(), sig, TP, SL, HOLD)
            t = df["open_time"].to_numpy()
            parts.append(pd.DataFrame({"sym": s, "entry": pd.to_datetime(t[e], utc=True),
                                       "exit": pd.to_datetime(t[x], utc=True), "ret": r, "why": w,
                                       "dir": sig[e - 1]}))
        out[name] = pd.concat(parts, ignore_index=True)
    return out


if __name__ == "__main__":
    res = run()
    for name, tr in res.items():
        r = tr["ret"]
        loss = -r[r < 0].sum()
        print(f"\n=== {name} ===")
        print(f"trades {len(tr)} ({(tr.dir == 1).sum()} long / {(tr.dir == -1).sum()} short), "
              f"win rate {(r > 0).mean():.1%}, avg {r.mean() * 100:+.3f}% per trade, "
              f"PF {r[r > 0].sum() / loss:.2f}, t {r.mean() / r.std() * np.sqrt(len(r)):.2f}")
        print("exits:", {k: f"{v:.0%}" for k, v in tr["why"].map({0: "TP", 1: "SL", 2: "5h timeout"}).value_counts(normalize=True).items()},
              "| gross win rate before fees:", f"{((r + 0.0009 + 0.0005 * (tr.why > 0)) > 0).mean():.1%}")
        for side, g in tr.groupby("dir"):
            print(f"  {'LONG ' if side == 1 else 'SHORT'}: {len(g)} trades, win {(g.ret > 0).mean():.1%}, avg {g.ret.mean() * 100:+.3f}%")
        eqs = []
        for lev in (1, 3, 5, 10, 20, 50):
            eq, liqs = equity(tr, lev)
            eqs.append(f"{lev}x ${eq:,.0f}" + (f" ({liqs} liq)" if liqs else ""))
        print("  $500, whole account as margin:", " | ".join(eqs))
        # risk-based: 1% of equity lost at the 2% stop -> 0.5x notional
        eq1 = 500 * np.prod(1 + 0.5 * tr.sort_values("exit")["ret"].to_numpy())
        print(f"  $500, 1% risk per trade (0.5x notional, all signals taken): ${eq1:,.0f}")
        tr["m"] = tr.entry.dt.tz_localize(None).dt.to_period("M")
        mm = tr.groupby("m").ret.agg(["size", "mean"])
        print(f"  months with positive average trade: {(mm['mean'] > 0).sum()}/{len(mm)}")
