"""5-minute scalping strategies with high leverage, on one year of real Binance USD-M 5m data.

Setups (signal on a closed 5m bar, market entry at the next open):
  ema_cross   EMA9 crosses EMA21 in the direction of the 1h EMA50 trend
  vwap_rev    close stretched > 1.5 ATR from the daily VWAP -> fade back toward it
  bb_rsi      close outside the 2-sigma Bollinger band with RSI(2) extreme -> fade
  breakout    close through the prior 12-bar high/low on 2x volume with same-side taker delta
  delta_mom   3 consecutive bars of same-side taker delta > 20% of volume, close at a 6-bar extreme

Exits: take-profit / stop-loss in 5m-ATR multiples, 2h time stop. Stop checked first on each bar.
Costs: two scenarios. "taker" = market entry 0.05% + stop 0.05% (+0.02% slippage each), TP maker 0.02%.
       "maker" = optimistic: every fill at maker 0.02% with no slippage (a lower bound on costs).
Leverage: the whole account is posted as margin on each trade (notional = L x equity), one position at
a time across all pairs. Isolated liquidation is modelled: if price moves against the position by
1/L - 0.5% (maintenance margin) before the exit, the trade loses 100% of the margin.

Run: python -m research.scalp5m
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from numba import njit

from binansignals.client import BinanceFutures, drop_unclosed
from binansignals.config import PAIRS
from binansignals.indicators import atr, ema, rsi

MMR = 0.005
H = 24  # 2 hours


def signals(df: pd.DataFrame) -> dict[str, np.ndarray]:
    c, v = df["close"], df["volume"]
    a = atr(df, 14)
    e9, e21 = ema(c, 9), ema(c, 21)
    h1 = ema(c, 50 * 12)                      # 1h EMA50 on 5m bars
    trend = np.sign(c - h1)
    day = df["open_time"].dt.floor("1D")
    tp = (df["high"] + df["low"] + c) / 3
    vwap = (tp * v).groupby(day).cumsum() / v.groupby(day).cumsum()
    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    r2 = rsi(c, 2)
    dr = (2 * df["taker_buy_volume"] - v) / v.replace(0, np.nan)
    out = {}
    up_x = (e9 > e21) & (e9.shift() <= e21.shift())
    dn_x = (e9 < e21) & (e9.shift() >= e21.shift())
    out["ema_cross"] = np.where(up_x & (trend > 0), 1, np.where(dn_x & (trend < 0), -1, 0))
    out["vwap_rev"] = np.where(c < vwap - 1.5 * a, 1, np.where(c > vwap + 1.5 * a, -1, 0))
    out["bb_rsi"] = np.where((c < ma - 2 * sd) & (r2 < 5), 1, np.where((c > ma + 2 * sd) & (r2 > 95), -1, 0))
    hi, lo, vs = df["high"].rolling(12).max().shift(), df["low"].rolling(12).min().shift(), v.rolling(48).mean()
    out["breakout"] = np.where((c > hi) & (v > 2 * vs) & (dr > 0.1), 1,
                               np.where((c < lo) & (v > 2 * vs) & (dr < -0.1), -1, 0))
    b3, s3 = (dr > 0.2).rolling(3).sum() == 3, (dr < -0.2).rolling(3).sum() == 3
    out["delta_mom"] = np.where(b3 & (c >= df["high"].rolling(6).max()), 1,
                                np.where(s3 & (c <= df["low"].rolling(6).min()), -1, 0))
    return {k: np.asarray(x, dtype=np.int64) for k, x in out.items()} | {"atr": a.to_numpy()}


@njit(cache=True)
def _trades(o, h, l, c, a, sig, tp, sl, H):
    n = len(o)
    ent = np.empty(n, np.int64)
    ext = np.empty(n, np.int64)
    gross = np.empty(n)
    mae = np.empty(n)
    kind = np.empty(n, np.int64)   # 0 tp (maker exit), 1 stop/time (taker exit)
    m, busy = 0, -1
    for i in range(n - 1):
        d = sig[i]
        if d == 0 or i <= busy or not np.isfinite(a[i]) or a[i] <= 0:
            continue
        e = i + 1
        en = o[e]
        stop, tgt = en - d * sl * a[i], en + d * tp * a[i]
        worst, px, k, j = 0.0, 0.0, 1, e
        for j in range(e, min(n, e + H)):
            adverse = (en - l[j]) / en if d == 1 else (h[j] - en) / en
            if (l[j] <= stop) if d == 1 else (h[j] >= stop):
                gap = (d == 1 and o[j] < stop) or (d == -1 and o[j] > stop)
                px = o[j] if gap else stop
                worst = max(worst, (en - px) * d / en)
                k = 1
                break
            worst = max(worst, adverse)
            if (h[j] >= tgt) if d == 1 else (l[j] <= tgt):
                px, k = tgt, 0
                break
        else:
            j = min(n - 1, e + H - 1)
            px, k = c[j], 1
        ent[m], ext[m], gross[m], mae[m], kind[m] = e, j, (px - en) * d / en, worst, k
        m += 1
        busy = j
    return ent[:m], ext[:m], gross[:m], mae[:m], kind[:m]


def net(gross, kind, scenario):
    if scenario == "taker":
        return gross - 0.0007 - np.where(kind == 0, 0.0002, 0.0007)
    return gross - 0.0004


def equity(tr: pd.DataFrame, lev: float, start=500.0):
    """Whole account as margin, one position at a time (chronological, skip overlaps).
    Returns (final equity, peak equity, days until 90% of the starting money was lost or None, liquidations)."""
    eq, busy_until, peak, liqs = start, pd.Timestamp(0, tz="UTC"), start, 0
    liq = 1 / lev - MMR
    t0, dead = None, None
    for r in tr.sort_values("entry").itertuples():
        if r.entry <= busy_until:
            continue
        t0 = t0 or r.entry
        busy_until = r.exit
        if r.mae >= liq:
            pnl, liqs = -1.0, liqs + 1
        else:
            pnl = max(lev * r.ret, -1.0)
        eq *= 1 + pnl
        peak = max(peak, eq)
        if dead is None and eq <= 0.1 * start:
            dead = (r.exit - t0).days
        if eq < 0.01:
            break
    return eq, peak, dead, liqs


def run():
    c = BinanceFutures()
    data = {p["symbol"]: drop_unclosed(c.cached_history(p["symbol"], "5m", 365)) for p in PAIRS}
    sigs = {s: signals(df) for s, df in data.items()}
    rows, keep = [], {}
    for strat, (tp, sl) in itertools.product(["ema_cross", "vwap_rev", "bb_rsi", "breakout", "delta_mom"],
                                             [(1.0, 1.0), (1.5, 1.0), (2.0, 1.0), (1.0, 1.5), (0.5, 1.5)]):
        parts = []
        for s, df in data.items():
            t = df["open_time"].to_numpy()
            e, x, g, mae, k = _trades(df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(),
                                      df["close"].to_numpy(), sigs[s]["atr"], sigs[s][strat], tp, sl, H)
            parts.append(pd.DataFrame({"entry": pd.to_datetime(t[e], utc=True), "exit": pd.to_datetime(t[x], utc=True),
                                       "gross": g, "mae": mae, "kind": k}))
        tr = pd.concat(parts, ignore_index=True)
        row = {"strategy": strat, "tp_atr": tp, "sl_atr": sl, "trades": len(tr),
               "gross_bp": tr["gross"].mean() * 1e4}
        for sc in ("taker", "maker"):
            r = net(tr["gross"].to_numpy(), tr["kind"].to_numpy(), sc)
            row[f"{sc}_win"] = (r > 0).mean()
            row[f"{sc}_avg_bp"] = r.mean() * 1e4
        tr["ret"] = net(tr["gross"].to_numpy(), tr["kind"].to_numpy(), "taker")
        rows.append(row)
        keep[(strat, tp, sl)] = tr
    return pd.DataFrame(rows), keep


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    res, keep = run()
    print(res.round(3).to_string())
    best = res.sort_values("taker_avg_bp", ascending=False).head(3)
    print("\n$500, whole account as margin, one position at a time, realistic (taker) costs:")
    for b in best.itertuples():
        tr = keep[(b.strategy, b.tp_atr, b.sl_atr)]
        print(f"{b.strategy} tp{b.tp_atr}/sl{b.sl_atr}")
        for lev in (1, 5, 10, 25, 50, 100):
            eq, peak, dead, liqs = equity(tr, lev)
            print(f"   {lev:>3}x: end ${eq:,.2f}, peak ${peak:,.0f}, lost 90% after "
                  f"{'never' if dead is None else f'{dead} days'}, liquidations {liqs}")
