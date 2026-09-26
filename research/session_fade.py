"""Session "buy every dip / sell every rise" (session fade) backtest on real 15m Binance futures data.

Rules, per trading session (UTC):
* At the session start, record the session open price O and the previous day's ATR (daily, causal).
* Resting limit orders: BUY at O - k*ATR and SELL SHORT at O + k*ATR (the "dip" and the "rise").
  A fill needs price to trade through the level by 0.05% (conservative queue model); maker fee.
* Take-profit: back to the session open O (limit, maker fee).
* Stop-loss: s*ATR beyond the entry (stop-market, taker fee + slippage), or no stop ("session end only").
* Everything still open at the session end is closed at the close (taker + slippage).
* After an exit, the same level can fill again later in the session ("every dip").
* On the entry bar the stop is checked (worst case) but the target is not.
Optional trend filter: only buy dips when the daily trend is up, only sell rises when it is down.

Run: python -m research.session_fade
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

SESSIONS = {"Asia 00-08": (0, 8), "London 07-15": (7, 15), "New York 13-21": (13, 21), "Full day 00-24": (0, 24)}
KS = [0.25, 0.5, 0.75, 1.0]          # dip/rise size, in daily ATRs from the session open
STOPS = [0.25, 0.5, 1.0, 0.0]        # stop distance in daily ATRs; 0 = no stop (session end only)
PEN = 0.0005
TAKER = TAKER_FEE + SLIPPAGE


@njit(cache=True)
def _simulate(o, h, l, c, sess_id, sess_open, datr, allow_long, allow_short, k, s, pen, maker, taker):
    n = len(o)
    out_ret = np.empty(n)
    out_i = np.empty(n, np.int64)
    out_dir = np.empty(n, np.int64)
    m = 0
    pos, entry, stop, tgt = 0, 0.0, 0.0, 0.0
    for i in range(n):
        sid = sess_id[i]
        if sid < 0 or not np.isfinite(datr[i]):
            continue
        O, A = sess_open[i], datr[i]
        last_bar = i == n - 1 or sess_id[i + 1] != sid
        if pos == 0:
            lo_lvl, hi_lvl = O - k * A, O + k * A
            if allow_long[i] and l[i] <= lo_lvl * (1 - pen):
                pos, entry = 1, min(lo_lvl, o[i])
                stop, tgt = entry - s * A, O
            elif allow_short[i] and h[i] >= hi_lvl * (1 + pen):
                pos, entry = -1, max(hi_lvl, o[i])
                stop, tgt = entry + s * A, O
            if pos != 0:
                # worst case on the fill bar: the stop may also be hit after the fill
                if s > 0 and ((pos == 1 and l[i] <= stop) or (pos == -1 and h[i] >= stop)):
                    out_ret[m] = (stop - entry) * pos / entry - maker - taker
                    out_i[m], out_dir[m] = i, pos
                    m += 1
                    pos = 0
                elif last_bar:
                    out_ret[m] = (c[i] - entry) * pos / entry - maker - taker
                    out_i[m], out_dir[m] = i, pos
                    m += 1
                    pos = 0
                continue
        else:
            px, fee, done = 0.0, 0.0, False
            if s > 0 and ((pos == 1 and l[i] <= stop) or (pos == -1 and h[i] >= stop)):
                gap = (pos == 1 and o[i] < stop) or (pos == -1 and o[i] > stop)
                px, fee, done = (o[i] if gap else stop), taker, True
            elif (pos == 1 and h[i] >= tgt) or (pos == -1 and l[i] <= tgt):
                px, fee, done = tgt, maker, True
            elif last_bar:
                px, fee, done = c[i], taker, True
            if done:
                out_ret[m] = (px - entry) * pos / entry - maker - fee
                out_i[m], out_dir[m] = i, pos
                m += 1
                pos = 0
    return out_ret[:m], out_i[:m], out_dir[:m]


def prepare(df: pd.DataFrame, session: tuple[int, int]):
    d1 = resample_ohlcv(df, "1D")
    d1["atr_prev"] = atr(d1, 14).shift(1)          # yesterday's ATR: known at today's open
    e20, e50 = ema(d1["close"], 20), ema(d1["close"], 50)
    d1["trend"] = np.sign(e20 - e50).shift(1)      # yesterday's daily trend
    day = df["open_time"].dt.floor("1D")
    m = d1.set_index("open_time")
    datr = day.map(m["atr_prev"]).to_numpy(float)
    trend = day.map(m["trend"]).fillna(0).to_numpy(float)
    hour = df["open_time"].dt.hour.to_numpy()
    a, b = session
    inside = (hour >= a) & (hour < b)
    sess_key = np.where(inside, (df["open_time"].dt.floor("1D").astype("int64") // 3_600_000_000_000 + a), -1)
    sess_id = np.where(inside, pd.factorize(sess_key)[0], -1)
    opens = pd.Series(df["open"].to_numpy()).groupby(sess_id).transform("first").to_numpy()
    return sess_id.astype(np.int64), opens, datr, trend


def run():
    c = BinanceFutures()
    data = {p["symbol"]: drop_unclosed(c.cached_history(p["symbol"], "15m", 1095)) for p in PAIRS}
    rows = []
    for (sname, sess), filt in itertools.product(SESSIONS.items(), (False, True)):
        prepped = {}
        for sym, df in data.items():
            sid, opens, datr, trend = prepare(df, sess)
            al = trend > 0 if filt else np.ones(len(df), bool)
            ash = trend < 0 if filt else np.ones(len(df), bool)
            prepped[sym] = (df, sid, opens, datr, al, ash)
        for k, s in itertools.product(KS, STOPS):
            rets, times = [], []
            for sym, (df, sid, opens, datr, al, ash) in prepped.items():
                r, idx, _ = _simulate(df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(),
                                      df["close"].to_numpy(), sid, opens, datr, al, ash, k, s, PEN, MAKER_FEE, TAKER)
                rets.append(r)
                times.append(df["open_time"].to_numpy()[idx])
            r, t = np.concatenate(rets), pd.to_datetime(np.concatenate(times), utc=True)
            row = {"session": sname, "trend_filter": filt, "dip_k_atr": k, "stop_atr": s or "none"}
            for part, mask in (("dev", t < DEV_END), ("holdout", t >= DEV_END)):
                x = r[mask]
                loss = -x[x < 0].sum()
                row |= {f"{part}_n": len(x), f"{part}_win": (x > 0).mean() if len(x) else np.nan,
                        f"{part}_avg_bp": x.mean() * 1e4 if len(x) else np.nan,
                        f"{part}_pf": x[x > 0].sum() / loss if loss else np.nan,
                        f"{part}_t": x.mean() / x.std(ddof=1) * np.sqrt(len(x)) if len(x) > 2 else np.nan}
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    res = run()
    pd.set_option("display.width", 250)
    res.to_csv("reports/session_fade_grid.csv", index=False)
    cols = ["session", "trend_filter", "dip_k_atr", "stop_atr", "dev_n", "dev_win", "dev_avg_bp", "dev_pf", "dev_t",
            "holdout_n", "holdout_win", "holdout_avg_bp", "holdout_pf", "holdout_t"]
    print(res[cols].round(3).to_string())
