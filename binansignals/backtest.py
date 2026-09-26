"""Bar-by-bar backtester and walk-forward optimiser on real Binance history.

Execution model (deliberately conservative):
* signal on a closed 1h bar -> market entry at the NEXT bar's open (taker fee + slippage)
* protective stop is a stop-market (taker fee + slippage); gaps through the stop fill at the open
* TP1 is a resting limit (maker fee) that closes 50%; the stop then moves to break-even and the
  remaining 50% trails a chandelier stop (highest high / lowest low since entry -/+ k*ATR)
* if a bar touches both the stop and a target, the stop is assumed to fill first
* real historical funding is charged/credited on every 8h settlement the position is open
Results are measured in R (multiples of the initial risk), so fees scale with stop distance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import MAKER_FEE, SLIPPAGE, TAKER_FEE, WF_TEST_DAYS, WF_TRAIN_DAYS
from .strategy import entries, param_key, param_sets


@dataclass
class Arrays:
    t: np.ndarray       # bar open times (int64 ns)
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    atr: np.ndarray
    f_t: np.ndarray     # funding times (int64 ns)
    f_r: np.ndarray     # funding rates


def to_arrays(df: pd.DataFrame, funding: pd.DataFrame) -> Arrays:
    ft = funding["time"].astype("int64").to_numpy() if len(funding) else np.array([], dtype="int64")
    fr = funding["rate"].to_numpy() if len(funding) else np.array([])
    return Arrays(df["open_time"].astype("int64").to_numpy(), df["open"].to_numpy(), df["high"].to_numpy(),
                  df["low"].to_numpy(), df["close"].to_numpy(), df["atr"].to_numpy(), ft, fr)


def simulate_trade(A: Arrays, i: int, d: int, stop: float, p: dict):
    """Simulate one trade signalled on bar i. Returns a dict or None if not fillable."""
    n = len(A.o)
    if i + 1 >= n:
        return None
    e = i + 1
    entry = A.o[e]
    risk = (entry - stop) * d
    if not np.isfinite(risk) or risk <= 0.3 * A.atr[i]:
        return None
    partial = p["tp1_r"] > 0
    # Without a partial, the "target" is only an arming level (+1R) for break-even + trailing.
    tp1 = entry + d * (p["tp1_r"] if partial else 1.0) * risk
    cost_r = (TAKER_FEE + SLIPPAGE) * entry / risk
    r, size, cur_stop, took_tp1 = 0.0, 1.0, stop, False
    extreme = entry
    exit_j, reason = None, None
    partial_t = None
    for j in range(e, n):
        o, h, l = A.o[j], A.h[j], A.l[j]
        # 1) stop (gap-aware)
        hit_stop = (l <= cur_stop) if d == 1 else (h >= cur_stop)
        if hit_stop:
            px = cur_stop
            if j > e and ((d == 1 and o < cur_stop) or (d == -1 and o > cur_stop)):
                px = o
            r += size * (px - entry) * d / risk - size * (TAKER_FEE + SLIPPAGE) * px / risk
            exit_j, reason = j, ("stop" if not took_tp1 else "trail")
            break
        # 2) first target
        if not took_tp1 and ((h >= tp1) if d == 1 else (l <= tp1)):
            if partial:
                r += 0.5 * p["tp1_r"] - 0.5 * MAKER_FEE * tp1 / risk
                size, partial_t = 0.5, A.t[j]
            took_tp1 = True
            cur_stop = entry
        # 3) time stop
        if j - e >= p["max_bars"]:
            px = A.c[j]
            r += size * (px - entry) * d / risk - size * (TAKER_FEE + SLIPPAGE) * px / risk
            exit_j, reason = j, "time"
            break
        # 4) update trailing stop for the next bar (only after TP1)
        extreme = max(extreme, h) if d == 1 else min(extreme, l)
        if took_tp1:
            trail = extreme - d * p["trail_atr"] * A.atr[j]
            cur_stop = max(cur_stop, trail) if d == 1 else min(cur_stop, trail)
    if exit_j is None:  # still open at the end of data -> mark to market
        j = n - 1
        px = A.c[j]
        r += size * (px - entry) * d / risk - size * (TAKER_FEE + SLIPPAGE) * px / risk
        exit_j, reason = j, "open"
    r -= cost_r
    # funding: longs pay positive rates, shorts receive them (and vice versa)
    if len(A.f_t):
        t0, t1 = A.t[e], A.t[exit_j] + (A.t[1] - A.t[0])
        lo, hi = np.searchsorted(A.f_t, t0, "right"), np.searchsorted(A.f_t, t1, "right")
        for k in range(lo, hi):
            w = 0.5 if (partial_t is not None and A.f_t[k] > partial_t) else 1.0
            r -= w * d * A.f_r[k] * entry / risk
    return {"entry_i": e, "exit_i": exit_j, "dir": d, "entry": entry, "stop": stop,
            "risk_pct": risk / entry, "r": r, "reason": reason, "tp1": took_tp1}


def run_params(df: pd.DataFrame, A: Arrays, p: dict) -> pd.DataFrame:
    sig = entries(df, p)
    dirs, stops = sig["dir"].to_numpy(), sig["stop"].to_numpy()
    idx = np.flatnonzero(dirs != 0)
    trades, busy_until = [], -1
    for i in idx:
        if i <= busy_until or not np.isfinite(A.atr[i]):
            continue
        tr = simulate_trade(A, i, int(dirs[i]), float(stops[i]), p)
        if tr is None:
            continue
        busy_until = tr["exit_i"]
        trades.append(tr)
    out = pd.DataFrame(trades)
    if not out.empty:
        out["entry_time"] = pd.to_datetime(A.t[out["entry_i"]], utc=True)
        out["exit_time"] = pd.to_datetime(A.t[out["exit_i"]], utc=True)
    return out


def metrics(tr: pd.DataFrame, risk=0.01) -> dict:
    if tr is None or tr.empty:
        return {"trades": 0, "win_rate": 0.0, "avg_r": 0.0, "total_r": 0.0, "profit_factor": 0.0,
                "max_dd_r": 0.0, "return_pct": 0.0, "max_dd_pct": 0.0, "t_stat": 0.0}
    r = tr["r"].to_numpy()
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    cum = np.cumsum(r)
    dd_r = float((np.maximum.accumulate(np.concatenate([[0], cum])) - np.concatenate([[0], cum])).max())
    eq = np.cumprod(1 + risk * r)
    peak = np.maximum.accumulate(np.concatenate([[1], eq]))
    dd_pct = float(((peak - np.concatenate([[1], eq])) / peak).max())
    return {
        "trades": int(len(r)), "win_rate": float((r > 0).mean()), "avg_r": float(r.mean()),
        "total_r": float(r.sum()), "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
        "max_dd_r": dd_r, "return_pct": float(eq[-1] - 1), "max_dd_pct": dd_pct,
        "t_stat": float(r.mean() / r.std(ddof=1) * np.sqrt(len(r))) if len(r) > 1 and r.std(ddof=1) > 0 else 0.0,
    }


def _score(tr: pd.DataFrame) -> float:
    """Robust selection score: t-stat of mean R (penalises small samples and noisy edges)."""
    if len(tr) < 8:
        return -np.inf
    r = tr["r"].to_numpy()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(len(r))) if sd > 0 else -np.inf


def walk_forward(df: pd.DataFrame, funding: pd.DataFrame, tf: str = "1h",
                 train_days=WF_TRAIN_DAYS, test_days=WF_TEST_DAYS) -> dict:
    """Rolling walk-forward: pick the best rule set on each train window, trade it blind on
    the following test window. Only the concatenated out-of-sample trades are reported."""
    A = to_arrays(df, funding)
    all_trades = {}
    for p in param_sets(tf):
        all_trades[param_key(p)] = (p, run_params(df, A, p))
    start = df["open_time"].iloc[0] + pd.Timedelta(days=30)  # indicator warm-up
    end = df["open_time"].iloc[-1]
    train, test = pd.Timedelta(days=train_days), pd.Timedelta(days=test_days)
    oos, windows = [], []
    t0 = start
    while t0 + train < end:
        tr_end, te_end = t0 + train, min(t0 + train + test, end + pd.Timedelta(hours=1))
        best_key, best_s = None, 0.0  # require a positive in-sample edge, else stay flat
        for key, (p, tr) in all_trades.items():
            if tr.empty:
                continue
            s = _score(tr[(tr["entry_time"] >= t0) & (tr["entry_time"] < tr_end)])
            if s > best_s:
                best_key, best_s = key, s
        w = {"train_start": t0, "test_start": tr_end, "test_end": te_end, "params": best_key, "score": best_s}
        if best_key:
            tr = all_trades[best_key][1]
            sel = tr[(tr["entry_time"] >= tr_end) & (tr["entry_time"] < te_end)].copy()
            sel["params"] = best_key
            oos.append(sel)
            w["oos"] = metrics(sel)
        windows.append(w)
        t0 = t0 + test
    oos_df = pd.concat(oos, ignore_index=True) if oos else pd.DataFrame()
    # The rule set live signals will use: best on the most recent train window.
    live_t0 = end - train
    live_key, live_s = None, 0.0
    for key, (p, tr) in all_trades.items():
        if tr.empty:
            continue
        s = _score(tr[tr["entry_time"] >= live_t0])
        if s > live_s:
            live_key, live_s = key, s
    return {
        "oos_trades": oos_df, "oos_metrics": metrics(oos_df), "windows": windows,
        "live_params": all_trades[live_key][0] if live_key else None, "live_score": live_s,
        "in_sample_all": {k: metrics(v[1]) for k, v in all_trades.items()},
    }
