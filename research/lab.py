"""Research harness used to look for higher-win-rate strategies (see reports/research.md).

Everything runs on real Binance USD-M history pulled through binansignals.client (cached in data/).
Protocol: design on the DEV period only (before DEV_END); the HOLDOUT period is evaluated once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from binansignals.client import BinanceFutures, drop_unclosed
from binansignals.config import MAKER_FEE, PAIRS, SLIPPAGE, TAKER_FEE
from binansignals.indicators import adx, atr, attach_htf, ema, htf_trend, resample_ohlcv, rsi

DEV_END = pd.Timestamp("2025-10-01", tz="UTC")
DAYS = 1095


def load(sym: str, tf: str, c: BinanceFutures | None = None) -> pd.DataFrame:
    return drop_unclosed((c or BinanceFutures()).cached_history(sym, tf, DAYS))


def load_funding(sym: str, c: BinanceFutures | None = None) -> pd.DataFrame:
    start = int((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=DAYS)).value // 1_000_000)
    return (c or BinanceFutures()).funding_history(sym, start)


def feats(df: pd.DataFrame, btc: pd.DataFrame | None = None) -> pd.DataFrame:
    """Causal per-bar features plus 4h and daily trend context."""
    df = df.copy()
    c = df["close"]
    df["atr"] = atr(df, 14)
    df["atr_pct"] = df["atr"] / c
    df["rsi2"], df["rsi14"] = rsi(c, 2), rsi(c, 14)
    df["ema20"], df["ema50"], df["ema200"] = ema(c, 20), ema(c, 50), ema(c, 200)
    df["bb_z"] = (c - c.rolling(20).mean()) / c.rolling(20).std()
    df["delta"] = 2 * df["taker_buy_volume"] - df["volume"]
    df["dr"] = df["delta"] / df["volume"].replace(0, np.nan)
    df["vol_z"] = df["volume"] / df["volume"].rolling(50).median()
    df["clv"] = (c - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    for n in (24, 72):
        df[f"lo{n}"] = df["low"].rolling(n).min().shift(1)
        df[f"hi{n}"] = df["high"].rolling(n).max().shift(1)
    df["adx1h"] = adx(df)["adx"]
    s4 = htf_trend(resample_ohlcv(df, "4h"), 20, 50)  # identity resample when df is already 4h
    df = attach_htf(df, s4.rename(columns={"htf_bias": "b4", "htf_adx": "adx4",
                                           "htf_ema_fast": "e4f", "htf_ema_slow": "e4s"}))
    sd = htf_trend(resample_ohlcv(df, "1D"), 20, 50)
    df = attach_htf(df, sd.rename(columns={"htf_bias": "bd", "htf_adx": "adxd",
                                           "htf_ema_fast": "edf", "htf_ema_slow": "eds"}))
    if btc is not None:
        sb = htf_trend(resample_ohlcv(btc, "1D"), 20, 50)[["close_time", "htf_bias"]]
        df = attach_htf(df, sb.rename(columns={"htf_bias": "btc_bd"}))
    else:
        df["btc_bd"] = df["bd"]
    return df.reset_index(drop=True)


def universe(tf: str) -> dict[str, pd.DataFrame]:
    c = BinanceFutures()
    btc = load("BTCUSDT", tf, c)
    return {p["symbol"]: feats(load(p["symbol"], tf, c), None if p["symbol"] == "BTCUSDT" else btc) for p in PAIRS}


def barrier(df, idx, dirs, lim_atr, tp_atr, sl_atr, max_bars, valid=3, pen=0.05, fund=None) -> pd.DataFrame:
    """Triple-barrier trades (non-overlapping per pair).

    lim_atr == 0: market entry at the next open (taker + slippage). Otherwise a resting limit at
    close -/+ lim_atr*ATR, valid `valid` bars, filled only if price trades THROUGH it by pen*ATR
    (maker fee). TP is a limit (maker); SL is a stop-market (taker + slippage, gap-aware, checked
    before TP on the same bar); time exit at the close. fund=(times_ns, rates) charges funding.
    """
    o, h, l, c, a = (df[k].to_numpy() for k in ("open", "high", "low", "close", "atr"))
    t = df["open_time"].astype("int64").to_numpy()
    n, out, busy = len(o), [], -1
    for i, d in zip(idx, dirs):
        if i <= busy or i + 1 >= n or not np.isfinite(a[i]):
            continue
        A, e = a[i], None
        lim = c[i] - d * lim_atr * A
        for k in range(i + 1, min(n, i + 1 + valid)):
            if lim_atr == 0:
                e, entry, efee = k, o[k], TAKER_FEE + SLIPPAGE
                break
            if (d == 1 and l[k] <= lim - pen * A) or (d == -1 and h[k] >= lim + pen * A):
                e, efee = k, MAKER_FEE
                entry = min(lim, o[k]) if d == 1 else max(lim, o[k])
                break
        if e is None:
            continue
        risk = sl_atr * A
        stop, tgt = entry - d * risk, entry + d * tp_atr * A
        for j in range(e, min(n, e + max_bars + 1)):
            if (l[j] <= stop) if d == 1 else (h[j] >= stop):
                gap = j > e and ((d == 1 and o[j] < stop) or (d == -1 and o[j] > stop))
                px, xfee, why = (o[j] if gap else stop), TAKER_FEE + SLIPPAGE, "sl"
                break
            # target only from the bar after the fill (intrabar order on the fill bar is unknown)
            if j > e and ((h[j] >= tgt) if d == 1 else (l[j] <= tgt)):
                px, xfee, why = tgt, MAKER_FEE, "tp"
                break
        else:
            j = min(n - 1, e + max_bars)
            px, xfee, why = c[j], TAKER_FEE + SLIPPAGE, "time"
        net = (px - entry) * d / entry - efee - xfee
        if fund is not None and len(fund[0]):
            lo_ = np.searchsorted(fund[0], t[e], "right")
            hi_ = np.searchsorted(fund[0], t[j] + (t[1] - t[0]), "right")
            net -= d * fund[1][lo_:hi_].sum()
        out.append((t[e], j - e, d, entry, net, net * entry / risk, why))
        busy = j
    r = pd.DataFrame(out, columns=["time", "bars", "dir", "entry", "ret", "r", "why"])
    r["time"] = pd.to_datetime(r["time"], utc=True)
    return r


def summarize(tr: pd.DataFrame, label: str = "") -> dict:
    if tr.empty:
        return {"label": label, "n": 0}
    r = tr["ret"]
    losses = -r[r < 0].sum()
    return {"label": label, "n": len(tr), "win": round(float((r > 0).mean()), 3),
            "avg_ret_bp": round(float(r.mean() * 1e4), 1), "avg_r": round(float(tr["r"].mean()), 3),
            "pf": round(float(r[r > 0].sum() / losses), 2) if losses else float("inf"),
            "t": round(float(r.mean() / r.std(ddof=1) * np.sqrt(len(r))), 2) if len(r) > 2 else 0.0}


# ---- machine-learning features / labels -------------------------------------------------

def ml_features(df: pd.DataFrame, btc: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    c, a, lc = df["close"], df["atr"], np.log(df["close"])
    for k in (1, 3, 6, 18, 42):
        X[f"ret{k}"] = (lc - lc.shift(k)) / df["atr_pct"]
    X["atr_pct"] = df["atr_pct"]
    X["vol_regime"] = df["atr_pct"] / df["atr_pct"].rolling(180).mean()
    for k in ("rsi2", "rsi14", "bb_z", "dr", "vol_z", "clv", "adx1h", "bd", "adxd", "b4", "btc_bd"):
        X[k] = df[k]
    for e in ("ema20", "ema50", "ema200"):
        X[f"d_{e}"] = (c - df[e]) / a
    for n in (6, 18):
        X[f"dr{n}"] = df["delta"].rolling(n).sum() / df["volume"].rolling(n).sum()
    X["clv3"] = df["clv"].rolling(3).mean()
    for n in (20, 55):
        X[f"d_hi{n}"] = (c - df["high"].rolling(n).max()) / a
        X[f"d_lo{n}"] = (c - df["low"].rolling(n).min()) / a
    X["d_d50"], X["d_d20"] = (c - df["eds"]) / c, (c - df["edf"]) / c
    X["hour"], X["dow"] = df["open_time"].dt.hour, df["open_time"].dt.dayofweek
    bl = np.log(btc.set_index("open_time")["close"])
    for k in (1, 6, 42):
        X[f"btc_ret{k}"] = df["open_time"].map(bl - bl.shift(k)).values
    X["rs42"] = (lc - lc.shift(42)).values - X["btc_ret42"].values
    f = fund.sort_values("time").copy()
    f["fz"] = (f["rate"] - f["rate"].rolling(90).mean()) / f["rate"].rolling(90).std()
    f["f3"] = f["rate"].rolling(3).mean()
    m = pd.merge_asof(df[["close_time"]].reset_index(), f.rename(columns={"time": "ft"}),
                      left_on="close_time", right_on="ft", direction="backward").set_index("index")
    X["fund"], X["fund3"], X["fund_z"] = m["rate"].values, m["f3"].values, m["fz"].values
    return X


def tb_labels(df: pd.DataFrame, tp: float, sl: float, H: int):
    """1 if the take-profit barrier is hit before the stop within H bars (stop checked first)."""
    o, h, l, a = (df[k].to_numpy() for k in ("open", "high", "low", "atr"))
    n = len(o)
    yl, ys = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n - 1):
        if not np.isfinite(a[i]):
            continue
        e = i + 1
        for d, y in ((1, yl), (-1, ys)):
            stop, tgt, res = o[e] - d * sl * a[i], o[e] + d * tp * a[i], 0.0
            for j in range(e, min(n, e + H)):
                if (l[j] <= stop) if d == 1 else (h[j] >= stop):
                    break
                if (h[j] >= tgt) if d == 1 else (l[j] <= tgt):
                    res = 1.0
                    break
            else:
                if e + H > n:
                    res = np.nan
            y[i] = res
    return yl, ys
