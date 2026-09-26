"""Vectorised technical indicators (Wilder-style where applicable)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift()
    return pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return rma(true_range(df), n)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up, dn = rma(d.clip(lower=0), n), rma(-d.clip(upper=0), n)
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = rma(true_range(df), n)
    pdi = 100 * rma(pd.Series(plus_dm, index=df.index), n) / tr
    mdi = 100 * rma(pd.Series(minus_dm, index=df.index), n) / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return pd.DataFrame({"adx": rma(dx, n), "pdi": pdi, "mdi": mdi})


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate closed bars into a higher timeframe, indexed by open_time."""
    g = df.set_index("open_time").resample(rule, label="left", closed="left")
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "volume": g["volume"].sum(),
        "taker_buy_volume": g["taker_buy_volume"].sum(), "n": g["close"].count(),
    }).dropna(subset=["close"])
    expected = pd.Timedelta(rule) / (df["open_time"].iloc[1] - df["open_time"].iloc[0])
    out = out[out["n"] >= expected].drop(columns="n")  # drop incomplete HTF bars
    out = out.reset_index()
    out["open_time"] = out["open_time"].astype("datetime64[ns, UTC]")
    out["close_time"] = out["open_time"] + pd.Timedelta(rule)
    return out


def htf_trend(htf: pd.DataFrame, fast: int = 50, slow: int = 200, adx_n: int = 14) -> pd.DataFrame:
    """Higher-timeframe trend state, keyed by the HTF bar close time."""
    e_fast, e_slow = ema(htf["close"], fast), ema(htf["close"], slow)
    a = adx(htf, adx_n)
    slope = e_fast - e_fast.shift(3)
    bias = np.where((htf["close"] > e_fast) & (e_fast > e_slow) & (slope > 0), 1,
                    np.where((htf["close"] < e_fast) & (e_fast < e_slow) & (slope < 0), -1, 0))
    return pd.DataFrame({
        "close_time": htf["close_time"], "htf_bias": bias, "htf_adx": a["adx"].values,
        "htf_ema_fast": e_fast.values, "htf_ema_slow": e_slow.values,
    })


def attach_htf(ltf: pd.DataFrame, htf_state: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    """As-of join: each LTF bar sees only HTF bars that had closed by its own close."""
    st = htf_state.rename(columns={c: c + suffix for c in htf_state.columns if c != "close_time"})
    st = st.rename(columns={"close_time": "_htf_close"}).sort_values("_htf_close")
    out = pd.merge_asof(ltf.sort_values("close_time"), st, left_on="close_time",
                        right_on="_htf_close", direction="backward")
    return out.drop(columns="_htf_close")
