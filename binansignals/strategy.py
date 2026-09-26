"""Signal rules. Everything here is causal: a row only uses data from closed bars.

Two trend-following entry models, both gated by a higher-timeframe (4h) directional bias,
which is where these strategies make their money (they are designed to stay flat in chop):

* pullback  - trend continuation after a retrace into the 1h EMA21 value zone, triggered by a
              close back in trend direction that takes out the prior bar's extreme and is
              confirmed by aggressive order flow (taker delta) in the same direction.
* breakout  - Donchian-channel breakout in the direction of the 4h bias, confirmed by a volume
              expansion and positive/negative taker delta (initiative buyers/sellers).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .indicators import adx, atr, attach_htf, ema, htf_trend, resample_ohlcv, rsi


# Signal timeframe -> (bias timeframe, bias EMA fast, bias EMA slow, max holding bars)
TIMEFRAMES = {
    "1h": ("4h", 50, 200, 240),
    "4h": ("1D", 20, 50, 90),
}


def prepare(bars: pd.DataFrame, tf: str = "1h", market: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute all features on closed bars of the signal timeframe (columns from client._klines_df).

    `market` is BTCUSDT on the same timeframe: altcoins are only traded when Bitcoin's own
    higher-timeframe trend is not pointing the other way (alts rarely trend against BTC).
    """
    htf_rule, fast, slow, _ = TIMEFRAMES[tf]
    df = bars.copy().reset_index(drop=True)
    df["ema21"] = ema(df["close"], 21)
    df["ema50"] = ema(df["close"], 50)
    df["atr"] = atr(df, 14)
    df["rsi"] = rsi(df["close"], 14)
    df["adx"] = adx(df, 14)["adx"]
    df["vol_sma"] = df["volume"].rolling(20).mean()
    # Order flow from Binance's own taker-buy volume (aggressive buys vs sells per bar).
    df["delta"] = 2 * df["taker_buy_volume"] - df["volume"]
    df["delta_ratio"] = df["delta"] / df["volume"].replace(0, np.nan)
    df["cvd"] = df["delta"].cumsum()
    df["cvd_slope"] = df["delta"].rolling(6).sum() / df["volume"].rolling(6).sum()
    for n in BREAKOUT_GRID["n"]:
        df[f"dc_high{n}"] = df["high"].rolling(n).max().shift(1)
        df[f"dc_low{n}"] = df["low"].rolling(n).min().shift(1)
    df = attach_htf(df, htf_trend(resample_ohlcv(df, htf_rule), fast, slow))
    if market is not None:
        m = htf_trend(resample_ohlcv(market, htf_rule), fast, slow)[["close_time", "htf_bias"]]
        df = attach_htf(df, m.rename(columns={"htf_bias": "mkt_bias"}))
        df["mkt_bias"] = df["mkt_bias"].fillna(0)
    else:
        df["mkt_bias"] = df["htf_bias"]
    return df.reset_index(drop=True)


PULLBACK_GRID = {
    "adx_min": [15, 22],
    "of_min": [0.0, 0.06],
    "tp1_r": [0, 1.0, 1.5],   # 0 = no partial: arm break-even + trailing at +1R
    "trail_atr": [2.5, 3.5],
}
BREAKOUT_GRID = {
    "n": [20, 55],
    "vol_mult": [1.2, 1.6],
    "tp1_r": [0, 1.0, 1.5],   # 0 = no partial: arm break-even + trailing at +1R
    "trail_atr": [2.5, 3.5],
}


def param_sets(tf: str = "1h"):
    out = []
    for grid, name in ((PULLBACK_GRID, "pullback"), (BREAKOUT_GRID, "breakout")):
        keys = list(grid)
        for combo in itertools.product(*(grid[k] for k in keys)):
            p = dict(zip(keys, combo))
            p["strategy"] = name
            p["max_bars"] = TIMEFRAMES[tf][3]
            out.append(p)
    return out


def param_key(p: dict) -> str:
    return p["strategy"] + "|" + ",".join(f"{k}={p[k]}" for k in sorted(p) if k not in ("strategy", "max_bars"))


def entries(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Return per-bar direction (+1/-1/0) and the protective stop for that signal bar."""
    sig = _pullback(df, p) if p["strategy"] == "pullback" else _breakout(df, p)
    against_market = sig["dir"].to_numpy() * df["mkt_bias"].to_numpy() < 0
    sig.loc[against_market, ["dir", "stop"]] = [0, np.nan]
    return sig


def _pullback(df, p, lookback=6):
    a = df["atr"]
    near_long = ((df["low"] - df["ema21"]) / a).rolling(lookback).min() <= 0.3
    near_short = ((df["high"] - df["ema21"]) / a).rolling(lookback).max() >= -0.3
    trend_ok = df["htf_adx"] >= p["adx_min"]
    long_ = ((df["htf_bias"] == 1) & trend_ok & near_long & (df["close"] > df["ema21"])
             & (df["close"] > df["ema50"]) & (df["close"] > df["high"].shift(1))
             & (df["delta_ratio"] > p["of_min"]) & (df["rsi"] < 72))
    short = ((df["htf_bias"] == -1) & trend_ok & near_short & (df["close"] < df["ema21"])
             & (df["close"] < df["ema50"]) & (df["close"] < df["low"].shift(1))
             & (df["delta_ratio"] < -p["of_min"]) & (df["rsi"] > 28))
    swing_lo = df["low"].rolling(lookback + 1).min() - 0.25 * a
    swing_hi = df["high"].rolling(lookback + 1).max() + 0.25 * a
    stop = np.where(long_, _clamp_stop(df["close"], swing_lo, a, 1),
                    np.where(short, _clamp_stop(df["close"], swing_hi, a, -1), np.nan))
    return pd.DataFrame({"dir": np.where(long_, 1, np.where(short, -1, 0)), "stop": stop})


def _breakout(df, p):
    a, n = df["atr"], p["n"]
    vol_ok = df["volume"] > p["vol_mult"] * df["vol_sma"]
    long_ = ((df["htf_bias"] == 1) & (df["close"] > df[f"dc_high{n}"]) & vol_ok
             & (df["delta_ratio"] > 0.05) & (df["adx"] >= 18))
    short = ((df["htf_bias"] == -1) & (df["close"] < df[f"dc_low{n}"]) & vol_ok
             & (df["delta_ratio"] < -0.05) & (df["adx"] >= 18))
    # Stop back inside the old range: midpoint of the breakout bar, at least 1.5 ATR away.
    mid = (df["high"] + df["low"]) / 2
    stop_l = _clamp_stop(df["close"], np.minimum(mid, df["close"] - 1.5 * a), a, 1)
    stop_s = _clamp_stop(df["close"], np.maximum(mid, df["close"] + 1.5 * a), a, -1)
    stop = np.where(long_, stop_l, np.where(short, stop_s, np.nan))
    return pd.DataFrame({"dir": np.where(long_, 1, np.where(short, -1, 0)), "stop": stop})


def _clamp_stop(close, raw_stop, a, d, lo=0.8, hi=3.0):
    dist = ((close - raw_stop) * d).clip(lower=lo * a, upper=hi * a)
    return close - d * dist
