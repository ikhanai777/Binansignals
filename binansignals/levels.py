"""Support / resistance from swing pivots and volume-at-price."""
from __future__ import annotations

import numpy as np
import pandas as pd


def swing_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    """Fractal swing highs/lows. A pivot needs `right` closed bars after it to be confirmed."""
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    rows = []
    for i in range(left, len(df) - right):
        if h[i] == h[i - left:i + right + 1].max():
            rows.append((df["open_time"].iloc[i], h[i], "R"))
        if l[i] == l[i - left:i + right + 1].min():
            rows.append((df["open_time"].iloc[i], l[i], "S"))
    return pd.DataFrame(rows, columns=["time", "price", "kind"])


def cluster_levels(pivots: pd.DataFrame, tol: float, now: pd.Timestamp, half_life_days: float = 45) -> list[dict]:
    """Merge pivots within `tol` of each other into zones; strength = recency-weighted touches."""
    if pivots.empty:
        return []
    p = pivots.sort_values("price").reset_index(drop=True)
    zones, cur = [], [0]
    for i in range(1, len(p)):
        # chain to the previous pivot, but cap zone width so dense areas don't merge into one blob
        if (p["price"].iloc[i] - p["price"].iloc[cur[-1]] <= tol
                and p["price"].iloc[i] - p["price"].iloc[cur[0]] <= 2 * tol):
            cur.append(i)
        else:
            zones.append(cur)
            cur = [i]
    zones.append(cur)
    out = []
    for z in zones:
        g = p.iloc[z]
        age = (now - g["time"]).dt.total_seconds() / 86400
        out.append({
            "price": float(g["price"].mean()), "low": float(g["price"].min()), "high": float(g["price"].max()),
            "touches": int(len(g)), "strength": float((0.5 ** (age / half_life_days)).sum()),
            "last_touch": g["time"].max(),
        })
    return out


def key_levels(bars_4h: pd.DataFrame, bars_1d: pd.DataFrame, atr_4h: float, price: float, n: int = 4) -> dict:
    """Nearest meaningful supports below and resistances above the current price."""
    now = bars_4h["open_time"].iloc[-1]
    piv = pd.concat([swing_pivots(bars_4h.tail(540), 3, 3), swing_pivots(bars_1d.tail(365), 2, 2)])
    zones = cluster_levels(piv, tol=0.6 * atr_4h, now=now)
    zones = [z for z in zones if z["strength"] >= 0.25 or z["touches"] >= 2]
    sup = sorted([z for z in zones if z["price"] < price], key=lambda z: -z["price"])[:n]
    res = sorted([z for z in zones if z["price"] > price], key=lambda z: z["price"])[:n]
    return {"support": sup, "resistance": res}


def volume_profile(bars: pd.DataFrame, bins: int = 80, value_area: float = 0.70) -> dict:
    """Volume-at-price with aggressor split, built from real (e.g. 1m) Binance klines.

    Each bar's volume is spread uniformly across its high-low range, which is a standard
    approximation when tick data for the whole window is not pulled.
    """
    lo, hi = float(bars["low"].min()), float(bars["high"].max())
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    vol = np.zeros(bins)
    buy = np.zeros(bins)
    for h, l, v, b in bars[["high", "low", "volume", "taker_buy_volume"]].itertuples(index=False):
        i0 = min(np.searchsorted(edges, l, "right") - 1, bins - 1)
        i1 = min(np.searchsorted(edges, h, "right") - 1, bins - 1)
        k = i1 - i0 + 1
        vol[i0:i1 + 1] += v / k
        buy[i0:i1 + 1] += b / k
    poc = int(vol.argmax())
    lo_i, hi_i, acc, total = poc, poc, vol[poc], vol.sum()
    while acc < value_area * total and (lo_i > 0 or hi_i < bins - 1):
        down = vol[lo_i - 1] if lo_i > 0 else -1
        up = vol[hi_i + 1] if hi_i < bins - 1 else -1
        if up >= down:
            hi_i += 1
            acc += vol[hi_i]
        else:
            lo_i -= 1
            acc += vol[lo_i]
    delta = 2 * buy - vol
    mean = vol.mean()
    hvn = [float(centers[i]) for i in range(1, bins - 1)
           if vol[i] > 1.5 * mean and vol[i] >= vol[i - 1] and vol[i] >= vol[i + 1]]
    return {
        "poc": float(centers[poc]), "vah": float(edges[hi_i + 1]), "val": float(edges[lo_i]),
        "hvn": hvn, "window_start": bars["open_time"].iloc[0], "window_end": bars["close_time"].iloc[-1],
        "delta_at_poc": float(delta[poc]), "total_delta": float(delta.sum()), "total_volume": float(total),
    }
