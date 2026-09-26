"""Footprint (bid/ask volume at price) from real Binance aggregated trades, plus book and
derivatives positioning (funding, open interest, long/short ratios)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def footprint(trades: pd.DataFrame, bar: str = "5min", tick: float | None = None,
              imbalance: float = 3.0, min_stack: int = 3) -> dict:
    """Build a footprint chart from aggTrades.

    Buyer-initiated volume ("ask" side) is aggTrade.m == False; seller-initiated ("bid")
    is m == True. Diagonal imbalance: ask volume at price p vs bid volume one tick below.
    """
    if trades.empty:
        return {}
    px = trades["price"]
    if tick is None:
        tick = _nice_tick(float(px.iloc[-1]) * 0.0005)  # ~5 bps buckets
    t = trades.assign(
        bar=trades["time"].dt.floor(bar),
        level=(px / tick).round().astype("int64"),
        ask=np.where(~trades["is_buyer_maker"], trades["qty"], 0.0),
        bid=np.where(trades["is_buyer_maker"], trades["qty"], 0.0),
    )
    bars = []
    for b, g in t.groupby("bar"):
        lv = g.groupby("level")[["ask", "bid"]].sum().sort_index()
        lv = lv.reindex(range(lv.index.min(), lv.index.max() + 1), fill_value=0.0)
        ask, bid = lv["ask"].to_numpy(), lv["bid"].to_numpy()
        buy_imb = np.zeros(len(lv), bool)
        sell_imb = np.zeros(len(lv), bool)
        tot = ask + bid
        meaningful = tot >= 0.25 * tot.mean()  # ignore thin prints at the edges
        buy_imb[1:] = meaningful[1:] & (ask[1:] >= imbalance * np.maximum(bid[:-1], 1e-12))
        sell_imb[:-1] = meaningful[:-1] & (bid[:-1] >= imbalance * np.maximum(ask[1:], 1e-12))
        bars.append({
            "time": b, "open": float(g["price"].iloc[0]), "high": float(g["price"].max()),
            "low": float(g["price"].min()), "close": float(g["price"].iloc[-1]),
            "volume": float(tot.sum()), "delta": float(ask.sum() - bid.sum()),
            "poc": float(lv.index[tot.argmax()] * tick),
            "stacked_buy": [_zone(lv.index, s, e, tick) for s, e in _runs(buy_imb, min_stack)],
            "stacked_sell": [_zone(lv.index, s, e, tick) for s, e in _runs(sell_imb, min_stack)],
            "levels": {float(i * tick): (float(a), float(bd)) for i, a, bd in zip(lv.index, ask, bid)},
        })
    df = pd.DataFrame(bars)
    df["cvd"] = df["delta"].cumsum()
    last = df.iloc[-1]
    # Absorption: heavy one-sided aggression that failed to move price in its direction.
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    move = (df["close"] - df["open"]) / rng
    heavy = df["volume"] > 1.5 * df["volume"].median()
    absorb_buy = heavy & (df["delta"] < -0.15 * df["volume"]) & (move > -0.1)   # sellers absorbed
    absorb_sell = heavy & (df["delta"] > 0.15 * df["volume"]) & (move < 0.1)    # buyers absorbed
    return {
        "tick": tick, "bar": bar, "start": df["time"].iloc[0], "end": trades["time"].iloc[-1],
        "n_trades": int(len(trades)), "bars": df,
        "window_delta": float(df["delta"].sum()), "window_volume": float(df["volume"].sum()),
        "last_delta": float(last["delta"]),
        "stacked_buy_zones": [z for zs in df["stacked_buy"] for z in zs][-5:],
        "stacked_sell_zones": [z for zs in df["stacked_sell"] for z in zs][-5:],
        "absorption_buy_bars": int(absorb_buy.sum()), "absorption_sell_bars": int(absorb_sell.sum()),
    }


def _runs(mask: np.ndarray, min_len: int):
    out, start = [], None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    return out


def _zone(index, s, e, tick):
    return (float(index[s] * tick), float(index[e] * tick))


def _nice_tick(x: float) -> float:
    exp = np.floor(np.log10(x))
    base = x / 10 ** exp
    nice = 1 if base < 2 else 2 if base < 5 else 5
    return float(nice * 10 ** exp)


def book_imbalance(depth: dict, price: float, pct: float = 0.01) -> dict:
    """Resting liquidity within +/-pct of price and the largest walls on each side."""
    bids = [(p, q) for p, q in depth["bids"] if p >= price * (1 - pct)]
    asks = [(p, q) for p, q in depth["asks"] if p <= price * (1 + pct)]
    bv, av = sum(p * q for p, q in bids), sum(p * q for p, q in asks)
    wall_b = max(depth["bids"], key=lambda x: x[0] * x[1]) if depth["bids"] else (None, 0)
    wall_a = max(depth["asks"], key=lambda x: x[0] * x[1]) if depth["asks"] else (None, 0)
    return {
        "bid_usd": bv, "ask_usd": av, "imbalance": (bv - av) / (bv + av) if bv + av else 0.0,
        "bid_wall": wall_b[0], "bid_wall_usd": wall_b[0] * wall_b[1] if wall_b[0] else 0.0,
        "ask_wall": wall_a[0], "ask_wall_usd": wall_a[0] * wall_a[1] if wall_a[0] else 0.0,
    }


def positioning(premium: dict, oi: pd.DataFrame, ls: pd.DataFrame, top: pd.DataFrame, bars_1h: pd.DataFrame) -> dict:
    """Derivatives crowding: funding, OI change vs price change, retail vs top-trader ratios."""
    out = {"funding": float(premium["lastFundingRate"]), "mark": float(premium["markPrice"]),
           "next_funding": pd.to_datetime(int(premium["nextFundingTime"]), unit="ms", utc=True)}
    if not oi.empty and len(oi) >= 25:
        out["oi_usd"] = float(oi["oi_value"].iloc[-1])
        out["oi_chg_24h"] = float(oi["oi"].iloc[-1] / oi["oi"].iloc[-25] - 1)
        p = bars_1h.set_index("close_time")["close"]
        p0 = p.asof(oi["time"].iloc[-25])
        out["px_chg_24h"] = float(p.iloc[-1] / p0 - 1) if p0 == p0 else float("nan")
    if not ls.empty:
        out["retail_ls"] = float(ls["ratio"].iloc[-1])
        out["retail_long_pct"] = float(ls["long_pct"].iloc[-1])
    if not top.empty:
        out["top_ls"] = float(top["ratio"].iloc[-1])
    return out
