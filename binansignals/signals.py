"""Live signal engine: walk-forward-validated trend rules + S/R, footprint, positioning, news."""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from . import news as newsmod
from .backtest import walk_forward
from .client import BinanceFutures, drop_unclosed
from .config import (BACKTEST_DAYS, MAX_LEVERAGE, MIN_OOS_PROFIT_FACTOR, MIN_OOS_TRADES, PAIRS,
                     RISK_PER_TRADE)
from .levels import key_levels, volume_profile
from .orderflow import book_imbalance, footprint, positioning
from .strategy import entries, param_key, prepare

TF = "4h"


def run(pairs=None, equity: float = 1000.0, risk: float = RISK_PER_TRADE, log=print) -> dict:
    c = BinanceFutures()
    cfgs = [p for p in PAIRS if not pairs or p["pair"] in pairs or p["symbol"] in pairs]
    listed = c.exchange_symbols()
    log("Fetching news & sentiment ...")
    headlines = newsmod.fetch_headlines(48)
    fg = newsmod.fear_greed()
    delist = newsmod.binance_delistings([p["symbol"] for p in cfgs])
    mkt_news = newsmod.market_news(headlines)
    btc = drop_unclosed(c.cached_history("BTCUSDT", TF, BACKTEST_DAYS))
    results = []
    for cfg in cfgs:
        sym = cfg["symbol"]
        if listed.get(sym, {}).get("status") != "TRADING":
            results.append({"pair": cfg["pair"], "symbol": sym, "name": cfg["name"], "status": "UNAVAILABLE",
                            "reason": f"{sym} is not trading on Binance USD-M futures"})
            continue
        log(f"Analysing {cfg['pair']} ({sym}) ...")
        try:
            results.append(analyse_pair(c, cfg, btc, headlines, delist, equity, risk))
        except Exception as e:  # keep the run alive; report the failure honestly
            results.append({"pair": cfg["pair"], "symbol": sym, "name": cfg["name"], "status": "ERROR",
                            "reason": repr(e)})
    return {"generated": pd.Timestamp.now(tz="UTC"), "timeframe": TF, "equity": equity, "risk": risk,
            "fear_greed": fg, "market_news": mkt_news, "headline_count": len(headlines),
            "pairs": results}


def analyse_pair(c: BinanceFutures, cfg: dict, btc: pd.DataFrame, headlines, delist, equity, risk) -> dict:
    sym = cfg["symbol"]
    bars = drop_unclosed(c.cached_history(sym, TF, BACKTEST_DAYS))
    funding = c.funding_history(sym, int(bars["open_time"].iloc[0].value // 1_000_000))
    df = prepare(bars, TF, None if sym == "BTCUSDT" else btc)
    wf = walk_forward(df, funding, TF)
    oos = wf["oos_metrics"]
    validated = oos["trades"] >= MIN_OOS_TRADES and oos["profit_factor"] >= MIN_OOS_PROFIT_FACTOR
    params = wf["live_params"]

    # ---- live market context --------------------------------------------------------
    prem = c.premium_index(sym)
    price = float(prem["markPrice"])
    d1 = drop_unclosed(c.klines(sym, "1d", 400))
    h1 = drop_unclosed(c.klines(sym, "1h", 200))
    now_ms = int(time.time() * 1000)
    m1 = drop_unclosed(c.klines_range(sym, "1m", now_ms - 72 * 3_600_000))
    last = df.iloc[-1]
    a = float(last["atr"])
    lv = key_levels(bars, d1, a, price)
    vp = volume_profile(m1)
    trades = c.agg_trades(sym, max_trades=8000, window_ms=4 * 3_600_000)
    fp = footprint(trades, bar="5min")
    book = book_imbalance(c.depth(sym, 500), price)
    pos = positioning(prem, c.open_interest_hist(sym, "1h", 48), c.long_short_ratio(sym, "1h", 24),
                      c.top_trader_ratio(sym, "1h", 24), h1)
    cn = newsmod.coin_news(headlines, cfg)

    # ---- signal / setup ---------------------------------------------------------------
    plan = _plan(df, params, price) if params else None
    bias = int(last["htf_bias"])
    mkt = int(last["mkt_bias"])
    if sym in delist:
        status, plan = "NO TRADE", None
        reason = "Binance delisting notice: " + delist[sym][0]
    elif plan is None and params is None:
        status, reason = "NO TRADE", "No rule set showed a positive edge over the last 180 days (stay flat)."
    elif plan is None:
        status = "NO TRADE"
        reason = ("No directional bias: daily trend filter is neutral (range / transition)." if bias == 0
                  else "Daily bias conflicts with BTC's daily trend; alts are not traded against BTC.")
    else:
        status, reason = plan["status"], plan["why"]

    out = {
        "pair": cfg["pair"], "symbol": sym, "name": cfg["name"], "price": price, "status": status,
        "reason": reason, "last_bar": last["close_time"], "bias": bias, "btc_bias": mkt,
        "atr": a, "atr_pct": a / price, "rsi": float(last["rsi"]), "adx_d1": float(last["htf_adx"]),
        "ema21": float(last["ema21"]), "ema50": float(last["ema50"]),
        "validated": validated, "oos": oos, "oos_windows": len(wf["windows"]),
        "rule_set": param_key(params) if params else None, "rule_score": wf["live_score"],
        "levels": lv, "profile": vp, "book": book, "positioning": pos, "news": cn,
        "footprint": _fp_summary(fp), "cvd_slope_4h": float(last["cvd_slope"]),
        "plan": plan,
    }
    if plan:
        out["score"], out["score_parts"] = _confluence(out, plan["dir"])
        out["sizing"] = _size(plan["entry"], plan["stop"], equity, risk)
        out["grade"] = _grade(out["score"], validated, status)
    return out


def _plan(df: pd.DataFrame, p: dict, price: float) -> dict | None:
    """ACTIVE if the validated rule fired on one of the last 2 closed bars and is still valid;
    otherwise a PENDING plan describing exactly what must happen for an entry."""
    sig = entries(df, p)
    last = df.iloc[-1]
    a = float(last["atr"])
    for k in (1, 2):
        d = int(sig["dir"].iloc[-k])
        if d == 0:
            continue
        stop = float(sig["stop"].iloc[-k])
        sig_close = float(df["close"].iloc[-k])
        r = (sig_close - stop) * d
        if (price - stop) * d <= 0 or (price - sig_close) * d > 0.5 * r:
            continue  # stopped out or already ran away -> don't chase
        return _levels(d, price, stop, p, "ACTIVE",
                       f"{p['strategy']} rule fired on the 4h bar closed {df['close_time'].iloc[-k]:%Y-%m-%d %H:%M} UTC")
    d = int(last["htf_bias"])
    if d == 0 or d * int(last["mkt_bias"]) < 0:
        return None
    if p["strategy"] == "pullback":
        zone = float(last["ema21"])
        entry = zone if (price - zone) * d > 0 else price
        swing = (df["low"].tail(7).min() if d == 1 else df["high"].tail(7).max())
        raw = min(swing, entry - 0.8 * a) - 0.25 * a if d == 1 else max(swing, entry + 0.8 * a) + 0.25 * a
        dist = min(max((entry - raw) * d, 0.8 * a), 3 * a)
        why = (f"Daily trend {'up' if d == 1 else 'down'}; wait for a pullback into the 4h EMA21 zone "
               f"(~{_fmt(zone)}) and a 4h close back {'above' if d == 1 else 'below'} EMA21 and the prior bar's "
               f"{'high' if d == 1 else 'low'} with {'positive' if d == 1 else 'negative'} taker delta.")
    else:
        n = p["n"]
        lvl = float(last[f"dc_high{n}"] if d == 1 else last[f"dc_low{n}"])
        lvl = max(lvl, float(df["high"].iloc[-1])) if d == 1 else min(lvl, float(df["low"].iloc[-1]))
        entry = lvl + d * 0.1 * a
        dist = min(max(1.5 * a, 0.8 * a), 3 * a)
        why = (f"Daily trend {'up' if d == 1 else 'down'}; enter on a 4h CLOSE {'above' if d == 1 else 'below'} "
               f"the {n}-bar Donchian {'high' if d == 1 else 'low'} (~{_fmt(lvl)}) on volume > "
               f"{p['vol_mult']}x average and {'buy' if d == 1 else 'sell'}-side taker delta.")
    return _levels(d, entry, entry - d * dist, p, "PENDING", why)


def _levels(d, entry, stop, p, status, why):
    r = (entry - stop) * d
    return {
        "status": status, "dir": d, "side": "LONG" if d == 1 else "SHORT", "why": why,
        "strategy": p["strategy"], "entry": entry, "stop": stop, "risk_pct": r / entry,
        "tp1": entry + d * r * (p["tp1_r"] or 1.0), "tp2": entry + d * 2 * r, "tp3": entry + d * 3 * r,
        "partial": bool(p["tp1_r"]), "trail_atr": p["trail_atr"],
        "management": ((f"Close 50% at TP1 ({p['tp1_r']}R), move stop to entry, " if p["tp1_r"]
                        else "At +1R move stop to entry, ")
                       + f"then trail the rest {p['trail_atr']}x ATR(4h) behind the best price. "
                       f"TP2/TP3 are reference levels for manual scaling."),
    }


def _confluence(o: dict, d: int):
    parts = []

    def add(name, ok, pts, note):
        parts.append({"factor": name, "points": pts if ok else 0, "max": pts, "note": note})

    pos, fp, lv, cn = o["positioning"], o["footprint"], o["levels"], o["news"]
    add("Daily trend aligned", o["bias"] == d, 20, f"daily bias {o['bias']:+d}")
    add("BTC trend aligned", o["btc_bias"] == d or o["symbol"] == "BTCUSDT", 10, f"BTC daily bias {o['btc_bias']:+d}")
    add("Trend strength (daily ADX >= 20)", o["adx_d1"] >= 20, 10, f"ADX {o['adx_d1']:.1f}")
    add("Price vs 4h EMA50", (o["price"] - o["ema50"]) * d > 0, 5, f"EMA50 {_fmt(o['ema50'])}")
    add("4h CVD slope aligned", o["cvd_slope_4h"] * d > 0, 10, f"{o['cvd_slope_4h']:+.3f}")
    add("Live footprint delta aligned", fp.get("window_delta", 0) * d > 0, 10,
        f"delta {fp.get('window_delta', 0):+,.0f} over {fp.get('minutes', 0):.0f} min")
    tgt = o["plan"]["tp2"]
    opp = lv["resistance"] if d == 1 else lv["support"]
    blocker = next((z for z in opp if (z["price"] - o["plan"]["entry"]) * d > 0 and (tgt - z["price"]) * d > 0), None)
    add("Room to 2R (no strong S/R in the way)", blocker is None or blocker["strength"] < 1.0, 10,
        "clear" if blocker is None else
        f"level {_fmt(blocker['price'])} (×{blocker['touches']}, recency-weighted strength {blocker['strength']:.2f})")
    f = pos.get("funding", 0.0)
    add("Funding not crowded against trade", f * d < 0.0003, 5, f"{f * 100:+.4f}% / 8h")
    oi, px = pos.get("oi_chg_24h"), pos.get("px_chg_24h")
    add("OI confirms (new money with the move)", oi is not None and px == px and oi > 0 and px * d > 0, 5,
        f"OI {oi * 100:+.1f}%, price {px * 100:+.1f}% (24h)" if oi is not None and px == px else "n/a")
    ls = pos.get("retail_ls")
    add("Retail not one-sided with trade", ls is None or (ls < 2.5 if d == 1 else ls > 0.6), 5,
        f"retail L/S {ls:.2f}" if ls else "n/a")
    add("News not against trade", cn["sentiment"] * d > -0.34 or cn["count"] < 2, 10,
        f"{cn['count']} headlines, sentiment {cn['sentiment']:+.2f}")
    return sum(p["points"] for p in parts), parts


def _grade(score, validated, status):
    if not validated:
        return "C (rule set not validated out-of-sample for this pair)"
    if score >= 75:
        return "A" if status == "ACTIVE" else "A (pending trigger)"
    if score >= 60:
        return "B" if status == "ACTIVE" else "B (pending trigger)"
    return "C (weak confluence)"


def _size(entry, stop, equity, risk):
    risk_usd = equity * risk
    qty = risk_usd / abs(entry - stop)
    notional = qty * entry
    lev = max(1, math.ceil(notional / equity))
    capped = lev > MAX_LEVERAGE
    if capped:
        notional = equity * MAX_LEVERAGE
        qty = notional / entry
        lev = MAX_LEVERAGE
    return {"equity": equity, "risk": risk, "risk_usd": qty * abs(entry - stop), "qty": qty, "notional": notional,
            "leverage": lev, "capped": capped}


def _fp_summary(fp: dict) -> dict:
    if not fp:
        return {}
    bars = fp["bars"]
    return {
        "start": fp["start"], "end": fp["end"], "minutes": (fp["end"] - fp["start"]).total_seconds() / 60,
        "n_trades": fp["n_trades"], "tick": fp["tick"], "bar": fp["bar"],
        "window_delta": fp["window_delta"], "window_volume": fp["window_volume"],
        "delta_pct": fp["window_delta"] / fp["window_volume"] if fp["window_volume"] else 0.0,
        "stacked_buy_zones": fp["stacked_buy_zones"], "stacked_sell_zones": fp["stacked_sell_zones"],
        "absorption_buy_bars": fp["absorption_buy_bars"], "absorption_sell_bars": fp["absorption_sell_bars"],
        "recent_bars": [{"time": r.time, "open": r.open, "high": r.high, "low": r.low, "close": r.close,
                         "volume": r.volume, "delta": r.delta, "poc": r.poc}
                        for r in bars.tail(6).itertuples()],
    }


def _fmt(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    digits = max(2, 5 - int(math.floor(math.log10(abs(x)))) - 1) if x else 2
    return f"{x:,.{min(digits, 6)}f}"
