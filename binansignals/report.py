"""Markdown / JSON rendering of live signals and backtest results."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from .signals import _fmt

DISCLAIMER = ("Not financial advice. Signals are generated mechanically from live Binance data; the "
              "backtest statistics are historical and out-of-sample, and they do not guarantee future results. "
              "Leveraged futures can lose more than the stake.")

ORDER = {"ACTIVE": 0, "PENDING": 1, "NO TRADE": 2, "UNAVAILABLE": 3, "ERROR": 4}


def _pct(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x * 100:+.{d}f}%"


def signals_markdown(res: dict) -> str:
    pairs = sorted(res["pairs"], key=lambda p: (ORDER.get(p["status"], 9), -p.get("score", 0)))
    fg = res.get("fear_greed")
    L = [f"# Binance USD-M Futures Signals — {res['generated']:%Y-%m-%d %H:%M} UTC", "",
         f"Signal timeframe **{res['timeframe']}** · trend filter **1D** · account **{res['equity']:,.0f} USDT** "
         f"risking **{res['risk'] * 100:.1f}%** per trade", ""]
    if fg:
        L.append(f"Crypto Fear & Greed: **{fg['value']} ({fg['label']})**, a week ago {fg['week_ago']}.  ")
    mn = res["market_news"]
    L.append(f"Market news (48h, {res['headline_count']} headlines): sentiment **{mn['sentiment']:+.2f}** "
             f"across {mn['count']} macro/market items.")
    if mn["high_impact"]:
        L += ["", "High-impact market headlines:"] + [f"- {t}" for t in mn["high_impact"]]
    L += ["", "## Summary", "",
          "| Pair | Status | Side | Entry | Stop | TP1 | TP2 | TP3 | Score | Grade | OOS PF | OOS trades |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in pairs:
        pl = p.get("plan")
        oos = p.get("oos", {})
        pf = oos.get("profit_factor")
        if pl:
            L.append(f"| {p['pair']} | {p['status']} | {pl['side']} | {_fmt(pl['entry'])} | {_fmt(pl['stop'])} | "
                     f"{_fmt(pl['tp1'])} | {_fmt(pl['tp2'])} | {_fmt(pl['tp3'])} | {p['score']} | {p['grade'].split(' ')[0]} | "
                     f"{pf:.2f} | {oos.get('trades')} |")
        else:
            L.append(f"| {p['pair']} | {p['status']} | — | — | — | — | — | — | — | — | "
                     f"{'' if pf is None else f'{pf:.2f}'} | {oos.get('trades', '')} |")
    L += ["", "**ACTIVE** = the validated rule fired on one of the last two closed 4h bars and price has not run "
          "more than 0.5R past it: enter at market. **PENDING** = directional bias is present; place the order "
          "only when the stated trigger happens. **NO TRADE** = no bias or no validated edge: stay flat.", ""]
    for p in pairs:
        L += _pair_md(p)
    L += ["## Method", "", *METHOD, "", f"_{DISCLAIMER}_", ""]
    return "\n".join(L)


def _pair_md(p: dict) -> list[str]:
    L = [f"## {p['pair']} — {p['name']}" + (f" (traded as {p['symbol']})" if p["symbol"] != p["pair"] else ""), ""]
    if p["status"] in ("UNAVAILABLE", "ERROR"):
        return L + [f"**{p['status']}**: {p['reason']}", ""]
    pl = p.get("plan")
    L.append(f"**{p['status']}{' ' + pl['side'] if pl else ''}** · mark {_fmt(p['price'])} · {p['reason']}")
    L.append("")
    if pl:
        s = p["sizing"]
        L += [f"- Entry **{_fmt(pl['entry'])}** · Stop **{_fmt(pl['stop'])}** ({pl['risk_pct'] * 100:.2f}% away)",
              f"- TP1 **{_fmt(pl['tp1'])}** · TP2 **{_fmt(pl['tp2'])}** · TP3 **{_fmt(pl['tp3'])}**",
              f"- Management: {pl['management']}",
              f"- Size for {s['equity']:,.0f} USDT @ {s['risk'] * 100:.1f}% risk: {s['qty']:.6g} {p['symbol'].replace('USDT', '')} "
              f"(notional {s['notional']:,.0f} USDT, isolated leverage {s['leverage']}x"
              f"{', capped' if s['capped'] else ''}) → max loss ≈ {s['risk_usd']:,.2f} USDT",
              f"- Confluence **{p['score']}/100**, grade **{p['grade']}**", ""]
        L += ["| Factor | Pts | Detail |", "|---|---|---|"]
        L += [f"| {c['factor']} | {c['points']}/{c['max']} | {c['note']} |" for c in p["score_parts"]]
        L.append("")
    o = p["oos"]
    L.append(f"**Walk-forward out-of-sample** ({p['oos_windows']} windows): {o['trades']} trades, win rate "
             f"{o['win_rate'] * 100:.0f}%, avg {o['avg_r']:+.2f}R, PF {o['profit_factor']:.2f}, total {o['total_r']:+.1f}R, "
             f"max DD {o['max_dd_r']:.1f}R · {'validated ✅' if p['validated'] else 'NOT validated ⚠️'} · "
             f"current rule set: `{p['rule_set'] or 'none (flat)'}`")
    L.append("")
    lv, vp, fp, pos, bk = p["levels"], p["profile"], p["footprint"], p["positioning"], p["book"]
    sup = ", ".join(f"{_fmt(z['price'])} (×{z['touches']})" for z in lv["support"]) or "none"
    res = ", ".join(f"{_fmt(z['price'])} (×{z['touches']})" for z in lv["resistance"]) or "none"
    L += [f"- Trend: daily bias {p['bias']:+d}, BTC bias {p['btc_bias']:+d}, daily ADX {p['adx_d1']:.1f}, "
          f"4h RSI {p['rsi']:.0f}, 4h EMA21 {_fmt(p['ema21'])}, EMA50 {_fmt(p['ema50'])}, ATR {p['atr_pct'] * 100:.2f}%",
          f"- Support: {sup}",
          f"- Resistance: {res}",
          f"- Volume profile (72h, 1m bars): POC {_fmt(vp['poc'])}, value area {_fmt(vp['val'])}–{_fmt(vp['vah'])}, "
          f"net taker delta {vp['total_delta']:+,.0f}"]
    if fp:
        zones = ", ".join(f"{_fmt(a)}–{_fmt(b)} buy" for a, b in fp["stacked_buy_zones"])
        zones += (", " if zones and fp["stacked_sell_zones"] else "") + ", ".join(
            f"{_fmt(a)}–{_fmt(b)} sell" for a, b in fp["stacked_sell_zones"])
        L.append(f"- Footprint (last {fp['minutes']:.0f} min, {fp['n_trades']:,} aggTrades, {fp['tick']:g} ticks): "
                 f"delta {fp['window_delta']:+,.2f} ({fp['delta_pct'] * 100:+.1f}% of volume), stacked imbalances: "
                 f"{zones or 'none'}, absorption bars buy/sell {fp['absorption_buy_bars']}/{fp['absorption_sell_bars']}")
    L.append(f"- Derivatives: funding {pos['funding'] * 100:+.4f}%/8h, OI 24h {_pct(pos.get('oi_chg_24h'), 1)} vs price "
             f"{_pct(pos.get('px_chg_24h'), 1)}, retail L/S {pos.get('retail_ls', float('nan')):.2f}, "
             f"top-trader L/S {pos.get('top_ls', float('nan')):.2f}")
    L.append(f"- Order book ±1%: bid/ask imbalance {bk['imbalance']:+.2f}, largest bid wall {_fmt(bk['bid_wall'])} "
             f"(${bk['bid_wall_usd']:,.0f}), ask wall {_fmt(bk['ask_wall'])} (${bk['ask_wall_usd']:,.0f})")
    n = p["news"]
    L.append(f"- News (48h): {n['count']} headlines, sentiment {n['sentiment']:+.2f}"
             + (f"; high-impact: {'; '.join(n['high_impact'])}" if n["high_impact"] else ""))
    for h in n["headlines"][:3]:
        L.append(f"  - [{h['title']}]({h['link']}) — {h['source']}, {h['time']:%m-%d %H:%M}")
    L.append("")
    return L


METHOD = [
    "1. **Data** — every number comes from live Binance USD-M futures endpoints (klines incl. taker-buy volume, "
    "aggTrades, order book, funding, open interest, long/short ratios), public crypto RSS feeds, alternative.me "
    "Fear & Greed and Binance announcements. Nothing is simulated.",
    "2. **Directional bias** — daily EMA20/EMA50 stack with a rising (falling) EMA20 and price on the correct side. "
    "Altcoins additionally require that BTC's daily bias is not opposite. No bias → no trade.",
    "3. **Entry rules (4h)** — *pullback*: retrace into the EMA21 zone, then a close back in trend direction "
    "through the prior bar's extreme with taker delta confirming; *breakout*: close beyond the 20/55-bar Donchian "
    "channel on expanded volume with taker delta confirming.",
    "4. **Walk-forward validation** — 48 rule variants are backtested on ~3 years of real 4h history with fees "
    "(0.05% taker / 0.02% maker), 0.02% slippage, historical funding and conservative intrabar fills. Every 60 "
    "days the best variant on the previous 180 days (t-stat of mean R) is traded blind on the next 60. Only those "
    "out-of-sample trades are reported. A pair is *validated* with ≥25 OOS trades and PF ≥ 1.15; the live rule set "
    "is the best variant on the latest 180 days, and a pair stays flat if no variant had a positive edge.",
    "5. **Confluence score** (context, not backtested): trend alignment, BTC alignment, ADX, 4h CVD, live "
    "footprint delta, S/R room to 2R, funding, OI, retail positioning and news sentiment.",
    "6. **Risk** — fixed-fractional sizing from the stop distance; leverage is just what that notional needs.",
]


def backtest_markdown(rows: list[dict], portfolio: dict, generated) -> str:
    L = [f"# Walk-forward backtest (out-of-sample only) — {generated:%Y-%m-%d %H:%M} UTC", "",
         "4h signals, daily trend filter, BTC regime filter for alts, fees + slippage + real funding included. "
         "R = multiples of the initial risk; % columns assume 1% risk per trade compounded.", "",
         "| Pair | Symbol | From | Trades | Win % | Avg R | PF | Total R | Max DD (R) | Return @1% | Max DD @1% | Validated |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m = r["oos"]
        L.append(f"| {r['pair']} | {r['symbol']} | {r['from']:%Y-%m-%d} | {m['trades']} | {m['win_rate'] * 100:.0f} | "
                 f"{m['avg_r']:+.2f} | {m['profit_factor']:.2f} | {m['total_r']:+.1f} | {m['max_dd_r']:.1f} | "
                 f"{m['return_pct'] * 100:+.1f}% | {m['max_dd_pct'] * 100:.1f}% | {'yes' if r['validated'] else 'no'} |")
    m = portfolio
    L += ["", f"**All pairs combined** (every OOS trade, 1% risk each, chronological): {m['trades']} trades, "
          f"win rate {m['win_rate'] * 100:.0f}%, avg {m['avg_r']:+.3f}R, PF {m['profit_factor']:.2f}, "
          f"total {m['total_r']:+.1f}R, max DD {m['max_dd_r']:.1f}R, t-stat of mean R {m['t_stat']:.2f}.", "",
          _verdict(m), "",
          f"_{DISCLAIMER}_", ""]
    return "\n".join(L)


def _verdict(m: dict) -> str:
    if m["trades"] == 0:
        return "No out-of-sample trades."
    if m["avg_r"] <= 0:
        return "Honest reading: out-of-sample expectancy is not positive. Do not trade this blindly."
    sig = ("statistically significant (t ≥ 2)" if m["t_stat"] >= 2 else
           f"positive but NOT statistically significant (t = {m['t_stat']:.2f} < 2), so it could still be luck")
    return (f"Honest reading: the out-of-sample edge is {sig}. Profits come from a minority of large trend "
            "trades, so expect long flat or losing stretches. Pairs that are not validated deserve extra caution, "
            "and correlated positions across pairs make portfolio drawdowns deeper than single-pair ones.")


def to_json(obj) -> str:
    def enc(o):
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        if isinstance(o, pd.DataFrame):
            return o.to_dict("records")
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return str(o)

    def clean(x):
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        return x

    return json.dumps(clean(obj), default=enc, indent=2)
