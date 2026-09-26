"""Gold on Binance: XAUUSDT (TradFi perp, since 2025-12), PAXGUSDT perp (since 2025-03) and, for a longer
history, PAXGUSDT spot 4h (gold-backed token, since 2020-08). Runs the same walk-forward 4h trend system
(no BTC filter) and compares it with buy-and-hold.

Run: python -m research.gold
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from binansignals.backtest import walk_forward
from binansignals.client import BinanceFutures, _klines_df, drop_unclosed
from binansignals.strategy import prepare


def paxg_spot_4h() -> pd.DataFrame:
    rows, cur = [], int(pd.Timestamp("2020-01-01", tz="UTC").value // 1_000_000)
    while True:
        r = requests.get("https://data-api.binance.vision/api/v3/klines", timeout=20,
                         params={"symbol": "PAXGUSDT", "interval": "4h", "startTime": cur, "limit": 1000}).json()
        if not r:
            break
        rows += r
        cur = r[-1][0] + 14_400_000
        if len(r) < 1000:
            break
    return drop_unclosed(_klines_df(rows))


def report(name, bars, funding, windows=((180, 60), (120, 30))):
    df = prepare(bars, "4h")
    print(f"{name}: {bars.open_time.iloc[0].date()} -> {bars.open_time.iloc[-1].date()}, "
          f"buy&hold {bars.close.iloc[-1] / bars.close.iloc[0] - 1:+.0%}")
    for tr, te in windows:
        wf = walk_forward(df, funding, "4h", tr, te)
        m, t = wf["oos_metrics"], wf["oos_trades"]
        eq = 500 * np.prod(1 + 0.01 * t["r"].to_numpy()) if len(t) else 500.0
        print(f"  WF {tr}/{te}: {m['trades']} OOS trades, win {m['win_rate']:.0%}, avg {m['avg_r']:+.2f}R, "
              f"PF {m['profit_factor']:.2f}, t {m['t_stat']:.2f}, $500 @1% risk -> ${eq:,.0f}")


if __name__ == "__main__":
    c = BinanceFutures()
    for sym in ("XAUUSDT", "PAXGUSDT"):
        bars = drop_unclosed(c.cached_history(sym, "4h", 600))
        report(f"{sym} perp", bars, c.funding_history(sym, int(bars["open_time"].iloc[0].value // 1_000_000)),
               ((120, 30), (90, 30)))
    report("PAXGUSDT spot (gold proxy)", paxg_spot_4h(), pd.DataFrame(columns=["time", "rate"]))
