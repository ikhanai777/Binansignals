"""CLI: python -m binansignals {signals,backtest} [options]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .config import PAIRS, RISK_PER_TRADE

OUT = Path(__file__).resolve().parent.parent / "reports"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="binansignals", description="Binance USD-M futures signals for 12 pairs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("signals", help="generate live signals")
    s.add_argument("--pairs", help="comma-separated subset, e.g. BTCUSDT,ETHUSDT")
    s.add_argument("--equity", type=float, default=1000.0, help="account size in USDT for position sizing")
    s.add_argument("--risk", type=float, default=RISK_PER_TRADE, help="fraction of equity risked per trade")
    s.add_argument("--out", type=Path, default=OUT)
    b = sub.add_parser("backtest", help="walk-forward backtest report")
    b.add_argument("--pairs")
    b.add_argument("--out", type=Path, default=OUT)
    k = sub.add_parser("carry", help="live funding-carry monitor (long spot + short perp)")
    k.add_argument("--pairs")
    k.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    pairs = a.pairs.split(",") if a.pairs else None
    log = lambda m: print(m, file=sys.stderr)  # noqa: E731

    if a.cmd == "signals":
        from .report import signals_markdown, to_json
        from .signals import run
        res = run(pairs, a.equity, a.risk, log)
        md = signals_markdown(res)
        (a.out / "signals.md").write_text(md)
        (a.out / "signals.json").write_text(to_json(res))
        print(md)
        log(f"\nWrote {a.out / 'signals.md'} and {a.out / 'signals.json'}")
    elif a.cmd == "carry":
        from .carry import carry_markdown, carry_table
        md = carry_markdown(carry_table(pairs))
        (a.out / "carry.md").write_text(md)
        print(md)
    else:
        from .backtest import metrics, walk_forward
        from .client import BinanceFutures, drop_unclosed
        from .config import BACKTEST_DAYS, MIN_OOS_PROFIT_FACTOR, MIN_OOS_TRADES
        from .report import backtest_markdown
        from .signals import TF
        from .strategy import prepare
        c = BinanceFutures()
        btc = drop_unclosed(c.cached_history("BTCUSDT", TF, BACKTEST_DAYS))
        rows, trades = [], []
        for cfg in PAIRS:
            if pairs and cfg["pair"] not in pairs and cfg["symbol"] not in pairs:
                continue
            sym = cfg["symbol"]
            log(f"Backtesting {sym} ...")
            bars = drop_unclosed(c.cached_history(sym, TF, BACKTEST_DAYS))
            fund = c.funding_history(sym, int(bars["open_time"].iloc[0].value // 1_000_000))
            wf = walk_forward(prepare(bars, TF, None if sym == "BTCUSDT" else btc), fund, TF)
            m = wf["oos_metrics"]
            rows.append({"pair": cfg["pair"], "symbol": sym, "from": bars["open_time"].iloc[0], "oos": m,
                         "validated": m["trades"] >= MIN_OOS_TRADES and m["profit_factor"] >= MIN_OOS_PROFIT_FACTOR})
            if not wf["oos_trades"].empty:
                trades.append(wf["oos_trades"].assign(symbol=sym))
        allt = pd.concat(trades).sort_values("exit_time") if trades else pd.DataFrame()
        md = backtest_markdown(rows, metrics(allt), pd.Timestamp.now(tz="UTC"))
        (a.out / "backtest.md").write_text(md)
        if not allt.empty:
            allt.drop(columns=["entry_i", "exit_i"]).to_csv(a.out / "oos_trades.csv", index=False)
        print(md)


if __name__ == "__main__":
    main()
