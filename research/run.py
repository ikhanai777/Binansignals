"""Reproduce the win-rate research: python -m research.run {rules,ml,carry}

rules  scan the rule-based "high win-rate" families on the DEV period
ml     walk-forward gradient-boosting model (DEV and the untouched HOLDOUT)
carry  delta-neutral funding carry on real funding history
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from binansignals.config import PAIRS

from .lab import (DEV_END, barrier, load_funding, ml_features, summarize, tb_labels, universe)


def families(df, name):
    up = (df["bd"] == 1) & (df["btc_bd"] >= 0)
    dn = (df["bd"] == -1) & (df["btc_bd"] <= 0)
    if name == "trend_all":        # every bar in a daily uptrend/downtrend (baseline drift)
        L, S = up, dn
    elif name == "dip_rsi2":       # short-term oversold inside the daily trend
        L, S = up & (df["rsi2"] < 10), dn & (df["rsi2"] > 90)
    elif name == "ema_pull":       # pullback to EMA20 with EMA50 intact
        L = up & (df["low"] <= df["ema20"]) & (df["close"] > df["ema50"])
        S = dn & (df["high"] >= df["ema20"]) & (df["close"] < df["ema50"])
    elif name == "absorb":         # heavy aggressive selling absorbed (close high in the bar)
        L = up & (df["vol_z"] > 2.0) & (df["dr"] < -0.05) & (df["clv"] > 0.5)
        S = dn & (df["vol_z"] > 2.0) & (df["dr"] > 0.05) & (df["clv"] < 0.5)
    elif name == "sweep":          # stop-run through the 24-bar low that closes back inside
        L = up & (df["low"] < df["lo24"]) & (df["close"] > df["lo24"])
        S = dn & (df["high"] > df["hi24"]) & (df["close"] < df["hi24"])
    d = np.where(L, 1, np.where(S, -1, 0))
    return np.flatnonzero(d), d[d != 0]


def cmd_rules(tf="4h"):
    u = universe(tf)
    cfgs = [(0, 1.0, 2.0, 24), (0.5, 1.0, 2.0, 24), (0.5, 1.0, 3.0, 48), (0.5, 1.5, 2.0, 48),
            (0.5, 2.0, 2.0, 48), (1.0, 1.0, 2.0, 24), (0.5, 0.75, 2.0, 24)]
    rows = []
    for f in ("trend_all", "dip_rsi2", "ema_pull", "absorb", "sweep"):
        for lim, tp, sl, mb in cfgs:
            trs = []
            for sym, df in u.items():
                dev = df[df["open_time"] < DEV_END].reset_index(drop=True)
                idx, d = families(dev, f)
                trs.append(barrier(dev, idx, d, lim, tp, sl, mb))
            rows.append(summarize(pd.concat(trs), f"{f} limit{lim} tp{tp} sl{sl} max{mb}"))
    print(pd.DataFrame(rows).to_string())


def cmd_ml(tp=2.0, sl=2.0, H=18, thresholds=(0.6, 0.65)):
    from sklearn.ensemble import HistGradientBoostingClassifier
    u = universe("4h")
    funds = {p["symbol"]: load_funding(p["symbol"]) for p in PAIRS}
    frames = []
    for sym, df in u.items():
        X = ml_features(df, u["BTCUSDT"], funds[sym])
        X["yl"], X["ys"] = tb_labels(df, tp, sl, H)
        X["sym"], X["t"], X["i"] = sym, df["open_time"].values, np.arange(len(df))
        frames.append(X.iloc[250:])
    D = pd.concat(frames, ignore_index=True)
    D["t"] = pd.to_datetime(D["t"], utc=True)
    feats = [c for c in D.columns if c not in ("yl", "ys", "sym", "t", "i")]
    preds, cut, end = [], pd.Timestamp("2024-06-01", tz="UTC"), D["t"].max() + pd.Timedelta(hours=4)
    while cut < end:   # retrain every 30 days on labels that closed before the cutoff (purged)
        nxt = min(cut + pd.Timedelta(days=30), end)
        tr = D[D["t"] < cut - pd.Timedelta(hours=4 * (H + 1))]
        te = D[(D["t"] >= cut) & (D["t"] < nxt)].copy()
        for y, col in (("yl", "pl"), ("ys", "ps")):
            trn = tr.dropna(subset=[y])
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.04, max_leaf_nodes=15,
                                               min_samples_leaf=200, l2_regularization=1.0, random_state=0)
            te[col] = m.fit(trn[feats], trn[y]).predict_proba(te[feats])[:, 1]
        preds.append(te)
        cut = nxt
    P = pd.concat(preds)
    for period, g0 in (("DEV walk-forward", P[P["t"] < DEV_END]), ("HOLDOUT", P[P["t"] >= DEV_END])):
        for thr in thresholds:
            trs = []
            for sym, g in g0.groupby("sym"):
                sig = pd.concat([g[(g.pl > thr) & (g.pl > g.ps)].assign(d=1),
                                 g[(g.ps > thr) & (g.ps > g.pl)].assign(d=-1)]).sort_values("i")
                f = funds[sym]
                trs.append(barrier(u[sym], sig["i"].to_numpy(), sig["d"].to_numpy(), 0, tp, sl, H,
                                   fund=(f["time"].astype("int64").to_numpy(), f["rate"].to_numpy())))
            print(period, f"p>{thr}", summarize(pd.concat(trs)))


def cmd_carry():
    cost = 2 * (0.001 + 0.0005 + 0.0002)  # spot + perp taker fees and slippage, in and out
    rows = []
    for p in PAIRS:
        f = load_funding(p["symbol"]).set_index("time")["rate"]
        f = f[f.index >= pd.Timestamp("2023-10-01", tz="UTC")]
        mon = f.resample("ME").sum()
        yrs = (f.index[-1] - f.index[0]).days / 365
        ho = f[f.index >= DEV_END]
        rows.append({"pair": p["pair"], "apr_%": round((f.sum() - cost) / yrs * 100, 1),
                     "months_positive": f"{(mon > 0).mean():.0%}", "worst_month_%": round(mon.min() * 100, 2),
                     "holdout_apr_%": round(ho.sum() / ((f.index[-1] - DEV_END).days / 365) * 100, 1)})
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    {"rules": cmd_rules, "ml": cmd_ml, "carry": cmd_carry}[sys.argv[1]]()
