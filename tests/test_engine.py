"""Offline tests on recorded real Binance data (tests/fixtures)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from binansignals.backtest import Arrays, metrics, run_params, simulate_trade, to_arrays
from binansignals.indicators import atr, htf_trend, resample_ohlcv, rsi
from binansignals.levels import key_levels, volume_profile
from binansignals.orderflow import footprint
from binansignals.strategy import entries, param_sets, prepare

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def bars():
    df = pd.read_csv(FIX / "btcusdt_4h.csv")
    for c in ("open_time", "close_time"):
        df[c] = pd.to_datetime(df[c], utc=True).astype("datetime64[ns, UTC]")
    return df


@pytest.fixture(scope="module")
def feats(bars):
    return prepare(bars, "4h")


def test_indicator_ranges(bars):
    r = rsi(bars["close"]).dropna()
    assert r.between(0, 100).all()
    assert (atr(bars).dropna() > 0).all()


def test_resample_only_complete_bars(bars):
    d1 = resample_ohlcv(bars, "1D")
    assert (d1["close_time"] - d1["open_time"] == pd.Timedelta("1D")).all()
    assert d1["close_time"].iloc[-1] <= bars["close_time"].iloc[-1]
    first = d1.iloc[1]
    day = bars[(bars["open_time"] >= first["open_time"]) & (bars["open_time"] < first["close_time"])]
    assert first["high"] == day["high"].max() and first["volume"] == pytest.approx(day["volume"].sum())


def test_htf_state_has_no_lookahead(bars, feats):
    d1 = htf_trend(resample_ohlcv(bars, "1D"), 20, 50).dropna(subset=["htf_ema_fast"])
    for _, row in feats.dropna(subset=["htf_ema_fast"]).sample(50, random_state=1).iterrows():
        visible = d1[d1["close_time"] <= row["close_time"]].iloc[-1]
        assert row["htf_ema_fast"] == pytest.approx(visible["htf_ema_fast"])


def test_stops_are_on_the_protective_side(feats):
    for p in param_sets("4h"):
        sig = entries(feats, p)
        long_, short = sig["dir"] == 1, sig["dir"] == -1
        assert (sig.loc[long_, "stop"] < feats.loc[long_, "close"]).all()
        assert (sig.loc[short, "stop"] > feats.loc[short, "close"]).all()


def test_backtest_trades_do_not_overlap(feats):
    A = to_arrays(feats, pd.DataFrame(columns=["time", "rate"]))
    tr = run_params(feats, A, param_sets("4h")[0])
    assert not tr.empty
    assert (tr["entry_i"].iloc[1:].to_numpy() > tr["exit_i"].iloc[:-1].to_numpy()).all()
    assert np.isfinite(tr["r"]).all()
    m = metrics(tr)
    assert m["trades"] == len(tr) and 0 <= m["win_rate"] <= 1


def test_stop_fills_before_target_on_same_bar():
    # Fill-logic check on a hand-built 3-bar path: bar 1 touches both the stop and TP1.
    A = Arrays(t=np.arange(3, dtype="int64") * 3_600_000_000_000, o=np.array([100.0, 100, 100]),
               h=np.array([100.0, 103, 100]), l=np.array([100.0, 97, 100]), c=np.array([100.0, 100, 100]),
               atr=np.array([2.0, 2, 2]), f_t=np.array([], dtype="int64"), f_r=np.array([]))
    p = {"tp1_r": 1.0, "trail_atr": 3, "max_bars": 10}
    tr = simulate_trade(A, 0, 1, 98.0, p)
    assert tr["reason"] == "stop" and tr["r"] < -1.0  # full loss plus costs


def test_footprint_delta_matches_raw_trades():
    t = pd.read_csv(FIX / "ethusdt_aggtrades.csv")
    t["time"] = pd.to_datetime(t["time"], utc=True, format="ISO8601")
    fp = footprint(t, bar="1min")
    buys = t.loc[~t["is_buyer_maker"], "qty"].sum()
    sells = t.loc[t["is_buyer_maker"], "qty"].sum()
    assert fp["window_delta"] == pytest.approx(buys - sells)
    assert fp["window_volume"] == pytest.approx(t["qty"].sum())
    b = fp["bars"].iloc[-1]
    assert sum(a + s for a, s in b["levels"].values()) == pytest.approx(b["volume"])


def test_volume_profile_conserves_volume(bars):
    vp = volume_profile(bars.tail(200))
    assert vp["total_volume"] == pytest.approx(bars.tail(200)["volume"].sum())
    assert vp["val"] <= vp["poc"] <= vp["vah"]


def test_levels_bracket_price(bars, feats):
    price = float(bars["close"].iloc[-1])
    d1 = resample_ohlcv(bars, "1D")
    lv = key_levels(bars, d1, float(feats["atr"].iloc[-1]), price)
    assert all(z["price"] < price for z in lv["support"])
    assert all(z["price"] > price for z in lv["resistance"])
