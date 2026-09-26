"""Thin client for Binance USD-M futures public market data (no API key needed)."""
from __future__ import annotations

import gzip
import os
import time
from pathlib import Path

import pandas as pd
import requests

from .config import FAPI_HOSTS

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
INTERVAL_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000,
    "4h": 14_400_000, "1d": 86_400_000,
}
CACHE_DIR = Path(os.environ.get("BINANSIGNALS_CACHE", Path(__file__).resolve().parent.parent / "data"))


class BinanceFutures:
    def __init__(self, hosts=None, timeout=20):
        self.hosts = list(hosts or FAPI_HOSTS)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "binansignals/1.0"

    def get(self, path: str, params: dict | None = None):
        last_err = None
        for host in list(self.hosts):
            for attempt in range(4):
                try:
                    r = self.session.get(host + path, params=params, timeout=self.timeout)
                except requests.RequestException as e:
                    last_err = e
                    time.sleep(2 ** attempt)
                    continue
                if r.status_code in (403, 451):
                    # Geo-blocked host: drop it and fall through to the next one.
                    last_err = RuntimeError(f"{host} returned {r.status_code}")
                    if len(self.hosts) > 1:
                        self.hosts.remove(host)
                    break
                if r.status_code in (418, 429) or r.status_code >= 500:
                    last_err = RuntimeError(f"{host}{path} -> {r.status_code}")
                    time.sleep(int(r.headers.get("Retry-After", 2 ** (attempt + 1))))
                    continue
                if r.status_code != 200:
                    raise RuntimeError(f"{host}{path} {params} -> {r.status_code}: {r.text[:200]}")
                self._throttle(r)
                return r.json()
        raise RuntimeError(f"All Binance hosts failed for {path}: {last_err}")

    @staticmethod
    def _throttle(r):
        """Stay well under Binance's 2400 request-weight per minute limit."""
        used = int(r.headers.get("X-MBX-USED-WEIGHT-1M", 0) or 0)
        if used > 1800:
            time.sleep(61 - time.time() % 60)

    # ---- market data -------------------------------------------------------
    def klines(self, symbol: str, interval: str, limit: int = 1500,
               start_ms: int | None = None, end_ms: int | None = None) -> pd.DataFrame:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms
        return _klines_df(self.get("/fapi/v1/klines", params))

    def klines_range(self, symbol: str, interval: str, start_ms: int, end_ms: int | None = None) -> pd.DataFrame:
        """Paginate klines from start_ms to end_ms (or now)."""
        step = INTERVAL_MS[interval]
        end_ms = end_ms or int(time.time() * 1000)
        frames, cur = [], start_ms
        while cur < end_ms:
            df = self.klines(symbol, interval, 1500, start_ms=cur, end_ms=end_ms)
            if df.empty:
                break
            frames.append(df)
            nxt = int(df["open_time"].iloc[-1].value // 1_000_000) + step
            if nxt <= cur:
                break
            cur = nxt
            time.sleep(0.15)
        if not frames:
            return _klines_df([])
        out = pd.concat(frames).drop_duplicates("open_time").sort_values("open_time")
        return out.reset_index(drop=True)

    def cached_history(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """Full history for backtests, cached on disk and topped up incrementally."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / f"{symbol}_{interval}.csv.gz"
        now = int(time.time() * 1000)
        start = now - days * 86_400_000
        df = None
        if path.exists():
            with gzip.open(path, "rt") as f:
                df = pd.read_csv(f, parse_dates=["open_time", "close_time"])
            for c in ("open_time", "close_time"):
                df[c] = _ns(pd.to_datetime(df[c], utc=True))
            first = int(df["open_time"].iloc[0].value // 1_000_000)
            if first > start + INTERVAL_MS[interval] * 24 and first > self._listing_ms(symbol, interval, start):
                df = None  # cache does not reach back far enough -> refetch
        if df is None:
            df = self.klines_range(symbol, interval, start)
        else:
            last = int(df["open_time"].iloc[-1].value // 1_000_000)
            new = self.klines_range(symbol, interval, last)
            df = pd.concat([df, new]).drop_duplicates("open_time", keep="last").sort_values("open_time")
        df = df.reset_index(drop=True)
        with gzip.open(path, "wt") as f:
            df.to_csv(f, index=False)
        return df[df["open_time"] >= pd.Timestamp(start, unit="ms", tz="UTC")].reset_index(drop=True)

    def _listing_ms(self, symbol, interval, start):
        first = self.klines(symbol, interval, 1, start_ms=start)
        return int(first["open_time"].iloc[0].value // 1_000_000) if not first.empty else start

    def funding_history(self, symbol: str, start_ms: int) -> pd.DataFrame:
        rows, cur = [], start_ms
        while True:
            batch = self.get("/fapi/v1/fundingRate", {"symbol": symbol, "startTime": cur, "limit": 1000})
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < 1000:
                break
            cur = int(batch[-1]["fundingTime"]) + 1
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["time", "rate"])
        return pd.DataFrame({
            "time": _ns(pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)),
            "rate": df["fundingRate"].astype(float),
        })

    def agg_trades(self, symbol: str, max_trades: int = 20_000, window_ms: int | None = None) -> pd.DataFrame:
        """Most recent aggregated trades (tick-level prints for the footprint).

        Walks backwards from the latest trade id until max_trades or window_ms is covered.
        """
        latest = self.get("/fapi/v1/aggTrades", {"symbol": symbol, "limit": 1000})
        rows = list(latest)
        if not rows:
            return pd.DataFrame(columns=["time", "price", "qty", "is_buyer_maker"])
        newest_t = rows[-1]["T"]
        while len(rows) < max_trades:
            first_id = rows[0]["a"]
            if first_id <= 0:
                break
            from_id = max(0, first_id - 1000)
            batch = self.get("/fapi/v1/aggTrades", {"symbol": symbol, "fromId": from_id, "limit": 1000})
            batch = [b for b in batch if b["a"] < first_id]
            if not batch:
                break
            rows = batch + rows
            if window_ms and newest_t - rows[0]["T"] >= window_ms:
                break
            time.sleep(0.05)
        df = pd.DataFrame(rows)
        return pd.DataFrame({
            "time": _ns(pd.to_datetime(df["T"], unit="ms", utc=True)),
            "price": df["p"].astype(float),
            "qty": df["q"].astype(float),
            "is_buyer_maker": df["m"].astype(bool),
        })

    def depth(self, symbol: str, limit: int = 500) -> dict:
        d = self.get("/fapi/v1/depth", {"symbol": symbol, "limit": limit})
        return {
            "bids": [(float(p), float(q)) for p, q in d["bids"]],
            "asks": [(float(p), float(q)) for p, q in d["asks"]],
        }

    def premium_index(self, symbol: str) -> dict:
        return self.get("/fapi/v1/premiumIndex", {"symbol": symbol})

    def open_interest_hist(self, symbol: str, period: str = "1h", limit: int = 48) -> pd.DataFrame:
        d = self.get("/futures/data/openInterestHist", {"symbol": symbol, "period": period, "limit": limit})
        df = pd.DataFrame(d)
        if df.empty:
            return df
        df["time"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        df["oi"] = df["sumOpenInterest"].astype(float)
        df["oi_value"] = df["sumOpenInterestValue"].astype(float)
        return df[["time", "oi", "oi_value"]]

    def long_short_ratio(self, symbol: str, period: str = "1h", limit: int = 24) -> pd.DataFrame:
        d = self.get("/futures/data/globalLongShortAccountRatio", {"symbol": symbol, "period": period, "limit": limit})
        df = pd.DataFrame(d)
        if df.empty:
            return df
        df["time"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        df["ratio"] = df["longShortRatio"].astype(float)
        df["long_pct"] = df["longAccount"].astype(float)
        return df[["time", "ratio", "long_pct"]]

    def top_trader_ratio(self, symbol: str, period: str = "1h", limit: int = 24) -> pd.DataFrame:
        d = self.get("/futures/data/topLongShortPositionRatio", {"symbol": symbol, "period": period, "limit": limit})
        df = pd.DataFrame(d)
        if df.empty:
            return df
        df["time"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        df["ratio"] = df["longShortRatio"].astype(float)
        return df[["time", "ratio"]]

    def exchange_symbols(self) -> dict:
        info = self.get("/fapi/v1/exchangeInfo")
        return {s["symbol"]: s for s in info["symbols"]}


def _klines_df(raw) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=KLINE_COLS)
    if df.empty:
        return df.drop(columns=["ignore"])
    for c in ("open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_volume", "taker_buy_quote_volume"):
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype("int64")
    df["open_time"] = _ns(pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True))
    df["close_time"] = _ns(pd.to_datetime(df["close_time"].astype("int64") + 1, unit="ms", utc=True))
    return df.drop(columns=["ignore"])


def _ns(s: pd.Series) -> pd.Series:
    """Normalise timestamps to ns resolution so int64 views are comparable everywhere."""
    return s.astype("datetime64[ns, UTC]")


def drop_unclosed(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the still-forming candle so signals only use closed bars."""
    now = pd.Timestamp.now(tz="UTC")
    return df[df["close_time"] <= now].reset_index(drop=True)
