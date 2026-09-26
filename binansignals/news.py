"""Live news / sentiment context: crypto RSS headlines, Fear & Greed, Binance delistings.

News is used as a risk filter, not as a trade trigger: headline history cannot be reliably
backtested, so it only lowers confidence / flags a pair, it never creates a signal.
"""
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import pandas as pd
import requests

FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
    "The Block": "https://www.theblock.co/rss.xml",
}
BULL = ["surge", "surges", "rally", "rallies", "soar", "soars", "jump", "jumps", "gain", "gains", "climb",
        "record high", "all-time high", "breakout", "approve", "approved", "approval", "inflow", "inflows",
        "adoption", "partnership", "launches", "upgrade", "bullish", "accumulate", "accumulation", "buyback",
        "rebound", "recover", "outperform", "etf launch", "rate cut", "cuts rates", "whales buy"]
BEAR = ["plunge", "plunges", "crash", "crashes", "tumble", "tumbles", "slump", "drop", "drops", "fall", "falls",
        "sell-off", "selloff", "hack", "hacked", "exploit", "drained", "lawsuit", "sues", "charged", "fraud",
        "outflow", "outflows", "ban", "bans", "liquidation", "liquidations", "delist", "delisting", "bearish",
        "reject", "rejected", "rate hike", "hikes rates", "investigation", "subpoena", "outage", "halt", "dump"]
HIGH_IMPACT = ["fomc", "federal reserve", "powell", "cpi", "inflation data", "jobs report", "nonfarm", "payrolls",
               "tariff", "sec ", "etf", "hack", "exploit", "delist", "halt", "emergency", "bankrupt", "insolvency"]
MARKET_WORDS = ["crypto market", "bitcoin", "fed", "fomc", "cpi", "inflation", "tariff", "etf", "sec",
                "stablecoin", "treasury", "rate"]


def fetch_headlines(hours: int = 48, timeout: int = 15) -> list[dict]:
    items, cutoff = [], pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    for src, url in FEEDS.items():
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 binansignals"})
            r.raise_for_status()
            root = ET.fromstring(r.content)
        except Exception:
            continue
        for it in root.iter("item"):
            title = html.unescape((it.findtext("title") or "").strip())
            desc = re.sub(r"<[^>]+>", " ", html.unescape(it.findtext("description") or ""))[:400]
            try:
                ts = pd.Timestamp(parsedate_to_datetime(it.findtext("pubDate"))).tz_convert("UTC")
            except Exception:
                continue
            if ts >= cutoff and title:
                items.append({"source": src, "title": title, "summary": desc.strip(), "time": ts,
                              "link": (it.findtext("link") or "").strip()})
    seen, out = set(), []
    for i in sorted(items, key=lambda x: x["time"], reverse=True):
        k = i["title"].lower()
        if k not in seen:
            seen.add(k)
            out.append(i)
    return out


def _count(text: str, words: list[str]) -> int:
    return sum(len(re.findall(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", text)) for w in words)


def score_text(text: str) -> float:
    t = text.lower()
    b, s = _count(t, BULL), _count(t, BEAR)
    return 0.0 if b + s == 0 else (b - s) / (b + s)


def matches_coin(item: dict, pair_cfg: dict) -> bool:
    """Names match case-insensitively; short tickers (DOT, LINK, SOL, ADA, POL...) only in CAPS."""
    text = item["title"] + " " + item["summary"]
    for kw in pair_cfg["keywords"]:
        if len(kw) <= 4:
            if re.search(r"(?<![A-Za-z$])\$?" + re.escape(kw.upper()) + r"(?![A-Za-z])", text):
                return True
        elif re.search(r"(?i)(?<![a-z])" + re.escape(kw) + r"(?![a-z])", text):
            return True
    return False


def coin_news(headlines: list[dict], pair_cfg: dict) -> dict:
    rel = [h for h in headlines if matches_coin(h, pair_cfg)]
    scores = [score_text(h["title"] + " " + h["summary"]) for h in rel]
    flags = [h["title"] for h in rel if _count((h["title"] + " ").lower(), HIGH_IMPACT)]
    return {
        "count": len(rel), "sentiment": sum(scores) / len(scores) if scores else 0.0,
        "headlines": [{"title": h["title"], "source": h["source"], "time": h["time"], "link": h["link"],
                       "score": round(s, 2)} for h, s in list(zip(rel, scores))[:5]],
        "high_impact": flags[:3],
    }


def market_news(headlines: list[dict]) -> dict:
    rel = [h for h in headlines if _count(h["title"].lower(), MARKET_WORDS)]
    scores = [score_text(h["title"] + " " + h["summary"]) for h in rel]
    return {
        "count": len(rel), "sentiment": sum(scores) / len(scores) if scores else 0.0,
        "high_impact": [h["title"] for h in rel if _count((h["title"] + " ").lower(), HIGH_IMPACT)][:6],
    }


def fear_greed() -> dict | None:
    try:
        d = requests.get("https://api.alternative.me/fng/?limit=8", timeout=15).json()["data"]
        return {"value": int(d[0]["value"]), "label": d[0]["value_classification"],
                "week_ago": int(d[-1]["value"])}
    except Exception:
        return None


def binance_delistings(symbols: list[str]) -> dict:
    """Recent Binance delisting announcements mentioning any of our base assets."""
    url = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
    try:
        d = requests.get(url, params={"type": 1, "catalogId": 161, "pageNo": 1, "pageSize": 20},
                         timeout=15, headers={"User-Agent": "Mozilla/5.0"}).json()
        arts = [a for c in d["data"]["catalogs"] for a in c.get("articles", [])]
    except Exception:
        return {}
    hits = {}
    for s in symbols:
        base = s.replace("USDT", "")
        for a in arts:
            t = a.get("title", "")
            # "XYZ/BTC pairs" is a spot pair removal, not a delisting of BTC itself.
            if "delist" in t.lower() and re.search(r"(?<![A-Z/])" + base + r"(?![A-Z/])", t):
                hits.setdefault(s, []).append(a["title"])
    return hits
