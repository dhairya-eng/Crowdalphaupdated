# crowdalpha/adapter.py
from __future__ import annotations
import math, datetime as dt
from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field
from langsmith import traceable

# ---- Phase-1 modules (yours) ----
from yFinance import get_yfinance_data
from indicators import get_ta_indicators
from redditFetch import get_reddit_posts_raw
from finnhub import get_company_news, get_stock_price

# =========================
# Shared Schemas (1 place)
# =========================
class Candle(BaseModel):
    t: str
    o: float
    h: float
    l: float
    c: float
    v: float

class PricesOut(BaseModel):
    source: Literal["yfinance","finnhub"]
    ticker: str
    timeframe: Literal["1d"]
    candles: List[Candle]
    coverage_pct: float
    last_bar_time: Optional[str] = None
    note: Optional[str] = None

class TAFeatures(BaseModel):
    RSI: Optional[float] = None
    MACD: Optional[Dict[str, float]] = None  # {"macd","signal","hist"}
    ATR: Optional[float] = None
    ADX: Optional[float] = None
    SMA20: Optional[float] = None
    SMA50: Optional[float] = None
    EMA20: Optional[float] = None
    BBANDS: Optional[Dict[str, float]] = None
    trend_score: float = 0.0
    vol_score: float = 0.0

class RedditOut(BaseModel):
    source: Literal["reddit"]
    count: int
    sentiment: Dict[str, float]          # {"score":-1..1, "confidence":0..1}
    topics: List[str]
    top_quotes: List[str]
    event_flags: Dict[str, bool]         # {"earnings":bool,"halt":bool,"fraud":bool}

class NewsItem(BaseModel):
    source: Literal["finnhub"]
    headline: str
    summary: Optional[str] = None
    url: Optional[str] = None
    datetime: Optional[str] = None
    category: Optional[str] = None

class FinnhubNewsOut(BaseModel):
    source: Literal["finnhub"]
    items: List[NewsItem]

class QuoteOut(BaseModel):
    source: Literal["finnhub"]
    current_price: float
    change: float
    percent_change: float
    high_price: float
    low_price: float
    open_price: float
    previous_close: float
    timestamp: str

# =========================
# Helpers
# =========================
def _coverage_from_len(n: int, target: int = 60) -> float:
    return min(1.0, n / float(target)) if target > 0 else 0.0

_KEYWORDS = {
    "earnings": ["earnings", "guidance", "eps", "revenue", "q1", "q2", "q3", "q4", "forecast"],
    "halt": ["trading halt", "halted"],
    "fraud": ["fraud", "scandal", "sec probe", "investigation"],
}

def _detect_topics_from_posts(posts: List[dict]) -> List[str]:
    text = " ".join([(p.get("title") or "") + " " + (p.get("selftext") or "") for p in posts]).lower()
    topics = []
    for k, kws in _KEYWORDS.items():
        if any(kw in text for kw in kws):
            topics.append(k)
    return topics or ["general"]

# =========================
# Adapters (LangSmith-traced)
# =========================

@traceable(name="tool:get_prices_yf", run_type="tool")
def get_prices_yf(symbol: str, timeframe: str = "1d") -> PricesOut:
    """
    Wraps yFinance.get_yfinance_data to normalized OHLCV candles.
    """
    data = get_yfinance_data(symbol)  # {ticker, current_price, currency, history:list[dict], news:list}
    ticker = data["ticker"]
    rows = data["history"]  # each dict has Date, Open, High, Low, Close, Volume
    candles: List[Candle] = []
    for r in rows:
        ts = r.get("Date")
        # yfinance reset_index gives pandas Timestamp or string; normalize to ISO Z
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime().replace(tzinfo=None).isoformat() + "Z"
        else:
            ts = str(ts)
            if not ts.endswith("Z"):
                ts += "Z"
        candles.append(Candle(
            t=ts,
            o=float(r["Open"]), h=float(r["High"]),
            l=float(r["Low"]), c=float(r["Close"]),
            v=float(r.get("Volume", 0.0))
        ))
    return PricesOut(
        source="yfinance",
        ticker=ticker,
        timeframe="1d",
        candles=candles,
        coverage_pct=_coverage_from_len(len(candles), target=60),
        last_bar_time=candles[-1].t if candles else None,
        note=None
    )

@traceable(name="tool:compute_ta_from_phase1", run_type="tool")
def compute_ta_from_phase1(symbol: str, period: str = "12mo", interval: str = "1d") -> TAFeatures:
    """
    Wraps indicators.get_ta_indicators (last 5 rows) to TAFeatures.
    Derives trend_score and vol_score.
    """
    df = get_ta_indicators(symbol, period=period, interval=interval)
    if df is None or df.empty:
        return TAFeatures()
    row = df.iloc[-1]
    # Base indicators
    rsi = float(row["RSI"])
    sma20 = float(row["SMA_20"])
    ema20 = float(row["EMA_20"])
    macd = float(row["MACD"])
    macdsig = float(row["MACD_signal"])
    macdhist = macd - macdsig
    # Derived scores
    trend_score = max(-1.0, min(1.0, ((ema20 - sma20) / max(1e-6, sma20)) * 5))
    try:
        closes = df["Close"].astype(float).values
        price = float(closes[-1])
        vol_score = float(min(1.0, max(0.0, (closes[-5:].std(ddof=1) / max(1e-6, price)) * 20)))
    except Exception:
        vol_score = 0.0
    return TAFeatures(
        RSI=rsi,
        SMA20=sma20,
        EMA20=ema20,
        MACD={"macd": macd, "signal": macdsig, "hist": macdhist},
        ATR=None, ADX=None, SMA50=None, BBANDS=None,
        trend_score=trend_score, vol_score=vol_score
    )

@traceable(name="tool:get_reddit_signals", run_type="tool")
def get_reddit_signals(symbol: str,
                       limit_per_sub: int = 60,
                       time_filter: str = "week") -> RedditOut:
    """
    Wraps redditFetch.get_reddit_posts_raw and summarizes sentiment + topics.
    """
    posts = get_reddit_posts_raw(symbol, limit_per_sub=limit_per_sub, time_filter=time_filter, with_sentiment=True)
    n = len(posts)
    if n == 0:
        return RedditOut(
            source="reddit", count=0,
            sentiment={"score": 0.0, "confidence": 0.0},
            topics=[], top_quotes=[],
            event_flags={"earnings": False, "halt": False, "fraud": False}
        )
    comp = [p.get("sent_compound") for p in posts if p.get("sent_compound") is not None]
    score = sum(comp)/len(comp) if comp else 0.0
    conf = 1.0 if n >= 30 else (0.7 if n >= 10 else (0.4 if n >= 5 else 0.0))
    topics = _detect_topics_from_posts(posts)
    flags = {k: (k in topics) for k in ["earnings","halt","fraud"]}
    top_quotes = [p["title"] for p in sorted(posts, key=lambda x: x.get("score", 0), reverse=True)[:3]]
    return RedditOut(
        source="reddit", count=n,
        sentiment={"score": float(score), "confidence": float(conf)},
        topics=topics, top_quotes=top_quotes, event_flags=flags
    )

@traceable(name="tool:get_company_news_finnhub", run_type="tool")
def get_company_news_finnhub(symbol: str, from_date: str, to_date: str) -> FinnhubNewsOut:
    """
    Wraps finnhub.get_company_news to a normalized news list.
    """
    articles = get_company_news(symbol=symbol, from_date=from_date, to_date=to_date)  # list[NewsArticleFinnHub]
    items: List[NewsItem] = []
    for a in articles:
        obj = getattr(a, "__dict__", a)
        headline = getattr(a, "headline", None) or obj.get("headline") or obj.get("title")
        summary  = getattr(a, "summary", None)  or obj.get("summary")
        url      = getattr(a, "url", None)      or obj.get("url")
        dt_str   = str(getattr(a, "datetime", None) or obj.get("datetime"))
        category = getattr(a, "category", None) or obj.get("category")
        items.append(NewsItem(source="finnhub", headline=headline, summary=summary, url=url, datetime=dt_str, category=category))
    return FinnhubNewsOut(source="finnhub", items=items)

@traceable(name="tool:get_stock_quote_finnhub", run_type="tool")
def get_stock_quote_finnhub(symbol: str) -> QuoteOut:
    """
    Wraps finnhub.get_stock_price to a uniform quote shape (optional).
    """
    q = get_stock_price(symbol)
    obj = getattr(q, "__dict__", q)
    ts = obj.get("timestamp")
    if hasattr(ts, "isoformat"):
        ts = ts.isoformat()
    return QuoteOut(
        source="finnhub",
        current_price=float(obj["current_price"]),
        change=float(obj["change"]),
        percent_change=float(obj["percent_change"]),
        high_price=float(obj["high_price"]),
        low_price=float(obj["low_price"]),
        open_price=float(obj["open_price"]),
        previous_close=float(obj["previous_close"]),
        timestamp=str(ts),
    )
