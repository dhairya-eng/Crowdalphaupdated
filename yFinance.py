import yfinance as yf
import numpy as np
from urllib.parse import urlparse

def _sanitize_symbol(s: str) -> str:
    s = (s or "").strip().upper().lstrip("$")
    return s

# def _is_motley(item: dict) -> bool:
#     pub = (item.get("publisher") or "").lower()
#     link = (item.get("link") or item.get("url") or "")
#     host = urlparse(link).hostname or ""
#     return ("motley" in pub) or ("fool.com" in host)

def get_yfinance_data(symbol: str):
    symbol = _sanitize_symbol(symbol)
    if not symbol or len(symbol) > 6 or " " in symbol:
        raise ValueError("Please provide a valid stock ticker (e.g., AAPL, MSFT, TSLA).")

    t = yf.Ticker(symbol)

    # History: try 2y first, fall back to 5d just to compute a current price
    hist = t.history(period="2y", interval="1d", auto_adjust=False)
    if hist.empty:
        hist = t.history(period="5d", interval="1d", auto_adjust=False)

    if hist.empty:
        raise ValueError(f"{symbol}: no price data found (symbol may be invalid or delisted).")

    current = float(hist["Close"].dropna().iloc[-1])

    # Currency: be defensive
    currency = "USD"
    try:
        fi = getattr(t, "fast_info", {}) or {}
        currency = fi.get("currency") or currency
    except Exception:
        pass
    if not currency:
        try:
            info = t.info or {}
            currency = info.get("currency", "USD")
        except Exception:
            currency = "USD"

    news = t.news or []
    # mf_news = [n for n in news if _is_motley(n)]

    return {
        "ticker": symbol,
        "current_price": current,
        "currency": currency,
        "history": hist.reset_index().to_dict(orient="records"),
        "news": news,
    }

