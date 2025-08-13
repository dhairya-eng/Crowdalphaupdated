from __future__ import annotations
from typing import List
import pandas as pd

# IMPORTANT: keep stdout clean (no prints). yfinance progress disabled below.
from utils.yfinance_helper import Candle, HistoryFrame, QuoteYF

# Allowed combos that yfinance supports well
_ALLOWED_INTERVALS = {"1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"}

def _to_iso_z(ts) -> str:
    if hasattr(ts, "to_pydatetime"):
        return ts.to_pydatetime().replace(tzinfo=None).isoformat() + "Z"
    s = str(ts)
    return s if s.endswith("Z") else s + "Z"

def get_price_history(symbol: str, period: str = "12mo", interval: str = "1d", auto_adjust: bool = True) -> HistoryFrame:
    """
    Fetch OHLCV candles via yfinance. Returns normalized candles with ISO timestamps.
    """
    if interval not in _ALLOWED_INTERVALS:
        raise ValueError(f"Unsupported interval '{interval}'. Allowed: {sorted(_ALLOWED_INTERVALS)}")

    import yfinance as yf
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=auto_adjust,
    )
    if df is None or df.empty:
        return HistoryFrame(ticker=symbol, timeframe=interval, period=period, candles=[])

    df = df.reset_index()
    # yfinance uses column names like 'Open','High','Low','Close','Adj Close','Volume'
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("datetime") or "Date"

    # ensure numeric
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    candles: List[Candle] = []
    for _, r in df.iterrows():
        if pd.isna(r.get("Close")):
            continue
        t_iso = _to_iso_z(r[date_col])
        candles.append(
            Candle(
                t=t_iso,
                o=float(r["Open"]),
                h=float(r["High"]),
                l=float(r["Low"]),
                c=float(r["Close"]),
                v=float(r.get("Volume", 0.0) or 0.0),
            )
        )
    return HistoryFrame(ticker=symbol, timeframe=interval, period=period, candles=candles)

def get_quote(symbol: str) -> QuoteYF:
    """
    Lightweight snapshot using yfinance.Ticker.fast_info / info fallbacks.
    """
    import yfinance as yf
    t = yf.Ticker(symbol)
    current = None
    prev = None
    cur = None
    opn = None
    hi = None
    lo = None

    # Prefer fast_info when available
    try:
        fi = t.fast_info
        current = float(fi.get("last_price")) if fi.get("last_price") is not None else None
        prev = float(fi.get("previous_close")) if fi.get("previous_close") is not None else None
        cur = fi.get("currency")
        opn = float(fi.get("open")) if fi.get("open") is not None else None
        hi = float(fi.get("day_high")) if fi.get("day_high") is not None else None
        lo = float(fi.get("day_low")) if fi.get("day_low") is not None else None
    except Exception:
        pass

    # Fallback to .info if needed
    if current is None or prev is None:
        try:
            info = t.info or {}
            current = float(info.get("currentPrice")) if info.get("currentPrice") is not None else current
            prev = float(info.get("previousClose")) if info.get("previousClose") is not None else prev
            cur = info.get("currency") or cur
            opn = float(info.get("open")) if info.get("open") is not None else opn
            hi = float(info.get("dayHigh")) if info.get("dayHigh") is not None else hi
            lo = float(info.get("dayLow")) if info.get("dayLow") is not None else lo
        except Exception:
            pass

    return QuoteYF(
        ticker=symbol,
        current_price=current,
        previous_close=prev,
        currency=cur,
        open_price=opn,
        day_high=hi,
        day_low=lo,
    )
