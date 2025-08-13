# from __future__ import annotations
# from typing import List
# import pandas as pd

# # IMPORTANT: keep stdout clean (no prints). yfinance progress disabled below.
# from utils.yfinance_helper import Candle, HistoryFrame, QuoteYF

# #---normalize the input---#
# # --- add near the top of the file (after imports) ---
# def _norm_str(x, default=""):
#     # Accept tuple/list/singletons and coerce to str
#     if isinstance(x, (list, tuple)):
#         x = x[0] if len(x) > 0 else default
#     if x is None:
#         x = default
#     return str(x)

# def _norm_bool(x, default=True):
#     if isinstance(x, (list, tuple)):
#         x = x[0] if len(x) > 0 else default
#     if isinstance(x, str):
#         xl = x.strip().lower()
#         if xl in ("true","1","yes","y"): return True
#         if xl in ("false","0","no","n"): return False
#         return default
#     return bool(x) if x is not None else default


# # Allowed combos that yfinance supports well
# _ALLOWED_INTERVALS = {"1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"}

# def _to_iso_z(ts) -> str:
#     if hasattr(ts, "to_pydatetime"):
#         return ts.to_pydatetime().replace(tzinfo=None).isoformat() + "Z"
#     s = str(ts)
#     return s if s.endswith("Z") else s + "Z"

# def get_price_history(symbol: str, period: str = "12mo", interval: str = "1d", auto_adjust: bool = True) -> HistoryFrame:
#     """
#     Fetch OHLCV candles via yfinance. Returns normalized candles with ISO timestamps.
#     """
#     symbol = _norm_str(symbol).upper().strip()
#     period = _norm_str(period, "12mo").lower().strip()
#     interval = _norm_str(interval, "1d").lower().strip()
#     auto_adjust = _norm_bool(auto_adjust, True)
#     if interval not in _ALLOWED_INTERVALS:
#         raise ValueError(f"Unsupported interval '{interval}'. Allowed: {sorted(_ALLOWED_INTERVALS)}")

#     import yfinance as yf
#     df = yf.download(
#         symbol,
#         period=period,
#         interval=interval,
#         progress=False,
#         auto_adjust=auto_adjust,
#     )
#     if df is None or df.empty:
#         return HistoryFrame(ticker=symbol, timeframe=interval, period=period, candles=[])

#     df = df.reset_index()
#     # yfinance uses column names like 'Open','High','Low','Close','Adj Close','Volume'
#     cols = {c.lower(): c for c in df.columns}
#     date_col = cols.get("date") or cols.get("datetime") or "Date"

#     # ensure numeric
#     for c in ["Open","High","Low","Close","Volume"]:
#         if c in df.columns:
#             df[c] = pd.to_numeric(df[c], errors="coerce")

#     candles: List[Candle] = []
#     for _, r in df.iterrows():
#         if pd.isna(r.get("Close")):
#             continue
#         t_iso = _to_iso_z(r[date_col])
#         candles.append(
#             Candle(
#                 t=t_iso,
#                 o=float(r["Open"]),
#                 h=float(r["High"]),
#                 l=float(r["Low"]),
#                 c=float(r["Close"]),
#                 v=float(r.get("Volume", 0.0) or 0.0),
#             )
#         )
#     return HistoryFrame(ticker=symbol, timeframe=interval, period=period, candles=candles)

# def get_quote(symbol: str) -> QuoteYF:
#     """
#     Lightweight snapshot using yfinance.Ticker.fast_info / info fallbacks.
#     """
#     import yfinance as yf
#     t = yf.Ticker(symbol)
#     current = None
#     prev = None
#     cur = None
#     opn = None
#     hi = None
#     lo = None

#     # Prefer fast_info when available
#     try:
#         fi = t.fast_info
#         current = float(fi.get("last_price")) if fi.get("last_price") is not None else None
#         prev = float(fi.get("previous_close")) if fi.get("previous_close") is not None else None
#         cur = fi.get("currency")
#         opn = float(fi.get("open")) if fi.get("open") is not None else None
#         hi = float(fi.get("day_high")) if fi.get("day_high") is not None else None
#         lo = float(fi.get("day_low")) if fi.get("day_low") is not None else None
#     except Exception:
#         pass

#     # Fallback to .info if needed
#     if current is None or prev is None:
#         try:
#             info = t.info or {}
#             current = float(info.get("currentPrice")) if info.get("currentPrice") is not None else current
#             prev = float(info.get("previousClose")) if info.get("previousClose") is not None else prev
#             cur = info.get("currency") or cur
#             opn = float(info.get("open")) if info.get("open") is not None else opn
#             hi = float(info.get("dayHigh")) if info.get("dayHigh") is not None else hi
#             lo = float(info.get("dayLow")) if info.get("dayLow") is not None else lo
#         except Exception:
#             pass

#     return QuoteYF(
#         ticker=symbol,
#         current_price=current,
#         previous_close=prev,
#         currency=cur,
#         open_price=opn,
#         day_high=hi,
#         day_low=lo,
#     )
# helpers/yfinance.py
from __future__ import annotations
from typing import List, Optional
import sys
import numpy as np
import pandas as pd

from utils.yfinance_helper import Candle, HistoryFrame, QuoteYF

# Allowed intervals that yfinance supports
_ALLOWED_INTERVALS = {"1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"}


def _elog(msg: str):
    print(f"[yfinance_helper] {msg}", file=sys.stderr, flush=True)

def _norm_str(x, default=""):
    if isinstance(x, (list, tuple)):
        x = x[0] if len(x) > 0 else default
    if x is None:
        x = default
    return str(x)

def _norm_bool(x, default=True):
    if isinstance(x, (list, tuple)):
        x = x[0] if len(x) > 0 else default
    if isinstance(x, str):
        xl = x.strip().lower()
        if xl in ("true","1","yes","y"): return True
        if xl in ("false","0","no","n"): return False
        return default
    return bool(x) if x is not None else default

def _to_iso_z(ts) -> str:
    if hasattr(ts, "to_pydatetime"):
        return ts.to_pydatetime().replace(tzinfo=None).isoformat() + "Z"
    s = str(ts)
    return s if s.endswith("Z") else s + "Z"

def _flatten_single_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """If yfinance returns a MultiIndex like ('Close','AAPL'), extract the last level."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    try:
        if symbol in df.columns.get_level_values(-1):
            return df.xs(symbol, axis=1, level=-1, drop_level=True)
    except Exception:
        pass
    # fallback: join levels
    df = df.copy()
    df.columns = ["_".join([str(x) for x in tup if str(x) != ""]) for tup in df.columns.to_list()]
    return df

def get_price_history(symbol: str, period: str = "12mo", interval: str = "1d", auto_adjust: bool = True) -> HistoryFrame:
    """
    Robust OHLCV fetch via yfinance with fallbacks & normalization.
    Returns normalized candles with ISO timestamps.
    """
    symbol = _norm_str(symbol).upper().strip()
    period = _norm_str(period, "12mo").lower().strip()
    interval = _norm_str(interval, "1d").lower().strip()
    auto_adjust = _norm_bool(auto_adjust, True)

    if interval not in _ALLOWED_INTERVALS:
        raise ValueError(f"Unsupported interval '{interval}'. Allowed: {sorted(_ALLOWED_INTERVALS)}")

    try:
        import yfinance as yf
    except Exception as e:
        _elog(f"import yfinance failed: {e}")
        return HistoryFrame(ticker=symbol, timeframe=interval, period=period, candles=[])

    df = pd.DataFrame()
    # Attempt 1: download
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=auto_adjust)
        _elog(f"yf.download({symbol},{period},{interval}) -> {None if df is None else df.shape}")
    except Exception as e:
        _elog(f"yf.download raised: {type(e).__name__}: {e}")
        df = pd.DataFrame()

    # Attempt 2: Ticker().history
    if df is None or df.empty:
        try:
            t = yf.Ticker(symbol)
            df = t.history(period=period, interval=interval, auto_adjust=auto_adjust)
            _elog(f"Ticker().history -> {None if df is None else df.shape}")
        except Exception as e:
            _elog(f"Ticker().history raised: {type(e).__name__}: {e}")
            df = pd.DataFrame()

    if df is None or df.empty:
        return HistoryFrame(ticker=symbol, timeframe=interval, period=period, candles=[])

    # Normalize & coerce numerics
    df = _flatten_single_symbol(df, symbol).reset_index()

    # find date col
    date_col = "Date" if "Date" in df.columns else ("Datetime" if "Datetime" in df.columns else None)
    if date_col is None:
        _elog(f"no Date/Datetime in columns={list(df.columns)}")
        return HistoryFrame(ticker=symbol, timeframe=interval, period=period, candles=[])

    # relaxed mapping for canonical columns
    target_cols = ["Open","High","Low","Close","Volume"]
    def _find_col(name: str) -> Optional[str]:
        if name in df.columns: return name
        low = {c.lower().replace(" ", ""): c for c in df.columns}
        return low.get(name.lower().replace(" ", ""), None)
    colmap = {k: _find_col(k) for k in target_cols}

    for m in ["Open","High","Low","Close"]:
        if colmap.get(m) is None:
            _elog(f"missing required column {m}; cols={list(df.columns)}")
            return HistoryFrame(ticker=symbol, timeframe=interval, period=period, candles=[])

    # Build candles
    # Ensure numeric coercion via Series to avoid "arg must be list/tuple/Series" TypeError
    for k, src in colmap.items():
        if src is not None:
            try:
                df[src] = pd.to_numeric(pd.Series(df[src]), errors="coerce")
            except Exception as e:
                _elog(f"numeric coercion failed for {k} from {src}: {e}")

    candles: List[Candle] = []
    for _, r in df.iterrows():
        if pd.isna(r[colmap["Close"]]):
            continue
        try:
            candles.append(
                Candle(
                    t=_to_iso_z(r[date_col]),
                    o=float(r[colmap["Open"]]),
                    h=float(r[colmap["High"]]),
                    l=float(r[colmap["Low"]]),
                    c=float(r[colmap["Close"]]),
                    v=float(r[colmap["Volume"]]) if colmap["Volume"] is not None and not pd.isna(r[colmap["Volume"]]) else 0.0,
                )
            )
        except Exception:
            # Skip impossible rows
            continue

    return HistoryFrame(ticker=symbol, timeframe=interval, period=period, candles=candles)

def get_quote(symbol: str) -> QuoteYF:
    """
    Lightweight snapshot using yfinance.Ticker.fast_info / info fallbacks.
    """
    symbol = _norm_str(symbol).upper().strip()
    import yfinance as yf
    t = yf.Ticker(symbol)

    current = prev = opn = hi = lo = None
    cur = None

    # fast_info
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

    # info fallback
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
