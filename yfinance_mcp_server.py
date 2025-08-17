# yfinance_mcp_server.py
import sys, traceback
from mcp.server.fastmcp import FastMCP
from helpers.yfinance import get_price_history, get_quote
from utils.yfinance_helper import HistoryFrame, QuoteYF

mcp = FastMCP("yfinance")

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

def _elog(msg: str):
    print(f"[yfinance_mcp] {msg}", file=sys.stderr, flush=True)

@mcp.tool(
    name="get_price_history_tool",
    description="Get OHLCV candles via yfinance. Args: symbol, period (e.g., 12mo/5d), interval (e.g., 1d/1h/15m). Returns HistoryFrame.",
)
def get_price_history_tool(
    symbol: str,
    period: str = "12mo",
    interval: str = "1d",
    auto_adjust: bool = True,
    last_n:int|None=100,
) -> HistoryFrame:
    try:
        symbol_n   = _norm_str(symbol).upper().strip()
        period_n   = _norm_str(period, "12mo").lower().strip()
        interval_n = _norm_str(interval, "1d").lower().strip()
        auto_adj_n = _norm_bool(auto_adjust, True)
        frame = get_price_history(symbol_n, period=period_n, interval=interval_n, auto_adjust=auto_adj_n)
        if last_n and frame.candles and len(frame.candles) > last_n:
            frame.candles = frame.candles[-last_n:]
        return frame
    except Exception as e:
        _elog(f"get_price_history_tool ERROR: {type(e).__name__}: {e}")
        traceback.print_exc(file=sys.stderr)
        return HistoryFrame(ticker=_norm_str(symbol).upper(), timeframe=_norm_str(interval), period=_norm_str(period), candles=[])

@mcp.tool(
    name="get_quote_tool",
    description="Get a snapshot quote via yfinance. Returns {ticker,current_price,previous_close,currency,open_price,day_high,day_low}.",
)
def get_quote_tool(symbol: str) -> QuoteYF:
    try:
        symbol_n = _norm_str(symbol).upper().strip()
        return get_quote(symbol_n)
    except Exception as e:
        _elog(f"get_quote_tool ERROR: {type(e).__name__}: {e}")
        traceback.print_exc(file=sys.stderr)
        return QuoteYF(ticker=_norm_str(symbol).upper(), current_price=None, previous_close=None, currency=None, open_price=None, day_high=None, day_low=None)

if __name__ == "__main__":
    mcp.run(transport="stdio")
