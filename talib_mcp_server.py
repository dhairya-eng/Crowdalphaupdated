# from mcp.server.fastmcp import FastMCP
# from helpers.talib import get_ta_indicators, derive_scores
# from utils.talib_helper import TAIndicatorsFrame, DerivedScores

# mcp = FastMCP("indicators")

# @mcp.tool(
#     name="get_ta_indicators_tool",
#     description="Compute RSI, SMA_20, EMA_20, MACD, MACD_signal for a symbol. Returns last N rows."
# )
# def get_ta_indicators_tool(
#     symbol: str,
#     period: str = "12mo",
#     interval: str = "1d",
#     last_n: int = 5,
# ) -> TAIndicatorsFrame:
#     try:
#         return get_ta_indicators(symbol, period=period, interval=interval, last_n=last_n)
#     except Exception as e:
#         # Return a structured error instead of crashing stdio session
#         # Keep return type consistent to satisfy validation
#         return TAIndicatorsFrame(rows=[])

# @mcp.tool(
#     name="derive_scores_tool",
#     description="Derive trend_score (EMA20 vs SMA20) and vol_score (5-bar stdev/price)."
# )
# def derive_scores_tool(
#     symbol: str,
#     period: str = "12mo",
#     interval: str = "1d",
# ) -> DerivedScores:
#     try:
#         return derive_scores(symbol, period=period, interval=interval)
#     except Exception as e:
#        # Fallback neutral scores on error
#         return DerivedScores(trend_score=0.0, vol_score=0.0)

# if __name__ == "__main__":
#     mcp.run(transport="stdio")

# trade_indicators_mcp_server.py
import sys
import traceback
import pandas as pd
from mcp.server.fastmcp import FastMCP

from helpers.talib import(
    compute_for_symbol,
    derive_basic_signal_for_symbol,
    TradeIndicatorsFrame,
    BasicSignal,
)

mcp = FastMCP("trade_indicators")

def _elog(msg: str):
    print(f"[trade_indicators_server] {msg}", file=sys.stderr, flush=True)

@mcp.tool(
    name="get_trade_indicators_tool",
    description="Compute decision-ready indicators for a symbol. Returns last N rows."
)
def get_trade_indicators_tool(
    symbol: str,
    period: str = "12mo",
    interval: str = "1d",
    last_n: int = 5,
) -> TradeIndicatorsFrame:
    try:
        return compute_for_symbol(symbol, period=period, interval=interval, last_n=last_n)
    except Exception as e:
        _elog(f"get_trade_indicators_tool ERROR: {type(e).__name__}: {e}")
        traceback.print_exc(file=sys.stderr)
        return TradeIndicatorsFrame(rows=[])

@mcp.tool(
    name="derive_basic_signal_tool",
    description="Derive a basic BUY/SELL/HOLD signal with reasons."
)
def derive_basic_signal_tool(
    symbol: str,
    period: str = "12mo",
    interval: str = "1d",
) -> BasicSignal:
    try:
        return derive_basic_signal_for_symbol(symbol, period=period, interval=interval)
    except Exception as e:
        _elog(f"derive_basic_signal_tool ERROR: {type(e).__name__}: {e}")
        traceback.print_exc(file=sys.stderr)
        return BasicSignal(action="HOLD", confidence=0.0, reasons=[f"Error: {type(e).__name__}"])
# --- add to trade_indicators_mcp_server.py ---
@mcp.tool(
    name="debug_trade_env_tool",
    description="Report server python, imports, and a quick fetch sanity check."
)
def debug_trade_env_tool(symbol: str = "AAPL") -> dict:
    import sys, platform, importlib
    info = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "imports": {},
        "quick_fetch_preview": None,
    }
    for mod in ["yfinance", "talib", "numpy", "pandas"]:
        try:
            importlib.import_module(mod)
            info["imports"][mod] = "OK"
        except Exception as e:
            info["imports"][mod] = f"FAIL: {type(e).__name__}: {e}"
    try:
        from helpers.talib import fetch_history_yf
        df = fetch_history_yf(symbol, period="3mo", interval="1d")
        if df is None:
            info["quick_fetch_preview"] = "None"
        elif df.empty:
            info["quick_fetch_preview"] = "EMPTY DF"
        else:
            # Avoid pandas internals that may raise; create a simple, safe preview
            n = min(3, len(df))
            info["quick_fetch_preview"] = {
                "len": len(df),
                "cols": list(df.columns),
                "rows": [{c: (None if pd.isna(df.iloc[i][c]) else float(df.iloc[i][c]) if c != "Date" else str(df.iloc[i][c]))
                          for c in df.columns} for i in range(n)]
            }
    except Exception as e:
        info["quick_fetch_preview"] = f"FETCH ERROR: {type(e).__name__}: {e}"
    return info



if __name__ == "__main__":
    # Run as stdio MCP server
    mcp.run(transport="stdio")
