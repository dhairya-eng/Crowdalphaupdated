from mcp.server.fastmcp import FastMCP
from helpers.talib import get_ta_indicators, derive_scores
from utils.talib_helper import TAIndicatorsFrame, DerivedScores

mcp = FastMCP("indicators")

@mcp.tool(
    name="get_ta_indicators_tool",
    description="Compute RSI, SMA_20, EMA_20, MACD, MACD_signal for a symbol. Returns last N rows."
)
def get_ta_indicators_tool(
    symbol: str,
    period: str = "12mo",
    interval: str = "1d",
    last_n: int = 5,
) -> TAIndicatorsFrame:
    try:
        return get_ta_indicators(symbol, period=period, interval=interval, last_n=last_n)
    except Exception as e:
        # Return a structured error instead of crashing stdio session
        return {"error": f"{type(e).__name__}: {e}"}

@mcp.tool(
    name="derive_scores_tool",
    description="Derive trend_score (EMA20 vs SMA20) and vol_score (5-bar stdev/price)."
)
def derive_scores_tool(
    symbol: str,
    period: str = "12mo",
    interval: str = "1d",
) -> DerivedScores:
    try:
        return derive_scores(symbol, period=period, interval=interval)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

if __name__ == "__main__":
    mcp.run(transport="stdio")
