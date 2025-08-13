from mcp.server.fastmcp import FastMCP
from helpers.talib import get_stock_indicators
from utils.talib_helper import StockQuoteTA, indicators
mcp = FastMCP("talib")
@mcp.tool(
    name="get_stock_indicators_tool",
    description="""Fetch stock indicators for a given stock symbol.
        This method returns various technical indicators such as RSI, SMA, EMA, MACD, and MACD Signal.
        """,    )
def get_stock_indicators_tool(symbol: str) -> StockQuoteTA:
    return get_stock_indicators(symbol)
if __name__ == "__main__":
    mcp.run(transport="stdio")