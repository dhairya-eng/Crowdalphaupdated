from mcp.server.fastmcp import FastMCP
from helpers.yfinance import get_price_history, get_quote
from utils.yfinance_helper import HistoryFrame, QuoteYF

mcp = FastMCP("yfinance")

@mcp.tool(
    name="get_price_history_tool",
    description="Get OHLCV candles via yfinance. Args: symbol, period (e.g., 12mo/5d), interval (e.g., 1d/1h/15m). Returns HistoryFrame.",
)
def get_price_history_tool(
    symbol: str,
    period: str = "12mo",
    interval: str = "1d",
    auto_adjust: bool = True,
) -> HistoryFrame:
    return get_price_history(symbol, period=period, interval=interval, auto_adjust=auto_adjust)

@mcp.tool(
    name="get_quote_tool",
    description="Get a snapshot quote via yfinance. Returns {ticker,current_price,previous_close,currency,open_price,day_high,day_low}.",
)
def get_quote_tool(symbol: str) -> QuoteYF:
    return get_quote(symbol)

if __name__ == "__main__":
    mcp.run(transport="stdio")
