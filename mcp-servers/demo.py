from mcp.server.fastmcp import FastMCP

mcp = FastMCP("reddit")


@mcp.tool()
def get_news_for_stock_from_reddit(stock_symbol: str) -> str:
    """Get the latest news for a specific stock from Reddit"""
    return f"The current news for this {stock_symbol} is bullish, people are buying a lot because the company is doing great"

@mcp.tool()
def get_news_for_stock_from_twitter(stock_symbol: str) -> str:
    """Get the latest news for a specific stock from Twitter"""
    return f"The current news for this {stock_symbol} is bullish, people are buying a lot because the company is doing great"



if __name__ == "__main__":
    # res = get_news_for_stock_from_reddit("AAPL")
    # print(res)
    mcp.run(transport="stdio")
