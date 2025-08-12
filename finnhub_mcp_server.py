from mcp.server.fastmcp import FastMCP
from helpers.finnhub import get_stock_price, get_company_news_minified, get_top_news_minified, get_some_relevant_stock_metrics
from utils.finnhub_helpers import StockQuoteFinnHub, PredictiveStockMetrics, RelevantNewsItemFinnHub

mcp = FastMCP("finnhub")

@mcp.tool(
    name="get_stock_price_tool",
    description="""Get the current stock price for a specific stock symbol. This method returns detailed stock information.
        It includes the current price, change, percent change, high price, low price, open price, previous close, and timestamp.
        """,
)
def get_stock_price_tool(stock_symbol: str) -> StockQuoteFinnHub:
    return get_stock_price(stock_symbol)


@mcp.tool(
    name="get_company_news_tool",
    description="""Get the latest news for a specific company from various sources.
        This method returns a summary of the most relevant news articles related to the company.
        """,
)
def get_company_news_tool(company_symbol: str) -> list[RelevantNewsItemFinnHub]:
    return get_company_news_minified(company_symbol)


@mcp.tool(
    name="get_top_news_tool",
    description="""Get the current top news articles from various sources.
        This method returns the headline and summary of the top 10 most relevant news articles.
        """,
)
def get_top_news_tool() -> list[RelevantNewsItemFinnHub]:
    return get_top_news_minified()


@mcp.tool(
    name="get_some_relevant_stock_metrics_tool",
    description="""Get some relevant stock metrics for a specific stock symbol.
        This method returns key financial metrics that can help in analyzing the stock's performance.
        """,
)
def get_some_relevant_stock_metrics_tool(stock_symbol: str) -> PredictiveStockMetrics:
    return get_some_relevant_stock_metrics(stock_symbol)

if __name__ == "__main__":
    # res = get_news_for_stock_from_reddit("AAPL")
    # print(res)
    mcp.run(transport="stdio")
