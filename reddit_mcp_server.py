from mcp.server.fastmcp import FastMCP
from helpers.reddit import get_reddit_posts_minified, aggregate_reddit_sentiment
from utils.reddit_helper import RedditPostMinified, RedditAggregate

mcp = FastMCP("reddit")

@mcp.tool(
    name="get_reddit_posts_tool",
    description="Fetch recent Reddit posts for a symbol (minified). Returns list[{title,score,url,created_utc,sent_compound,subreddit}].",
)
def get_reddit_posts_tool(
    symbol: str,
    limit_per_sub: int = 60,
    time_filter: str = "week",
) -> list[RedditPostMinified]:
    return get_reddit_posts_minified(symbol, limit_per_sub=limit_per_sub, time_filter=time_filter)

@mcp.tool(
    name="aggregate_reddit_sentiment_tool",
    description="Aggregate Reddit sentiment for a symbol. Returns {count, score_avg, confidence, topics, top_titles[3]}.",
)
def aggregate_reddit_sentiment_tool(
    symbol: str,
    limit_per_sub: int = 60,
    time_filter: str = "week",
) -> RedditAggregate:
    return aggregate_reddit_sentiment(symbol, limit_per_sub=limit_per_sub, time_filter=time_filter)

if __name__ == "__main__":
    mcp.run(transport="stdio")
