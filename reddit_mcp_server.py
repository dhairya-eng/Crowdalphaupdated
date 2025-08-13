# reddit_mcp_server.py
import sys, traceback
from mcp.server.fastmcp import FastMCP
from helpers.reddit import get_reddit_posts_minified, aggregate_reddit_sentiment
from utils.reddit_helper import RedditPostMinified, RedditAggregate

mcp = FastMCP("reddit")

def _norm_str(x, default=""):
    if isinstance(x, (list, tuple)):
        x = x[0] if len(x) > 0 else default
    if x is None:
        x = default
    return str(x)

def _norm_int(x, default=10):
    try:
        if isinstance(x, (list, tuple)):
            x = x[0] if len(x) > 0 else default
        return int(x)
    except Exception:
        return default

def _elog(msg: str):
    print(f"[reddit_mcp] {msg}", file=sys.stderr, flush=True)

@mcp.tool(
    name="get_reddit_posts_tool",
    description="Fetch recent Reddit posts for a symbol (minified). Returns list[{title,score,url,created_utc,sent_compound,subreddit}].",
)
def get_reddit_posts_tool(
    symbol: str,
    limit_per_sub: int = 60,
    time_filter: str = "week",
) -> list[RedditPostMinified]:
    try:
        sym = _norm_str(symbol).upper().strip()
        lim = max(1, min(_norm_int(limit_per_sub, 30), 100))  # cap to keep payload small
        tf  = _norm_str(time_filter, "week").lower().strip()
        return get_reddit_posts_minified(sym, limit_per_sub=lim, time_filter=tf)
    except Exception as e:
        _elog(f"get_reddit_posts_tool ERROR: {type(e).__name__}: {e}")
        traceback.print_exc(file=sys.stderr)
        return []  # typed: list[RedditPostMinified]

@mcp.tool(
    name="aggregate_reddit_sentiment_tool",
    description="Aggregate Reddit sentiment for a symbol. Returns {count, score_avg, confidence, topics, top_titles[3]}.",
)
def aggregate_reddit_sentiment_tool(
    symbol: str,
    limit_per_sub: int = 60,
    time_filter: str = "week",
) -> RedditAggregate:
    try:
        sym = _norm_str(symbol).upper().strip()
        lim = max(1, min(_norm_int(limit_per_sub, 30), 100))
        tf  = _norm_str(time_filter, "week").lower().strip()
        return aggregate_reddit_sentiment(sym, limit_per_sub=lim, time_filter=tf)
    except Exception as e:
        _elog(f"aggregate_reddit_sentiment_tool ERROR: {type(e).__name__}: {e}")
        traceback.print_exc(file=sys.stderr)
        return RedditAggregate(count=0, score_avg=0.0, confidence=0.0, topics=["general"], top_titles=[])
        
if __name__ == "__main__":
    mcp.run(transport="stdio")
