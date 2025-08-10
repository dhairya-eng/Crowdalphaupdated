# Usage of Reddit and yFinance APIs to fetch stock data and Reddit posts to provide clean object to LLMs later
# phase1_adapter.py — normalized payload for LLM / UI
from redditFetch import fetch_reddit_posts
from yFinance import get_yfinance_data

def gather_phase1(symbol: str, reddit_query: str | None = None) -> dict:
    symbol = symbol.upper()
    query = reddit_query or symbol

    yfd = get_yfinance_data(symbol)
    rdf = fetch_reddit_posts(query, with_sentiment=True)

    # keep reddit rows compact for LLM
    reddit_items = rdf[[
        c for c in ["subreddit","title","selftext","url","score","num_comments","sent_compound"]
        if c in rdf.columns
    ]].to_dict(orient="records")

    return {
        "symbol": symbol,
        "yfinance": {
            "current_price": yfd["current_price"],
            "currency": yfd["currency"],
            "history": yfd["history"],   # list[dict] daily OHLCV
            "news": yfd["news"],         # list[dict]
        },
        "reddit": {
            "query": query,
            "posts": reddit_items,       # list[dict] for LLM
        }
    }
