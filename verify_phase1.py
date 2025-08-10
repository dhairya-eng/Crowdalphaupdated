import argparse

from redditFetch import fetch_reddit_posts
from yFinance import get_yfinance_data   # your quiet file from earlier


def run_all(symbol: str, reddit_query: str | None = None):
    symbol = symbol.upper()
    query = reddit_query or symbol

    # yfinance
    yfd = get_yfinance_data(symbol)
    yf_ok = (yfd["current_price"] is not None) and (yfd["history"] and len(yfd["history"]) > 0)

    # reddit
    rdf = fetch_reddit_posts(query, with_sentiment=True)
    r_ok = len(rdf) > 0
    cmean = round(rdf["sent_compound"].dropna().mean(), 3) if r_ok and "sent_compound" in rdf else None


    # summary (compact)
    summary = {
        "symbol": symbol,
        "yfinance": {
            "ok": yf_ok,
            "current_price": yfd["current_price"],
            "history_rows": len(yfd["history"]),
            "news_count": len(yfd["news"]),
        },
        "reddit": {
            "ok": r_ok,
            "posts": int(len(rdf)),
            "compound_mean": cmean
        }
    }
    print(summary)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--query", help="Custom Reddit search text")
    args = ap.parse_args()
    run_all(args.symbol, args.query)
