import os, time, datetime as dt
from typing import List
import pandas as pd
import praw
from dotenv import load_dotenv

load_dotenv()

# Try to enable VADER sentiment; keep running even if not present
_SIA, _SIA_OK = None, False
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    _SIA = SentimentIntensityAnalyzer()
    _SIA_OK = True
except Exception:
    _SIA_OK = False  # run once to enable: python -c "import nltk; nltk.download('vader_lexicon')"

REDDIT = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT", "crowdalpha/phase1")
)

DEFAULT_SUBS = ["stocks", "investing", "wallstreetbets", "StockMarket"]

def fetch_reddit_posts(query: str,
                       subreddits: List[str] = DEFAULT_SUBS,
                       limit_per_sub: int = 60,
                       time_filter: str = "week",
                       with_sentiment: bool = True) -> pd.DataFrame:
    rows = []
    for sub in subreddits:
        for p in REDDIT.subreddit(sub).search(query, time_filter=time_filter, limit=limit_per_sub):
            if p.stickied: 
                continue
            text = f"{p.title}\n\n{p.selftext or ''}".strip()
            sent = {"pos": None, "neg": None, "neu": None, "compound": None}
            if with_sentiment and _SIA_OK:
                s = _SIA.polarity_scores(text)
                sent = {"pos": s["pos"], "neg": s["neg"], "neu": s["neu"], "compound": s["compound"]}
            rows.append({
                "subreddit": sub,
                "id": p.id,
                "title": p.title,
                "selftext": p.selftext,
                "url": f"https://www.reddit.com{p.permalink}",
                "score": p.score,
                "num_comments": p.num_comments,
                "created_utc": dt.datetime.utcfromtimestamp(p.created_utc),
                **{f"sent_{k}": v for k, v in sent.items()}
            })
        time.sleep(0.3)
    df = pd.DataFrame(rows).drop_duplicates(subset=["id"]).sort_values("score", ascending=False).reset_index(drop=True)
    return df

# --- NEW: get raw posts as a list[dict] (perfect for LLM input) ---
def get_reddit_posts_raw(query: str,
                         subreddits: List[str] = DEFAULT_SUBS,
                         limit_per_sub: int = 60,
                         time_filter: str = "week",
                         with_sentiment: bool = True) -> List[dict]:
    df = fetch_reddit_posts(query, subreddits, limit_per_sub, time_filter, with_sentiment)
    cols = ["subreddit", "id", "title", "selftext", "url", "score", "num_comments", "created_utc",
            "sent_pos", "sent_neg", "sent_neu", "sent_compound"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].to_dict(orient="records")


# --- NEW: pretty-print all posts to console (quick sanity check) ---
def print_reddit_posts(posts: List[dict], max_chars: int = 200):
    for i, p in enumerate(posts, 1):
        body = (p.get("selftext") or "").replace("\n", " ")
        if len(body) > max_chars:
            body = body[:max_chars] + "..."
        print(f"{i:02d}. [{p.get('subreddit')}] {p.get('title')}")
        print(f"    ↳ {p.get('url')}")
        print(f"    score={p.get('score')} comments={p.get('num_comments')} compound={p.get('sent_compound')}")
        print(f"    {body}\n")




