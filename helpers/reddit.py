from __future__ import annotations
from typing import List
import contextlib, io

from utils.reddit_helper import RedditPostMinified, RedditAggregate
from redditFetch import get_reddit_posts_raw  # your Phase‑1 module

_KEYWORDS = {
    "earnings": ["earnings", "guidance", "eps", "revenue", "q1", "q2", "q3", "q4", "forecast"],
    "halt": ["trading halt", "halted"],
    "fraud": ["fraud", "scandal", "sec probe", "investigation"],
}
from datetime import datetime, timezone

def _to_epoch(val) -> float | None:
    # Accept float/int epoch, ISO strings, or None
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    try:
        s = str(val).strip()
        # Handle trailing 'Z' as UTC
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        # fromisoformat handles 'YYYY-MM-DDTHH:MM:SS[.fff][+/-HH:MM]'
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None

def _as_opt_float(x):
    try:
        return None if x is None else float(x)
    except Exception:
        return None

def _as_opt_int(x):
    try:
        return None if x is None else int(x)
    except Exception:
        return None

def _as_opt_str(x):
    try:
        return None if x is None else str(x)
    except Exception:
        return None
    
def _minify_posts(posts: List[dict]) -> List[RedditPostMinified]:
    out: List[RedditPostMinified] = []
    for p in posts:
        out.append(
            RedditPostMinified(
                title=_as_opt_str(p.get("title")),
                score=_as_opt_int(p.get("score")),
                url=_as_opt_str(p.get("url")),
                created_utc=_to_epoch(p.get("created_utc")),   # <-- coerce to float seconds
                sent_compound=_as_opt_float(p.get("sent_compound")),
                subreddit=_as_opt_str(p.get("subreddit")),
            )
        )
    return out

def _topics_from_posts(posts: List[dict]) -> List[str]:
    text = " ".join(
        ((p.get("title") or "") + " " + (p.get("selftext") or "")) for p in posts
    ).lower()
    topics: List[str] = []
    for k, kws in _KEYWORDS.items():
        if any(kw in text for kw in kws):
            topics.append(k)
    return topics or ["general"]

def get_reddit_posts_minified(
    symbol: str,
    limit_per_sub: int = 60,
    time_filter: str = "week",
) -> List[RedditPostMinified]:
    # keep MCP stdio clean even if underlying code prints
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        posts = get_reddit_posts_raw(
            symbol, limit_per_sub=limit_per_sub, time_filter=time_filter, with_sentiment=True
        )
    return _minify_posts(posts)

def aggregate_reddit_sentiment(
    symbol: str,
    limit_per_sub: int = 60,
    time_filter: str = "week",
) -> RedditAggregate:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        posts = get_reddit_posts_raw(
            symbol, limit_per_sub=limit_per_sub, time_filter=time_filter, with_sentiment=True
        )

    n = len(posts)
    comp = [p.get("sent_compound") for p in posts if p.get("sent_compound") is not None]
    score = (sum(comp) / len(comp)) if comp else 0.0
    # crude confidence by sample size
    conf = 1.0 if n >= 30 else (0.7 if n >= 10 else (0.4 if n >= 5 else 0.0))

    topics = _topics_from_posts(posts)
    top_titles = [p.get("title") for p in sorted(posts, key=lambda x: x.get("score", 0), reverse=True)[:3]]

    return RedditAggregate(
        count=n,
        score_avg=float(score),
        confidence=float(conf),
        topics=topics,
        top_titles=top_titles,
    )
