from dataclasses import dataclass
from typing import Optional, List

@dataclass
class RedditPostMinified:
    title: Optional[str]
    score: Optional[int]
    url: Optional[str]
    created_utc: Optional[float]
    sent_compound: Optional[float]
    subreddit: Optional[str]

@dataclass
class RedditAggregate:
    count: int
    score_avg: float        # mean of VADER compound
    confidence: float       # rough n-based confidence
    topics: List[str]       # ["earnings","halt","fraud"] or ["general"]
    top_titles: List[str]   # top 3 by score
