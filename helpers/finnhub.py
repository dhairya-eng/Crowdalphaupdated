from dataclasses import dataclass
from datetime import datetime
import requests


@dataclass
class StockQuoteFinnHub:
    current_price: float
    change: float
    percent_change: float
    high_price: float
    low_price: float
    open_price: float
    previous_close: float
    timestamp: datetime

@dataclass
class NewsArticleFinnHub:
    category: str
    datetime: int
    headline: str
    id: int
    image: str
    related: str
    source: str
    summary: str
    url: str

class NewsArticlesFinnHub:
    articles: list[NewsArticleFinnHub]