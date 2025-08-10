import requests
from dotenv import load_dotenv
import os
from helpers.finnhub import StockQuoteFinnHub, NewsArticleFinnHub
from datetime import datetime

load_dotenv()

FINHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

BASE_URL = "https://finnhub.io/api/v1"

def get_stock_price(symbol: str) -> StockQuoteFinnHub:
    url = f"{BASE_URL}/quote?symbol={symbol}&token={FINHUB_API_KEY}"
    response = requests.get(url)
    data = response.json()
    return StockQuoteFinnHub(
        current_price=data["c"],
        change=data["d"],
        percent_change=data["dp"],
        high_price=data["h"],
        low_price=data["l"],
        open_price=data["o"],
        previous_close=data["pc"],
        timestamp=datetime.fromtimestamp(data["t"])
    )

def get_company_news(symbol: str, from_date: str, to_date: str):
    """Fetch news for a specific company between two dates."""
    url = f"{BASE_URL}/company-news"
    params = {
        "symbol": symbol,
        "from": from_date,
        "to": to_date,
        "token": FINHUB_API_KEY
    }
    data = requests.get(url, params=params).json()
    return [NewsArticleFinnHub(**article) for article in data]

def get_top_news(category: str = "general"):
    """Fetch top market news by category (general, forex, crypto, merger)."""
    url = f"{BASE_URL}/news"
    params = {
        "category": category,
        "token": FINHUB_API_KEY
    }
    return requests.get(url, params=params).json()

def get_stock_metrics(symbol: str, metric: str = "all"):
    """Fetch stock metrics for a given symbol."""
    url = f"{BASE_URL}/stock/metric"
    params = {
        "symbol": symbol,
        "metric": metric,
        "token": FINHUB_API_KEY
    }
    return requests.get(url, params=params).json()

def main():
    # stock_data = get_stock_price("AAPL")
    # print(stock_data)

    news_for_comany = get_company_news(symbol="AAPL", from_date="2025-01-01", to_date="2025-12-31")
    print(news_for_comany)
if __name__ == "__main__":
    main()