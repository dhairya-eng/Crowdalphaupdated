import requests
from dotenv import load_dotenv
import os
from utils.finnhub_helpers import StockQuoteFinnHub, NewsArticleFinnHub, RelevantNewsItemFinnHub, PredictiveStockMetrics
from datetime import datetime
from datetime import timedelta

load_dotenv()

FINHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
 
BASE_URL = "https://finnhub.io/api/v1"

def get_x_days_ago(x: int) -> str:
    """Get the date X days ago as a string."""
    return (datetime.today() - timedelta(days=x)).strftime("%Y-%m-%d")

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

def get_company_news(symbol: str, from_date: str, to_date: str ):
    """Fetch news for a specific company between two dates."""
    url = f"{BASE_URL}/company-news"
    params = {
        "symbol": symbol,
        "from": from_date,
        "to": to_date,
        "token": FINHUB_API_KEY
    }
    return requests.get(url, params=params).json()

def get_company_news_minified(symbol: str, from_date: str = get_x_days_ago(7), to_date: str = get_x_days_ago(0), number_of_items: int = 10):
    """Fetch top 10 news for a specific company between two dates. By default, it fetches news from the last 7 days."""
    articles = get_company_news(symbol, from_date, to_date)
    results: list[RelevantNewsItemFinnHub] = []
    if not isinstance(articles, list):
        return []
    for item in articles:
        headline = item.get("headline", "")
        summary = item.get("summary", "")
        if headline or summary:
            results.append(RelevantNewsItemFinnHub(headline=headline, summary=summary))
    return results[0:number_of_items]

def get_top_news(category: str = "general"):
    """Fetch top market news by category (general, forex, crypto, merger)."""
    url = f"{BASE_URL}/news"
    params = {
        "category": category,
        "token": FINHUB_API_KEY
    }
    return requests.get(url, params=params).json()

def get_top_news_minified(category: str = "general", number_of_items: int = 10) -> list[RelevantNewsItemFinnHub]:
    """Return only headline and summary as RelevantNewsItemFinnHub dataclass list."""
    raw_items = get_top_news(category)
    results: list[RelevantNewsItemFinnHub] = []
    if not isinstance(raw_items, list):
        return []
    for item in raw_items:
        headline = item.get("headline", "")
        summary = item.get("summary", "")
        if headline or summary:
            results.append(RelevantNewsItemFinnHub(headline=headline, summary=summary))
    return results[0:number_of_items]

def get_stock_metrics(symbol: str, metric: str = "all"):
    """Fetch stock metrics for a given symbol."""
    url = f"{BASE_URL}/stock/metric"
    params = {
        "symbol": symbol,
        "metric": metric,
        "token": FINHUB_API_KEY
    }
    return requests.get(url, params=params).json()

def get_some_relevant_stock_metrics(symbol: str) -> PredictiveStockMetrics:
    """
    Convert Finnhub /stock/metric response JSON into PredictiveStockMetrics.
    Expects a dict with a 'metric' key (as returned by your get_stock_metrics()).
    """
    raw = get_stock_metrics(symbol)
    m = raw.get("metric", {}) if isinstance(raw, dict) else {}
    long_term_debt_to_equity = (
        m.get("longTermDebt/equityQuarterly", m.get("longTermDebt/equityAnnual"))
    )
    total_debt_to_equity = (
        m.get("totalDebt/totalEquityQuarterly", m.get("totalDebt/totalEquityAnnual"))
    )

    return PredictiveStockMetrics(
        price_return_13w=m.get("13WeekPriceReturnDaily"),
        price_return_52w=m.get("52WeekPriceReturnDaily"),
        rel_to_sp500_26w=m.get("priceRelativeToS&P50026Week"),
        rel_to_sp500_52w=m.get("priceRelativeToS&P50052Week"),
        operating_margin_ttm=m.get("operatingMarginTTM"),
        gross_margin_ttm=m.get("grossMarginTTM"),
        roa_ttm=m.get("roaTTM"),
        roe_ttm=m.get("roeTTM"),
        cash_flow_per_share_ttm=m.get("cashFlowPerShareTTM"),
        capex_cagr_5y=m.get("capexCagr5Y"),
        revenue_growth_ttm_yoy=m.get("revenueGrowthTTMYoy"),
        tbv_cagr_5y=m.get("tbvCagr5Y"),
        eps_growth_ttm_yoy=m.get("epsGrowthTTMYoy"),
        eps_growth_q_yoy=m.get("epsGrowthQuarterlyYoy"),
        revenue_growth_q_yoy=m.get("revenueGrowthQuarterlyYoy"),
        pfcf_ttm=m.get("pfcfShareTTM"),
        pe_ttm=m.get("peTTM"),
        pb_ratio=m.get("pb"),
        long_term_debt_to_equity=long_term_debt_to_equity,
        total_debt_to_equity=total_debt_to_equity,
        net_interest_coverage=m.get("netInterestCoverageTTM"),
        beta=m.get("beta"),
        avg_trading_vol_3m=m.get("3MonthAverageTradingVolume"),
        avg_trading_vol_10d=m.get("10DayAverageTradingVolume"),
    )




def main():
    # stock_data = get_stock_price("AAPL")
    # print(stock_data)

    # news_for_comany = get_company_news(symbol="AAPL", from_date="2025-01-01", to_date="2025-12-31")
    # print(news_for_comany)

    # general_news = get_top_news_minified(category="general", number_of_items=5)
    # print(general_news)

    # today = get_x_days_ago(0)
    # seven_days_ago_str = get_x_days_ago(7)
    # print(f"Today: {today}, 7 days ago: {seven_days_ago_str}")

    # company_news = get_company_news_minified("AAPL", number_of_items=5)
    # print(company_news)

    # metr = get_some_relevant_stock_metrics("AAPL")

    # print(metr)
    pass

if __name__ == "__main__":
    main()