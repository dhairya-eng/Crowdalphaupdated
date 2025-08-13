import requests
from dataclasses import dataclass
from datetime import datetime
from utils.talib_helper import StockQuoteTA, indicators


def get_stock_indicators(symbol: str) -> StockQuoteTA:
    """Fetch stock indicators for a given symbol."""
    url = f"https://api.talib.com/v1/indicators/{symbol}"
    response = requests.get(url)
    data = response.json()

    if not data or "indicators" not in data:
        return StockQuoteTA(indicators=[])

    indicators_list = []
    for item in data["indicators"]:
        indicators_list.append(
            indicators(
                rsi=item.get("rsi"),
                sma_20=item.get("sma_20"),
                ema_20=item.get("ema_20"),
                macd=item.get("macd"),
                macd_signal=item.get("macd_signal")
            )
        )

    return StockQuoteTA(indicators=indicators_list)