from dataclasses import dataclass
from datetime import datetime
import requests
@dataclass
class indicators:
    def __init__(self, rsi, sma_20, ema_20, macd, macd_signal):
        self.rsi = rsi
        self.sma_20 = sma_20
        self.ema_20 = ema_20
        self.macd = macd
        self.macd_signal = macd_signal

class StockQuoteTA:
    indicators: list[indicators]