from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Candle:
    t: str  # ISO8601 Z
    o: float
    h: float
    l: float
    c: float
    v: float

@dataclass
class HistoryFrame:
    ticker: str
    timeframe: str          # e.g., "1d", "1h", "15m"
    period: str             # e.g., "12mo", "5d"
    candles: List[Candle]

@dataclass
class QuoteYF:
    ticker: str
    current_price: Optional[float]
    previous_close: Optional[float]
    currency: Optional[str]
    open_price: Optional[float]
    day_high: Optional[float]
    day_low: Optional[float]
