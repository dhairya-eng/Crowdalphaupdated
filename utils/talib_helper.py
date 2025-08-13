from dataclasses import dataclass
from typing import Optional, List

@dataclass
class TAIndicatorsRow:
    Date: str
    Close: float
    RSI: Optional[float]
    SMA_20: Optional[float]
    EMA_20: Optional[float]
    MACD: Optional[float]
    MACD_signal: Optional[float]

@dataclass
class TAIndicatorsFrame:
    rows: List[TAIndicatorsRow]  # typically last N rows

@dataclass
class DerivedScores:
    trend_score: float   # [-1..1], >0 uptrend
    vol_score: float     # [0..1], normalized short-term vol
