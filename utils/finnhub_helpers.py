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

@dataclass
class RelevantNewsItemFinnHub:
    headline: str
    summary: str



@dataclass
class PredictiveStockMetrics:
    # Momentum / trend
    price_return_13w: float | None = None # 13WeekPriceReturnDaily
    price_return_52w: float | None = None # 52WeekPriceReturnDaily
    rel_to_sp500_26w: float | None = None # priceRelativeToS&P50026Week
    rel_to_sp500_52w: float | None = None # priceRelativeToS&P50052Week
    # Profitability / quality
    operating_margin_ttm: float | None = None       # operatingMarginTTM
    gross_margin_ttm: float | None = None           # grossMarginTTM
    roa_ttm: float | None = None                    # roaTTM
    roe_ttm: float | None = None                    # roeTTM
    cash_flow_per_share_ttm: float | None = None    # cashFlowPerShareTTM

    # Investment / asset growth
    capex_cagr_5y: float | None = None              # capexCagr5Y (lower is better)
    revenue_growth_ttm_yoy: float | None = None     # revenueGrowthTTMYoy
    tbv_cagr_5y: float | None = None                # tbvCagr5Y

    # Fundamental momentum
    eps_growth_ttm_yoy: float | None = None         # epsGrowthTTMYoy
    eps_growth_q_yoy: float | None = None           # epsGrowthQuarterlyYoy
    revenue_growth_q_yoy: float | None = None       # revenueGrowthQuarterlyYoy

    # Valuation (conditioning variables)
    pfcf_ttm: float | None = None                   # pfcfShareTTM
    pe_ttm: float | None = None                     # peTTM
    pb_ratio: float | None = None                   # pb

    # Risk / safety controls
    long_term_debt_to_equity: float | None = None   # longTermDebt/equityQuarterly or Annual
    total_debt_to_equity: float | None = None       # totalDebt/totalEquityQuarterly or Annual
    net_interest_coverage: float | None = None      # netInterestCoverageTTM
    beta: float | None = None                       # beta

    # Liquidity / attention (context)
    avg_trading_vol_3m: float | None = None         # 3MonthAverageTradingVolume
    avg_trading_vol_10d: float | None = None        # 10DayAverageTradingVolume


class NewsArticlesFinnHub:
    articles: list[NewsArticleFinnHub]