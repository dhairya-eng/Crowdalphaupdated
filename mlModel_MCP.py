# mcp_realtime_model.py
# pip install mcp yfinance pandas numpy scikit-learn uvloop
import asyncio, time
import numpy as np, pandas as pd, yfinance as yf
from mcp.server.fastmcp import FastMCP
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge

app = FastMCP("realtime-ta-model")

# # ---------------- Indicators (pure pandas) ----------------
# def rsi(close: pd.Series, period: int = 14) -> pd.Series:
#     delta = close.diff()
#     up = delta.clip(lower=0.0)
#     down = -delta.clip(upper=0.0)
#     roll_up = up.ewm(alpha=1/period, adjust=False).mean()
#     roll_down = down.ewm(alpha=1/period, adjust=False).mean()
#     rs = roll_up / (roll_down.replace(0, np.nan))
#     return 100 - (100 / (1 + rs))

# def macd(close: pd.Series, fast=12, slow=26, signal=9):
#     ema_fast = close.ewm(span=fast, adjust=False).mean()
#     ema_slow = close.ewm(span=slow, adjust=False).mean()
#     macd_line = ema_fast - ema_slow
#     sig_line  = macd_line.ewm(span=signal, adjust=False).mean()
#     hist = macd_line - sig_line
#     return macd_line, sig_line, hist

# def bollinger(close: pd.Series, period=20, num_std=2.0):
#     mid = close.rolling(period).mean()
#     std = close.rolling(period).std(ddof=0)
#     upper = mid + num_std * std
#     lower = mid - num_std * std
#     percent_b = (close - lower) / (upper - lower).replace(0, np.nan)
#     bandwidth = (upper - lower) / mid.replace(0, np.nan)
#     return upper, mid, lower, percent_b, bandwidth

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np, pandas as pd

    def to_series(x, idx):
        # robustly convert anything (Series/ndarray/DataFrame) to 1-D Series with the right index
        if isinstance(x, pd.DataFrame):
            x = x.iloc[:, 0]
        elif isinstance(x, np.ndarray):
            x = np.asarray(x).reshape(-1)  # (n,1) -> (n,)
        return pd.Series(x, index=idx)

    c = df['close']

    # --- Bollinger (20, 2σ) ---
    period, num_std = 20, 2.0
    mid = c.rolling(period).mean()
    std = c.rolling(period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    bb_width  = (upper - lower) / mid.replace(0, np.nan)
    percent_b = (c - lower) / (upper - lower).replace(0, np.nan)

    # --- RSI(14) ---
    period_rsi = 14
    delta = c.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period_rsi, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period_rsi, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi14 = 100 - (100 / (1 + rs))
    drsi3 = rsi14 - rsi14.shift(3)

    # --- MACD(12,26,9) ---
    ema_fast = c.ewm(span=12, adjust=False).mean()
    ema_slow = c.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_sig
    dhist3    = macd_hist - macd_hist.shift(3)

    cols = {
        'bb_width_20':       to_series(bb_width,  df.index),
        'percent_b_20':      to_series(percent_b, df.index),
        'rsi14':             to_series(rsi14,     df.index),
        'rsi14_delta_3':     to_series(drsi3,     df.index),
        'macd':              to_series(macd_line, df.index),
        'macd_signal':       to_series(macd_sig,  df.index),
        'macd_hist':         to_series(macd_hist, df.index),
        'macd_hist_delta_3': to_series(dhist3,    df.index),
    }

    X = pd.concat(cols, axis=1)
    # optional: light winsorization for robustness
    X = X.apply(lambda s: s.clip(lower=s.quantile(0.01), upper=s.quantile(0.99)))
    return X.dropna()

def fetch_bars(symbol: str, interval: str, lookback: str) -> pd.DataFrame:
    df = yf.download(symbol, period=lookback, interval=interval, auto_adjust=True, progress=False)
    return df.dropna().rename(columns=str.lower)

def train_on_demand(df: pd.DataFrame, H: int, interval: str):
    X = build_features(df)

    # Largest warmup among indicators: MACD slow=26, BB=20, RSI=14 → 26 bars
    max_window = 26
    n = len(X)
    # Heuristic: need ~10x the largest window, but at least 120 rows
    min_rows = max(10 * max_window, 120)

    if n < min_rows:
        raise ValueError(f"Not enough history: have {n} rows, need ≥{min_rows} for interval={interval}.")

    ret_fut = np.log(df['close']).shift(-H) - np.log(df['close'])
    if isinstance(ret_fut, pd.DataFrame):
        ret_fut = ret_fut.squeeze(axis=1)
    ret_fut.name = 'y'

    data = pd.concat([X, ret_fut], axis=1).dropna()
    X, y = data.drop(columns=['y']), data['y']

    split = int(len(X) * 0.8)
    model = Pipeline([('scaler', RobustScaler()), ('reg', Ridge(alpha=1.0))])
    model.fit(X.iloc[:split], y.iloc[:split])

    x_last = X.iloc[[-1]]
    yhat = float(model.predict(x_last)[0])
    feats_payload = {k: float(x_last[k].iloc[0]) for k in x_last.columns}
    ts_last = X.index[-1]
    return ts_last, yhat, feats_payload


# simple per-(symbol,interval) TTL cache to avoid retraining several times/second
_CACHE = {}
_CACHE_TTL = 45  # seconds

@app.tool()
def indicators_get_latest(symbol: str, interval: str = "1m", lookback: str = "2d") -> dict:
    """
    Return latest Bollinger/RSI/MACD values for a symbol.
    """
    df = fetch_bars(symbol, interval, lookback)
    X = build_features(df)
    if X.empty:
        return {"error": "not_enough_data"}
    last = X.iloc[[-1]]
    ts = X.index[-1]
    return {
        "as_of": str(pd.Timestamp(ts).tz_localize("UTC")),
        "symbol": symbol.upper(),
        "interval": interval,
        "features": {k: float(last[k].iloc[0]) for k in last.columns}
    }

@app.tool()
def model_predict(symbol: str, interval: str = "1m", lookback: str = "7d", horizon: int = 5) -> dict:
    key = (symbol.upper(), interval, horizon)
    now = time.time()

    # try primary lookback
    try:
        df = fetch_bars(symbol, interval, lookback)
        ts, yhat, feats = train_on_demand(df, H=horizon, interval=interval)
    except ValueError as e:
        # Auto-expand lookback once based on interval
        fallback = None
        if "1d" in interval:
            fallback = "2y"
        elif "m" in interval:  # intraday
            fallback = "30d"
        if fallback and "Not enough history" in str(e):
            df = fetch_bars(symbol, interval, fallback)
            ts, yhat, feats = train_on_demand(df, H=horizon, interval=interval)
        else:
            return {"error": str(e)}

    exp_bps = (np.exp(yhat) - 1.0) * 1e4
    ts_utc = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return {
        "as_of": str(ts_utc),
        "symbol": symbol.upper(),
        "interval": interval,
        "horizon_bars": horizon,
        "features": feats,
        "prediction": {"log_return": yhat, "expected_bps": float(exp_bps)}
    }

if __name__ == "__main__":
    # Run as stdio MCP server
    app.run(transport="stdio")
