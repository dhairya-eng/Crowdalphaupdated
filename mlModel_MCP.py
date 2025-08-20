# mlModel_MCP.py
# CrowdAlpha Technicals MCP server (no plotting): returns last price, ML forecast, features, and simple signals.

import sys, traceback, math
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge

# MCP plumbing (same style as your other working servers)
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("crowdalpha_technicals")

# ---------------- Utilities ----------------

_INTERVAL_SEC = {
    "1m": 60, "2m": 120, "5m": 300, "15m": 900, "30m": 1800,
    "60m": 3600, "90m": 5400, "1h": 3600, "1d": 86400,
}

def _to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df

def _close_series(df: pd.DataFrame) -> pd.Series:
    """Return a 1-D float Series for 'close' even if df['close'] is 2-D."""
    if "close" not in df.columns:
        raise KeyError("DataFrame missing 'close' column after download/rename.")
    s = df["close"]
    # If it's a DataFrame (e.g., duplicate columns / multi-ticker edge case), take the first column
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    # Ensure it's a Series of float, with a consistent name
    s = pd.Series(s, copy=False)
    s = s.astype(float)
    s.name = "close"
    return s


def fetch_bar(symbol: str, interval: str, lookback: str, *, prepost: bool = True) -> pd.DataFrame:
    import inspect
    # Build kwargs compatible with both old and new yfinance
    kwargs = dict(
        period=lookback,
        interval=interval,
        prepost=prepost,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    # Only pass 'repair' if the installed yfinance supports it
    if "repair" in inspect.signature(yf.download).parameters:
        kwargs["repair"] = True

    # Try download, then light fallback if empty (Yahoo flakiness)
    df = yf.download(symbol, **kwargs)
    if df is None or df.empty:
        # one retry with longer lookback
        fallback = "2y" if lookback.lower() in {"1y", "1yr"} else "max"
        kwargs["period"] = fallback
        df = yf.download(symbol, **kwargs)

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.dropna().rename(columns=str.lower)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def assert_fresh(df: pd.DataFrame, interval: str, *, max_lag_mult: float = 2.5) -> Dict[str, Any]:
    now = pd.Timestamp.now(tz="UTC")
    last_ts = df.index[-1]
    if getattr(last_ts, "tzinfo", None) is None:
        last_ts = last_ts.tz_localize("UTC")
    else:
        last_ts = last_ts.tz_convert("UTC")
    lag_s = float((now - last_ts).total_seconds())
    tol = max_lag_mult * _INTERVAL_SEC.get(interval, 60)
    return {"last_ts": str(last_ts), "stale_seconds": lag_s, "is_stale": lag_s > tol}

# ---------------- Features & Model ----------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    px = _close_series(df)

    # Bollinger (20,2)
    mid = px.rolling(20).mean()
    std = px.rolling(20).std(ddof=0)
    up = mid + 2 * std
    lo = mid - 2 * std
    bb_width  = (up - lo) / mid.replace(0.0, np.nan)
    percent_b = (px - lo) / (up - lo).replace(0.0, np.nan)

    # RSI(14) EWMA
    delta = px.diff()
    upm = delta.clip(lower=0)
    dnm = -delta.clip(upper=0)
    roll_up = upm.ewm(alpha=1/14, adjust=False).mean()
    roll_dn = dnm.ewm(alpha=1/14, adjust=False).mean().replace(0.0, np.nan)
    rs = roll_up / roll_dn
    rsi14 = 100 - (100 / (1 + rs))
    drsi3 = rsi14 - rsi14.shift(3)

    # MACD (12,26,9)
    ema12 = px.ewm(span=12, adjust=False).mean()
    ema26 = px.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_sig
    macd_hist_delta3 = macd_hist - macd_hist.shift(3)

    X = pd.concat({
        "bb_width": bb_width,
        "percent_b": percent_b,
        "rsi14": rsi14,
        "drsi3": drsi3,
        "macd_line": macd_line,
        "macd_sig": macd_sig,
        "macd_hist": macd_hist,
        "macd_hist_delta3": macd_hist_delta3,
    }, axis=1).dropna()

    if isinstance(X.columns, pd.MultiIndex):
        X.columns = X.columns.get_level_values(0)
    return X

def predict(df: pd.DataFrame, H: int) -> Tuple[pd.Timestamp, float, Dict[str, float]]:
    px = _close_series(df)

    # future log-return over horizon H
    ret_fut = np.log(px.shift(-H) / px)
    if isinstance(ret_fut, pd.DataFrame):
        ret_fut = ret_fut.iloc[:, 0]
    y_df = ret_fut.to_frame("y")

    X = build_features(df)
    data = X.join(y_df, how="inner").dropna()

    min_rows = max(120, H + 10)
    if len(data) < min_rows:
        raise ValueError(f"Not enough aligned rows: {len(data)} (need {min_rows})")

    X_all = data.drop(columns=["y"])
    y_all = data["y"]
    split = int(len(X_all) * 0.8)

    model = Pipeline([("scaler", RobustScaler()), ("reg", Ridge(alpha=1.0))])
    model.fit(X_all.iloc[:split], y_all.iloc[:split])

    X_last = X_all.iloc[split:split+1]
    if X_last.empty:
        X_last = X_all.iloc[[-1]]

    yhat = float(model.predict(X_last)[0])
    ts_last = X_last.index[-1]

    last_price = float(px.iloc[-1])
    predicted_price = last_price * float(np.exp(yhat))

    feats = {"last_price": last_price, "predicted_price": predicted_price}
    feats.update({k: float(X_last[k].iloc[0]) for k in X_last.columns})
    return ts_last, yhat, feats

def compute_signals_latest(df: pd.DataFrame) -> Dict[str, float | bool]:
    """Return only the latest scalar signals (no Series)."""
    px = _close_series(df)

    # MACD (12,26,9)
    ema12 = px.ewm(span=12, adjust=False).mean()
    ema26 = px.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    macds = macd.ewm(span=9, adjust=False).mean()

    if len(macd) < 2:
        return {"macd_bull_cross": False, "macd_bear_cross": False, "rsi14": float("nan")}

    macd_prev, macds_prev = float(macd.iloc[-2]), float(macds.iloc[-2])
    macd_now,  macds_now  = float(macd.iloc[-1]), float(macds.iloc[-1])
    macd_bull = (macd_prev < macds_prev) and (macd_now > macds_now)
    macd_bear = (macd_prev > macds_prev) and (macd_now < macds_now)

    # RSI(14) EWMA
    delta = px.diff()
    upm = delta.clip(lower=0)
    dnm = -delta.clip(upper=0)
    r_up = upm.ewm(alpha=1/14, adjust=False).mean()
    r_dn = dnm.ewm(alpha=1/14, adjust=False).mean().replace(0.0, np.nan)
    rs = r_up / r_dn
    rsi = 100 - (100 / (1 + rs))
    rsi_val = float(rsi.iloc[-1]) if len(rsi) else float("nan")

    return {"macd_bull_cross": bool(macd_bull), "macd_bear_cross": bool(macd_bear), "rsi14": rsi_val}


# ---------------- MCP Tool ----------------

@mcp.tool(
    name="analyze_technicals",
    description=(
        "No-plot analysis: fetch OHLC, compute features, predict H-step log-return, "
        "and return last price, predicted price, expected %, latest signals, and freshness."
    ),
)
def analyze_technicals(
    symbol: str = "AAPL",           # <--- DEFAULT so it's not required
    interval: str = "1d",
    lookback: str = "1y",
    horizon: int = 5,
    prepost: bool = True
) -> Dict[str, Any]:
    """
    Example:
      analyze_technicals()                          # defaults to AAPL, 1d, 1y, H=5
      analyze_technicals("AAPL", horizon=10)        # custom horizon
    """
    try:
        # Normalize inputs a bit
        symbol = (symbol or "AAPL").upper().strip()
        interval = interval.strip()
        lookback = lookback.strip()
        if horizon < 1:
            horizon = 1

        df = fetch_bar(symbol, interval, lookback, prepost=prepost)
        if df.empty:
            return {"error": f"No data for {symbol} (interval={interval}, lookback={lookback})."}

        fresh = assert_fresh(df, interval)
        def _enough(d: pd.DataFrame) -> bool:
            return len(d) >= 180  # room for 20/26 EMA warmup + 80/20 split

        if not _enough(df):
            for alt in ["2y", "5y", "max"]:
                df_alt = fetch_bar(symbol, interval, alt, prepost=prepost)
                if _enough(df_alt):
                    df, lookback = df_alt, alt
                    break
            if not _enough(df):
                return {"error": f"Not enough data for {symbol} with interval={interval}. Try a longer lookback (e.g., 2y/5y/max)."}

        # Train/predict
        try:
            ts, yhat, feats = predict(df, horizon)
        except Exception as e:
            # If the typical cause is not enough rows, suggest a bigger lookback
            msg = f"{type(e).__name__}: {e}"
            if "Not enough aligned rows" in str(e):
                msg += f" | Tip: try lookback='2y' or smaller horizon (got horizon={horizon})."
            return {"error": msg}

        sig = compute_signals_latest(df)

        exp_pct = (math.exp(yhat) - 1.0) * 100.0
        macd_bull = bool(sig.get("macd_bull_cross", False))
        verdict = "BUY" if (exp_pct > 0.0 and macd_bull) else ("HOLD" if abs(exp_pct) < 0.3 else "SELL")

        current_price   = float(feats["last_price"])
        predicted_price = float(feats["predicted_price"])

        return {
            "symbol": symbol,
            "interval": interval,
            "lookback": lookback,
            "horizon": horizon,
            "asof": str(ts),
            "freshness": fresh,

            # prominent, easy to read
            "current_price": current_price,
            "predicted_price": predicted_price,

            "prediction": {
                "log_return_hat": yhat,
                "expected_pct": exp_pct,
                "last_price": current_price,
                "predicted_price": predicted_price,
            },
            "verdict": verdict,
            "latest_signals": sig,
            "features": {k: float(v) for k, v in feats.items() if k not in ("last_price", "predicted_price")},
        }


    except Exception as e:
        import traceback, sys
        traceback.print_exc(file=sys.stderr)  # log details to stderr (doesn't break stdio)
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    # Match your existing servers’ pattern
    mcp.run(transport="stdio")
