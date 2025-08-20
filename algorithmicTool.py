import matplotlib
matplotlib.use("Agg")      # headless backend for stdio servers
import matplotlib.pyplot as plt
import os, tempfile
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

_INTERVAL_SEC = {
    "1m": 60, "2m": 120, "5m": 300, "15m": 900, "30m": 1800,
    "60m": 3600, "90m": 5400, "1h": 3600, "1d": 86400,
}

def _save_fig(fig) -> str:
    tmpdir = os.environ.get("MCP_TMP_DIR", tempfile.gettempdir())
    path = os.path.join(tmpdir, f"crowdalpha_chart_{next(tempfile._get_candidate_names())}.png")
    fig.savefig(path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    return path
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
def make_bollinger_chart(df: pd.DataFrame, symbol: str) -> str:
    px = _close_series(df)
    mid = px.rolling(20).mean()
    std = px.rolling(20).std(ddof=0)
    up = mid + 2*std
    lo = mid - 2*std
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(px, label="Close")
    ax.plot(mid, label="Mid(20)")
    ax.plot(up, label="Upper")
    ax.plot(lo, label="Lower")
    ax.set_title(f"{symbol} — Bollinger Bands (20, 2)")
    ax.legend()
    return _save_fig(fig)

def make_rsi_chart(df: pd.DataFrame, symbol: str) -> str:
    px = _close_series(df)
    delta = px.diff()
    upm = delta.clip(lower=0)
    dnm = -delta.clip(upper=0)
    r_up = upm.ewm(alpha=1/14, adjust=False).mean()
    r_dn = dnm.ewm(alpha=1/14, adjust=False).mean().replace(0.0, np.nan)
    rs = r_up / r_dn
    rsi = 100 - (100 / (1 + rs))
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(rsi, label="RSI(14)")
    ax.axhline(30, linestyle="--")
    ax.axhline(70, linestyle="--")
    ax.set_title(f"{symbol} — RSI(14)")
    ax.legend()
    return _save_fig(fig)

def make_macd_chart(df: pd.DataFrame, symbol: str) -> str:
    px = _close_series(df)
    ema12 = px.ewm(span=12, adjust=False).mean()
    ema26 = px.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    macds = macd.ewm(span=9, adjust=False).mean()
    hist  = macd - macds
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(macd, label="MACD")
    ax.plot(macds, label="Signal")
    ax.bar(hist.index, hist, width=1.0, alpha=0.5, label="Hist")
    ax.set_title(f"{symbol} — MACD (12, 26, 9)")
    ax.legend()
    return _save_fig(fig)

def compute_signal_series(df: pd.DataFrame) -> pd.DataFrame:
    """Full series of buy/sell markers based on MACD cross + RSI filter."""
    px = _close_series(df)
    ema12 = px.ewm(span=12, adjust=False).mean()
    ema26 = px.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    macds = macd.ewm(span=9, adjust=False).mean()

    delta = px.diff()
    upm = delta.clip(lower=0)
    dnm = -delta.clip(upper=0)
    r_up = upm.ewm(alpha=1/14, adjust=False).mean()
    r_dn = dnm.ewm(alpha=1/14, adjust=False).mean().replace(0.0, np.nan)
    rs = r_up / r_dn
    rsi = 100 - (100 / (1 + rs))

    cross_up = (macd.shift(1) < macds.shift(1)) & (macd > macds)
    cross_dn = (macd.shift(1) > macds.shift(1)) & (macd < macds)
    buy  = cross_up & (rsi < 60)
    sell = cross_dn & (rsi > 40)
    return pd.DataFrame({"buy": buy.fillna(False), "sell": sell.fillna(False)})

def make_future_chart(df: pd.DataFrame, symbol: str, predicted_price: float, H: int) -> str:
    px = _close_series(df)
    sigs = compute_signal_series(df)

    last_ts = px.index[-1]
    # Put the predicted point at a future x-value; for daily bars add H days.
    fut_ts = last_ts + pd.Timedelta(days=H)

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(px, label="Close")

    # Past buy/sell markers
    buys  = sigs.index[sigs["buy"]]
    sells = sigs.index[sigs["sell"]]
    if len(buys):
        ax.scatter(buys, px.loc[buys], marker="^", s=40, label="Buy")
    if len(sells):
        ax.scatter(sells, px.loc[sells], marker="v", s=40, label="Sell")

    # Future predicted point
    ax.scatter([fut_ts], [predicted_price], marker="*", s=120, label=f"Pred +{H}", zorder=5)
    ax.set_title(f"{symbol} — Future Trend & Signals")
    ax.legend()
    return _save_fig(fig)

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
@mcp.tool(
    name="analyze_technicals_with_charts",
    description=(
        "Compute MACD/Bollinger/RSI and an ML forecast; return JSON + PNG charts "
        "(Bollinger, RSI, MACD, and Future Trend with buy/sell markers)."
    ),
)
def analyze_technicals_with_charts(
    symbol: str = "AAPL",
    interval: str = "1d",
    lookback: str = "1y",
    horizon: int = 5,
    prepost: bool = True
) -> Dict[str, Any]:
    try:
        symbol = (symbol or "AAPL").upper().strip()
        interval = interval.strip(); lookback = lookback.strip()
        if horizon < 1: horizon = 1

        df = fetch_bar(symbol, interval, lookback, prepost=prepost)
        if df.empty:
            return {"error": f"No data for {symbol} (interval={interval}, lookback={lookback})."}

        # ensure enough rows; auto-extend lookback if needed
        def _enough(d: pd.DataFrame) -> bool: return len(d) >= 180
        if not _enough(df):
            for alt in ["2y", "5y", "max"]:
                dfa = fetch_bar(symbol, interval, alt, prepost=prepost)
                if _enough(dfa):
                    df, lookback = dfa, alt
                    break
            if not _enough(df):
                return {"error": f"Not enough data for {symbol} with {interval}. Try lookback=2y/5y/max."}

        fresh = assert_fresh(df, interval)
        ts, yhat, feats = predict(df, horizon)
        sig_last = compute_signals_latest(df)

        exp_pct = (math.exp(yhat) - 1.0) * 100.0
        verdict = "BUY" if (exp_pct > 0.0 and bool(sig_last.get("macd_bull_cross", False))) else ("HOLD" if abs(exp_pct) < 0.3 else "SELL")

        current_price   = float(feats["last_price"])
        predicted_price = float(feats["predicted_price"])

        # ---- Charts (PNG files -> return URIs) ----
        b_path = make_bollinger_chart(df, symbol)
        r_path = make_rsi_chart(df, symbol)
        m_path = make_macd_chart(df, symbol)
        f_path = make_future_chart(df, symbol, predicted_price, horizon)

        charts = [
            {"name": f"{symbol} Bollinger", "path": b_path, "uri": f"file://{b_path}", "format": "png"},
            {"name": f"{symbol} RSI",       "path": r_path, "uri": f"file://{r_path}", "format": "png"},
            {"name": f"{symbol} MACD",      "path": m_path, "uri": f"file://{m_path}", "format": "png"},
            {"name": f"{symbol} Future",    "path": f_path, "uri": f"file://{f_path}", "format": "png"},
        ]

        return {
            "symbol": symbol,
            "interval": interval,
            "lookback": lookback,
            "horizon": horizon,
            "asof": str(ts),
            "freshness": fresh,

            "current_price": current_price,
            "predicted_price": predicted_price,

            "prediction": {
                "log_return_hat": yhat,
                "expected_pct": exp_pct,
                "last_price": current_price,
                "predicted_price": predicted_price,
            },
            "verdict": verdict,
            "latest_signals": sig_last,
            "features": {k: float(v) for k, v in feats.items() if k not in ("last_price", "predicted_price")},
            "charts": charts,
            "strategy": {
                "entry":  "MACD bull cross + RSI<60",
                "exit":   "MACD bear cross + RSI>40 or risk-based stop",
                "notes":  "Markers show historical buy/sell points; star marks ML-predicted future price."
            }
        }

    except Exception as e:
        import traceback, sys
        traceback.print_exc(file=sys.stderr)
        return {"error": f"{type(e).__name__}: {e}"}

if __name__ == "__main__":
    # Match your existing servers’ pattern
    mcp.run(transport="stdio")