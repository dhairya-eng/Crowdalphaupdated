from __future__ import annotations
from typing import List, Optional
import math
import numpy as np
import pandas as pd

# IMPORTANT: no prints here (MCP uses stdout). If you need logs, use stderr or a logger.
from utils.talib_helper import TAIndicatorsRow, TAIndicatorsFrame, DerivedScores

# ---- price fetch ----
def _fetch_history_yf(symbol: str, period: str = "12mo", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV with yfinance. Returns DataFrame with columns:
    ['Open','High','Low','Close','Adj Close','Volume'] and a DatetimeIndex.
    """
    import yfinance as yf
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index().rename(columns={"Date": "Date"})
    # normalize types
    for col in ["Open","High","Low","Close","Adj Close","Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["Date","Open","High","Low","Close","Adj Close","Volume"]]

# ---- indicators (TA-Lib if available, else numpy/pandas fallbacks) ----
def _ema(series: np.ndarray, span: int) -> Optional[np.ndarray]:
    if len(series) < span:
        return None
    alpha = 2 / (span + 1)
    ema = np.empty_like(series, dtype=float)
    ema[:] = np.nan
    ema_idx = span - 1
    ema[ema_idx] = np.nanmean(series[:span])
    for i in range(ema_idx + 1, len(series)):
        ema[i] = alpha * series[i] + (1 - alpha) * ema[i - 1]
    return ema

def _rsi(close: np.ndarray, period: int = 14) -> Optional[np.ndarray]:
    if len(close) < period + 1:
        return None
    delta = np.diff(close)
    up = np.clip(delta, 0, None)
    down = np.clip(-delta, 0, None)
    roll_up = pd.Series(up).rolling(period, min_periods=period).mean()
    roll_down = pd.Series(down).rolling(period, min_periods=period).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan], rsi.values])
    return rsi

def _macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if len(close) < slow + signal:
        return None, None
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    if ema_fast is None or ema_slow is None:
        return None, None
    macd_line = ema_fast - ema_slow
    # signal is EMA of macd_line
    macd_valid = np.where(~np.isnan(macd_line))[0]
    if len(macd_valid) == 0:
        return macd_line, None
    start = macd_valid[0]
    sig = np.empty_like(macd_line)
    sig[:] = np.nan
    # compute EMA over macd_line[start:]
    alpha = 2 / (signal + 1)
    sig_val = np.nanmean(macd_line[start : start + signal]) if start + signal <= len(macd_line) else macd_line[start]
    for i in range(start + signal if start + signal < len(macd_line) else start + 1, len(macd_line)):
        sig_val = alpha * macd_line[i] + (1 - alpha) * sig_val
        sig[i] = sig_val
    return macd_line, sig

def _sma(close: np.ndarray, window: int) -> Optional[np.ndarray]:
    if len(close) < window:
        return None
    return pd.Series(close).rolling(window, min_periods=window).mean().values

def _as_1d_float(series_like) -> Optional[np.ndarray]:
    """Coerce to 1-D float ndarray; return None if impossible."""
    if series_like is None:
        return None
    try:
        arr = pd.to_numeric(series_like, errors="coerce").to_numpy(dtype=float).reshape(-1)
        if arr.size == 0 or np.all(np.isnan(arr)):
            return None
        return arr
    except Exception:
        try:
            arr = np.asarray(series_like, dtype=float).reshape(-1)
            if arr.size == 0 or np.all(np.isnan(arr)):
                return None
            return arr
        except Exception:
            return None

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds RSI(14), SMA_20, EMA_20, MACD, MACD_signal to df (per-close).
    Robust to odd types/NaNs; returns a cleaned subset or empty DF.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Ensure required cols exist
    for col in ["Open","High","Low","Close","Volume"]:
        if col not in df.columns:
            return pd.DataFrame()

    # Clean numeric + drop rows with NaN Close
    df = df.copy()
    for col in ["Open","High","Low","Close","Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"])
    if df.empty:
        return pd.DataFrame()

    close = _as_1d_float(df["Close"])
    if close is None or close.size < 30:  # need enough history for MACD/EMA
        return pd.DataFrame()

    # Try TA-Lib, else fallback
    try:
        import talib
        rsi = talib.RSI(close, timeperiod=14)
        macd_line, macd_signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        sma20 = pd.Series(close).rolling(20, min_periods=20).mean().to_numpy()
        ema20 = talib.EMA(close, timeperiod=20)
    except Exception:
        rsi = _rsi(close, 14)
        sma20 = _sma(close, 20)
        ema20 = _ema(close, 20)
        macd_line, macd_signal = _macd(close, 12, 26, 9)

    df["RSI"] = rsi if rsi is not None and len(rsi) == len(df) else np.nan
    df["SMA_20"] = sma20 if sma20 is not None and len(sma20) == len(df) else np.nan
    df["EMA_20"] = ema20 if ema20 is not None and len(ema20) == len(df) else np.nan
    df["MACD"] = macd_line if macd_line is not None and len(macd_line) == len(df) else np.nan
    df["MACD_signal"] = macd_signal if macd_signal is not None and len(macd_signal) == len(df) else np.nan

    # Keep only rows where all indicators exist
    df = df.dropna(subset=["RSI","SMA_20","EMA_20","MACD","MACD_signal"], how="any")
    return df

def get_ta_indicators(symbol: str, period: str = "12mo", interval: str = "1d", last_n: int = 5) -> TAIndicatorsFrame:
    hist = _fetch_history_yf(symbol, period=period, interval=interval)
    if hist is None or hist.empty:
        return TAIndicatorsFrame(rows=[])

    indf = compute_indicators(hist)
    if indf is None or indf.empty:
        return TAIndicatorsFrame(rows=[])

    tail = indf.tail(last_n)
    rows: List[TAIndicatorsRow] = []
    # Normalize date column presence
    date_col = "Date" if "Date" in tail.columns else ("Datetime" if "Datetime" in tail.columns else None)
    for _, r in tail.iterrows():
        dt_val = r[date_col] if date_col else None
        date_str = dt_val.isoformat() if hasattr(dt_val, "isoformat") else (str(dt_val) if dt_val is not None else "")
        rows.append(
            TAIndicatorsRow(
                Date=date_str,
                Close=float(r["Close"]),
                RSI=float(r["RSI"]),
                SMA_20=float(r["SMA_20"]),
                EMA_20=float(r["EMA_20"]),
                MACD=float(r["MACD"]),
                MACD_signal=float(r["MACD_signal"]),
            )
        )
    return TAIndicatorsFrame(rows=rows)
def derive_scores(symbol: str, period: str = "12mo", interval: str = "1d") -> DerivedScores:
    """
    Compute trend_score from (EMA_20 - SMA_20)/SMA_20 and vol_score from
    stdev of last 5 closes / last close. Both are normalized to [0..1] (vol) and [-1..1] (trend).
    """
    hist = _fetch_history_yf(symbol, period=period, interval=interval)
    if hist.empty:
        return DerivedScores(trend_score=0.0, vol_score=0.0)
    indf = compute_indicators(hist)
    if indf.empty:
        return DerivedScores(trend_score=0.0, vol_score=0.0)

    last = indf.iloc[-1]
    sma20 = float(last["SMA_20"])
    ema20 = float(last["EMA_20"])
    close_vals = indf["Close"].astype(float).values
    price = float(close_vals[-1])

    # trend score: amplified slope difference, clamped
    trend_raw = (ema20 - sma20) / max(1e-6, sma20)
    trend_score = max(-1.0, min(1.0, trend_raw * 5.0))

    # vol score: 5-bar stdev normalized to price, clamped
    if len(close_vals) >= 5:
        vol_raw = np.std(close_vals[-5:], ddof=1) / max(1e-6, price)
    else:
        vol_raw = 0.0
    vol_score = float(min(1.0, max(0.0, vol_raw * 20.0)))

    return DerivedScores(trend_score=float(trend_score), vol_score=float(vol_score))
