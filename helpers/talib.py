# helpers/trade_indicators.py
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, List, Any, Tuple

import numpy as np
import pandas as pd

# =========================
# Logging (stderr only)
# =========================
def _d(msg: str):
    if os.getenv("DEBUG_INDICATORS", "1") not in ("0", "false", "False"):
        print(f"[indicators] {msg}", file=sys.stderr, flush=True)

# =========================
# Data structures
# =========================
@dataclass
class TradeIndicatorsRow:
    Date: str
    Close: float
    # Momentum / Oscillators
    RSI_14: Optional[float]
    Stoch_K: Optional[float]
    Stoch_D: Optional[float]
    # Trend / Averages
    SMA_20: Optional[float]
    EMA_20: Optional[float]
    EMA_50: Optional[float]
    EMA_200: Optional[float]
    # MACD
    MACD: Optional[float]
    MACD_signal: Optional[float]
    MACD_hist: Optional[float]
    # Volatility
    ATR_14: Optional[float]
    BB_mid_20: Optional[float]
    BB_upper_20: Optional[float]
    BB_lower_20: Optional[float]
    # Strength / Volume
    ADX_14: Optional[float]
    OBV: Optional[float]

@dataclass
class TradeIndicatorsFrame:
    rows: List[TradeIndicatorsRow]

@dataclass
class BasicSignal:
    action: str           # "BUY" | "SELL" | "HOLD"
    confidence: float     # 0..1
    reasons: List[str]

# =========================
# TA-Lib gate
# =========================
_HAVE_TALIB = False
try:
    import talib  # type: ignore
    _HAVE_TALIB = True
    _d("TA-Lib available: using accelerated path")
except Exception as e:
    _d(f"TA-Lib not available, using fallbacks: {type(e).__name__}: {e}")

# =========================
# Fallback math helpers
# =========================
def _sma(a: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(a, dtype=float)
    return s.rolling(window, min_periods=window).mean().to_numpy()

def _ema(a: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(a, dtype=float)
    return s.ewm(span=window, adjust=False, min_periods=window).mean().to_numpy()

def _rsi(a: np.ndarray, period: int = 14) -> np.ndarray:
    if a.size < period + 1:
        return np.full_like(a, np.nan, dtype=float)
    delta = np.diff(a)
    up = np.clip(delta, 0, None)
    down = np.clip(-delta, 0, None)
    roll_up = pd.Series(up).rolling(period, min_periods=period).mean()
    roll_down = pd.Series(down).rolling(period, min_periods=period).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan], rsi.values])
    if rsi.size < a.size:
        rsi = np.concatenate([rsi, np.full(a.size - rsi.size, np.nan)])
    return rsi

def _stoch(high: np.ndarray, low: np.ndarray, close: np.ndarray, k=14, d=3) -> tuple[np.ndarray, np.ndarray]:
    hh = pd.Series(high, dtype=float).rolling(k, min_periods=k).max().to_numpy()
    ll = pd.Series(low, dtype=float).rolling(k, min_periods=k).min().to_numpy()
    denom = (hh - ll)
    denom[denom == 0] = np.nan
    k_raw = 100.0 * (close - ll) / denom
    k_smooth = pd.Series(k_raw).rolling(d, min_periods=d).mean().to_numpy()
    d_line = pd.Series(k_smooth).rolling(d, min_periods=d).mean().to_numpy()
    return k_smooth, d_line

def _macd(a: np.ndarray, fast=12, slow=26, signal=9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ema_fast = _ema(a, fast)
    ema_slow = _ema(a, slow)
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False, min_periods=signal).mean().to_numpy()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.concatenate([[np.nan], close[:-1]])
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    return np.nanmax(np.vstack([tr1, tr2, tr3]), axis=0)

def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period=14) -> np.ndarray:
    tr = _true_range(high, low, close)
    return pd.Series(tr).rolling(period, min_periods=period).mean().to_numpy()

def _bollinger(a: np.ndarray, window=20, num_std=2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = pd.Series(a, dtype=float)
    mid = s.rolling(window, min_periods=window).mean()
    std = s.rolling(window, min_periods=window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid.to_numpy(), upper.to_numpy(), lower.to_numpy()

def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period=14) -> np.ndarray:
    up_move = np.diff(high)
    down_move = -np.diff(low)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = _true_range(high, low, close)

    def _wilder_smooth(x):
        out = np.full_like(x, np.nan, dtype=float)
        if x.size < period:
            return out
        first = np.nanmean(x[1:period+1])
        out[period] = first
        alpha = (period - 1) / period
        for i in range(period + 1, x.size):
            out[i] = alpha * out[i - 1] + x[i]
        return out

    tr_sm = _wilder_smooth(tr)
    plus_dm_sm = _wilder_smooth(np.concatenate([[np.nan], plus_dm]))
    minus_dm_sm = _wilder_smooth(np.concatenate([[np.nan], minus_dm]))
    plus_di = 100.0 * (plus_dm_sm / (tr_sm + 1e-12))
    minus_di = 100.0 * (minus_dm_sm / (tr_sm + 1e-12))
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)
    adx = _wilder_smooth(dx)
    return adx

def _obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    obv = np.zeros_like(close, dtype=float)
    obv[:] = np.nan
    if close.size == 0:
        return obv
    obv[0] = 0.0
    for i in range(1, close.size):
        if np.isnan(close[i]) or np.isnan(close[i-1]) or np.isnan(volume[i]):
            obv[i] = obv[i-1]
        elif close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    return obv

# =========================
# Core indicator engine
# =========================
def _as_float_1d(x: Any) -> np.ndarray:
    try:
        return pd.to_numeric(x, errors="coerce").to_numpy(dtype=float).reshape(-1)
    except Exception:
        return np.asarray(x, dtype=float).reshape(-1)

def compute_trade_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute decision‑ready indicators.
    Required df columns: ['Date' or 'Datetime', 'Open','High','Low','Close','Volume'].
    Returns a copy with the indicators added (may contain NaNs for early bars).
    """
    if df is None or df.empty:
        _d("compute_trade_indicators: empty input")
        return pd.DataFrame()

    df = df.copy()
    date_col = "Date" if "Date" in df.columns else ("Datetime" if "Datetime" in df.columns else None)
    if date_col is None:
        _d("compute_trade_indicators: missing Date/Datetime")
        return pd.DataFrame()

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            _d(f"compute_trade_indicators: missing column {c}")
            return pd.DataFrame()
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["Close"])
    _d(f"dropna(Close): {before} -> {len(df)}")
    if df.empty:
        return df

    close = _as_float_1d(df["Close"])
    high = _as_float_1d(df["High"])
    low  = _as_float_1d(df["Low"])
    vol  = _as_float_1d(df["Volume"])

    if _HAVE_TALIB:
        rsi = talib.RSI(close, timeperiod=14)
        k, d = talib.STOCH(high, low, close)  # (5,3,3) default; acceptable proxy for fast stoch
        sma20 = pd.Series(close).rolling(20, min_periods=20).mean().to_numpy()
        ema20 = talib.EMA(close, timeperiod=20)
        ema50 = talib.EMA(close, timeperiod=50)
        ema200= talib.EMA(close, timeperiod=200)
        macd, macds, macdh = talib.MACD(close, 12, 26, 9)
        atr = talib.ATR(high, low, close, timeperiod=14)
        mid, upper, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        adx = talib.ADX(high, low, close, timeperiod=14)
        obv = talib.OBV(close, vol)
    else:
        rsi = _rsi(close, 14)
        k, d = _stoch(high, low, close, k=14, d=3)
        sma20 = _sma(close, 20)
        ema20 = _ema(close, 20)
        ema50 = _ema(close, 50)
        ema200= _ema(close, 200)
        macd, macds, macdh = _macd(close, 12, 26, 9)
        atr = _atr(high, low, close, 14)
        mid, upper, lower = _bollinger(close, 20, 2.0)
        adx = _adx(high, low, close, 14)
        obv = _obv(close, vol)

    df["RSI_14"] = rsi
    df["Stoch_K"] = k
    df["Stoch_D"] = d
    df["SMA_20"] = sma20
    df["EMA_20"] = ema20
    df["EMA_50"] = ema50
    df["EMA_200"] = ema200
    df["MACD"] = macd
    df["MACD_signal"] = macds
    df["MACD_hist"] = macdh
    df["ATR_14"] = atr
    df["BB_mid_20"] = mid
    df["BB_upper_20"] = upper
    df["BB_lower_20"] = lower
    df["ADX_14"] = adx
    df["OBV"] = obv
    return df

def to_trade_indicators_frame(df_with_ind: pd.DataFrame, last_n: int = 5) -> TradeIndicatorsFrame:
    if df_with_ind is None or df_with_ind.empty:
        return TradeIndicatorsFrame(rows=[])
    date_col = "Date" if "Date" in df_with_ind.columns else ("Datetime" if "Datetime" in df_with_ind.columns else None)
    tail = df_with_ind.tail(last_n)
    rows: List[TradeIndicatorsRow] = []
    for _, r in tail.iterrows():
        dt = r[date_col] if date_col else None
        date_str = dt.isoformat() if hasattr(dt, "isoformat") else (str(dt) if dt is not None else "")

        def g(col: str) -> Optional[float]:
            v = r.get(col, np.nan)
            try:
                fv = float(v)
                return fv if not np.isnan(fv) else None
            except Exception:
                return None

        rows.append(
            TradeIndicatorsRow(
                Date=date_str,
                Close=g("Close") or 0.0,
                RSI_14=g("RSI_14"),
                Stoch_K=g("Stoch_K"),
                Stoch_D=g("Stoch_D"),
                SMA_20=g("SMA_20"),
                EMA_20=g("EMA_20"),
                EMA_50=g("EMA_50"),
                EMA_200=g("EMA_200"),
                MACD=g("MACD"),
                MACD_signal=g("MACD_signal"),
                MACD_hist=g("MACD_hist"),
                ATR_14=g("ATR_14"),
                BB_mid_20=g("BB_mid_20"),
                BB_upper_20=g("BB_upper_20"),
                BB_lower_20=g("BB_lower_20"),
                ADX_14=g("ADX_14"),
                OBV=g("OBV"),
            )
        )
    return TradeIndicatorsFrame(rows=rows)

# =========================
# Minimal Yahoo fetcher
# =========================
# --- replace the whole fetch_history_yf with this version ---
def fetch_history_yf(symbol: str, period: str = "12mo", interval: str = "1d") -> pd.DataFrame:
    """Robust yfinance fetcher that flattens MultiIndex and safely coerces numerics."""
    try:
        import yfinance as yf
    except Exception as e:
        _d(f"yfinance import failed: {e}")
        return pd.DataFrame()

    df = pd.DataFrame()

    def _flatten_single_symbol(df_in: pd.DataFrame) -> pd.DataFrame:
        """If MultiIndex columns (e.g., ('Close','AAPL')), extract the single symbol layer."""
        if not isinstance(df_in.columns, pd.MultiIndex):
            return df_in
        # Try to pick the symbol level if present
        try:
            if symbol in df_in.columns.get_level_values(-1):
                out = df_in.xs(symbol, axis=1, level=-1, drop_level=True)
                return out
        except Exception:
            pass
        # Fallback: flatten by joining levels with '_'
        df_in = df_in.copy()
        df_in.columns = ["_".join([str(x) for x in tup if str(x) != ""]) for tup in df_in.columns.to_list()]
        return df_in

    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        _d(f"yf.download({symbol}, {period}, {interval}) -> {None if df is None else df.shape}")
    except Exception as e:
        _d(f"yf.download raised: {e}")
        df = pd.DataFrame()

    if df is None or df.empty:
        try:
            t = yf.Ticker(symbol)
            df = t.history(period=period, interval=interval, auto_adjust=False)
            _d(f"Ticker().history -> {None if df is None else df.shape}")
        except Exception as e:
            _d(f"Ticker().history raised: {e}")
            return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Normalize columns/index
    df = _flatten_single_symbol(df)
    df = df.reset_index()

    # Work out date column
    date_col = "Date" if "Date" in df.columns else ("Datetime" if "Datetime" in df.columns else None)
    if date_col is None:
        _d("fetch_history_yf: no Date/Datetime column present")
        return pd.DataFrame()

    # Canonical numeric columns we want to end with
    target_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    # Map alternate names if they exist (yfinance sometimes gives lowercase or spaces)
    # Build a relaxed lookup for each target
    def _find_col(name: str) -> Optional[str]:
        if name in df.columns:
            return name
        # loose matches
        low = {c.lower().replace(" ", ""): c for c in df.columns}
        key = name.lower().replace(" ", "")
        return low.get(key, None)

    cols_map = {name: _find_col(name) for name in target_cols}
    # At minimum we must have Open/High/Low/Close (Volume is useful, Adj Close optional)
    for must in ["Open", "High", "Low", "Close"]:
        if cols_map.get(must) is None:
            _d(f"fetch_history_yf: missing required column '{must}' in raw df columns={list(df.columns)}")
            return pd.DataFrame()

    # Build the output frame with exactly the columns we need
    out = pd.DataFrame()
    out[date_col] = df[date_col]
    for name in target_cols:
        src = cols_map.get(name)
        if src is None:
            # If truly missing (e.g., 'Adj Close'), create it as NaN to keep shape consistent
            out[name] = np.nan
        else:
            # Safe numeric coercion: always convert via Series, never pass weird objects to to_numeric
            try:
                out[name] = pd.to_numeric(pd.Series(df[src]), errors="coerce")
            except Exception as e:
                _d(f"coercion failed for {name} from '{src}': {type(e).__name__}: {e}")
                out[name] = pd.Series(df[src]).astype("float64") if np.issubdtype(pd.Series(df[src]).dtype, np.number) else np.nan

    # Rename date to 'Date'
    out.rename(columns={date_col: "Date"}, inplace=True)

    _d(f"fetch_history_yf: final columns={list(out.columns)} len={len(out)}")
    # Head preview to confirm non-empty and sane types
    try:
        _d(f"head Close={out['Close'].head(3).tolist()} Volume={out['Volume'].head(3).tolist() if 'Volume' in out.columns else 'N/A'}")
    except Exception:
        pass

    return out

# =========================
# Convenience APIs
# =========================
def compute_for_symbol(symbol: str, period="12mo", interval="1d", last_n=5) -> TradeIndicatorsFrame:
    hist = fetch_history_yf(symbol, period=period, interval=interval)
    if hist.empty:
        _d("compute_for_symbol: empty history")
        return TradeIndicatorsFrame(rows=[])
    indf = compute_trade_indicators(hist)
    return to_trade_indicators_frame(indf, last_n=last_n)

def derive_basic_signal_from_df(df_with_ind: pd.DataFrame) -> BasicSignal:
    if df_with_ind is None or df_with_ind.empty:
        return BasicSignal(action="HOLD", confidence=0.0, reasons=["No data"])
    last = df_with_ind.iloc[-1]

    reasons: List[str] = []
    score = 0

    # Trend bias via EMAs
    ema20 = float(last.get("EMA_20", np.nan))
    ema50 = float(last.get("EMA_50", np.nan))
    ema200= float(last.get("EMA_200", np.nan))
    close = float(last.get("Close", np.nan))

    if not np.isnan(ema20) and not np.isnan(ema50):
        if ema20 > ema50:
            score += 1; reasons.append("EMA20 > EMA50 (short-term uptrend)")
        elif ema20 < ema50:
            score -= 1; reasons.append("EMA20 < EMA50 (short-term downtrend)")

    if not np.isnan(ema50) and not np.isnan(ema200):
        if ema50 > ema200:
            score += 1; reasons.append("EMA50 > EMA200 (long-term uptrend)")
        elif ema50 < ema200:
            score -= 1; reasons.append("EMA50 < EMA200 (long-term downtrend)")

    # Momentum via RSI
    rsi = float(last.get("RSI_14", np.nan))
    if not np.isnan(rsi):
        if rsi > 60:
            score += 1; reasons.append(f"RSI(14) strong ({rsi:.1f})")
        elif rsi < 40:
            score -= 1; reasons.append(f"RSI(14) weak ({rsi:.1f})")

    # MACD confirmation
    macd = float(last.get("MACD", np.nan))
    macds= float(last.get("MACD_signal", np.nan))
    if not np.isnan(macd) and not np.isnan(macds):
        if macd > macds:
            score += 1; reasons.append("MACD above signal (bullish momentum)")
        elif macd < macds:
            score -= 1; reasons.append("MACD below signal (bearish momentum)")

    # ADX trend strength (optional)
    adx = float(last.get("ADX_14", np.nan))
    if not np.isnan(adx):
        if adx < 15:
            reasons.append(f"ADX(14) weak trend ({adx:.1f})")
        elif adx > 25:
            score += 0.5; reasons.append(f"ADX(14) solid trend ({adx:.1f})")

    action = "HOLD"
    if score >= 2:
        action = "BUY"
    elif score <= -2:
        action = "SELL"

    confidence = float(min(1.0, max(0.0, (abs(score) / 3.0))))
    return BasicSignal(action=action, confidence=confidence, reasons=reasons)

def derive_basic_signal_for_symbol(symbol: str, period="12mo", interval="1d") -> BasicSignal:
    hist = fetch_history_yf(symbol, period=period, interval=interval)
    if hist.empty:
        return BasicSignal(action="HOLD", confidence=0.0, reasons=["No price history"])
    indf = compute_trade_indicators(hist)
    return derive_basic_signal_from_df(indf)
