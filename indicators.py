import talib
import pandas as pd
import yfinance as yf

def get_ta_indicators(ticker: str, period: str = "12mo", interval: str = "1d"):
    print(f"\n📥 Downloading data for: {ticker}, Period: {period}, Interval: {interval}")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False)

    # Extract 'Close' column properly
    if isinstance(df.columns, pd.MultiIndex):
        close = df[("Close", ticker)]
        df.columns = df.columns.droplevel(1)
    else:
        close = df["Close"]

    df["Close"] = close  # set it early while everything is aligned
    df.reset_index(inplace=True)

    print("\n✅ Raw Data with Correct Close:")
    print(df[["Date", "Close"]].head())

    close_prices = df["Close"].astype(float).values

    # Calculate indicators
    df["RSI"] = talib.RSI(close_prices)
    df["SMA_20"] = talib.SMA(close_prices, timeperiod=20)
    df["EMA_20"] = talib.EMA(close_prices, timeperiod=20)
    macd, macdsignal, _ = talib.MACD(close_prices)
    df["MACD"] = macd
    df["MACD_signal"] = macdsignal

    print("\n✅ Data BEFORE dropna():")
    print(df.tail(10))

    df.dropna(inplace=True)

    print("\n✅ Final Data (after dropna()):")
    print(df.tail(5))

    return df.tail(5)[["Date", "Close", "RSI", "SMA_20", "EMA_20", "MACD", "MACD_signal"]]



if __name__ == "__main__":
    try:
        ticker = input("Enter a ticker symbol (e.g., AAPL, TSLA, MSFT): ").strip().upper()
        df = get_ta_indicators(ticker, period="12mo")  # ← increased from 3mo
        if df.empty:
            print("\n⚠️ No valid data returned. Try a longer period or check the ticker.")
        else:
            print("\n📊 Technical Indicators (last 5 days):\n")
            print(df)
    except Exception as e:
        print(f"\n❌ Error: {e}")

