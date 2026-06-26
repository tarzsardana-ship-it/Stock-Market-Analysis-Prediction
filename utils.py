import yfinance as yf
import pandas as pd
import numpy as np


# ==========================================
# FETCH STOCK DATA
# ==========================================

def get_stock_data(stock, period="1y"):

    df = yf.download(
        stock,
        period=period,
        auto_adjust=False,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    print(df.columns)
    print(type(df["Close"]))
    print(df.columns.tolist())
    print(df.head())
    return df


# ==========================================
# FEATURE ENGINEERING
# ==========================================

def add_indicators(df):

    # MA10
    df["MA10"] = df["Close"].rolling(10).mean()

    # MA20
    df["MA20"] = df["Close"].rolling(20).mean()

    # MA50
    df["MA50"] = df["Close"].rolling(50).mean()

    # Daily Return
    df["Daily_Return"] = df["Close"].pct_change()

    # RSI
    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    # EMA12
    df["EMA12"] = df["Close"].ewm(span=12).mean()

    # EMA26
    df["EMA26"] = df["Close"].ewm(span=26).mean()

    # MACD
    df["MACD"] = df["EMA12"] - df["EMA26"]

    # Signal
    df["Signal"] = df["MACD"].ewm(span=9).mean()

    # Bollinger Bands
    rolling_std = df["Close"].rolling(20).std()

    df["Upper_Band"] = df["MA20"] + (2 * rolling_std)

    df["Lower_Band"] = df["MA20"] - (2 * rolling_std)

    # Volatility
    df["Volatility"] = (
        df["Daily_Return"]
        .rolling(20)
        .std()
    )

    # Candlestick Features
    df["Body_Size"] = abs(
        df["Close"] - df["Open"]
    )

    df["Upper_Shadow"] = (
        df["High"]
        - df[["Open", "Close"]].max(axis=1)
    )

    df["Lower_Shadow"] = (
        df[["Open", "Close"]].min(axis=1)
        - df["Low"]
    )

    df["Bullish_Candle"] = (
        df["Close"] > df["Open"]
    ).astype(int)

    df["Bearish_Candle"] = (
        df["Close"] < df["Open"]
    ).astype(int)

    df.dropna(inplace=True)

    return df


# ==========================================
# PREDICTION INPUT
# ==========================================

def prepare_input(df, stock_name, encoder):

    latest = df.iloc[-1]

    stock_encoded = encoder.transform(
        [stock_name]
    )[0]

    features = pd.DataFrame([{

        "Open": latest["Open"],
        "High": latest["High"],
        "Low": latest["Low"],
        "Close": latest["Close"],
        "Volume": latest["Volume"],

        "MA10": latest["MA10"],
        "MA50": latest["MA50"],

        "RSI": latest["RSI"],

        "MACD": latest["MACD"],
        "Signal": latest["Signal"],

        "Upper_Band": latest["Upper_Band"],
        "Lower_Band": latest["Lower_Band"],

        "Daily_Return": latest["Daily_Return"],
        "Volatility": latest["Volatility"],

        "Body_Size": latest["Body_Size"],
        "Upper_Shadow": latest["Upper_Shadow"],
        "Lower_Shadow": latest["Lower_Shadow"],

        "Bullish_Candle": latest["Bullish_Candle"],
        "Bearish_Candle": latest["Bearish_Candle"],

        "Stock": stock_encoded

    }])

    return features


# ==========================================
# BUY SELL SIGNAL
# ==========================================

def get_signal(rsi, macd, signal):

    if rsi < 30 and macd > signal:
        return "BUY"

    elif rsi > 70 and macd < signal:
        return "SELL"

    else:
        return "HOLD"
    

# ==========================================
# AI RECOMMENDATION
# ==========================================

def ai_recommendation(
    current_price,
    predicted_price,
    rsi,
    macd,
    signal,
    ma10,
    ma50
):

    recommendation = []

    if predicted_price > current_price:
        recommendation.append(
            "Model predicts upward price movement."
        )
    else:
        recommendation.append(
            "Model predicts downward price movement."
        )

    if rsi < 30:
        recommendation.append(
            "RSI indicates oversold condition."
        )

    elif rsi > 70:
        recommendation.append(
            "RSI indicates overbought condition."
        )

    if macd > signal:
        recommendation.append(
            "MACD is above Signal Line (Bullish)."
        )
    else:
        recommendation.append(
            "MACD is below Signal Line (Bearish)."
        )

    if ma10 > ma50:
        recommendation.append(
            "Short-term trend is stronger than long-term trend."
        )
    else:
        recommendation.append(
            "Long-term trend is stronger than short-term trend."
        )

    score = 0

    if predicted_price > current_price:
        score += 1

    if rsi < 70:
        score += 1

    if macd > signal:
        score += 1

    if ma10 > ma50:
        score += 1

    if score >= 3:
        final_signal = "STRONG BUY"

    elif score == 2:
        final_signal = "BUY"

    elif score == 1:
        final_signal = "HOLD"

    else:
        final_signal = "SELL"

    return final_signal, recommendation   