import yfinance as yf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
stocks = [
    "^NSEI",
    "^BSESN",
    "^NSEBANK",
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "LT.NS",
    "ITC.NS",
    "BHARTIARTL.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "MARUTI.NS",
    "ASIANPAINT.NS",
    "WIPRO.NS",
    "TITAN.NS"
]

all_data = []

for stock in stocks:

    print(f"Downloading {stock}")

    df = yf.download(
        stock,
        start="2020-01-01",
        end="2025-01-01",
        auto_adjust=False,
        progress=False
    )

    # MultiIndex fix
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open','High','Low','Close','Volume']]

    df.reset_index(inplace=True)

    df['Stock'] = stock

    all_data.append(df)

final_df = pd.concat(all_data, ignore_index=True)

print(final_df.shape)

final_df.to_csv(
    "dataset/indian_stock_data.csv",
    index=False
)
# Data Preprocessing
df = pd.read_csv("dataset/indian_stock_data.csv")
#date format conversion
df['Date'] = pd.to_datetime(df['Date'])
#sorting the data by date
df = df.sort_values(['Stock','Date'])
print("Initial Dataset:")
print(df.shape)
# =====================================================
# MOVING AVERAGES
# =====================================================

df["MA10"] = (
    df.groupby("Stock")["Close"]
    .transform(lambda x: x.rolling(10).mean())
)

df["MA50"] = (
    df.groupby("Stock")["Close"]
    .transform(lambda x: x.rolling(50).mean())
)

# =====================================================
# DAILY RETURN
# =====================================================

df["Daily_Return"] = (
    df.groupby("Stock")["Close"]
    .pct_change()
)

# =====================================================
# RSI
# =====================================================

delta = df.groupby("Stock")["Close"].diff()

gain = delta.where(delta > 0, 0)

loss = -delta.where(delta < 0, 0)

avg_gain = gain.groupby(df["Stock"]).transform(
    lambda x: x.rolling(14).mean()
)

avg_loss = loss.groupby(df["Stock"]).transform(
    lambda x: x.rolling(14).mean()
)

rs = avg_gain / avg_loss

df["RSI"] = 100 - (100 / (1 + rs))

# =====================================================
# EMA + MACD
# =====================================================

df["EMA12"] = (
    df.groupby("Stock")["Close"]
    .transform(lambda x: x.ewm(span=12).mean())
)

df["EMA26"] = (
    df.groupby("Stock")["Close"]
    .transform(lambda x: x.ewm(span=26).mean())
)

df["MACD"] = df["EMA12"] - df["EMA26"]

df["Signal"] = (
    df.groupby("Stock")["MACD"]
    .transform(lambda x: x.ewm(span=9).mean())
)

# =====================================================
# BOLLINGER BANDS
# =====================================================

df["MA20"] = (
    df.groupby("Stock")["Close"]
    .transform(lambda x: x.rolling(20).mean())
)

rolling_std = (
    df.groupby("Stock")["Close"]
    .transform(lambda x: x.rolling(20).std())
)

df["Upper_Band"] = df["MA20"] + (2 * rolling_std)

df["Lower_Band"] = df["MA20"] - (2 * rolling_std)

# =====================================================
# VOLATILITY
# =====================================================

df["Volatility"] = (
    df.groupby("Stock")["Daily_Return"]
    .transform(lambda x: x.rolling(20).std())
)

# =====================================================
# CANDLESTICK FEATURES
# =====================================================

df["Body_Size"] = abs(
    df["Close"] - df["Open"]
)

df["Upper_Shadow"] = (
    df["High"] -
    df[["Open", "Close"]].max(axis=1)
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

# =====================================================
# TARGET
# NEXT DAY CLOSE PRICE
# =====================================================

df["Target"] = (
    df.groupby("Stock")["Close"]
    .shift(-1)
)

# =====================================================
# REMOVE NULL VALUES
# =====================================================
print("\nNull values before dropna:\n")
print(df.isnull().sum())

print("\nShape before dropna:", df.shape)
df.dropna(inplace=True)
print("Shape after dropna:", df.shape)

# =====================================================
# SAVE PROCESSED DATASET
# =====================================================

df.to_csv(
    "dataset/processed_stock_data.csv",
    index=False
)

print("Processed Dataset Shape:")
print(df.shape)

# =====================================================
# ENCODE STOCK NAMES
# =====================================================

encoder = LabelEncoder()

df["Stock"] = encoder.fit_transform(
    df["Stock"]
)

# =====================================================
# FEATURES
# =====================================================

X = df[
    [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",

        "MA10",
        "MA50",

        "RSI",

        "MACD",
        "Signal",

        "Upper_Band",
        "Lower_Band",

        "Daily_Return",
        "Volatility",

        "Body_Size",
        "Upper_Shadow",
        "Lower_Shadow",

        "Bullish_Candle",
        "Bearish_Candle",

        "Stock"
    ]
]

# =====================================================
# TARGET
# =====================================================

y = df["Target"]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# MODEL
# =====================================================

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

# =====================================================
# TRAIN
# =====================================================

model.fit(X_train, y_train)

# =====================================================
# PREDICT
# =====================================================

predictions = model.predict(X_test)

# =====================================================
# METRICS
# =====================================================

train_accuracy = model.score(
    X_train,
    y_train
)

test_accuracy = model.score(
    X_test,
    y_test
)

r2 = r2_score(
    y_test,
    predictions
)

mae = mean_absolute_error(
    y_test,
    predictions
)

print("\nModel Performance")
print("--------------------------")
print("Training Accuracy :", train_accuracy)
print("Testing Accuracy  :", test_accuracy)
print("R2 Score          :", r2)
print("MAE               :", mae)

# =====================================================
# SAVE MODEL
# =====================================================
import joblib
joblib.dump(
    model,
    "stock_prediction_model.pkl"
)

joblib.dump(
    encoder,
    "stock_encoder.pkl"
)

print("\nModel Saved Successfully")
#end of file