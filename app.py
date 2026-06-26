import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import plotly.graph_objects as go
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
from utils import (
    get_stock_data,
    add_indicators,
    prepare_input,
    ai_recommendation
)

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Stock Market Analysis & Prediction System",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# DARK THEME CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.stApp {
    background-color: #0E1117;
}

.title-text {
    text-align:center;
    color:#00FFAA;
    font-size:40px;
    font-weight:bold;
}

.subtitle {
    text-align:center;
    color:white;
    font-size:18px;
}

.card {
    background:#161B22;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 0px 15px rgba(0,255,170,0.15);
    text-align:center;
}

.metric-title {
    color:#A0A0A0;
    font-size:16px;
}

.metric-value {
    color:white;
    font-size:28px;
    font-weight:bold;
}

.buy-box {
    background:#0F5132;
    padding:15px;
    border-radius:10px;
    text-align:center;
    color:white;
    font-size:22px;
    font-weight:bold;
}

.sell-box {
    background:#842029;
    padding:15px;
    border-radius:10px;
    text-align:center;
    color:white;
    font-size:22px;
    font-weight:bold;
}

.hold-box {
    background:#664D03;
    padding:15px;
    border-radius:10px;
    text-align:center;
    color:white;
    font-size:22px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    model = joblib.load("stock_prediction_model.pkl")
    encoder = joblib.load("stock_encoder.pkl")
    return model, encoder

model, encoder = load_model()

# ==========================================
# TITLE
# ==========================================

st.markdown(
    "<h1 class='title-text'>📈 Stock Market Analysis & Prediction System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>Machine Learning + Technical Indicators Dashboard</p>",
    unsafe_allow_html=True
)

st.divider()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("⚙️ Settings")

stocks = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "NIFTY BANK": "^NSEBANK",
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services (TCS)": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India (SBI)": "SBIN.NS",
    "Larsen & Toubro (L&T)": "LT.NS",
    "ITC": "ITC.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Wipro": "WIPRO.NS",
    "Titan Company": "TITAN.NS"
}

selected_stock = st.sidebar.selectbox(
    "Select Stock",
    list(stocks.keys())
)
period = st.sidebar.selectbox(
    "Select Time Period",
    ["6mo", "1y", "2y", "5y"],
    index=1
)
refresh = st.sidebar.button("🔄 Refresh Data")

# ==========================================
# FETCH DATA
# ==========================================

with st.spinner("Fetching latest stock data..."):

    df = get_stock_data(
        stocks[selected_stock],
        period=period
    )

    if df.empty:
        st.error("Unable to fetch stock data.")
        st.stop()

    df = add_indicators(df)

    if df.empty:
        st.error("Not enough historical data to calculate indicators.")
        st.stop()

latest = df.iloc[-1]
from datetime import datetime, time

now = datetime.now().time()

market_open = time(9, 15)
market_close = time(15, 30)

if market_open <= now <= market_close:
    market_status = "🟢 OPEN"
else:
    market_status = "🔴 CLOSED"

live_price = float(latest["Close"])

if len(df) > 1:
    previous_close = float(df.iloc[-2]["Close"])
else:
    previous_close = live_price

price_change = live_price - previous_close
price_change_percent = (price_change / previous_close) * 100
# ==========================================
# PREPARE INPUT
# ==========================================

input_data = prepare_input(
    df,
    stocks[selected_stock],
    encoder
)

# ==========================================
# PREDICTION
# ==========================================

predicted_price = model.predict(
    input_data
)[0]

current_price = float(
    latest["Close"]
)
gap = predicted_price - current_price
predicted_open = current_price + (gap * 0.30)
change_percent = (
    (predicted_price - current_price)
    / current_price
) * 100

# ==========================================
# SIGNAL
# ==========================================

ai_signal, ai_points = ai_recommendation(
    current_price,
    predicted_price,
    latest["RSI"],
    latest["MACD"],
    latest["Signal"],
    latest["MA10"],
    latest["MA50"]
)

# ==========================================
# KPI CARDS
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Current Price",
        f"₹ {current_price:.2f}"
    )

with col2:
    st.metric(
        "Predicted Price",
        f"₹ {predicted_price:.2f}"
    )

with col3:
    st.metric(
        "Expected Change",
        f"{change_percent:.2f}%"
    )
st.markdown("## 📡 Live Market Status")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Market", market_status)

c2.metric(
    "Live Price",
    f"₹ {live_price:.2f}",
    f"{price_change:+.2f}"
)

c3.metric(
    "Previous Close",
    f"₹ {previous_close:.2f}"
)

c4.metric(
    "Change %",
    f"{price_change_percent:.2f}%"
)

st.caption(f"Last Updated: {datetime.now().strftime('%d-%m-%Y %I:%M:%S %p')}")   
# ==========================================
# TODAY MARKET SUMMARY
# ==========================================

st.markdown("## 📈 Today's Market Summary")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Open",
    f"₹ {float(latest['Open']):.2f}"
)

m2.metric(
    "High",
    f"₹ {float(latest['High']):.2f}"
)

m3.metric(
    "Low",
    f"₹ {float(latest['Low']):.2f}"
)

m4.metric(
    "Close",
    f"₹ {float(latest['Close']):.2f}"
)
# ==========================================
# TOMORROW PREDICTION
# ==========================================

st.markdown("## 🔮 Tomorrow Prediction")

if change_percent > 2:

    st.success(
        f"Tomorrow Expected Trend: Strong Bullish (+{change_percent:.2f}%)"
    )

elif change_percent > 0:

    st.info(
        f"Tomorrow Expected Trend: Bullish (+{change_percent:.2f}%)"
    )

elif change_percent < -2:

    st.error(
        f"Tomorrow Expected Trend: Strong Bearish ({change_percent:.2f}%)"
    )

else:

    st.warning(
        f"Tomorrow Expected Trend: Bearish ({change_percent:.2f}%)"
    )

st.write(
    f"📍 Current Close: ₹ {current_price:.2f}"
)

st.write(
    f"🎯 Predicted Next Close: ₹ {predicted_price:.2f}"
)
st.write(
    f"🔔 Predicted Tomorrow Open: ₹ {predicted_open:.2f}"
)
st.write(
    f"🎯 Predicted Tomorrow Close: ₹ {predicted_price:.2f}"
)
# ==========================================
# LIVE 1-MINUTE CANDLESTICK CHART
# ==========================================

st.markdown("## ⚡ Live 1-Minute Market Chart")

# Auto refresh every 60 seconds
st_autorefresh(interval=60000, key="live_chart_refresh")

live_df = yf.download(
    stocks[selected_stock],
    period="5d",
    interval="1m",
    auto_adjust=False,
    progress=False
)

if isinstance(live_df.columns, pd.MultiIndex):
    live_df.columns = live_df.columns.get_level_values(0)

live_df.reset_index(inplace=True)

live_df = live_df.dropna(subset=["Open", "High", "Low", "Close"])

# Convert Datetime column
live_df["Datetime"] = pd.to_datetime(live_df["Datetime"])

# Show today's data only
today = pd.Timestamp.now().date()

today_data = live_df[
    live_df["Datetime"].dt.date == today
]

# If market is closed or today has no data,
# show last trading day's chart
if today_data.empty:
    last_day = live_df["Datetime"].dt.date.max()
    live_df = live_df[
        live_df["Datetime"].dt.date == last_day
    ]
else:
    live_df = today_data

if not live_df.empty:

    fig_live = go.Figure(
        data=[
            go.Candlestick(
                x=live_df["Datetime"],
                open=live_df["Open"],
                high=live_df["High"],
                low=live_df["Low"],
                close=live_df["Close"],
                name="Live Market"
            )
        ]
    )

    fig_live.update_layout(
        dragmode="pan",
        template="plotly_dark",
        height=550,
        xaxis_title="Time",
        yaxis_title="Price (₹)",
        xaxis_rangeslider_visible=False
    )
    fig_live.update_xaxes(
    fixedrange=False
    )

    fig_live.update_yaxes(
    fixedrange=False
    )

    st.plotly_chart(
    fig_live,
    use_container_width=True,
    config={
        "scrollZoom": True,
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToAdd": [
            "zoom2d",
            "pan2d",
            "select2d",
            "lasso2d",
            "zoomIn2d",
            "zoomOut2d",
            "autoScale2d",
            "resetScale2d"
        ]
    }
)

    st.success(
        f"🟢 Latest Price : ₹ {float(live_df['Close'].iloc[-1]):.2f}"
    )

else:

    st.warning(
        "Market is closed. Showing last available intraday data."
    )
# ==========================================
# AI RECOMMENDATION PANEL
# ==========================================

st.markdown("## 🤖 AI Recommendation")

if ai_signal == "STRONG BUY":

    st.success(
        f"AI Recommendation: {ai_signal}"
    )

elif ai_signal == "BUY":

    st.info(
        f"AI Recommendation: {ai_signal}"
    )

elif ai_signal == "HOLD":

    st.warning(
        f"AI Recommendation: {ai_signal}"
    )

else:

    st.error(
        f"AI Recommendation: {ai_signal}"
    )

for point in ai_points:
    st.write("✅", point)
st.markdown("## 📝 AI Market Analysis")

if predicted_price > current_price:
    st.write("✅ Model predicts upward movement for the next session.")
else:
    st.write("⚠️ Model predicts downside risk for the next session.")

if latest["RSI"] > 70:
    st.write("⚠️ RSI indicates overbought conditions.")
elif latest["RSI"] < 30:
    st.write("✅ RSI indicates oversold conditions.")
else:
    st.write("ℹ️ RSI is in a neutral zone.")

if latest["MACD"] > latest["Signal"]:
    st.write("✅ MACD is bullish.")
else:
    st.write("⚠️ MACD is bearish.")

if latest["MA10"] > latest["MA50"]:
    st.write("✅ Short-term trend is stronger than long-term trend.")
else:
    st.write("⚠️ Long-term trend remains stronger.")   
# ==========================================
# STOCK INFORMATION
# ==========================================

st.markdown("## 🏢 Stock Information")

st.write(
    f"Selected Stock: {selected_stock}"
)   
# ==========================================
# INDICATORS SUMMARY
# ==========================================

st.markdown("## 📊 Technical Indicators")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "RSI",
    round(float(latest["RSI"]), 2)
)

c2.metric(
    "MACD",
    round(float(latest["MACD"]), 2)
)

c3.metric(
    "MA10",
    round(float(latest["MA10"]), 2)
)

c4.metric(
    "MA50",
    round(float(latest["MA50"]), 2)
)
st.divider()
# ==========================================
# CANDLESTICK CHART
# ==========================================

st.subheader("🕯️ Candlestick Chart")

fig = go.Figure(
    data=[
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"]
        )
    ]
)

fig.update_layout(
    template="plotly_dark",
    height=600,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "scrollZoom": True,
        "displayModeBar": True,
        "displaylogo": False
    }
)

# ==========================================
# MOVING AVERAGES
# ==========================================

st.subheader("📈 MA10 vs MA50")

fig_ma = go.Figure()

fig_ma.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Close"],
        mode="lines",
        name="Close Price"
    )
)

fig_ma.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["MA10"],
        mode="lines",
        name="MA10"
    )
)

fig_ma.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["MA50"],
        mode="lines",
        name="MA50"
    )
)

fig_ma.update_layout(
    template="plotly_dark",
    height=500
)

st.plotly_chart(
    fig_ma,
    use_container_width=True
)

# ==========================================
# RSI CHART
# ==========================================

st.subheader("📊 RSI Indicator")

fig_rsi = go.Figure()

fig_rsi.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["RSI"],
        mode="lines",
        name="RSI"
    )
)

fig_rsi.add_hline(
    y=70,
    line_dash="dash"
)

fig_rsi.add_hline(
    y=30,
    line_dash="dash"
)

fig_rsi.update_layout(
    template="plotly_dark",
    height=450
)

st.plotly_chart(
    fig_rsi,
    use_container_width=True
)

# ==========================================
# MACD CHART
# ==========================================

st.subheader("📉 MACD Indicator")

fig_macd = go.Figure()

fig_macd.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["MACD"],
        mode="lines",
        name="MACD"
    )
)

fig_macd.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Signal"],
        mode="lines",
        name="Signal Line"
    )
)

fig_macd.update_layout(
    template="plotly_dark",
    height=500
)

st.plotly_chart(
    fig_macd,
    use_container_width=True
)

# ==========================================
# BOLLINGER BANDS
# ==========================================

st.subheader("📡 Bollinger Bands")

fig_bb = go.Figure()

fig_bb.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Close"],
        mode="lines",
        name="Close"
    )
)

fig_bb.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Upper_Band"],
        mode="lines",
        name="Upper Band"
    )
)

fig_bb.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Lower_Band"],
        mode="lines",
        name="Lower Band"
    )
)

fig_bb.update_layout(
    template="plotly_dark",
    height=500
)

st.plotly_chart(
    fig_bb,
    use_container_width=True
)
# ==========================================
# TECHNICAL INDICATOR SUMMARY
# ==========================================

st.subheader(
    "📋 Technical Indicator Summary"
)

indicator_df = pd.DataFrame({

    "Indicator": [
        "RSI",
        "MACD",
        "Signal",
        "MA10",
        "MA50",
        "Volatility"
    ],

    "Value": [
        round(float(latest["RSI"]), 2),
        round(float(latest["MACD"]), 2),
        round(float(latest["Signal"]), 2),
        round(float(latest["MA10"]), 2),
        round(float(latest["MA50"]), 2),
        round(float(latest["Volatility"]), 4)
    ]

})

st.dataframe(
    indicator_df,
    use_container_width=True
)

# ==========================================
# DISCLAIMER
# ==========================================

st.warning(
    "This prediction is generated using Machine Learning and Technical Indicators. It should not be considered financial advice."
)
# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    """
    <center>
    <h4 style='color:#00FFAA'>
    Stock Market Analysis & Prediction System
    </h4>

    <p>
    Built using Machine Learning, Streamlit,
    Technical Indicators and Yahoo Finance
    </p>

    </center>
    """,
    unsafe_allow_html=True
)