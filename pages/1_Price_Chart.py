"""Candlestick chart for one stock."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils import get_prices, get_tickers

st.title("Price chart")

ticker = st.selectbox("Stock", get_tickers())
years = st.selectbox("Period", [1, 3, 5, 10], index=1, format_func=lambda y: f"Last {y} year(s)")
show_ma = st.checkbox("Show 20-day moving average", value=True)

df = get_prices(ticker)
df["MA20"] = df["Close"].rolling(20).mean()                      # compute BEFORE cutting the period
df = df[df.index >= df.index.max() - pd.DateOffset(years=years)]  # keep only the chosen period

# Candlestick: each candle = one day (open, high, low, close)
fig = go.Figure(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"], name=ticker,
))
if show_ma:
    fig.add_scatter(x=df.index, y=df["MA20"], name="20-day average", line=dict(width=1.5))

fig.update_layout(height=550, xaxis_rangeslider_visible=False, yaxis_title="Price (INR, adjusted)")
st.plotly_chart(fig)
st.caption("Prices are adjusted for splits and bonus shares, so old prices look lower than they were.")
