"""Profit of each strategy vs buy-and-hold."""
import plotly.express as px
import streamlit as st
from utils import get_backtest, get_equity, get_tickers

st.title("Backtest: models vs buy-and-hold")
st.caption("Rule: hold the stock when the model predicts up, otherwise stay in cash. "
           "Cost: 0.1% per buy or sell (an assumption).")

bt = get_backtest()
order = ["BuyHold", "LogReg", "RandForest", "XGBoost", "SameAsToday"]

stats = bt.groupby("strategy").agg(
    mean_return=("total_return_%", "mean"),
    median_return=("total_return_%", "median"),
    mean_drawdown=("max_drawdown_%", "mean"),
    avg_trades=("trades", "mean"),
).reindex(order).round(1)

st.subheader("Average over all stocks")
st.dataframe(stats)
st.caption("Median is shown too: a few big winners can pull the average up.")

fig = px.bar(stats.reset_index(), x="strategy", y=["mean_return", "median_return"],
             barmode="group", labels={"value": "Total return (%)"})
st.plotly_chart(fig)

# In how many stocks did each model beat buy-and-hold?
pivot = bt.pivot(index="ticker", columns="strategy", values="total_return_%")
st.subheader("Stocks where the strategy beat buy-and-hold")
st.dataframe({m: f"{int((pivot[m] > pivot['BuyHold']).sum())} of {len(pivot)}" for m in order[1:]},
             width=300)

# Profit curve of one stock
st.subheader("Profit curve for one stock")
ticker = st.selectbox("Stock", get_tickers())
log_scale = st.checkbox("Log scale", value=False)
eq = get_equity(ticker)                        # 1.0 = starting money
fig2 = px.line(eq, labels={"value": "Growth of 1 rupee", "variable": "Strategy", "index": "Date"})
if log_scale:
    fig2.update_yaxes(type="log")
st.plotly_chart(fig2)
st.dataframe(bt[bt["ticker"] == ticker].set_index("strategy").drop(columns="ticker"))
