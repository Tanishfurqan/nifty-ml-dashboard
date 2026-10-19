"""Shared loaders for the dashboard. Everything is read from dashboard_data/ (no training here)."""
from pathlib import Path
import pandas as pd
import streamlit as st

# dashboard_data/ sits next to this file (made by step8a_export.py)
DATA = Path(__file__).resolve().parent / "dashboard_data"


def _read(path):
    # index_col=0 = the first column (Date); parse_dates turns text into real dates
    return pd.read_csv(path, index_col=0, parse_dates=True)


# st.cache_data = load each file once, then reuse it (makes the app fast)
@st.cache_data
def get_tickers():
    return pd.read_csv(DATA / "tickers.csv")["ticker"].tolist()


@st.cache_data
def get_prices(ticker):
    return _read(DATA / "prices" / f"{ticker}.csv")        # Open, High, Low, Close, Volume


@st.cache_data
def get_equity(ticker):
    return _read(DATA / "equity" / f"{ticker}_equity.csv")  # profit curve of each strategy


@st.cache_data
def get_accuracy():
    return pd.read_csv(DATA / "accuracy_summary.csv")       # accuracy per stock and model


@st.cache_data
def get_backtest():
    return pd.read_csv(DATA / "backtest_summary.csv")       # backtest per stock and strategy
