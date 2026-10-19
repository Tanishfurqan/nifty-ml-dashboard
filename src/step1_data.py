"""
Step 1: Download 10 years of daily data for the Nifty 50 stocks.
Output: one CSV file per stock inside the data/ folder.
"""
from pathlib import Path   # built into Python; handles file and folder paths
import pandas as pd        # tables (DataFrame), dates, saving CSV
import yfinance as yf      # downloads stock prices from Yahoo Finance

from config import CANDIDATES as TICKERS    # the stock list lives in config.py

# <project folder>/data
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

END = pd.Timestamp.today().normalize()      # today's date
START = END - pd.DateOffset(years=10)       # 10 years before today


def download(ticker):
    """Download daily prices for ONE stock and return a clean table."""
    df = yf.download(
        ticker,
        start=START,
        end=END,
        auto_adjust=True,   # fixes old prices for splits and bonus shares
        progress=False,     # hides the progress bar
    )
    # Newer yfinance gives two-level column names, like ("Close", "TCS.NS").
    # Keep only the first level so the CSV has one clean header.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    return df[["Open", "High", "Low", "Close", "Volume"]]


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)     # create data/ if missing

    for ticker in TICKERS:
        df = download(ticker)

        if df.empty:                                # nothing came back from Yahoo
            print(f"{ticker:14s} FAILED: no data downloaded")
            continue

        df.to_csv(DATA_DIR / f"{ticker}.csv")       # save as data/RELIANCE.NS.csv etc.
        print(f"{ticker:14s} rows={len(df):5d}  "
              f"{df.index.min().date()} -> {df.index.max().date()}  "
              f"missing={int(df.isna().sum().sum())}")


if __name__ == "__main__":
    main()
