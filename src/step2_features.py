"""
Step 2: Build features (model inputs) and target (the answer) for each stock.
Input:  data/<TICKER>.csv
Output: data/features/<TICKER>_features.csv
"""
from pathlib import Path
import numpy as np         # we use np.nan (the "empty value")
import pandas as pd

from config import CANDIDATES as TICKERS    # the stock list lives in config.py

MIN_ROWS = 1500   # skip stocks with fewer rows than this (too little history)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = DATA_DIR / "features"

# The ONLY columns the model is allowed to see as inputs.
FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_10d",                       # returns
    "close_vs_ma5", "close_vs_ma20", "ma5_vs_ma20",      # moving average ratios
    "rsi_14", "vol_20d", "volume_change",                # RSI, volatility, volume
]


def load_prices(ticker):
    """Read one stock's CSV and clean it."""
    path = DATA_DIR / f"{ticker}.csv"
    df = pd.read_csv(path)

    # Safety: some yfinance versions save 3 header rows. Skip the extra 2.
    if df.columns[0] == "Price":
        df = pd.read_csv(path, skiprows=[1, 2])
        df = df.rename(columns={"Price": "Date"})

    df["Date"] = pd.to_datetime(df["Date"])             # text -> real dates
    df = df.set_index("Date").sort_index()              # oldest first
    df = df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna()                                    # drop rows with empty values
    df = df[df["Volume"] > 0]                           # drop fake rows with zero volume
    return df


def rsi(close, period=14):
    """RSI: a 0-100 score. High = rose a lot recently, low = fell a lot recently."""
    delta = close.diff()                    # today's close minus yesterday's close
    gain = delta.clip(lower=0)              # keep only the up moves
    loss = -delta.clip(upper=0)             # keep only the down moves (as positive numbers)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def make_features(prices):
    """Create the input columns. Every one uses only today and PAST days."""
    df = prices.copy()
    close = df["Close"]

    df["ret_1d"] = close.pct_change(1)      # % change over 1, 5, 10 days
    df["ret_5d"] = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)

    ma5 = close.rolling(5).mean()           # 5-day and 20-day average close
    ma20 = close.rolling(20).mean()
    # RATIOS (price / average - 1), not raw prices, because raw prices
    # differ between stocks and drift over the years.
    df["close_vs_ma5"] = close / ma5 - 1
    df["close_vs_ma20"] = close / ma20 - 1
    df["ma5_vs_ma20"] = ma5 / ma20 - 1

    df["rsi_14"] = rsi(close, 14)
    df["vol_20d"] = df["ret_1d"].rolling(20).std()      # how jumpy the last 20 days were
    df["volume_change"] = df["Volume"].pct_change()     # % change in shares traded
    return df


def make_target(df):
    """Target = 1 if TOMORROW's close is higher than today's, else 0."""
    next_close = df["Close"].shift(-1)                  # tomorrow's close, placed on today's row
    df["target"] = (next_close > df["Close"]).astype(float)
    df.loc[next_close.isna(), "target"] = np.nan        # last row has no tomorrow
    return df


def leakage_check(prices):
    """Proof test: cutting off the last 30 days must NOT change any earlier feature value."""
    full = make_features(prices)[FEATURE_COLS].dropna()
    cut = make_features(prices.iloc[:-30])[FEATURE_COLS].dropna()
    diff = (full.loc[cut.index] - cut).abs().max().max()
    return diff < 1e-9                                  # True = no leakage


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in TICKERS:
        if not (DATA_DIR / f"{ticker}.csv").exists():
            print(f"{ticker:14s} SKIPPED: no CSV from step 1")
            continue
        prices = load_prices(ticker)
        if len(prices) < MIN_ROWS:
            print(f"{ticker:14s} SKIPPED: only {len(prices)} rows (too little history)")
            continue

        df = make_target(make_features(prices))
        # Remove rows where any feature or the target is empty.
        df = df.dropna(subset=FEATURE_COLS + ["target"])
        df["target"] = df["target"].astype(int)

        ok = leakage_check(prices)
        up_pct = df["target"].mean() * 100              # % of days that went up

        df.to_csv(OUT_DIR / f"{ticker}_features.csv")
        print(f"{ticker:14s} rows={len(df):5d}  "
              f"{df.index.min().date()} -> {df.index.max().date()}  "
              f"up_days={up_pct:.1f}%  leakage_check={'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
