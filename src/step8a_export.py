"""
Step 8a: Pack everything the dashboard needs into one small folder: dashboard_data/
"""
import shutil
import pandas as pd

from config import TICKERS                       # the stocks that have features
from step3_baselines import FEAT_DIR             # data/features
from step5_walkforward import RESULT_DIR        # data/results

DATA_DIR = FEAT_DIR.parent                       # data/
OUT = DATA_DIR.parent / "dashboard_data"         # <project>/dashboard_data
MODELS = ["LogReg", "RandForest", "XGBoost", "AlwaysUp", "SameAsToday"]


def accuracy_rows(ticker):
    """Accuracy of each model for ONE stock: average over folds, and folds that beat the baseline."""
    res = pd.read_csv(RESULT_DIR / f"{ticker}_preds.csv", index_col="Date", parse_dates=True)
    acc = pd.DataFrame({m: (res[f"pred_{m}"] == res["target"]).groupby(res["fold"]).mean() * 100
                        for m in MODELS})
    best_base = acc[["AlwaysUp", "SameAsToday"]].max(axis=1)   # the bar to beat, per fold
    return [{"ticker": ticker, "model": m,
             "accuracy_%": round(acc[m].mean(), 2),
             "folds_beating_baseline": int((acc[m] > best_base).sum()),
             "folds": len(acc)} for m in MODELS]


def main():
    for sub in ["prices", "preds", "equity"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)       # make the folders

    acc_rows = []
    for t in TICKERS:
        shutil.copy(DATA_DIR / f"{t}.csv", OUT / "prices" / f"{t}.csv")                 # prices for candlesticks
        shutil.copy(RESULT_DIR / f"{t}_preds.csv", OUT / "preds" / f"{t}_preds.csv")    # predictions
        eq = RESULT_DIR / f"{t}_equity.csv"                                              # profit curves (step 7)
        if eq.exists():
            shutil.copy(eq, OUT / "equity" / f"{t}_equity.csv")
        acc_rows += accuracy_rows(t)

    shutil.copy(RESULT_DIR / "backtest_summary.csv", OUT / "backtest_summary.csv")      # step 7 table
    pd.DataFrame(acc_rows).to_csv(OUT / "accuracy_summary.csv", index=False)            # accuracy table
    pd.Series(TICKERS).to_csv(OUT / "tickers.csv", index=False, header=["ticker"])      # stock list
    print(f"Saved {len(TICKERS)} stocks to {OUT}")


if __name__ == "__main__":
    main()
