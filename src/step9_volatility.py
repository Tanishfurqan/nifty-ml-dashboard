"""
Step 9: Volatility experiment.
Question: can ML predict if TOMORROW's move will be BIG or SMALL (up or down does not matter)?
Input:  data/features/<TICKER>_features.csv  (from step2)
Output: dashboard_data/volatility_summary.csv
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import TICKERS                              # the stocks with features
from step3_baselines import FEAT_DIR                    # data/features
from step5_walkforward import MIN_TRAIN, TEST_BLOCK     # SAME folds as step 5

OUT_DIR = FEAT_DIR.parent.parent / "dashboard_data"     # where the dashboard reads from

# Inputs for this experiment (all use only today and past days).
VOL_FEATURES = ["abs_ret_1d", "range_1d", "vol_5d", "vol_20d",
                "vol_ratio", "volume_change", "rsi_14", "close_vs_ma20"]


def get_models():
    """Fresh untrained models (small, safe settings)."""
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                                 random_state=42, n_jobs=-1, eval_metric="logloss"),
    }


def build_data(df):
    """Add the volatility features, the big/small target and the persistence baseline."""
    d = df.copy()
    d["abs_ret_1d"] = d["ret_1d"].abs()                       # size of today's move
    d["range_1d"] = (d["High"] - d["Low"]) / d["Close"]       # today's high-low range
    d["vol_5d"] = d["ret_1d"].rolling(5).std()                # jumpiness, last 5 days
    d["vol_ratio"] = d["vol_5d"] / d["vol_20d"]               # short-term vs 20-day jumpiness

    # "Typical" move size = median of the last 60 days (past only, known today).
    typical = d["abs_ret_1d"].rolling(60).median()
    nxt = d["abs_ret_1d"].shift(-1)                           # tomorrow's move size
    d["target_vol"] = (nxt > typical).astype(float)           # 1 = tomorrow is BIG
    d.loc[nxt.isna() | typical.isna(), "target_vol"] = np.nan # no answer available -> drop later

    # Baseline rule: "if today was big, tomorrow will be big".
    d["base_persist"] = (d["abs_ret_1d"] > typical).astype(int)

    return d.dropna(subset=VOL_FEATURES + ["target_vol"])


def run_stock(ticker):
    """Walk-forward for ONE stock. Returns accuracy per fold for models and baselines."""
    df = pd.read_csv(FEAT_DIR / f"{ticker}_features.csv",
                     index_col="Date", parse_dates=True).sort_index()
    d = build_data(df)
    X, y = d[VOL_FEATURES], d["target_vol"].astype(int)

    rows, start, fold = [], MIN_TRAIN, 0
    while start < len(d):
        end = min(start + TEST_BLOCK, len(d))
        y_test = y.iloc[start:end]
        acc = {"fold": fold}
        for name, model in get_models().items():
            model.fit(X.iloc[:start], y.iloc[:start])          # train on the past only
            acc[name] = (model.predict(X.iloc[start:end]) == y_test).mean() * 100
        acc["AlwaysBig"] = (y_test == 1).mean() * 100
        acc["Persistence"] = (d["base_persist"].iloc[start:end] == y_test).mean() * 100
        rows.append(acc)
        start, fold = end, fold + 1
    return pd.DataFrame(rows).set_index("fold")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for i, ticker in enumerate(TICKERS, 1):
        acc = run_stock(ticker)
        best_base = acc[["AlwaysBig", "Persistence"]].max(axis=1)   # bar to beat, per fold
        row = {"ticker": ticker, "folds": len(acc)}
        row.update(acc.mean().round(2).to_dict())                   # average accuracy per column
        for m in ["LogReg", "XGBoost"]:
            row[f"{m}_beats"] = int((acc[m] > best_base).sum())     # folds beating best baseline
        out.append(row)
        if i % 10 == 0:
            print(f"done {i}/{len(TICKERS)}")

    table = pd.DataFrame(out)
    table.to_csv(OUT_DIR / "volatility_summary.csv", index=False)  # the dashboard reads this

    print()
    print("Average accuracy over all stocks:")
    print(table[["LogReg", "XGBoost", "AlwaysBig", "Persistence"]].mean().round(1))
    print()
    for m in ["LogReg", "XGBoost"]:
        share = table[f"{m}_beats"].sum() / table["folds"].sum() * 100
        stocks = int((table[m] > table["Persistence"]).sum())
        print(f"{m}: beat the best baseline in {share:.0f}% of folds; "
              f"beat Persistence on average in {stocks} of {len(table)} stocks")


if __name__ == "__main__":
    main()
