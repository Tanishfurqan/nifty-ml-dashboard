"""
Step 6: XGBoost inside the same walk-forward folds as Step 5.
Input:  data/features/<TICKER>_features.csv
        data/results/<TICKER>_preds.csv   (made by step5)
Output: the same preds file, with a new column pred_XGBoost
"""
import pandas as pd
from xgboost import XGBClassifier            # boosted trees: each tree fixes the last one's mistakes

from step2_features import FEATURE_COLS      # the 9 allowed input columns
from step3_baselines import FEAT_DIR, TICKERS
from step5_walkforward import RESULT_DIR, MIN_TRAIN, TEST_BLOCK   # SAME folds as step5


def get_xgb():
    """A fresh, untrained XGBoost. Small settings on purpose, to avoid overfitting."""
    return XGBClassifier(
        n_estimators=100,       # 100 small trees
        max_depth=2,            # each tree asks at most 2 questions
        learning_rate=0.05,     # each tree corrects only a small part
        subsample=0.8,          # each tree sees 80% of the days
        colsample_bytree=0.8,   # each tree sees 80% of the features
        min_child_weight=20,    # a leaf needs enough days (like min_samples_leaf)
        random_state=42,        # same result every run
        n_jobs=-1,
        eval_metric="logloss",
    )


def xgb_walk_forward(df):
    """Same folds as step5. Returns XGBoost's prediction for every tested day."""
    X, y = df[FEATURE_COLS], df["target"]
    pieces = []
    start = MIN_TRAIN
    while start < len(df):
        end = min(start + TEST_BLOCK, len(df))
        model = get_xgb()                                  # new model each fold
        model.fit(X.iloc[:start], y.iloc[:start])          # train on the past only
        pred = model.predict(X.iloc[start:end])            # predict the next block
        pieces.append(pd.Series(pred, index=df.index[start:end]))
        start = end                                        # slide forward
    return pd.concat(pieces)


def main():
    rows = []
    for ticker in TICKERS:
        df = pd.read_csv(FEAT_DIR / f"{ticker}_features.csv",
                         index_col="Date", parse_dates=True).sort_index()
        res = pd.read_csv(RESULT_DIR / f"{ticker}_preds.csv",
                          index_col="Date", parse_dates=True)

        res["pred_XGBoost"] = xgb_walk_forward(df)         # same dates, so it lines up
        res.to_csv(RESULT_DIR / f"{ticker}_preds.csv")     # save for the backtest

        # Accuracy per fold for every model/baseline, then the average.
        acc = {c[5:]: (res[c] == res["target"]).groupby(res["fold"]).mean() * 100
               for c in res.columns if c.startswith("pred_")}
        acc = pd.DataFrame(acc)
        best_base = acc[["AlwaysUp", "SameAsToday"]].max(axis=1)   # bar to beat, per fold

        row = {"ticker": ticker}
        row.update(acc.mean().round(1).to_dict())
        row["XGB_beats"] = f"{int((acc['XGBoost'] > best_base).sum())}/{len(acc)}"
        row["XGB_pred_up_%"] = round(res["pred_XGBoost"].mean() * 100, 1)
        rows.append(row)
        print(f"done {ticker}")

    table = pd.DataFrame(rows).set_index("ticker")
    print()
    print("Average accuracy over all stocks and folds:")
    print(table[["LogReg", "RandForest", "XGBoost", "AlwaysUp", "SameAsToday"]].mean().round(1))


if __name__ == "__main__":
    main()
