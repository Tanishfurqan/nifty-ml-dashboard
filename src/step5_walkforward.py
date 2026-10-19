"""
Step 5: Walk-forward validation.
Train on the past, test on the next block of days, then move forward and repeat.
Input:  data/features/<TICKER>_features.csv
Output: data/results/<TICKER>_preds.csv  (predictions, used later by the backtest)
"""
import pandas as pd

from step2_features import FEATURE_COLS            # the 9 allowed input columns
from step3_baselines import FEAT_DIR, TICKERS      # feature folder and stock list
from step4_models import get_models                # fresh untrained models

RESULT_DIR = FEAT_DIR.parent / "results"           # <project>/data/results
MIN_TRAIN = 750      # first training window = about 3 years of days
TEST_BLOCK = 126     # each test block = about 6 months of days


def walk_forward(df):
    """Run all folds for ONE stock. Returns a table with one row per tested day."""
    X, y = df[FEATURE_COLS], df["target"]
    blocks = []
    fold = 0
    start = MIN_TRAIN            # row number where the first test block begins

    while start < len(df):
        end = min(start + TEST_BLOCK, len(df))

        X_train, y_train = X.iloc[:start], y.iloc[:start]   # the past only
        X_test = X.iloc[start:end]                          # the future it has not seen

        block = pd.DataFrame({
            "fold": fold,
            "target": y.iloc[start:end],
            "Close": df["Close"].iloc[start:end],
        }, index=df.index[start:end])

        for name, model in get_models().items():   # a NEW model each fold
            model.fit(X_train, y_train)
            block[f"pred_{name}"] = model.predict(X_test)

        block["pred_AlwaysUp"] = 1
        block["pred_SameAsToday"] = (df["ret_1d"].iloc[start:end] > 0).astype(int)

        blocks.append(block)
        start = end              # slide forward to the next test block
        fold += 1

    return pd.concat(blocks)


def summarize(ticker, res):
    """Accuracy of every model/baseline in each fold, then averaged."""
    acc = {}
    for col in [c for c in res.columns if c.startswith("pred_")]:
        acc[col[5:]] = (res[col] == res["target"]).groupby(res["fold"]).mean() * 100
    acc = pd.DataFrame(acc)

    best_base = acc[["AlwaysUp", "SameAsToday"]].max(axis=1)   # the bar to beat, per fold

    row = {"ticker": ticker}
    row.update(acc.mean().round(1).to_dict())
    for m in ["LogReg", "RandForest"]:
        row[f"{m}_beats"] = f"{int((acc[m] > best_base).sum())}/{len(acc)}"
    return row


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for ticker in TICKERS:
        df = pd.read_csv(FEAT_DIR / f"{ticker}_features.csv",
                         index_col="Date", parse_dates=True).sort_index()
        res = walk_forward(df)
        res.to_csv(RESULT_DIR / f"{ticker}_preds.csv")
        rows.append(summarize(ticker, res))
        print(f"done {ticker}: {res['fold'].nunique()} folds, {len(res)} test days")

    table = pd.DataFrame(rows).set_index("ticker")
    print()
    print(table.to_string())
    print()
    cols = ["LogReg", "RandForest", "AlwaysUp", "SameAsToday"]
    print("Average accuracy over all stocks and folds:")
    print(table[cols].mean().round(1))


if __name__ == "__main__":
    main()
