"""
Step 4: Train Logistic Regression and Random Forest, test on the newest 20% of days.
Input:  data/features/<TICKER>_features.csv  (from step2)
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier      # many decision trees that vote
from sklearn.linear_model import LogisticRegression      # simple straight-line model
from sklearn.pipeline import make_pipeline               # chains steps together
from sklearn.preprocessing import StandardScaler         # puts features on the same scale

from step2_features import FEATURE_COLS                  # the only allowed input columns
from step3_baselines import score_baselines, FEAT_DIR, TICKERS, TEST_FRACTION


def get_models():
    """Return a fresh, untrained copy of each model."""
    return {
        # StandardScaler is INSIDE the pipeline, so it learns its averages from
        # training days only. This avoids leaking test data.
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),

        # Shallow trees and big leaves stop the forest from memorising noise.
        "RandForest": RandomForestClassifier(
            n_estimators=200,       # 200 trees
            max_depth=5,            # each tree asks at most 5 questions
            min_samples_leaf=20,    # a leaf needs at least 20 days
            random_state=42,        # same result every run
            n_jobs=-1,              # use all CPU cores
        ),
    }


def run_ticker(ticker):
    """Train and test both models for ONE stock. Returns a list of result rows."""
    df = pd.read_csv(FEAT_DIR / f"{ticker}_features.csv",
                     index_col="Date", parse_dates=True).sort_index()

    split = int(len(df) * (1 - TEST_FRACTION))
    train, test = df.iloc[:split], df.iloc[split:]

    X_train, y_train = train[FEATURE_COLS], train["target"]
    X_test, y_test = test[FEATURE_COLS], test["target"]

    base = score_baselines(ticker)
    best_baseline = max(base["always_up_%"], base["same_as_today_%"])

    rows = []
    for name, model in get_models().items():
        model.fit(X_train, y_train)                        # TRAINING happens here
        train_acc = model.score(X_train, y_train) * 100    # score on days it studied
        pred = model.predict(X_test)                       # guesses on unseen days
        test_acc = (pred == y_test).mean() * 100           # score on unseen days
        rows.append({
            "ticker": ticker,
            "model": name,
            "train_%": round(train_acc, 1),
            "test_%": round(test_acc, 1),
            "pred_up_%": round(pred.mean() * 100, 1),      # how often it said "up"
            "best_baseline_%": best_baseline,
        })
    return rows


def main():
    all_rows = []
    for t in TICKERS:
        all_rows += run_ticker(t)
    table = pd.DataFrame(all_rows)
    print(table.to_string(index=False))
    print()
    print(table.groupby("model")[["train_%", "test_%", "pred_up_%", "best_baseline_%"]].mean().round(1))


if __name__ == "__main__":
    main()
