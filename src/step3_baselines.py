"""
Step 3: Score simple fixed rules (no learning) on the test period.
Input:  data/features/<TICKER>_features.csv
"""
from pathlib import Path
import pandas as pd

from config import TICKERS     # the stocks that have a features file

FEAT_DIR = Path(__file__).resolve().parent.parent / "data" / "features"
TEST_FRACTION = 0.2   # the LAST 20% of days is the test period


def score_baselines(ticker):
    df = pd.read_csv(FEAT_DIR / f"{ticker}_features.csv",
                     index_col="Date", parse_dates=True).sort_index()

    split = int(len(df) * (1 - TEST_FRACTION))   # row number where the test period starts
    test = df.iloc[split:]                        # newest days only
    y = test["target"]                            # the real answers: 1 = went up

    pred_always_up = pd.Series(1, index=test.index)         # Rule 1: always say "up"
    pred_same_as_today = (test["ret_1d"] > 0).astype(int)   # Rule 2: copy today's direction

    return {
        "ticker": ticker,
        "test_start": test.index.min().date(),
        "test_days": len(test),
        "up_days_%": round(y.mean() * 100, 1),
        "always_up_%": round((pred_always_up == y).mean() * 100, 1),
        "same_as_today_%": round((pred_same_as_today == y).mean() * 100, 1),
    }


def main():
    rows = [score_baselines(t) for t in TICKERS]
    table = pd.DataFrame(rows).set_index("ticker")
    print(table.to_string())
    print()
    print("Average always_up_%      :", round(table["always_up_%"].mean(), 1))
    print("Average same_as_today_%  :", round(table["same_as_today_%"].mean(), 1))


if __name__ == "__main__":
    main()
