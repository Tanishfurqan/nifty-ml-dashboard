"""
Step 11: Train ONE final copy of each direction model and save everything the "Try it" page needs.
Run from the project folder:  python src/step11_export_models.py
Output (inside dashboard_data/models/):
  dir_logreg.joblib, dir_rf.joblib, dir_xgb.joblib   the 3 trained models (LogReg includes its scaler)
  feature_ranges.json                                 feature names IN TRAINING ORDER + slider ranges
  latest_features.csv                                 newest real indicator values for each stock
  recent_prices.csv                                   last 180 days of prices for the charts
"""
import json
import joblib                                    # saves a trained model to a file
import pandas as pd

from config import TICKERS                       # the stocks that have a features file
from step2_features import FEATURE_COLS, load_prices, make_features   # the SAME code used in training
from step3_baselines import FEAT_DIR             # data/features
from step4_models import get_models              # LogReg (scaler inside) and RandForest, same settings
from step6_xgboost import get_xgb                # XGBoost, same settings

OUT = FEAT_DIR.parent.parent / "dashboard_data" / "models"
OUT.mkdir(parents=True, exist_ok=True)

# 1. Stack every stock's features file into one big table (inputs + the up/down answer).
frames = [pd.read_csv(FEAT_DIR / f"{t}_features.csv", index_col="Date", parse_dates=True).sort_index()
          for t in TICKERS]
data = pd.concat(frames)
X, y = data[FEATURE_COLS], data["target"]        # FEATURE_COLS fixes the column order

# 2. Train each model once on all days, then save it as a file.
#    (The walk-forward tests never saved their models, so these files are made here, once.)
base = get_models()
joblib.dump(base["LogReg"].fit(X, y), OUT / "dir_logreg.joblib")
joblib.dump(base["RandForest"].fit(X, y), OUT / "dir_rf.joblib")
joblib.dump(get_xgb().fit(X, y), OUT / "dir_xgb.joblib")

# 3. Feature names in training order, plus slider ranges (1st to 99th percentile).
meta = {
    "features": FEATURE_COLS,
    "q01": X.quantile(0.01).to_dict(),
    "median": X.median().to_dict(),
    "q99": X.quantile(0.99).to_dict(),
    "trained_rows": int(len(X)),
    "n_stocks": len(TICKERS),
    "trained_until": str(data.index.max().date()),
}
(OUT / "feature_ranges.json").write_text(json.dumps(meta))

# 4. Newest REAL indicator values per stock, and recent prices for the charts.
#    The features files drop the newest day (it has no "tomorrow" answer), so we rebuild the
#    features from the raw prices with the same make_features() used in training.
rows, recent = [], []
for t in TICKERS:
    prices = load_prices(t)
    last = make_features(prices)[FEATURE_COLS].dropna().iloc[-1]        # newest day with all inputs
    rows.append({"ticker": t, "date": last.name.date(), "close": float(prices.loc[last.name, "Close"]),
                 **last.to_dict()})
    recent.append(prices.tail(180).reset_index().assign(ticker=t))
pd.DataFrame(rows).to_csv(OUT / "latest_features.csv", index=False)
pd.concat(recent).to_csv(OUT / "recent_prices.csv", index=False)

print(f"Saved 3 models for {len(TICKERS)} stocks to {OUT}")