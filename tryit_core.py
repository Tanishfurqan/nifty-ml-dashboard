"""
Logic for the Try it page: load the saved models, build the input row, predict, explain.
There is no Streamlit code in this file, so it can be tested on its own.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb

# The 3 model files made by src/step11_export_models.py (nothing here is invented).
MODEL_FILES = {
    "Logistic Regression": "dir_logreg.joblib",
    "Random Forest": "dir_rf.joblib",
    "XGBoost": "dir_xgb.joblib",
}
OTHER_FILES = ["feature_ranges.json", "latest_features.csv", "recent_prices.csv"]
FIX = "Fix: run  python src/step11_export_models.py  from the project folder."


class ArtifactError(Exception):
    """A required file is missing or does not match the training setup."""


def load_artifacts(folder):
    """Load models, feature info, latest real values and recent prices. Raises ArtifactError with a clear message."""
    folder = Path(folder)
    missing = [n for n in list(MODEL_FILES.values()) + OTHER_FILES if not (folder / n).exists()]
    if missing:
        raise ArtifactError(f"Missing file(s) in {folder}: {', '.join(missing)}. {FIX}")
    try:
        models = {name: joblib.load(folder / f) for name, f in MODEL_FILES.items()}
    except Exception as err:                                   # e.g. saved with other library versions
        raise ArtifactError(f"Could not load a model file ({type(err).__name__}: {err}). "
                            f"It may have been saved with different library versions. {FIX}") from err

    meta = json.loads((folder / "feature_ranges.json").read_text())
    for name, model in models.items():                         # same feature names, same ORDER as training
        trained_on = list(getattr(model, "feature_names_in_", []))
        if trained_on != meta["features"]:
            raise ArtifactError(f"{name} was trained on {trained_on}, but the saved feature list is "
                                f"{meta['features']}. {FIX}")
    latest = pd.read_csv(folder / "latest_features.csv").set_index("ticker")
    prices = pd.read_csv(folder / "recent_prices.csv", parse_dates=["Date"])
    absent = [f for f in meta["features"] if f not in latest.columns]
    if absent:
        raise ArtifactError(f"latest_features.csv has no column(s) {absent}. {FIX}")
    return models, meta, latest, prices


def build_row(meta, actual, rsi=None, ret_pct=None, ratio=None):
    """One row of model inputs. Starts from the stock's REAL values; any value given replaces the real one.
    rsi is 0-100, ret_pct is in percent (1.0 = +1%), ratio = close / 20-day average (1.02 = 2% above)."""
    row = {f: float(actual[f]) for f in meta["features"]}
    if rsi is not None:
        row["rsi_14"] = float(rsi)
    if ret_pct is not None:
        row["ret_1d"] = float(ret_pct) / 100.0
    if ratio is not None:
        row["close_vs_ma20"] = float(ratio) - 1.0              # the model uses ratio minus 1
    return pd.DataFrame([row])[meta["features"]]               # training column order


def predict_all(models, x):
    """Probability of UP from each model. The saved LogReg pipeline applies its own scaler inside."""
    out = {}
    for name, model in models.items():
        col = list(model.classes_).index(1)                    # column that holds class 1 (= up)
        out[name] = float(model.predict_proba(x)[0, col])
    return out


def direction(p_up):
    return "Up" if p_up >= 0.5 else "Down"                     # same 0.5 cut-off the models use


def outside_range(meta, feature, value):
    """True if the value is outside the 1st-99th percentile range seen in training."""
    return value < meta["q01"][feature] or value > meta["q99"][feature]


def explain_logreg(pipe, x):
    """Logistic Regression: coefficient x scaled value for each input (log-odds units)."""
    scaler = pipe.named_steps["standardscaler"]
    clf = pipe.named_steps["logisticregression"]
    return pd.Series(clf.coef_[0] * scaler.transform(x)[0], index=x.columns)


def explain_xgb(model, x):
    """XGBoost: SHAP values from XGBoost's own tree-SHAP (pred_contribs). Needs no extra package."""
    contribs = model.get_booster().predict(xgb.DMatrix(x), pred_contribs=True)[0]
    return pd.Series(contribs[:-1], index=x.columns), float(contribs[-1])   # last value = average (bias)


def rf_importance(model, features):
    """Random Forest: global importance (how much each input mattered overall, NOT for this prediction)."""
    return pd.Series(model.feature_importances_, index=features)