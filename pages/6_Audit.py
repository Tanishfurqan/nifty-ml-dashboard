"""Audit page: shows how the models were trained and lets anyone re-check the accuracy."""
import pandas as pd
import streamlit as st
from utils import DATA, get_prices, get_tickers

st.title("Audit: how the models were trained")
st.markdown(
    "Every number in this dashboard comes from saved predictions. "
    "Each model was **re-trained from scratch in every fold**, using only days "
    "**before** the test period."
)

MODELS = ["LogReg", "RandForest", "XGBoost", "AlwaysUp", "SameAsToday"]
MIN_TRAIN, TEST_BLOCK = 750, 126                         # same numbers as step5_walkforward.py

ticker = st.selectbox("Stock", get_tickers())
res = pd.read_csv(DATA / "preds" / f"{ticker}_preds.csv", index_col=0, parse_dates=True)
prices = get_prices(ticker)

# One row per fold: how much it trained on, when it was tested, and each model's score
rows = []
for f, g in res.groupby("fold"):                         # g = the test days of fold f
    before = prices.index[prices.index < g.index.min()]  # all days before this test block
    row = {
        "fold": f,
        "train_rows": MIN_TRAIN + TEST_BLOCK * f,        # fold 0 trains on 750 days, +126 each fold
        "trained_until": before.max().date(),            # last day the model was allowed to see
        "test_start": g.index.min().date(),
        "test_end": g.index.max().date(),
        "test_days": len(g),
    }
    for m in MODELS:                                     # accuracy = share of matching guesses
        row[m] = round((g[f"pred_{m}"] == g["target"]).mean() * 100, 1)
    rows.append(row)

st.subheader("Folds")
st.dataframe(pd.DataFrame(rows).set_index("fold"))
st.caption("'trained_until' is always before 'test_start': the model never sees the days it is tested on.")

# Let the viewer recount the accuracy from the raw predictions
st.subheader("Check the accuracy yourself")
model = st.selectbox("Model", MODELS)
correct = res[f"pred_{model}"] == res["target"]          # True where the guess was right
st.write(f"**{int(correct.sum()):,}** correct out of **{len(res):,}** predictions "
         f"= **{correct.mean() * 100:.2f}%**")
st.dataframe(res[["fold", "Close", "target"] + [f"pred_{m}" for m in MODELS]])
st.download_button("Download raw predictions (CSV)", res.to_csv().encode(),
                   file_name=f"{ticker}_preds.csv")

# Training time log (only exists after run_all.py has been run)
st.subheader("Training run log")
log_file = DATA / "run_log.csv"
if log_file.exists():
    st.dataframe(pd.read_csv(log_file))
else:
    st.info("A training log appears here after the full pipeline is re-run (run_all.py).")