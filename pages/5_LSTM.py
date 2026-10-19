"""LSTM experiment: ONE small LSTM trained on all stocks together, tested in walk-forward folds."""
import pandas as pd
import plotly.express as px
import streamlit as st
from utils import DATA                                   # the dashboard_data folder

st.title("LSTM experiment: a model with memory")
st.markdown(
    "An LSTM reads the **last 20 days in order** and remembers what came before. "
    "Here one small LSTM was trained on **all stocks together** in each walk-forward fold "
    "(train on days before a date, test on the next 6 months)."
)

folds_file = DATA / "lstm_pooled_folds.csv"              # made by step10_lstm_pooled.py
stocks_file = DATA / "lstm_pooled_stocks.csv"
if not folds_file.exists():                              # results not ready yet
    st.info("No LSTM results yet. Run step10_lstm_pooled.py with QUICK_TEST = False.")
    st.stop()                                            # stop drawing the page here

folds = pd.read_csv(folds_file)                          # one row per fold
stocks = pd.read_csv(stocks_file, index_col=0)           # one row per stock
cols = ["LSTM", "LogReg", "AlwaysUp", "SameAsToday"]
avg = folds[cols].mean()                                 # average accuracy over folds

# Headline numbers
c = st.columns(4)
for box, name in zip(c, cols):
    box.metric(name, f"{avg[name]:.1f}%")

# Bar chart of the average accuracy
fig = px.bar(avg.reset_index(name="accuracy_%").rename(columns={"index": "model"}),
             x="model", y="accuracy_%", text_auto=".1f")
fig.add_hline(y=50, line_dash="dash", annotation_text="coin flip (50%)")
fig.update_yaxes(range=[40, 60])                         # zoomed axis: differences look bigger than they are
st.plotly_chart(fig)

# Accuracy in each fold (each fold = one 6-month test period)
st.subheader("Accuracy in each test period")
fig2 = px.line(folds, x="test_start", y=cols, markers=True,
               labels={"value": "accuracy (%)", "test_start": "test period starts"})
fig2.add_hline(y=50, line_dash="dash")
st.plotly_chart(fig2)

# How consistent is the LSTM?
best_base = folds[["AlwaysUp", "SameAsToday"]].max(axis=1)         # the bar to beat, per fold
beat_folds = int((folds["LSTM"] > best_base).sum())
beat_stocks = int((stocks["LSTM"] > stocks["AlwaysUp"]).sum())
st.write(f"The LSTM beat the best simple rule in **{beat_folds} of {len(folds)}** test periods, "
         f"and beat 'always up' in **{beat_stocks} of {len(stocks)}** stocks.")
st.write(f"On average it said 'up' on **{folds['lstm_up_%'].mean():.0f}%** of days "
         "(close to 100% would mean it just copied 'always up').")

# Verdict, worked out from the numbers (not typed by hand)
gap = avg["LSTM"] - max(avg["AlwaysUp"], avg["SameAsToday"])        # points above the best rule
st.info(
    f"The LSTM is {gap:+.1f} points versus the best simple rule. "
    + ("A gap under 1 point is within noise, so it found no edge." if gap < 1
       else "A small edge, but check it is steady across the test periods above.")
)
st.caption("The LogReg here is also trained on all stocks together, so its number can differ "
           "slightly from the per-stock LogReg on the other pages. Direction question, not volatility.")

with st.expander("Training details per fold"):
    st.dataframe(folds.round(1))                                    # seconds and epochs per fold