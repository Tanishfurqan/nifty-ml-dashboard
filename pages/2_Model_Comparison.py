"""Accuracy of each model vs the baselines."""
import plotly.express as px
import streamlit as st
from utils import get_accuracy

st.title("Model comparison")
acc = get_accuracy()
order = ["LogReg", "RandForest", "XGBoost", "AlwaysUp", "SameAsToday"]
models = order[:3]

# One row per model: average over all stocks
summary = acc.groupby("model").agg(
    avg_accuracy=("accuracy_%", "mean"),
    median_accuracy=("accuracy_%", "median"),
    beat=("folds_beating_baseline", "sum"),
    folds=("folds", "sum"),
).reindex(order)
summary["folds_beating_best_baseline_%"] = summary["beat"] / summary["folds"] * 100
summary.loc[["AlwaysUp", "SameAsToday"], "folds_beating_best_baseline_%"] = None  # baselines can't beat themselves

st.subheader("Average over all stocks")
st.dataframe(summary[["avg_accuracy", "median_accuracy", "folds_beating_best_baseline_%"]].round(2))
st.caption(
    "'Best baseline' = the better of the two baselines in each test period. "
    "Beating it by pure luck happens in fewer than half the periods, so about 30-40% is what luck looks like here."
)

fig = px.bar(summary.reset_index(), x="model", y="avg_accuracy", text_auto=".1f")
fig.add_hline(y=50, line_dash="dash", annotation_text="coin flip (50%)")
fig.update_yaxes(range=[40, 60])
st.plotly_chart(fig)

# Per-stock table: rows = stocks, columns = models
st.subheader("Per stock")
pivot = acc.pivot(index="ticker", columns="model", values="accuracy_%")[order]
st.dataframe(pivot.round(1))

st.subheader("In how many stocks did each model beat 'always up'?")
wins = {m: int((pivot[m] > pivot["AlwaysUp"]).sum()) for m in models}
st.dataframe({m: f"{n} of {len(pivot)}" for m, n in wins.items()}, width=300)
