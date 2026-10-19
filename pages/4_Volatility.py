"""Volatility experiment: can ML predict if tomorrow's move is BIG or SMALL?"""
import pandas as pd
import plotly.express as px
import streamlit as st
from utils import DATA

st.title("Volatility experiment: big move or small move tomorrow?")
st.markdown(
    "Different question: instead of up or down, can the models predict whether tomorrow's move "
    "will be **bigger than the stock's typical move** (median of the last 60 days)? "
    "Direction does not matter here."
)

vol = pd.read_csv(DATA / "volatility_summary.csv")       # made by step9_volatility.py
order = ["LogReg", "XGBoost", "AlwaysBig", "Persistence"]
avg = vol[order].mean()                                   # average over all stocks
folds = vol["folds"].sum()                                # total number of test periods

cols = st.columns(4)
for c, name in zip(cols, order):
    c.metric(name, f"{avg[name]:.1f}%")

fig = px.bar(avg.reset_index(name="accuracy_%").rename(columns={"index": "model"}),
             x="model", y="accuracy_%", text_auto=".1f")
fig.add_hline(y=50, line_dash="dash", annotation_text="coin flip (50%)")
fig.update_yaxes(range=[45, 60])                          # zoomed axis: differences look bigger than they are
st.plotly_chart(fig)
st.caption("The y-axis is zoomed to 45-60%.")

# One row per model: how much better than the smart baseline, and how consistent
rows = []
for m in ["LogReg", "XGBoost"]:
    rows.append({
        "model": m,
        "avg_accuracy_%": round(avg[m], 1),
        "points_above_Persistence": round(avg[m] - avg["Persistence"], 2),
        "stocks_beating_Persistence": f"{int((vol[m] > vol['Persistence']).sum())} of {len(vol)}",
        "folds_beating_best_baseline_%": round(vol[f"{m}_beats"].sum() / folds * 100),
    })
st.subheader("How consistent is it?")
st.dataframe(pd.DataFrame(rows).set_index("model"))
st.caption("By pure luck a model would beat Persistence in about half the stocks.")

best_gain = max(avg["LogReg"], avg["XGBoost"]) - avg["Persistence"]
st.info(
    f"The models beat the simple 'persistence' rule by about {best_gain:.1f} points. "
    "That is a small edge, consistent with the known idea that calm days follow calm days. "
    "It says how big tomorrow's move may be, not which way it goes."
)

st.subheader("Per stock")
st.dataframe(vol.set_index("ticker")[order].round(1))