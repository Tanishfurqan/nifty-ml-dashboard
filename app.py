"""Home page. Run the whole dashboard with:  streamlit run app.py"""
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from utils import DATA, get_accuracy, get_backtest, get_tickers

st.set_page_config(page_title="Nifty 50 ML Evaluation", layout="wide")
st.title("Can ML predict the next-day moves of Nifty 50 stocks?")
st.caption("Walk-forward tests, compared with simple baselines and buy-and-hold, after trading costs.")

# ---------- load the saved results (no training happens here) ----------
acc, bt, tickers = get_accuracy(), get_backtest(), get_tickers()
avg_acc = acc.groupby("model")["accuracy_%"].mean()          # average accuracy per model
avg_ret = bt.groupby("strategy")["total_return_%"].mean()    # average backtest return per strategy
models = ["LogReg", "RandForest", "XGBoost"]
best_model = avg_acc[models].idxmax()                        # name of the best direction model


def result_label(gap):
    """Gap = best model minus best simple rule, in points. Under 1 point = no edge."""
    return "Small edge" if gap >= 1 else "No edge"


# ---------- section 1: the two questions side by side ----------
st.subheader("Two questions, two answers")
st.caption("Same 47 stocks, same dates, same walk-forward testing. Only the question changes.")

best_dir = avg_acc[models].max()                                  # best direction model
rule_dir = max(avg_acc["AlwaysUp"], avg_acc["SameAsToday"])       # best simple rule for direction
rows = [{
    "Question": "Will tomorrow close UP or DOWN?",
    "Best model": f"{best_dir:.1f}%",
    "Best simple rule": f"{rule_dir:.1f}%",
    "Points above rule": round(best_dir - rule_dir, 1),
    "Result": result_label(best_dir - rule_dir),
}]

vol_file = DATA / "volatility_summary.csv"                        # made by step9_volatility.py
vol_ok = vol_file.exists()
if vol_ok:
    v = pd.read_csv(vol_file)[["LogReg", "XGBoost", "Persistence", "AlwaysBig"]].mean()
    best_vol = max(v["LogReg"], v["XGBoost"])
    rule_vol = max(v["Persistence"], v["AlwaysBig"])
    rows.append({
        "Question": "Will tomorrow's move be BIG or SMALL?",
        "Best model": f"{best_vol:.1f}%",
        "Best simple rule": f"{rule_vol:.1f}%",
        "Points above rule": round(best_vol - rule_vol, 1),
        "Result": result_label(best_vol - rule_vol),
    })
st.table(pd.DataFrame(rows))
st.caption("A 'rule' is a hand-written baseline (such as always-up). A model must beat it to count.")

# link to the details page, only if that page file exists
if vol_ok and (Path(__file__).parent / "pages" / "4_Volatility.py").exists():
    st.page_link("pages/4_Volatility.py", label="See the full volatility experiment")

# ---------- section 2: question 1 in detail (direction) ----------
st.header("Question 1: up or down?")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Stocks tested", len(tickers))
c2.metric(f"Direction: best model ({best_model})", f"{avg_acc[best_model]:.1f}%")
c3.metric("Direction: always-up baseline", f"{avg_acc['AlwaysUp']:.1f}%")
c4.metric("Coin flip", "50.0%")

st.subheader("Average accuracy")
chart = avg_acc.reindex(models + ["AlwaysUp", "SameAsToday"]).reset_index()
fig = px.bar(chart, x="model", y="accuracy_%", text_auto=".1f")
fig.add_hline(y=50, line_dash="dash", annotation_text="coin flip (50%)")
fig.update_yaxes(range=[40, 60])               # zoomed axis: differences look bigger than they are
st.plotly_chart(fig)
st.caption("The y-axis is zoomed to 40-60%. All bars are within about 1.5 points of each other.")

st.subheader("Average backtest return (0.1% cost per trade)")
best_ret = avg_ret[models].max()
st.write(f"Buy-and-hold: **{avg_ret['BuyHold']:.0f}%**. "
         f"Best model: **{best_ret:.0f}%**.")

# ---------- verdict ----------
st.subheader("Verdict")
st.info(
    "None of the models beat simple baselines at predicting next-day direction, "
    "and all of them lost to buy-and-hold after trading costs. "
    "This fits the idea that public price data is already priced in. "
    "Predicting the size of tomorrow's move worked slightly better, "
    "but it says nothing about which way the stock goes."
)

# ---------- limits ----------
st.subheader("Limits of this study")
st.markdown(
    "- Uses today's Nifty members over 10 years, so results may be slightly optimistic (survivorship bias).\n"
    "- Stocks recently listed were skipped (too little history).\n"
    "- Cost of 0.1% per trade is an assumption; real costs, tax and slippage are not modelled.\n"
    "- Stocks move together, so 47 stocks are fewer than 47 independent tests.\n"
    "- This is a learning project, not investment advice."
)