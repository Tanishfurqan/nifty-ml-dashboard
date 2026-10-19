"""
Try it: pick a real stock, see its actual indicators, change three inputs, and compare
what the saved Logistic Regression, Random Forest and XGBoost models say.
Nothing is trained here. Logic lives in tryit_core.py; this file is only the screen.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tryit_core import (ArtifactError, build_row, direction, explain_logreg, explain_xgb,
                        load_artifacts, outside_range, predict_all, rf_importance)
from utils import DATA, get_accuracy

GREEN, RED, BLUE = "#00c27a", "#ff4b5c", "#6c86b8"           # green = up/positive, red = down/negative
EXAMPLE = {"in_rsi": 65.0, "in_ret": 1.0, "in_ratio": 1.02}  # starting experiment values
NAMES = {
    "ADANIENT.NS": "Adani Enterprises", "ADANIPORTS.NS": "Adani Ports", "APOLLOHOSP.NS": "Apollo Hospitals",
    "ASIANPAINT.NS": "Asian Paints", "AXISBANK.NS": "Axis Bank", "BAJAJ-AUTO.NS": "Bajaj Auto",
    "BAJFINANCE.NS": "Bajaj Finance", "BAJAJFINSV.NS": "Bajaj Finserv", "BEL.NS": "Bharat Electronics",
    "BHARTIARTL.NS": "Bharti Airtel", "CIPLA.NS": "Cipla", "COALINDIA.NS": "Coal India",
    "DRREDDY.NS": "Dr. Reddy's", "EICHERMOT.NS": "Eicher Motors", "GRASIM.NS": "Grasim",
    "HCLTECH.NS": "HCL Technologies", "HDFCBANK.NS": "HDFC Bank", "HDFCLIFE.NS": "HDFC Life",
    "HINDALCO.NS": "Hindalco", "HINDUNILVR.NS": "Hindustan Unilever", "ICICIBANK.NS": "ICICI Bank",
    "INDIGO.NS": "IndiGo", "INFY.NS": "Infosys", "ITC.NS": "ITC", "JSWSTEEL.NS": "JSW Steel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank", "LT.NS": "Larsen & Toubro", "M&M.NS": "Mahindra & Mahindra",
    "MARUTI.NS": "Maruti Suzuki", "MAXHEALTH.NS": "Max Healthcare", "NESTLEIND.NS": "Nestle India",
    "NTPC.NS": "NTPC", "ONGC.NS": "ONGC", "POWERGRID.NS": "Power Grid", "RELIANCE.NS": "Reliance Industries",
    "SBILIFE.NS": "SBI Life", "SHRIRAMFIN.NS": "Shriram Finance", "SBIN.NS": "State Bank of India",
    "SUNPHARMA.NS": "Sun Pharma", "TATACONSUM.NS": "Tata Consumer", "TATASTEEL.NS": "Tata Steel",
    "TCS.NS": "Tata Consultancy Services", "TECHM.NS": "Tech Mahindra", "TITAN.NS": "Titan",
    "TRENT.NS": "Trent", "ULTRACEMCO.NS": "UltraTech Cement", "WIPRO.NS": "Wipro",
}

# ---------- look: cards on a dark navy background (the dark theme is in .streamlit/config.toml) ----------
st.markdown("""
<style>
.mcard {background:#13213b; border:1px solid #22345a; border-radius:14px; padding:16px 18px; height:100%;}
.mname {color:#9fb3d6; font-size:0.9rem; letter-spacing:0.04em; text-transform:uppercase;}
.mdir {font-size:2rem; font-weight:700; margin:4px 0;}
.up {color:#00c27a;} .down {color:#ff4b5c;}
.mprob {font-size:1.05rem; color:#e6edf7;}
.mdelta {font-size:0.9rem; margin-top:2px;}
.mnote {font-size:0.75rem; color:#8aa0c8; margin-top:10px;}
.badge {display:inline-block; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600;}
.b-real {background:#0f3d33; color:#00c27a;} .b-exp {background:#4a2a12; color:#ffb347;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource                                           # load the files only once
def get_artifacts():
    return load_artifacts(DATA / "models")


st.title("Try it: what do the models say?")
st.caption("Experimental model outputs for learning. Not a forecast, and never a buy or sell recommendation.")

try:
    models, meta, latest, prices = get_artifacts()
except ArtifactError as err:                                 # missing or mismatched files: say so, show nothing fake
    st.error(str(err))
    st.stop()

# ---------- 1. choose a stock ----------
tickers = latest.index.tolist()
ticker = st.selectbox("Stock", tickers, index=tickers.index("TCS.NS") if "TCS.NS" in tickers else 0,
                      format_func=lambda t: f"{NAMES.get(t, t)} ({t})")
actual = latest.loc[ticker]                                  # the REAL newest indicator values
act_rsi = float(actual["rsi_14"])
act_ret = float(actual["ret_1d"]) * 100                      # in percent
act_ratio = 1 + float(actual["close_vs_ma20"])               # close / 20-day average

# ---------- 2. actual values (real historical data) ----------
st.markdown('<span class="badge b-real">ACTUAL HISTORICAL VALUES</span>', unsafe_allow_html=True)
st.caption(f"As of {actual['date']}, calculated from real downloaded prices with the same code used in training.")
c = st.columns(4)
c[0].metric("Last close", f"{actual['close']:,.2f}")
c[1].metric("RSI (14-day)", f"{act_rsi:.1f}")
c[2].metric("Yesterday's return", f"{act_ret:+.2f}%")
c[3].metric("Price / 20-day average", f"{act_ratio:.3f}")

p = prices[prices["ticker"] == ticker].sort_values("Date")
p = p.assign(MA20=p["Close"].rolling(20).mean()).tail(120)   # average computed on 180 days, then last 120 shown
fig = go.Figure(go.Candlestick(x=p["Date"], open=p["Open"], high=p["High"], low=p["Low"], close=p["Close"],
                               increasing_line_color=GREEN, decreasing_line_color=RED, name="Price"))
fig.add_scatter(x=p["Date"], y=p["MA20"], name="20-day average", line=dict(color="#ffb347", width=1.5))
fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False,
                  legend=dict(orientation="h", y=1.05))
st.plotly_chart(fig)

# ---------- 3. experiment inputs (manual) ----------
st.markdown('<span class="badge b-exp">EXPERIMENTAL INPUTS (set by you)</span>', unsafe_allow_html=True)
for key, val in EXAMPLE.items():
    st.session_state.setdefault(key, val)                    # first visit: RSI 65, +1%, 1.02


def use_example():
    for key, val in EXAMPLE.items():
        st.session_state[key] = val


def use_actual(real_rsi, real_ret, real_ratio):              # copy this stock's real values into the sliders
    st.session_state["in_rsi"] = float(np.clip(round(real_rsi, 1), 0, 100))
    st.session_state["in_ret"] = float(np.clip(round(real_ret, 2), -10, 10))
    st.session_state["in_ratio"] = float(np.clip(round(real_ratio, 3), 0.8, 1.2))


b1, b2, _ = st.columns([1, 1, 2])
b1.button("Use example values", on_click=use_example)
b2.button("Reset to actual values", on_click=use_actual, args=(act_rsi, act_ret, act_ratio))

s = st.columns(3)
rsi = s[0].slider("RSI (14-day)", 0.0, 100.0, step=0.1, key="in_rsi")
ret = s[1].slider("Yesterday's return (%)", -10.0, 10.0, step=0.01, key="in_ret")
ratio = s[2].slider("Price / 20-day average", 0.800, 1.200, step=0.001, format="%.3f", key="in_ratio")
st.caption("The price-to-moving-average ratio here means close divided by the 20-day average. "
           "All other model inputs stay at this stock's actual values.")

x_exp = build_row(meta, actual, rsi=rsi, ret_pct=ret, ratio=ratio)   # real values + your 3 changes
x_act = build_row(meta, actual)                                        # real values only
with st.expander("All 9 model inputs (training order)"):
    show = pd.DataFrame({"actual": x_act.iloc[0], "used now": x_exp.iloc[0]})
    show["changed by you"] = np.where(~np.isclose(show["actual"], show["used now"]), "yes", "")
    st.dataframe(show)

for feat, value in [("rsi_14", rsi), ("ret_1d", ret / 100), ("close_vs_ma20", ratio - 1)]:
    if outside_range(meta, feat, value):
        st.warning(f"{feat} = {value:.4f} is outside the range the models mostly saw in training "
                   f"({meta['q01'][feat]:.4f} to {meta['q99'][feat]:.4f}). The output is less meaningful.")
st.caption("Manually changed inputs may not represent realistic market conditions: "
           "in real markets these inputs move together, and here they do not.")

# ---------- 4. predictions ----------
st.header("Model predictions")
p_exp, p_act = predict_all(models, x_exp), predict_all(models, x_act)
cols = st.columns(3)
for col, (name, prob) in zip(cols, p_exp.items()):
    dirn = direction(prob)
    delta = round((prob - p_act[name]) * 100, 1) + 0.0        # in points; "+ 0.0" turns -0.0 into 0.0
    dcolor = GREEN if delta > 0 else RED if delta < 0 else "#8aa0c8"
    col.markdown(f"""
<div class="mcard">
  <div class="mname">{name}</div>
  <div class="mdir {'up' if dirn == 'Up' else 'down'}">{'▲' if dirn == 'Up' else '▼'} {dirn}</div>
  <div class="mprob">Probability of up: <b>{prob * 100:.1f}%</b></div>
  <div class="mdelta" style="color:{dcolor}">{delta:+.1f} pts vs actual inputs ({p_act[name] * 100:.1f}%)</div>
  <div class="mnote">Experimental model output, not a guaranteed forecast.</div>
</div>""", unsafe_allow_html=True)

names = list(p_exp)
cmp = go.Figure()
cmp.add_bar(x=names, y=[p_act[n] * 100 for n in names], name="Actual inputs", marker_color=BLUE)
cmp.add_bar(x=names, y=[p_exp[n] * 100 for n in names], name="Your experimental inputs",
            marker_color=[GREEN if p_exp[n] >= 0.5 else RED for n in names])
cmp.add_hline(y=50, line_dash="dash", annotation_text="50% = coin flip")
cmp.update_yaxes(range=[40, 60], title="Probability of up (%)")
cmp.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  barmode="group", height=340, margin=dict(l=10, r=10, t=10, b=10),
                  legend=dict(orientation="h", y=1.12))
st.plotly_chart(cmp)
st.caption("The y-axis is zoomed to 40-60%, so small differences look bigger than they are.")

# ---------- 5. what influenced the prediction ----------
st.header("What influenced the prediction?")
which = st.radio("Model", ["Logistic Regression", "XGBoost", "Random Forest"], horizontal=True)
if which == "Logistic Regression":
    contrib = explain_logreg(models[which], x_exp)
    note = "Coefficient x scaled input, in log-odds. Positive pushes towards Up, negative towards Down."
elif which == "XGBoost":
    contrib, _ = explain_xgb(models[which], x_exp)
    note = "SHAP values from XGBoost's built-in tree-SHAP, in log-odds. Positive pushes towards Up."
else:
    contrib = rf_importance(models[which], meta["features"])
    note = ("Global importance over all training days (adds up to 1). It describes the whole model, "
            "not this single prediction, and has no direction.")
contrib = contrib.reindex(contrib.abs().sort_values().index)             # biggest at the top of the chart
colors = [BLUE] * len(contrib) if which == "Random Forest" else [GREEN if v > 0 else RED for v in contrib]
imp = go.Figure(go.Bar(x=contrib.values, y=contrib.index, orientation="h", marker_color=colors))
imp.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  height=340, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(imp)
st.caption(note + " This shows how the model does its arithmetic. It does not mean a feature caused any stock to move.")

# ---------- 6. honest results ----------
st.header("How reliable is this?")
try:
    acc = get_accuracy()
    avg = acc.groupby("model")["accuracy_%"].mean()
    best = max(avg["LogReg"], avg["RandForest"], avg["XGBoost"])
    gap = best - avg["AlwaysUp"]
    st.warning(
        f"Existing walk-forward evaluation (average over {acc['ticker'].nunique()} stocks): "
        f"Logistic Regression {avg['LogReg']:.1f}%, Random Forest {avg['RandForest']:.1f}%, "
        f"XGBoost {avg['XGBoost']:.1f}%, always-up baseline {avg['AlwaysUp']:.1f}%, coin flip 50.0%. "
        + ("The models have not shown an advantage over the simple always-up baseline."
           if gap < 1 else "The best model is above the baseline; check the Model Comparison page for how consistent that is.")
    )
except FileNotFoundError:
    st.warning("The evaluation file (accuracy_summary.csv) was not found, so no accuracy numbers are shown here.")
st.caption(f"The 3 models on this page were trained once on all {meta['trained_rows']:,} days up to "
           f"{meta['trained_until']} ({meta['n_stocks']} stocks), so their probabilities here have already seen "
           "the past. Only the walk-forward results above are an honest test.")

with st.expander("How the inputs flow into the models"):
    st.markdown(
        "1. Real prices are downloaded, and the same `make_features()` code used in training calculates the 9 inputs "
        "from past days only.\n"
        "2. Your 3 sliders replace RSI, yesterday's return, and the 20-day price ratio. The other 6 inputs stay as they are.\n"
        "3. The 9 values go to each model in the exact training order.\n"
        "4. **Logistic Regression** first passes them through its saved StandardScaler. **Random Forest** and **XGBoost** "
        "were trained on unscaled values, so they get the raw numbers.\n"
        "5. Each model returns a probability of Up. Up means 50% or more, the same cut-off the models use."
    )