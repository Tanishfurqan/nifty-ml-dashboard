"""
Step 7: Backtest. Turn each model's predictions into trades and compare with buy-and-hold.
Input:  data/results/<TICKER>_preds.csv   (made by step5 and step6)
Output: data/results/<TICKER>_equity.csv and backtest_summary.csv (used by the dashboard)
"""
import pandas as pd

from step3_baselines import TICKERS
from step5_walkforward import RESULT_DIR

COST = 0.001   # 0.1% cost on every buy or sell (an ASSUMPTION; change it to test)
STRATEGIES = ["LogReg", "RandForest", "XGBoost", "SameAsToday"]


def run_strategy(res, pos):
    """pos = 1 (hold stock) or 0 (cash) for each day. Returns position, next-day return, trade flags."""
    # Return from today's close to tomorrow's close.
    next_ret = res["Close"].shift(-1) / res["Close"] - 1
    df = pd.DataFrame({"next_ret": next_ret, "pos": pos}).dropna()   # last day has no tomorrow

    # A trade happens whenever the position changes (0->1 buy, 1->0 sell).
    trades = df["pos"].diff().abs()
    trades.iloc[0] = df["pos"].iloc[0]       # first day: buying from cash counts as a trade
    return df["pos"], df["next_ret"], trades


def summarize(pos, next_ret, trades, cost):
    """Daily return = position x market return, minus cost for any trade."""
    daily = pos * next_ret - trades * cost
    equity = (1 + daily).cumprod()           # 1.0 = starting money
    drawdown = equity / equity.cummax() - 1  # drop from the best level so far
    return equity, {
        "total_return_%": round((equity.iloc[-1] - 1) * 100, 1),
        "max_drawdown_%": round(drawdown.min() * 100, 1),
        "trades": int(trades.sum()),
        "in_market_%": round(pos.mean() * 100, 1),
    }


def backtest(cost):
    """Run every strategy on every stock. Returns a results table and equity curves."""
    rows, curves = [], {}
    for ticker in TICKERS:
        res = pd.read_csv(RESULT_DIR / f"{ticker}_preds.csv",
                          index_col="Date", parse_dates=True)
        curve = {}

        # Buy-and-hold: hold the stock every day, pay cost once at the start.
        pos, nret, trades = run_strategy(res, pd.Series(1, index=res.index))
        curve["BuyHold"], stats = summarize(pos, nret, trades, cost)
        rows.append({"ticker": ticker, "strategy": "BuyHold", **stats})

        # Each model: hold the stock only when it predicts "up".
        for name in STRATEGIES:
            pos, nret, trades = run_strategy(res, res[f"pred_{name}"])
            curve[name], stats = summarize(pos, nret, trades, cost)
            rows.append({"ticker": ticker, "strategy": name, **stats})

        curves[ticker] = pd.DataFrame(curve)
    return pd.DataFrame(rows), curves


def main():
    for cost in [0.0, COST]:
        table, curves = backtest(cost)
        print(f"===== Cost per trade = {cost * 100:.2f}% =====")
        if cost == COST:
            table.to_csv(RESULT_DIR / "backtest_summary.csv", index=False)   # the dashboard reads this
            print(f"{table['ticker'].nunique()} stocks; per-stock table saved to backtest_summary.csv")
            for t, c in curves.items():              # save profit curves for the dashboard
                c.to_csv(RESULT_DIR / f"{t}_equity.csv")
            print()
        print("Average over stocks:")
        print(table.groupby("strategy")[["total_return_%", "max_drawdown_%", "trades"]]
              .mean().round(1).sort_values("total_return_%", ascending=False))
        print()


if __name__ == "__main__":
    main()
