"""
Step 10 (pooled): ONE small LSTM trained on ALL stocks together in every walk-forward fold.
Input:  data/features/<TICKER>_features.csv  (from step2)
Output: dashboard_data/lstm_pooled_folds.csv and lstm_pooled_stocks.csv (full run only)
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view   # cuts data into "last 20 days" windows

from config import TICKERS                       # the 47 stocks that have features
from step2_features import FEATURE_COLS          # the 9 allowed input columns
from step3_baselines import FEAT_DIR             # data/features
from step4_models import get_models              # we reuse LogReg for a fair comparison
from step5_walkforward import MIN_TRAIN, TEST_BLOCK   # SAME fold sizes as step 5

OUT_DIR = FEAT_DIR.parent.parent / "dashboard_data"   # where the dashboard reads from

# ---------- SETTINGS ----------
QUICK_TEST = False   # True = only 2 folds (a timing test, saves nothing). False = full run.
SEQ = 20             # the LSTM looks at the last 20 days
HIDDEN = 32          # small on purpose: fast, and less overfitting
MAX_EPOCHS = 15      # most passes over the training data
PATIENCE = 3         # stop early if validation does not improve for 3 passes
BATCH = 256          # how many windows are learned from at a time


class LSTMNet(nn.Module):
    """One small LSTM layer + a final layer that outputs one number (up or down)."""
    def __init__(self, n_features, hidden):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)   # the memory part
        self.drop = nn.Dropout(0.3)               # switches off 30% randomly to reduce overfitting
        self.out = nn.Linear(hidden, 1)           # turns the memory into one number

    def forward(self, x):
        o, _ = self.lstm(x)                       # LSTM output for each of the 20 days
        return self.out(self.drop(o[:, -1, :])).squeeze(1)   # use only the LAST day's output


def train_lstm(Xtr, ytr, Xval, yval):
    """Train on Xtr; keep the version that scored best on Xval (early stopping)."""
    torch.manual_seed(42)                         # same result every run
    model = LSTMNet(Xtr.shape[2], HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()              # loss for yes/no answers
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)       # numpy -> torch tensors
    Xval_t, yval_t = torch.tensor(Xval), torch.tensor(yval)

    best_loss, best_state, bad, epochs = float("inf"), None, 0, 0
    for epoch in range(MAX_EPOCHS):
        epochs += 1
        model.train()
        order = torch.randperm(len(Xtr_t))        # shuffle ONLY inside the training period
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            opt.zero_grad()
            loss_fn(model(Xtr_t[idx]), ytr_t[idx]).backward()   # find how to improve
            opt.step()                                           # improve a little
        model.eval()
        with torch.no_grad():                     # no learning, just measuring
            val_loss = loss_fn(model(Xval_t), yval_t).item()
        if val_loss < best_loss - 1e-4:           # improved: remember this version
            best_loss, bad = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:                   # no progress: stop early
                break
    model.load_state_dict(best_state)             # go back to the best version
    return model, epochs


def load_all():
    """Read every stock and cut it into 20-day windows. Windows are NOT scaled yet."""
    Ws, ys, ds, rs, ts = [], [], [], [], []
    for ticker in TICKERS:
        df = pd.read_csv(FEAT_DIR / f"{ticker}_features.csv",
                         index_col="Date", parse_dates=True).sort_index()
        X = df[FEATURE_COLS].values.astype(np.float32)
        # sliding_window_view gives (windows, 9 features, 20 days); transpose -> (windows, 20 days, 9)
        w = sliding_window_view(X, SEQ, axis=0).transpose(0, 2, 1)
        Ws.append(np.ascontiguousarray(w))
        # A window ends on row k+SEQ-1, so it takes that row's date, answer and return.
        ys.append(df["target"].values[SEQ - 1:])
        ds.append(df.index.values[SEQ - 1:])
        rs.append(df["ret_1d"].values[SEQ - 1:])
        ts.append(np.full(len(w), ticker))
    return (np.concatenate(Ws), np.concatenate(ys).astype(np.float32),
            np.concatenate(ds), np.concatenate(rs), np.concatenate(ts))


def main():
    t_all = time.time()
    W, y, d, ret, tick = load_all()            # all stocks stacked into one big table
    y_int = y.astype(int)
    X_last = W[:, -1, :]                       # each window's last day (used by LogReg)
    print(f"{len(W)} windows from {len(TICKERS)} stocks")

    # Folds are cut by DATE, so every stock is split on the same day (no stock sees the future).
    cal = np.unique(d)                         # sorted list of all dates
    starts = list(range(MIN_TRAIN, len(cal), TEST_BLOCK))
    if QUICK_TEST:
        starts = starts[:2]

    fold_rows, P = [], {k: [] for k in ["tick", "y", "lstm", "lr", "same"]}
    for k, start in enumerate(starts):
        t0 = time.time()
        d0 = cal[start]                                                   # first test day
        end = start + TEST_BLOCK
        d1 = cal[end] if end < len(cal) else cal[-1] + np.timedelta64(1, "D")   # day after the test block
        train = d < d0                          # windows that END before the test block
        test = (d >= d0) & (d < d1)             # windows that END inside the test block

        # Scale using TRAINING days only (no leakage from the test block).
        mean = X_last[train].mean(axis=0)
        std = X_last[train].std(axis=0) + 1e-8
        scale = lambda a: ((a - mean) / std).astype(np.float32)

        # The newest 15% of training DATES = validation set for early stopping.
        idx = np.where(train)[0]
        idx = idx[np.argsort(d[idx], kind="stable")]       # oldest -> newest
        cut = int(len(idx) * 0.85)
        tr_idx, val_idx = idx[:cut], idx[cut:]
        model, epochs = train_lstm(scale(W[tr_idx]), y[tr_idx], scale(W[val_idx]), y[val_idx])

        model.eval()
        with torch.no_grad():
            p_lstm = (model(torch.tensor(scale(W[test]))) > 0).numpy().astype(int)   # >0 = "up"

        # Logistic Regression on the SAME windows' last day, and the baselines.
        lr = get_models()["LogReg"]
        lr.fit(X_last[train], y_int[train])
        p_lr = lr.predict(X_last[test])
        p_same = (ret[test] > 0).astype(int)    # SameAsToday rule
        y_te = y_int[test]

        fold_rows.append({
            "fold": k, "test_start": str(d0)[:10], "test_rows": int(test.sum()),
            "LSTM": (p_lstm == y_te).mean() * 100, "LogReg": (p_lr == y_te).mean() * 100,
            "AlwaysUp": (y_te == 1).mean() * 100, "SameAsToday": (p_same == y_te).mean() * 100,
            "lstm_up_%": p_lstm.mean() * 100, "epochs": epochs, "seconds": time.time() - t0,
        })
        sec = fold_rows[-1]["seconds"]
        print(f"fold {k}: {sec:.0f} seconds, {epochs} epochs, "
              f"LSTM said up on {fold_rows[-1]['lstm_up_%']:.0f}% of days")
        if k == 0:    # rough guess: later folds have more data, so they take about 1.5-3x longer
            print(f"   rough total estimate: {sec * len(starts) * 1.5 / 60:.0f}"
                  f" to {sec * len(starts) * 3 / 60:.0f} minutes ({len(starts)} folds)")
        for name, arr in zip(P, [tick[test], y_te, p_lstm, p_lr, p_same]):
            P[name].append(arr)                 # keep every prediction for the per-stock table

    folds = pd.DataFrame(fold_rows)
    print()
    print(folds.round(1).to_string(index=False))
    cols = ["LSTM", "LogReg", "AlwaysUp", "SameAsToday"]
    print("\nAverage accuracy over folds:")
    print(folds[cols].mean().round(1))
    best_base = folds[["AlwaysUp", "SameAsToday"]].max(axis=1)           # bar to beat, per fold
    print(f"\nLSTM beat the best baseline in {int((folds['LSTM'] > best_base).sum())} of {len(folds)} folds")

    # Accuracy per stock (all folds together).
    y_all = np.concatenate(P["y"])
    stock = pd.DataFrame({
        "ticker": np.concatenate(P["tick"]),
        "LSTM": (np.concatenate(P["lstm"]) == y_all) * 100.0,
        "LogReg": (np.concatenate(P["lr"]) == y_all) * 100.0,
        "AlwaysUp": (y_all == 1) * 100.0,
        "SameAsToday": (np.concatenate(P["same"]) == y_all) * 100.0,
    }).groupby("ticker").mean().round(2)
    print(f"LSTM beat 'always up' in {int((stock['LSTM'] > stock['AlwaysUp']).sum())} of {len(stock)} stocks")
    print(f"Total time: {(time.time() - t_all) / 60:.1f} minutes")

    if not QUICK_TEST:                          # only a full run saves the files the dashboard reads
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        folds.to_csv(OUT_DIR / "lstm_pooled_folds.csv", index=False)
        stock.to_csv(OUT_DIR / "lstm_pooled_stocks.csv")
        print("Saved the results to dashboard_data/")


if __name__ == "__main__":
    main()