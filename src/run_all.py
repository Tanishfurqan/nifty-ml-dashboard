"""
Re-run the whole pipeline in order and log how long each step took.
  python src/run_all.py            -> steps 1, 2, 5, 6, 7, 8a, 9
  python src/run_all.py --lstm     -> the same, plus the LSTM (needs PyTorch, slower)
"""
import subprocess, sys, time
from pathlib import Path
import pandas as pd

SRC = Path(__file__).resolve().parent                    # the src folder
OUT = SRC.parent / "dashboard_data"

STEPS = ["step1_data.py", "step2_features.py", "step5_walkforward.py",
         "step6_xgboost.py", "step7_backtest.py", "step8a_export.py", "step9_volatility.py"]
if "--lstm" in sys.argv:
    STEPS.append("step10_lstm_pooled.py")                # optional: heavy, needs PyTorch

log = []
for s in STEPS:
    t0 = time.time()
    done = subprocess.run([sys.executable, str(SRC / s)])      # run the step like: python src/<step>
    log.append({"step": s, "minutes": round((time.time() - t0) / 60, 1),
                "finished_at": time.strftime("%Y-%m-%d %H:%M"), "ok": done.returncode == 0})
    if done.returncode != 0:                                   # stop if a step fails
        break

OUT.mkdir(exist_ok=True)
pd.DataFrame(log).to_csv(OUT / "run_log.csv", index=False)     # the Audit page shows this
print(pd.DataFrame(log).to_string(index=False))
