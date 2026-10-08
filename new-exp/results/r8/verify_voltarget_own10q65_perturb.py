import sys, time
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import numpy as np, pandas as pd
import r8_common as R
import r8_f_voltarget as mod

c = R.load()
p = next(q for q in mod.CONFIGS if q["name"] == "own10_q65")
print("params:", p, flush=True)
dates = ("2021-03-10", "2020-03-12", "2025-03-20", "2026-06-25", "2018-11-15", "2012-09-20", "2016-06-24", "2024-08-05")
runner_dates = ("2014-03-14", "2017-10-02", "2020-06-17", "2022-06-14", "2025-08-12")
print("dates in index:", [pd.Timestamp(d) in c["C"].index for d in dates], "overlap with runner:", set(dates) & set(runner_dates), flush=True)
t0 = time.time()
ok, worst = R.check_perturbation(lambda cc: mod.apply(cc, R.base_weights(cc), p), c, dates=dates)
print("RESULT module.apply perturbation:", ok, worst, "secs", round(time.time() - t0), flush=True)

# power test: an un-lagged variant of the same overlay must be flagged by the guard
def leaky(cc):
    W = R.base_weights(cc)
    rl = R.run(cc, W)                     # NOT lagged
    sig = rl.rolling(10).std() * mod.ANN
    s = (p["star"] / sig).replace([np.inf, -np.inf], np.nan).clip(upper=1.0).fillna(1.0)
    return W.mul(s, axis=0)
print("--- power test: un-lagged own10 variant (should be flagged)", flush=True)
R.check_perturbation(leaky, c, dates=("2018-11-15", "2021-03-10", "2025-03-20"))
