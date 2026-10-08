import sys, time
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import numpy as np, pandas as pd
import r8_common as R
import r8_f_voltarget as mod

c = R.load()
p = next(q for q in mod.CONFIGS if q["name"] == "own20_exp80")
print("params:", p, flush=True)
dates = ("2021-03-10", "2020-03-12", "2025-03-20", "2026-06-25", "2018-11-15", "2013-01-24", "2015-08-24", "2023-09-14")
print("dates in index:", [d for d in dates if pd.Timestamp(d) in c["C"].index], flush=True)
t0 = time.time()
ok, worst = R.check_perturbation(lambda cc: mod.apply(cc, R.base_weights(cc), p), c, dates=dates)
print("RESULT module.apply perturbation:", ok, worst, "secs", round(time.time() - t0), flush=True)

# power test of the guard: deliberately leaky variants must be flagged
def leaky_nolag(cc):
    W = R.base_weights(cc)
    rl = R.run(cc, W)                     # NOT lagged -> uses open t+1
    sig = rl.rolling(20).std() * mod.ANN
    star = sig.expanding(min_periods=504).quantile(0.8)
    s = (star / sig).replace([np.inf, -np.inf], np.nan).clip(upper=1.0).fillna(1.0)
    return W.mul(s, axis=0)

def leaky_fullq(cc):
    W = R.base_weights(cc)
    rl = R.lagged(R.run(cc, W))
    sig = rl.rolling(20).std() * mod.ANN
    star = sig.quantile(0.8)              # full-sample quantile
    s = (star / sig).replace([np.inf, -np.inf], np.nan).clip(upper=1.0).fillna(1.0)
    return W.mul(s, axis=0)

for nm, f in (("leaky_nolag", leaky_nolag), ("leaky_fullsample_q", leaky_fullq)):
    print("---", nm, flush=True)
    R.check_perturbation(f, c, dates=("2018-11-15", "2021-03-10", "2025-03-20"))

# truncation test (module weights -> returns), different cut dates
for T0 in ("2019-06-28", "2023-03-15"):
    R.check_truncation(lambda cc: R.run(cc, mod.apply(cc, R.base_weights(cc), p)), c, T0=T0)
