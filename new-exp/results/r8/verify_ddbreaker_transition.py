import sys
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import numpy as np, pandas as pd
import r8_common as R
import r8_f_ddbreaker as mod
c = R.load()
p = next(q for q in mod.CONFIGS if q["name"] == "w126_t10_c50_d10")
expo = pd.read_csv("/home/user/quant_crypto_strategy/new-exp/results/r8/verify_ddbreaker_expo.csv", index_col=0, parse_dates=True).iloc[:, 0]
start = expo.index[(expo < 1) & (expo.shift(1) == 1) & (expo.index >= "2012-01-01")]
reent = expo.index[(expo == 1) & (expo.shift(1) < 1) & (expo.index >= "2012-01-01")]
sel = list(start[::max(1, len(start)//7)][:7]) + list(reent[::max(1, len(reent)//4)][:4])
dates = tuple(d.strftime("%Y-%m-%d") for d in sel)
print("transition dates (cut starts + re-entries):", dates)
orig = R.lagged
R.lagged = lambda r: r
mod._CACHE.clear()
ok_m, w_m = R.check_perturbation(lambda cc: mod.apply(cc, R.base_weights(cc), p), c, dates=dates, tol=1e-9, verbose=False)
print("MUTANT (un-lagged) flagged:", not ok_m, "worst", w_m)
R.lagged = orig
mod._CACHE.clear()
ok, w = R.check_perturbation(lambda cc: mod.apply(cc, R.base_weights(cc), p), c, dates=dates, tol=1e-9, verbose=True)
print("REAL module on transition dates ok:", ok, "worst", w)
