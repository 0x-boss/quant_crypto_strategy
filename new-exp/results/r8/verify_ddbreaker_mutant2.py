import sys
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import numpy as np, pandas as pd
import r8_common as R
import r8_f_ddbreaker as mod
c = R.load()
p = next(q for q in mod.CONFIGS if q["name"] == "w126_t10_c50_d10")
W0 = R.base_weights(c)
r = R.run(c, W0)
def expo_for(known):
    eq = np.cumprod(1 + known); n = len(eq); ex = np.ones(n); anchor = 0; cs = None
    for t in range(n):
        if cs is None:
            pk = eq[max(anchor, t - 125):t + 1].max()
            if eq[t] / pk - 1 <= -0.10: cs = t; ex[t] = 0.5
        else:
            if t - cs >= 10: cs = None; anchor = t; ex[t] = 1.0
            else: ex[t] = 0.5
    return pd.Series(ex, index=W0.index[:n])
ex_m = expo_for(r.fillna(0).to_numpy(float))               # mutant: un-lagged
d = ex_m.index[(ex_m < 1) & (ex_m.shift(1) == 1) & (ex_m.index >= "2012-01-01")]
dates = tuple(x.strftime("%Y-%m-%d") for x in d[::max(1, len(d)//6)][:6])
print("mutant cut-start dates:", dates)
R.lagged = lambda x: x
mod._CACHE.clear()
ok_m, w_m = R.check_perturbation(lambda cc: mod.apply(cc, R.base_weights(cc), p), c, dates=dates, tol=1e-9, verbose=True)
print("MUTANT flagged:", not ok_m, w_m)
