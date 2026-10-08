import sys
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import numpy as np, pandas as pd
import r8_common as R
c = R.load(["mom_6_1", "high52", "M300", "C", "O"])
s = c["mom_6_1"].where(c["M300"]); s = s.where(c["high52"] > 0.8)
rk = s.rank(axis=1, ascending=False, method="first")
sl = ((rk <= 10) & s.notna()).astype(float) / 10
mom = c["mom_6_1"].values.astype(np.float64); h52 = c["high52"].values.astype(np.float64); M = c["M300"].values
elig = M & np.isfinite(mom) & np.isfinite(h52) & (h52 >= 0.8)
for t in range(len(sl)):
    ii = np.flatnonzero(elig[t])
    mine = np.zeros(mom.shape[1])
    if ii.size: mine[ii[np.argsort(-mom[t, ii], kind="stable")][:10]] = 0.1
    d = np.abs(mine - sl.values[t])
    if d.max() > 1e-9:
        print(sl.index[t].date(), "sum diff", d.sum(), "n elig", ii.size, "module n sel", int((sl.values[t] > 0).sum()), "mine", int((mine > 0).sum()))
        j = np.flatnonzero(d > 1e-9); print("  cols", list(c["mom_6_1"].columns[j]), "mom", mom[t, j], "h52", h52[t, j], "rank", rk.values[t, j])
