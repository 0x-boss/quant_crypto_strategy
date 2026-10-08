import sys, json, time, os
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import numpy as np, pandas as pd
import r8_common as R
import lib
import r8_f_signal as mod
p = next(q for q in mod.CONFIGS if q["name"] == "m61_h52")
c = R.load()
idx = c["C"].index; cols = c["C"].columns

# (a) where does the 0.04 all-dates diff come from?  (re-implementation with the same algorithm, compare per-date)
W_mod = mod.weights(c, p).reindex(index=idx, columns=cols).fillna(0)
mom = c["mom_6_1"].values.astype(np.float64); h52 = c["high52"].values.astype(np.float64); M = c["M300"].values
elig = M & np.isfinite(mom) & np.isfinite(h52) & (h52 >= 0.8)
T, N = mom.shape
S = np.zeros((T, N))
for t in range(T):
    ii = np.flatnonzero(elig[t])
    if ii.size:
        S[t, ii[np.argsort(-mom[t, ii], kind="stable")][:10]] = 0.1
cs = np.cumsum(S, 0); W = np.empty_like(S)
for t in range(T):
    lo = max(0, t - 4); W[t] = (cs[t] - (cs[lo - 1] if lo > 0 else 0)) / (t - lo + 1)
d = np.abs(W - W_mod.values).max(1)
bad = np.flatnonzero(d > 1e-9)
print("dates with diff >1e-9:", len(bad), idx[bad][:5], idx[bad][-5:], " first date idx:", idx[0], " first date with any M300:", idx[M.any(1)][0])
print(" n names selected on those days (mine vs module):", [(str(idx[i].date()), int((S[i] > 0).sum()), int((W_mod.values[i] > 0).sum()), float(W[i].sum()), float(W_mod.values[i].sum())) for i in bad[:5]])

# (b) future-aware bad-print cleaning in lib.load_panels:  rn = C.shift(-1)/C - 1 masks a >50 % move reversed next day.  Quantify the effect on the candidate.
t0 = time.time()
d_ = os.path.join(lib.DATA, "panels")
rc = pd.read_parquet(os.path.join(d_, "raw_close.parquet"), columns=list(cols)).loc["2008-01-01":].astype(np.float64)
ac = pd.read_parquet(os.path.join(d_, "raw_adj_close.parquet"), columns=list(cols)).loc["2008-01-01":].astype(np.float64)
Cu = ac.where(rc > 0)
r = Cu / Cu.shift(1) - 1; rn = Cu.shift(-1) / Cu - 1
badm = (r.abs() > 0.5) & (((1 + r) * (1 + rn) - 1).abs() < 0.1)
cal = c["C"].index
badm = badm.reindex(cal).fillna(False)
Cu = Cu.reindex(cal)
nb = int(badm.values.sum())
inM = int((badm.values & c["M300"].values).sum())
print(f"bad-print cells flagged (all research cols): {nb}; of which inside the PIT top-300 on that day: {inM}  [{time.time()-t0:.0f}s]")
# does restoring them change the C used? (C in cache is NaN where bad, else adj close)
Cc = c["C"].astype(np.float64)
restored = Cu.where(badm, Cc)      # unmasked close where flagged, cache elsewhere
mom_u = (restored.shift(21) / restored.shift(126) - 1)
h52_u = restored / restored.rolling(252, min_periods=200).max()
cu = dict(c); cu["mom_6_1"] = mom_u.astype(np.float32); cu["high52"] = h52_u.astype(np.float32)
Wu = mod.weights(cu, p).reindex(index=idx, columns=cols).fillna(0)
sel = idx >= "2012-01-01"
dd = np.abs(Wu.values - W_mod.values)[sel]
print("candidate weights with bad prints NOT removed (signal only): max|dW| =", dd.max(), " days with diff:", int((dd.max(1) > 1e-9).sum()), "of", int(sel.sum()))
ru = R.run(c, Wu); rm = R.run(c, W_mod)
for nm, rr in (("clean", rm), ("raw-prints", ru)):
    st = R.evaluate(rr, nm, show=False); print(nm, "DEV Sh", round(st["dev"]["sharpe"], 3), "HOLD Sh", round(st["hold"]["sharpe"], 3), "DEV DD", round(st["dev"]["maxdd"], 3))
# same for baseline (signal only)
cb = dict(c); cb["mom_6_1"] = mom_u.astype(np.float32)
Wbu = R.base_weights(cb); Wb = R.base_weights(c)
dd2 = np.abs(Wbu.reindex(index=idx, columns=cols).fillna(0).values - Wb.reindex(index=idx, columns=cols).fillna(0).values)[sel]
print("baseline weights with bad prints not removed: max|dW| =", dd2.max(), " days with diff:", int((dd2.max(1) > 1e-9).sum()))
for nm, W_ in (("base-clean", Wb), ("base-raw", Wbu)):
    st = R.evaluate(R.run(c, W_), nm, show=False); print(nm, "DEV Sh", round(st["dev"]["sharpe"], 3), "HOLD Sh", round(st["hold"]["sharpe"], 3))
# (c) which columns are flagged in the M300 + date list
ii = np.argwhere(badm.values & c["M300"].values)
print([ (str(idx[a].date()), cols[b]) for a, b in ii[:15]])
