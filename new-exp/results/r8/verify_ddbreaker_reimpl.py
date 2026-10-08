import sys
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import numpy as np, pandas as pd
import r8_common as R
import r8_f_ddbreaker as mod

c = R.load(["O", "C", "M300", "mom_6_1"])
# ---- independent baseline book -------------------------------------------------------
sig = c["mom_6_1"].where(c["M300"])
rk = sig.rank(axis=1, ascending=False, method="first")
top = ((rk <= 10) & sig.notna()).astype(float) * 0.1
Wb = top.rolling(5, min_periods=1).mean()
Wb_ref = R.base_weights(c)
print("baseline weights max diff vs R.base_weights:", float((Wb - Wb_ref).abs().to_numpy().max()))

# ---- independent breaker --------------------------------------------------------------
WIN, THR, CUT, WAIT = 126, 0.10, 0.5, 10
r = R.run(c, Wb)                      # unscaled open-to-open net return, indexed by start day
known = r.shift(1).fillna(0.0).to_numpy(float)   # at close of t only the return indexed t-1 is known
eq = np.cumprod(1.0 + known)
n = len(eq)
expo = np.ones(n)
anchor = 0               # first row eligible for the peak (last re-entry day)
cut_since = None         # row where the current cut started
for t in range(n):
    if cut_since is None:
        start = max(anchor, t - WIN + 1)
        peak = eq[start:t + 1].max()
        if eq[t] <= (1 - THR) * peak + 1e-15 * 0 or (eq[t] / peak - 1) <= -THR:
            cut_since = t
            expo[t] = CUT
    else:
        if t - cut_since >= WAIT:
            cut_since = None
            anchor = t
            expo[t] = 1.0
        else:
            expo[t] = CUT
expo = pd.Series(expo, index=r.index.append(pd.Index([])) if False else Wb.index[:n])
W_mine = Wb.mul(expo, axis=0)

W_mod = mod.apply(c, Wb_ref, next(q for q in mod.CONFIGS if q["name"] == "w126_t10_c50_d10"))
sel = W_mine.index >= "2012-01-01"
a, b = W_mine.loc[sel], W_mod.reindex(W_mine.index).loc[sel]
print("max abs weight diff (>=2012):", float((a - b.reindex(columns=a.columns)).abs().to_numpy().max()))
print("max abs weight diff (all dates):", float((W_mine - W_mod.reindex(W_mine.index)).abs().to_numpy().max()))
print("cut fraction of days >=2012:", float((expo.loc["2012":] < 1).mean()), " n cut episodes:", int(((expo < 1) & (expo.shift(1) == 1)).sum()))
print("avg gross mine/mod:", float(W_mine.sum(1).loc["2012":].mean()), float(W_mod.sum(1).loc["2012":].mean()))
# exposure comparison
s_mod = (W_mod.sum(1) / Wb_ref.sum(1)).replace([np.inf, -np.inf], np.nan)
print("exposure diff (where baseline gross>0):", float((s_mod - expo).abs().dropna().max()))
expo.to_csv("/home/user/quant_crypto_strategy/new-exp/results/r8/verify_ddbreaker_expo.csv")
d = (a - b.reindex(columns=a.columns))
print("NaN count in diff:", int(d.isna().sum().sum()), " NaN in a:", int(a.isna().sum().sum()), " NaN in b:", int(b.isna().sum().sum()))
print("max abs diff, nan-safe (fillna 0 both):", float((a.fillna(0) - b.fillna(0)).abs().to_numpy().max()))
print("rows with NaN in Wb_ref:", int(Wb_ref.isna().any(axis=1).sum()), "first/last", Wb_ref.index[0], Wb_ref.index[-1], "expo len", len(expo), "W idx len", len(Wb))
a2, b2 = a.iloc[:-1], b.reindex(columns=a.columns).iloc[:-1]
print("FINAL max abs weight diff (>=2012, excl. last untradable row):", float((a2 - b2).abs().to_numpy().max()))
print("last row exposures: mine NaN, module =", float(W_mod.iloc[-1].sum() / Wb_ref.iloc[-1].sum()), " state of cut at last known row:", float(expo.iloc[-1]))
print("row index of last two rows", Wb.index[-2:], "r last idx", r.index[-1])
