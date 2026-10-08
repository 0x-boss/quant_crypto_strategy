"""Independent re-implementation of own20_exp80 from its TEXT description only (no call into r8_f_voltarget for the transformation).
Baseline: top-10 of the PIT top-300 by 6-1 momentum, equal weight 1/10, average of the last 5 daily books.
Transformation: row t scaled by min(1, sigma*_t / sigma_hat_t); sigma_hat_t = annualised std of the 20 most recent strategy returns KNOWN at the
close of t (return indexed d spans open d -> open d+1, so the last known at close t is the one indexed t-1); sigma*_t = 80th percentile (linear
interpolation) of all sigma_hat_u, u <= t.
"""
import sys, bisect, time
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import numpy as np, pandas as pd
import r8_common as R
import r8_f_voltarget as mod          # only used to get the module's weights for comparison

t0 = time.time()
c = R.load(["O", "mom_6_1", "M300"])
O = c["O"].astype(np.float64)
dates, cols = O.index, O.columns
T, N = O.shape
Ov = O.values

# ---------------- own baseline book ----------------
mom = c["mom_6_1"].where(c["M300"]).astype(np.float64).values
sleeve = np.zeros((T, N))
for t in range(T):
    v = mom[t]
    ok = np.where(np.isfinite(v))[0]
    if len(ok) == 0:
        continue
    order = ok[np.argsort(-v[ok], kind="stable")]       # ties -> column order (== rank method='first')
    sleeve[t, order[:10]] = 1.0 / 10
cs = np.cumsum(sleeve, axis=0)
Wb = np.zeros_like(sleeve)
for t in range(T):
    lo = max(0, t - 4)
    Wb[t] = (cs[t] - (cs[lo - 1] if lo > 0 else 0)) / (t - lo + 1)
Wb_ref = R.base_weights(c).values
print("baseline book max|diff| vs R.base_weights:", np.nanmax(np.abs(Wb - np.nan_to_num(Wb_ref))), flush=True)

# ---------------- own backtest (open-to-open, 0.1%/side on traded notional) ----------------
held = np.where(Wb.sum(0) > 0)[0]
def backtest(W, fee=R.FEE, lag=0):
    Wh = W[:, held]
    Oh = Ov[:, held]
    pos = np.zeros_like(Wh)
    pos[1 + lag:] = Wh[: T - 1 - lag]
    ratio = np.ones_like(Wh)
    nxt = Oh[1:] / Oh[:-1]
    ratio[:-1] = np.where(np.isfinite(nxt), nxt, 1.0)
    ret = np.zeros(T)
    drift = np.zeros(Wh.shape[1])
    for d in range(T - 1):
        w = pos[d]
        turn = np.abs(w - drift).sum()
        g = 1.0 - w.sum() + (w * ratio[d]).sum()
        ret[d] = (1 - fee * turn) * g - 1
        drift = w * ratio[d] / g
    return ret[:-1]                                      # length T-1, index = dates[:-1]

rb = backtest(Wb)
rb_ref = R.run(c, R.base_weights(c)).values
print("baseline return max|diff| vs R.run:", np.abs(rb - rb_ref).max(), flush=True)

# ---------------- own transformation ----------------
def transform(Wb, rb, win=20, q=0.8, min_obs=504, drop_zero=False):
    r_known = np.full(T, np.nan)
    r_known[1:T] = np.r_[rb, np.nan][: T - 1]            # known at close t = return indexed t-1  (rb has T-1 entries; index T-1 never needed)
    # r_known[t] = rb[t-1] for t=1..T-1
    r_known[1:] = rb
    sig = np.full(T, np.nan)
    for t in range(T):
        # the 20 returns known at close t: indexed t-20 ... t-1
        a, b = t - win, t - 1
        if a >= 0 and b <= T - 2:
            x = rb[a:b + 1]
            sig[t] = x.std(ddof=1) * np.sqrt(252)
    s = np.ones(T)
    srt = []
    star = np.full(T, np.nan)
    for t in range(T):
        if np.isfinite(sig[t]) and not (drop_zero and sig[t] == 0.0):
            bisect.insort(srt, sig[t])
        n = len(srt)
        if n >= min_obs:
            pos = q * (n - 1)
            lo, hi = int(np.floor(pos)), int(np.ceil(pos))
            star[t] = srt[lo] + (srt[hi] - srt[lo]) * (pos - lo)
        if np.isfinite(sig[t]) and np.isfinite(star[t]) and sig[t] > 0:
            s[t] = min(1.0, star[t] / sig[t])
        else:
            s[t] = 1.0
    return Wb * s[:, None], s, sig, star

Wm_df = mod.apply(c, R.base_weights(c), next(q for q in mod.CONFIGS if q["name"] == "own20_exp80"))
Wm = Wm_df.reindex(dates).values
i12 = np.searchsorted(dates.values, np.datetime64("2012-01-01"))

for drop_zero in (False, True):
    W2, s, sig, star = transform(Wb, rb, drop_zero=drop_zero)
    d_all = np.abs(W2[i12:T - 1] - np.nan_to_num(Wm[i12:T - 1], nan=0.0)).max()
    d_full = np.abs(W2[:T - 1] - np.nan_to_num(Wm[:T - 1], nan=0.0)).max()
    print(f"[drop_zero_vol_in_quantile={drop_zero}] max|dW| rows>=2012 (excl. final row): {d_all:.3e} | all rows: {d_full:.3e} | final-row module NaN count: "
          f"{int(np.isnan(Wm[T-1]).sum())}", flush=True)
    r_my = backtest(W2)
    r_mod = R.run(c, Wm_df).values
    print(f"    return series max|diff| (mine vs R.run(module W)): {np.abs(r_my - r_mod).max():.3e}")
    if not drop_zero:
        # diagnostics on the quirk: zeros / early-sample in expanding quantile
        zero_sig = int((sig == 0).sum())
        print("    sigma_hat==0 count (book empty, zero-vol included in the expanding sample):", zero_sig,
              "| first date star defined:", dates[np.where(np.isfinite(star))[0][0]].date(),
              "| first date s<1:", dates[np.where(s < 1)[0][0]].date())
        sdf = pd.Series(s, index=dates)
        print("    mean s 2012-21: %.3f | 2022-26: %.3f | frac days s<1 DEV: %.3f HOLD: %.3f" % (
            sdf["2012":"2021"].mean(), sdf["2022":].mean(), (sdf["2012":"2021"] < 1).mean(), (sdf["2022":] < 1).mean()))
print("secs", round(time.time() - t0))
