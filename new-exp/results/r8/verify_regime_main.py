import sys, json
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import numpy as np, pandas as pd
import lib
import r8_common as R
import r8_f_regime as mod

c = R.load()
p = next(q for q in mod.CONFIGS if q["name"] == "ph_disp_half80_x75")
Wb = R.base_weights(c)
Wm = mod.apply(c, Wb, p)

# ---------------------------------------------------------------- independent re-implementation (written from the text description only)
C21 = c["ret21"].astype(np.float64).to_numpy()
inU = c["M300"].to_numpy()
dates = c["ret21"].index
T = len(dates)
disp = np.full(T, np.nan)
for i in range(T):
    v = C21[i][inU[i] & np.isfinite(C21[i])]
    if v.size >= 10:
        q25, q75 = np.percentile(v, [25, 75])
        disp[i] = (q75 - q25) / 1.349

def reimpl_exposure(minobs=504, hi=0.80, lo=0.75):
    pct = np.full(T, np.nan)
    seen = []
    import bisect
    for i in range(T):
        if np.isfinite(disp[i]):
            bisect.insort(seen, disp[i])
            if len(seen) >= minobs:
                # share of past-and-present values that are <= today's value
                pct[i] = np.searchsorted(seen, disp[i], side="right") / len(seen)
    ex = np.ones(T)
    half = False
    for i in range(T):
        if np.isfinite(pct[i]):
            if (not half) and pct[i] >= hi:
                half = True
            elif half and pct[i] < lo:
                half = False
        ex[i] = 0.5 if half else 1.0
    return pd.Series(ex, index=dates), pd.Series(pct, index=dates)

# independent baseline book too: top-10 of mom_6_1 inside M300, equal weight 1/10, mean of last 5 daily books
sig = c["mom_6_1"].astype(np.float64).where(c["M300"])
ranks = sig.rank(axis=1, ascending=False, method="first")
daily = ((ranks <= 10) & sig.notna()).astype(float) / 10.0
Wb2 = sum(daily.shift(k).fillna(0.0) for k in range(5)) / 5.0
# NB: rolling(5, min_periods=1).mean() in the module divides by the number of available rows at the very start; compare from 2012 only
Wb2 = Wb2
ex, pct = reimpl_exposure()
W2 = Wb2.mul(ex, axis=0)
sl = slice("2012-01-01", None)
cols = Wm.columns.union(W2.columns)
dW = (Wm.reindex(columns=cols).fillna(0.0).loc[sl] - W2.reindex(columns=cols).fillna(0.0).loc[sl]).abs().to_numpy().max()
dB = (Wb.reindex(columns=cols).fillna(0.0).loc[sl] - Wb2.reindex(columns=cols).fillna(0.0).loc[sl]).abs().to_numpy().max()
print("first valid disp:", pd.Series(disp, index=dates).first_valid_index(), " first valid pct:", pct.first_valid_index())
print("MAX |W_module - W_reimpl| (>=2012):", dW, "  baseline-book diff:", dB)
e_mod = mod.exposure(c, Wb, p).reindex(dates)
print("exposure series max diff (all dates):", float((e_mod - ex).abs().max()), " (>=2012):", float((e_mod - ex).loc[sl].abs().max()))
# sensitivity of reimplementation to min-obs choice
for mo in (252, 126, 756):
    ex_m, _ = reimpl_exposure(minobs=mo)
    print(f"  minobs={mo}: max exposure diff vs module >=2012: {float((e_mod - ex_m).loc[sl].abs().max())}, share of days differing: {float((e_mod != ex_m).loc[sl].mean()):.4f}")

# ---------------------------------------------------------------- performance
def sharpe(x):
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(252))

def dev_ex(r, years=(2020, 2021)):
    x = r.loc[lib.DEV[0]:lib.DEV[1]].dropna()
    return sharpe(x[~x.index.year.isin(years)])

out = {}
rb, rm = R.run(c, Wb), R.run(c, Wm)
sb, sm = R.evaluate(rb, "BASELINE"), R.evaluate(rm, "CANDIDATE")
out["baseline"] = {k: sb[k] for k in ("dev", "hold", "all")}
out["candidate"] = {k: sm[k] for k in ("dev", "hold", "all")}
for tag, kw in (("fee2x", dict(fee=2 * R.FEE)), ("fee4x", dict(fee=4 * R.FEE)), ("lag1", dict(lag=1)), ("fee2x_lag1", dict(fee=2 * R.FEE, lag=1))):
    a, b = R.run(c, Wb, **kw), R.run(c, Wm, **kw)
    out[tag] = dict(base_dev=sharpe(a.loc[lib.DEV[0]:lib.DEV[1]]), cand_dev=sharpe(b.loc[lib.DEV[0]:lib.DEV[1]]),
                    base_hold=sharpe(a.loc[lib.HOLD[0]:lib.HOLD[1]]), cand_hold=sharpe(b.loc[lib.HOLD[0]:lib.HOLD[1]]))
    print(tag, out[tag])
out["ex2020_2021"] = dict(base=dev_ex(rb), cand=dev_ex(rm))
out["ex2020"] = dict(base=dev_ex(rb, (2020,)), cand=dev_ex(rm, (2020,)))
out["ex2021"] = dict(base=dev_ex(rb, (2021,)), cand=dev_ex(rm, (2021,)))
print("DEV Sharpe excl 2020+2021:", out["ex2020_2021"], " excl 2020:", out["ex2020"], " excl 2021:", out["ex2021"])

# by-year Sharpe and return
yr = []
for y in range(2012, 2027):
    a, b = rb.loc[str(y)].dropna(), rm.loc[str(y)].dropna()
    yr.append(dict(year=y, base_sh=sharpe(a), cand_sh=sharpe(b), base_ret=float((1 + a).prod() - 1), cand_ret=float((1 + b).prod() - 1),
                   half_share=float((ex.loc[str(y)] == 0.5).mean())))
yrdf = pd.DataFrame(yr)
print(yrdf.round(3).to_string())
out["by_year"] = yr
# leave-one-DEV-year-out
loyo = {}
for y in range(2012, 2022):
    loyo[y] = dict(base=dev_ex(rb, (y,)), cand=dev_ex(rm, (y,)))
out["loyo"] = loyo
print("leave-one-year-out DEV Sharpe:", {k: (round(v["base"], 3), round(v["cand"], 3)) for k, v in loyo.items()})

# occupancy
out["half_share_dev"] = float((ex.loc[lib.DEV[0]:lib.DEV[1]] == 0.5).mean())
out["half_share_hold"] = float((ex.loc[lib.HOLD[0]:lib.HOLD[1]] == 0.5).mean())
out["avg_gross"] = [float(Wm.sum(axis=1).loc[lib.DEV[0]:lib.DEV[1]].mean()), float(Wm.sum(axis=1).loc[lib.HOLD[0]:lib.HOLD[1]].mean())]
print("half share DEV/HOLD", out["half_share_dev"], out["half_share_hold"], "avg gross", out["avg_gross"])

# monthly-block bootstrap of the DEV Sharpe difference
d = (rm - rb).loc[lib.DEV[0]:lib.DEV[1]].dropna()
x_b = rb.loc[lib.DEV[0]:lib.DEV[1]].dropna(); x_m = rm.loc[lib.DEV[0]:lib.DEV[1]].dropna()
months = x_b.index.to_period("M")
groups = [np.where(months == m)[0] for m in months.unique()]
rng = np.random.default_rng(1)
diffs = []
for _ in range(2000):
    pick = rng.integers(0, len(groups), len(groups))
    idx = np.concatenate([groups[i] for i in pick])
    a, b = x_b.values[idx], x_m.values[idx]
    diffs.append(b.mean() / b.std() - a.mean() / a.std())
diffs = np.array(diffs) * np.sqrt(252)
out["boot_dev_sharpe_diff"] = dict(mean=float(diffs.mean()), p05=float(np.percentile(diffs, 5)), p95=float(np.percentile(diffs, 95)), p_pos=float((diffs > 0).mean()))
print("month-block bootstrap DEV Sharpe diff:", out["boot_dev_sharpe_diff"])

json.dump(out, open("/home/user/quant_crypto_strategy/new-exp/results/r8/verify_regime_main_out.json", "w"), default=float, indent=1)
ex.to_csv("/home/user/quant_crypto_strategy/new-exp/results/r8/verify_regime_expo.csv")
