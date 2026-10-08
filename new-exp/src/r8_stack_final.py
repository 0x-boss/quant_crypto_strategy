"""Round 8 stage 2b: final evaluation of the DEV-chosen stack cell (R8_STACK_PLAN.md).   python src/r8_stack_final.py"""
import json
import numpy as np
import pandas as pd

import lib
import r8_common as R
import r8_stack as S

ANN = np.sqrt(252)
choice = json.load(open(f"{R.OUTDIR}/stack_choice.json"))["chosen"]
cells = {S.name(cell): cell for cell in S.CELLS}
cell = cells[choice]
print("chosen cell:", choice, cell)
c = R.load()
W0 = R.base_weights(c)
r0 = R.run(c, W0)
W = S.build(c, cell)
r = R.run(c, W)
base, cand = R.evaluate(r0, "BASELINE", show=True), R.evaluate(r, f"STACK {choice}", show=True)
out = dict(chosen=choice)

# 1. look-ahead guard
ok, worst = R.check_perturbation(lambda cc: S.build(cc, cell), c)
out["perturbation_ok"], out["perturbation_worst"] = bool(ok), worst
tr_ok, tr_diff = R.check_truncation(lambda cc: R.run(cc, S.build(cc, cell)), c)
out["truncation_ok"] = bool(tr_ok)

# 2. acceptance, paired terms
hs, he = lib.HOLD
rh0, rh = r0.loc[hs:he], r.loc[hs:he]
m0, m1 = lib.monthly(rh0), lib.monthly(rh)
rng = np.random.default_rng(1)
idx = rng.integers(0, len(m0), (5000, len(m0)))
dmed = np.median(m1.values[idx], axis=1) - np.median(m0.values[idx], axis=1)
dshare = (m1.values[idx] >= 0.05).mean(axis=1) - (m0.values[idx] >= 0.05).mean(axis=1)
blk = lib.block_bootstrap  # daily paired block bootstrap for Sharpe / DD differences
def paired_block(a, b, n=2000, block=21, seed=3):
    rng = np.random.default_rng(seed)
    x, y = a.dropna().values, b.reindex(a.dropna().index).values
    T = len(x); nb = int(np.ceil(T / block))
    dsh, ddd = np.empty(n), np.empty(n)
    for i in range(n):
        st = rng.integers(0, T, nb)
        ii = (st[:, None] + np.arange(block)[None, :]) % T
        ii = ii.ravel()[:T]
        xa, ya = x[ii], y[ii]
        dsh[i] = ya.mean() / ya.std() * ANN - xa.mean() / xa.std() * ANN
        ea, eb = np.cumprod(1 + xa), np.cumprod(1 + ya)
        ddd[i] = (eb / np.maximum.accumulate(eb) - 1).min() - (ea / np.maximum.accumulate(ea) - 1).min()
    return dsh, ddd
dsh_h, ddd_h = paired_block(rh0, rh)
dsh_d, ddd_d = paired_block(r0.loc[lib.DEV[0]:lib.DEV[1]], r.loc[lib.DEV[0]:lib.DEV[1]])
q = lambda v: [float(np.percentile(v, 5)), float(np.percentile(v, 50)), float(np.percentile(v, 95))]
out["acceptance"] = dict(
    hold_cagr_base=base["hold"]["cagr"], hold_cagr_stack=cand["hold"]["cagr"], cagr_retention=cand["hold"]["cagr"] / base["hold"]["cagr"],
    calmar_hold=[base["hold"]["calmar"], cand["hold"]["calmar"]], ulcer_hold=[base["hold"]["ulcer"], cand["hold"]["ulcer"]],
    hold_median=[base["hold"]["med_m"], cand["hold"]["med_m"]], hold_share_ge5=[float((m0 >= .05).mean()), float((m1 >= .05).mean())],
    months_ge5_hold=[int((m0 >= .05).sum()), int((m1 >= .05).sum()), len(m0)],
    p_median_gt5=[R.month_bootstrap_median_gt(rh0), R.month_bootstrap_median_gt(rh)],
    paired_median_diff_ci=q(dmed), paired_share_ge5_diff_ci=q(dshare),
    paired_sharpe_diff_hold_ci=q(dsh_h), paired_maxdd_diff_hold_ci=q(ddd_h),
    paired_sharpe_diff_dev_ci=q(dsh_d), paired_maxdd_diff_dev_ci=q(ddd_d),
    maxdd=dict(dev=[base["dev"]["maxdd"], cand["dev"]["maxdd"]], hold=[base["hold"]["maxdd"], cand["hold"]["maxdd"]], all=[base["all"]["maxdd"], cand["all"]["maxdd"]]),
    sharpe=dict(dev=[base["dev"]["sharpe"], cand["dev"]["sharpe"]], hold=[base["hold"]["sharpe"], cand["hold"]["sharpe"]], all=[base["all"]["sharpe"], cand["all"]["sharpe"]]),
)
print(json.dumps(out["acceptance"], indent=1, default=float))

# 3. timing placebo: circular shifts of the exposure series applied to the unscaled selection-layer book
Wsel = S.select_layer(c, cell[0])
rsel = R.run(c, Wsel)
s_act = (W.fillna(0).sum(axis=1) / Wsel.fillna(0).sum(axis=1)).reindex(rsel.index).fillna(1.0).clip(upper=1.0)
def approx(sh):
    return rsel * sh
act = approx(s_act)
def stats_approx(x):
    p = {part: R._period(x, *rng_) for part, rng_ in (("dev", lib.DEV), ("hold", lib.HOLD))}
    return p
a = stats_approx(act)
sh_pl = {"dev": [], "hold": []}; dd_pl = {"dev": [], "hold": []}; med_pl = []
T = len(s_act)
for k in range(200):
    shift = int(rng.integers(63, T - 63))
    x = approx(pd.Series(np.roll(s_act.values, shift), index=s_act.index))
    p = stats_approx(x)
    for part in ("dev", "hold"):
        sh_pl[part].append(p[part]["sharpe"]); dd_pl[part].append(p[part]["maxdd"])
    med_pl.append(p["hold"]["med_m"])
out["placebo"] = {part: dict(actual_sharpe=a[part]["sharpe"], placebo_mean_sharpe=float(np.mean(sh_pl[part])), p_sharpe=float((np.array(sh_pl[part]) >= a[part]["sharpe"]).mean()),
                             actual_maxdd=a[part]["maxdd"], placebo_mean_maxdd=float(np.mean(dd_pl[part])), p_maxdd=float((np.array(dd_pl[part]) >= a[part]["maxdd"]).mean())) for part in ("dev", "hold")}
out["placebo"]["hold_median"] = dict(actual=a["hold"]["med_m"], placebo_mean=float(np.mean(med_pl)), p_ge=float((np.array(med_pl) >= a["hold"]["med_m"]).mean()))
out["placebo"]["avg_gross"] = float(s_act.loc["2012":].mean())
print(json.dumps(out["placebo"], indent=1, default=float))

# 4. cost / lag stress
def shp(x, part, ex=None):
    a_, b_ = {"dev": lib.DEV, "hold": lib.HOLD}[part]
    y = x.loc[a_:b_]
    if ex:
        y = y[~y.index.year.isin(ex)]
    return float(y.mean() / y.std() * ANN)
stress = {}
for label, kw in (("fee_x1", {}), ("fee_x2", dict(fee=2 * R.FEE)), ("fee_x4", dict(fee=4 * R.FEE)), ("lag1", dict(lag=1))):
    b_, s_ = R.run(c, W0, **kw), R.run(c, W, **kw)
    stress[label] = dict(dev=[shp(b_, "dev"), shp(s_, "dev")], hold=[shp(b_, "hold"), shp(s_, "hold")])
stress["dev_ex2020_2021"] = [shp(r0, "dev", (2020, 2021)), shp(r, "dev", (2020, 2021))]
stress["dev_ex2021_window"] = [S.sharpe_ex(r0, *lib.DEV), S.sharpe_ex(r, *lib.DEV)]
out["stress"] = stress
print(json.dumps(stress, indent=1))

# 5. sibling books: same exposure layer on other momentum sleeves (paired change vs their own unscaled book)
sib = {}
books = {"k20_mom61": R.base_weights(c, k=20), "mom12_1_k10": R.base_weights(c, sig="mom_12_1"), "top1000_mom61": R.base_weights(c, mask="M1000"),
         "res_mom12_1_k10": R.base_weights(c, sig="res_mom_12_1")}
for nm, Wb in books.items():
    rb = R.run(c, Wb)
    rs = R.run(c, S.expo_layer(c, Wb, cell[1])) if cell[1] != "none" else rb
    sb, ss = R.evaluate(rb, show=False), R.evaluate(rs, show=False)
    sib[nm] = {part: dict(sharpe=[sb[part]["sharpe"], ss[part]["sharpe"]], maxdd=[sb[part]["maxdd"], ss[part]["maxdd"]], median=[sb[part]["med_m"], ss[part]["med_m"]]) for part in ("dev", "hold")}
    print(nm, {p: (round(sib[nm][p]["sharpe"][0], 2), round(sib[nm][p]["sharpe"][1], 2), round(sib[nm][p]["maxdd"][0], 2), round(sib[nm][p]["maxdd"][1], 2)) for p in ("dev", "hold")})
out["siblings"] = sib

json.dump(out, open(f"{R.OUTDIR}/stack_final.json", "w"), indent=1, default=float)
pd.to_pickle(dict(base=r0, stack=r, s=s_act), f"{R.OUTDIR}/stack_final_returns.pkl")
print("written results/r8/stack_final.json")
