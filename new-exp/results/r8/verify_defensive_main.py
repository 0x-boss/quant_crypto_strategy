"""Independent verification of r8_f_defensive config trigF_usmv.
Part A: independent re-implementation from the TEXT description (own baseline book, own backtest, own trigger, USMV leg) vs module weights.
Part B: stats via R.run / R.evaluate, cost stress, lag, ex-2020/2021, decomposition.
"""
import sys, json, time
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import numpy as np, pandas as pd
import r8_common as R
import lib
import r8_f_defensive as mod

t0 = time.time()
OUT = {}
p = next(q for q in mod.CONFIGS if q["name"] == "trigF_usmv")
print("config:", p)
c = R.load(["O", "C", "mom_6_1", "M300"])
O = c["O"].astype(np.float64)
Cc = c["C"]
dates, cols = O.index, list(O.columns)
T, N = O.shape
Ov = O.values

# ------------------------------------------------------------------------------------------------------------------
# Part A: own implementation
# ------------------------------------------------------------------------------------------------------------------
mom = c["mom_6_1"].where(c["M300"]).astype(np.float64).values
sleeve = np.zeros((T, N))
for t in range(T):
    v = mom[t]
    ok = np.where(np.isfinite(v))[0]
    if len(ok) == 0:
        continue
    order = ok[np.argsort(-v[ok], kind="stable")]
    sleeve[t, order[:10]] = 0.1
cs = np.cumsum(sleeve, axis=0)
W0 = np.zeros_like(sleeve)
for t in range(T):
    lo = max(0, t - 4)
    W0[t] = (cs[t] - (cs[lo - 1] if lo > 0 else 0)) / (t - lo + 1)
W0_ref = R.base_weights(c)
print("own baseline vs R.base_weights max|d|:", float(np.abs(W0 - W0_ref.fillna(0).values).max()), flush=True)
OUT["base_book_maxdiff"] = float(np.abs(W0 - W0_ref.fillna(0).values).max())

usmv = cols.index("USMV")
held = sorted(set(np.where(W0.sum(0) > 0)[0].tolist()) | {usmv})
hidx = {j: i for i, j in enumerate(held)}


def backtest(Wfull, fee=R.FEE, lag=0):
    """position at open d = decision at close d-1-lag; return open d -> open d+1; fee on |w - drifted w|."""
    Wh = Wfull[:, held]
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
    return ret[:-1]


rb = backtest(W0)
rb_ref = R.run(c, W0_ref).values
print("own baseline return vs R.run max|d|:", float(np.abs(rb - rb_ref).max()), flush=True)
OUT["base_ret_maxdiff"] = float(np.abs(rb - rb_ref).max())

# sigma_hat[t]: annualised sample std (ddof 1) of the 20 returns known at the close of t = those indexed t-20..t-1
sig = np.full(T, np.nan)
for t in range(20, T):
    x = rb[t - 20: t]            # indices t-20..t-1
    if len(x) == 20:
        sig[t] = x.std(ddof=1) * np.sqrt(252)
sd = pd.Series(sig, index=dates)
star_exact = float(sd.loc["2012-01-01":"2021-12-31"].median())
print("DEV median sigma_hat (own):", star_exact, "module constant:", mod.STAR_DEV_MEDIAN)
OUT["star_exact_dev_median"] = star_exact
# the DEV median over days with sigma_hat>0 too
OUT["star_dev_median_pos"] = float(sd.loc["2012-01-01":"2021-12-31"].where(sd > 0).median())

ok_usmv = (Cc["USMV"].notna() & O["USMV"].notna()).values.astype(float)


def own_weights(star):
    s = np.ones(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_ = star / sig
    good = np.isfinite(ratio_)
    s[good] = np.minimum(1.0, ratio_[good])
    W = W0 * s[:, None]
    W[:, usmv] = (1.0 - s) * ok_usmv
    return W, s


Wown, s_own = own_weights(0.2501)
Wmod = mod.apply(c, W0_ref, p)
Wmod_v = Wmod.reindex(dates).reindex(columns=cols).fillna(0.0).values
m2012 = np.asarray(dates >= pd.Timestamp("2012-01-01"))
d_fixed = np.abs(Wown - Wmod_v)[m2012]
print("max |W_own - W_mod| (star=0.2501), rows>=2012:", float(d_fixed.max()), "all rows:", float(np.abs(Wown - Wmod_v).max()))
OUT["reimpl_maxdiff_2012"] = float(d_fixed.max())
OUT["reimpl_maxdiff_all"] = float(np.abs(Wown - Wmod_v).max())
Wown2, _ = own_weights(star_exact)
d2 = np.abs(Wown2 - Wmod_v)[m2012]
print("max |W_own(exact DEV median) - W_mod|:", float(d2.max()))
OUT["reimpl_maxdiff_2012_exactstar"] = float(d2.max())
# row-sum check
OUT["max_gross"] = float(Wmod_v.sum(1).max())
OUT["min_w"] = float(Wmod_v.min())
OUT["usmv_ever_unavailable_since_2012"] = int((ok_usmv[m2012] == 0).sum())
OUT["frac_days_s_lt_1_dev"] = float((s_own[(dates >= "2012-01-01") & (dates <= "2021-12-31")] < 1).mean())
OUT["frac_days_s_lt_1_hold"] = float((s_own[dates >= "2022-01-01"] < 1).mean())
OUT["mean_stock_exposure_dev"] = float(s_own[(dates >= "2012-01-01") & (dates <= "2021-12-31")].mean())
OUT["mean_stock_exposure_hold"] = float(s_own[dates >= "2022-01-01"].mean())
print("frac s<1 DEV/HOLD:", OUT["frac_days_s_lt_1_dev"], OUT["frac_days_s_lt_1_hold"], " mean s DEV/HOLD:", OUT["mean_stock_exposure_dev"], OUT["mean_stock_exposure_hold"])

# own backtest of candidate vs R.run
r_own = pd.Series(backtest(Wown), index=dates[:-1])
r_mod = R.run(c, Wmod)
print("own candidate return vs R.run(module) max|d|:", float((r_own - r_mod).abs().max()), flush=True)
OUT["cand_ret_maxdiff_own_vs_R"] = float((r_own - r_mod).abs().max())

# ------------------------------------------------------------------------------------------------------------------
# Part B: stats
# ------------------------------------------------------------------------------------------------------------------
r0 = R.run(c, W0_ref)
st0 = R.evaluate(r0, "BASELINE")
st1 = R.evaluate(r_mod, "trigF_usmv")
OUT["base"] = {k: st0[k] for k in ("dev", "hold", "all")}
OUT["cand"] = {k: st1[k] for k in ("dev", "hold", "all")}
print(R.fmt_row("baseline", st0))
print(R.fmt_row("trigF_usmv", st1))


def sh(r, a=None, b=None, excl_years=()):
    x = r.loc[a:b].dropna() if a else r.dropna()
    if excl_years:
        x = x[~x.index.year.isin(excl_years)]
    return float(x.mean() / x.std() * np.sqrt(252))


stress = {}
for tag, kw in (("fee1x", dict()), ("fee2x", dict(fee=2 * R.FEE)), ("fee4x", dict(fee=4 * R.FEE)), ("lag1", dict(lag=1)), ("lag1_fee2x", dict(lag=1, fee=2 * R.FEE))):
    rb_ = R.run(c, W0_ref, **kw)
    rc_ = R.run(c, Wmod, **kw)
    stress[tag] = dict(base_dev=sh(rb_, *lib.DEV), cand_dev=sh(rc_, *lib.DEV), base_hold=sh(rb_, *lib.HOLD), cand_hold=sh(rc_, *lib.HOLD),
                       cand_turn=float(rc_.attrs["turnover"].loc["2012":].mean()), base_turn=float(rb_.attrs["turnover"].loc["2012":].mean()))
    print(tag, {k: round(v, 3) for k, v in stress[tag].items()}, flush=True)
OUT["stress"] = stress

ex = {}
ex["base_dev_ex2020_2021"] = sh(r0, *lib.DEV, excl_years=(2020, 2021))
ex["cand_dev_ex2020_2021"] = sh(r_mod, *lib.DEV, excl_years=(2020, 2021))
ex["base_dev_ex2020"] = sh(r0, *lib.DEV, excl_years=(2020,))
ex["cand_dev_ex2020"] = sh(r_mod, *lib.DEV, excl_years=(2020,))
ex["base_dev_ex2021"] = sh(r0, *lib.DEV, excl_years=(2021,))
ex["cand_dev_ex2021"] = sh(r_mod, *lib.DEV, excl_years=(2021,))
print(ex)
OUT["ex"] = ex

# by-year Sharpe and return
yr = []
for y in range(2012, 2027):
    a, b = r0[str(y)], r_mod[str(y)]
    yr.append(dict(year=y, base_sh=float(a.mean() / a.std() * np.sqrt(252)), cand_sh=float(b.mean() / b.std() * np.sqrt(252)),
                   base_ret=float((1 + a).prod() - 1), cand_ret=float((1 + b).prod() - 1), mean_s=float(s_own[dates.year == y].mean())))
yd = pd.DataFrame(yr)
print(yd.round(3).to_string(index=False))
OUT["by_year"] = yr
OUT["n_years_cand_sh_gt_base_dev"] = int((yd[yd.year <= 2021].cand_sh > yd[yd.year <= 2021].base_sh).sum())
OUT["n_years_cand_sh_gt_base_all"] = int((yd.cand_sh > yd.base_sh).sum())

# decomposition: cash twin (freed fraction idle) and USMV buy&hold
Wcash = mod.apply(c, W0_ref, dict(kind="trig", mode="fixed", star=0.2501, dest="cash"))
r_cash = R.run(c, Wcash)
stc = R.evaluate(r_cash, "cash twin")
OUT["cash_twin"] = {k: stc[k] for k in ("dev", "hold", "all")}
# USMV buy and hold open-to-open
Wu = pd.DataFrame(0.0, index=dates, columns=cols)
Wu["USMV"] = ok_usmv
r_u = R.run(c, Wu)
stu = R.evaluate(r_u, "USMV buy&hold")
OUT["usmv_bh"] = {k: stu[k] for k in ("dev", "hold", "all")}

json.dump(OUT, open("/home/user/quant_crypto_strategy/new-exp/results/r8/verify_defensive_main_out.json", "w"), indent=1, default=float)
print("done", round(time.time() - t0), "s")
