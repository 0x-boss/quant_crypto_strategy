import sys, json, time
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import numpy as np, pandas as pd
import r8_common as R
import r8_f_defensive as mod
import lib

t0 = time.time()
c = R.load(["O", "C", "mom_6_1", "M300"])
W0 = R.base_weights(c)
p = next(q for q in mod.CONFIGS if q["name"] == "trigF_usmv_gld")
W = mod.apply(c, W0, p)
out = {}


def sh(x):
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(252))


def mdd(x):
    eq = (1 + x.dropna()).cumprod()
    return float((eq / eq.cummax() - 1).min())


def excl(x, years=(2020, 2021), a="2012-01-01", b="2021-12-31"):
    x = x.loc[a:b]
    return x[~x.index.year.isin(years)]


rc = R.run(c, W)
rb = R.run(c, W0)
print("=== baseline vs candidate through R.evaluate ===")
stc = R.evaluate(rc, "CANDIDATE trigF_usmv_gld")
stb = R.evaluate(rb, "BASELINE")
out["dev_sharpe_recomputed"] = sh(rc.loc[lib.DEV[0]:lib.DEV[1]])
out["hold_sharpe_recomputed"] = sh(rc.loc[lib.HOLD[0]:lib.HOLD[1]])
out["all_sharpe"] = sh(rc.loc["2012":]); out["all_maxdd"] = mdd(rc.loc["2012":])
out["turnover_per_day_2012+"] = float(rc.attrs["turnover"].loc["2012":].mean())
out["baseline_dev_sharpe"] = sh(rb.loc[lib.DEV[0]:lib.DEV[1]]); out["baseline_hold_sharpe"] = sh(rb.loc[lib.HOLD[0]:lib.HOLD[1]])
print("manual DEV/HOLD/ALL Sharpe cand:", out["dev_sharpe_recomputed"], out["hold_sharpe_recomputed"], out["all_sharpe"], "ALL maxDD", out["all_maxdd"], "turnover", out["turnover_per_day_2012+"])
print("manual DEV/HOLD Sharpe base:", out["baseline_dev_sharpe"], out["baseline_hold_sharpe"])
print("HOLD median month cand: %.4f   base: %.4f" % (stc["hold"]["med_m"], stb["hold"]["med_m"]))

print("\n=== cost / lag stress (DEV Sharpe) : candidate | baseline ===")
for lab, kw in (("1x", {}), ("2x fee", dict(fee=2 * R.FEE)), ("4x fee", dict(fee=4 * R.FEE)), ("lag1", dict(lag=1)), ("2x fee + lag1", dict(fee=2 * R.FEE, lag=1))):
    a = R.run(c, W, **kw); b = R.run(c, W0, **kw)
    da, db = a.loc[lib.DEV[0]:lib.DEV[1]], b.loc[lib.DEV[0]:lib.DEV[1]]
    ha, hb = a.loc[lib.HOLD[0]:lib.HOLD[1]], b.loc[lib.HOLD[0]:lib.HOLD[1]]
    out[f"stress_{lab}"] = dict(dev_cand=sh(da), dev_base=sh(db), hold_cand=sh(ha), hold_base=sh(hb), dev_dd_cand=mdd(da), dev_dd_base=mdd(db))
    print(f"{lab:14s} DEV cand {sh(da):.4f} base {sh(db):.4f} (delta {sh(da)-sh(db):+.4f}) | HOLD cand {sh(ha):.4f} base {sh(hb):.4f} | DEV DD cand {mdd(da):.3f} base {mdd(db):.3f}")
    if lab in ("1x", "2x fee", "4x fee"):
        ea, eb = excl(a), excl(b)
        out[f"excl2021_{lab}"] = dict(cand=sh(ea), base=sh(eb), dd_cand=mdd(ea), dd_base=mdd(eb), n=len(ea))
        print(f"   DEV excl. 2020,2021 ({len(ea)} days) : cand {sh(ea):.4f} base {sh(eb):.4f} (delta {sh(ea)-sh(eb):+.4f})  | maxDD cand {mdd(ea):.3f} base {mdd(eb):.3f}")

print("\n=== calendar-year Sharpe (cand | base) and return ===")
rows = []
for y in range(2012, 2027):
    a, b = rc[rc.index.year == y], rb[rb.index.year == y]
    rows.append((y, sh(a), sh(b), float((1 + a).prod() - 1), float((1 + b).prod() - 1), mdd(a), mdd(b)))
yr = pd.DataFrame(rows, columns=["year", "sh_cand", "sh_base", "ret_cand", "ret_base", "dd_cand", "dd_base"]).set_index("year")
print(yr.round(3).to_string())
out["years"] = yr.round(4).reset_index().to_dict("records")
for lab, ys in (("excl 2020", (2020,)), ("excl 2021", (2021,)), ("excl 2020+2021", (2020, 2021))):
    ea, eb = excl(rc, ys), excl(rb, ys)
    print(f"DEV {lab}: cand {sh(ea):.4f} base {sh(eb):.4f} delta {sh(ea)-sh(eb):+.4f}")
    out[f"dev_{lab}"] = dict(cand=sh(ea), base=sh(eb))
# also: excl 2020-21 on the full sample (2012-2026) for completeness
ea = rc.loc["2012":]; ea = ea[~ea.index.year.isin((2020, 2021))]
eb = rb.loc["2012":]; eb = eb[~eb.index.year.isin((2020, 2021))]
print("ALL 2012-2026 excl 2020,2021: cand %.4f base %.4f" % (sh(ea), sh(eb)))

print("\n=== causal / neighbouring variants (post-hoc info only) ===")
variants = {
    "ph_trigE (expanding median, causal sigma*)": next(q for q in mod.CONFIGS if q["name"] == "ph_trigE_usmv_gld"),
    "sigma*=0.20": dict(p, star=0.20),
    "sigma*=0.30": dict(p, star=0.30),
    "trigF_cash (no ETF)": next(q for q in mod.CONFIGS if q["name"] == "trigF_cash"),
}
for nm, pp in variants.items():
    Wv = mod.apply(c, W0, pp)
    for lab, kw in (("1x", {}), ("2x", dict(fee=2 * R.FEE))):
        a = R.run(c, Wv, **kw)
        da = a.loc[lib.DEV[0]:lib.DEV[1]]; ea = excl(a)
        print(f"{nm:44s} {lab}: DEV Sh {sh(da):.4f} DD {mdd(da):.3f} | DEV excl20/21 Sh {sh(ea):.4f} | HOLD Sh {sh(a.loc[lib.HOLD[0]:]):.4f}")
        out[f"variant_{nm}_{lab}"] = dict(dev=sh(da), dev_dd=mdd(da), dev_excl=sh(ea), hold=sh(a.loc[lib.HOLD[0]:]))

print("\n=== placebo: circularly shift the exposure s inside DEV (same DEV exposure distribution, destroyed timing) ===")
s = mod.trig_scale(c, W0, p)
idx_dev = s.loc[lib.DEV[0]:lib.DEV[1]].index
rng = np.random.default_rng(12345)
res = []
for k in rng.integers(120, len(idx_dev) - 120, size=40):
    sp = s.copy()
    sp.loc[idx_dev] = np.roll(s.loc[idx_dev].to_numpy(), int(k))
    Wp = W0.mul(sp, axis=0).copy()
    for tkr in ("USMV", "GLD"):
        Wp[tkr] = ((1.0 - sp) * 0.5 * mod.avail(c, tkr, Wp.index)).astype(np.float64)
    a = R.run(c, Wp)
    da = a.loc[lib.DEV[0]:lib.DEV[1]]
    res.append((int(k), sh(da), mdd(da), float(R.lib.monthly(da).mean())))
pl = pd.DataFrame(res, columns=["shift", "dev_sh", "dev_dd", "dev_mean_m"])
print(pl.describe().round(4).to_string())
act = out["dev_sharpe_recomputed"]
print("actual DEV Sh %.4f ; placebo mean %.4f max %.4f ; frac placebo >= actual: %.3f" % (act, pl.dev_sh.mean(), pl.dev_sh.max(), (pl.dev_sh >= act).mean()))
print("actual DEV DD %.3f ; placebo DD mean %.3f best %.3f" % (stc["dev"]["maxdd"], pl.dev_dd.mean(), pl.dev_dd.max()))
out["placebo"] = dict(act=act, mean=float(pl.dev_sh.mean()), max=float(pl.dev_sh.max()), std=float(pl.dev_sh.std()), frac_ge=float((pl.dev_sh >= act).mean()),
                      dd_mean=float(pl.dev_dd.mean()), dd_best=float(pl.dev_dd.max()))
json.dump(out, open("/home/user/quant_crypto_strategy/new-exp/results/r8/verify_defensive_usmvgld_stats_out.json", "w"), indent=1, default=float)
print("done in %.0fs" % (time.time() - t0))
