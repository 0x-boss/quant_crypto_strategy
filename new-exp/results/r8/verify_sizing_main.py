"""Independent verification of sizing config k10_iv60_cap20_vcap100_sec3.  Re-implementation written from the TEXT description only."""
import sys, json, time
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import numpy as np, pandas as pd
import lib
import r8_common as R
import r8_f_sizing as mod

OUT = "/home/user/quant_crypto_strategy/new-exp/results/r8/verify_sizing_out.json"
res = {}
t0 = time.time()
c = R.load()
p = next(q for q in mod.CONFIGS if q["name"] == "k10_iv60_cap20_vcap100_sec3")

# ---------------------------------------------------------------- independent re-implementation
K, VCAP, CAP, SMAX, H = 10, 1.0, 0.20, 3, 5

def cap_weights(w, cap):
    """w: positive Series summing to 1.  Final weight_i = min(cap, a*w_i) with a>=1 chosen so that weights sum to 1 (bisection)."""
    n = len(w)
    if n * cap <= 1.0:
        return pd.Series(cap, index=w.index)
    lo, hi = 1.0, 1e6
    x = w.values
    for _ in range(300):
        a = 0.5 * (lo + hi)
        if np.minimum(cap, a * x).sum() < 1.0:
            lo = a
        else:
            hi = a
    return pd.Series(np.minimum(cap, 0.5 * (lo + hi) * x), index=w.index)

def my_daily_books(c, na_as_one_sector=False):
    mom = c["mom_6_1"]; v60 = c["vol60"]; M = c["M300"]
    sec = c["sector"].reindex(mom.columns)
    secv = sec.fillna("NA").astype(str)
    books = []
    nn = []
    for t in range(len(mom.index)):
        s = mom.iloc[t].astype("float64")
        v = v60.iloc[t].astype("float64")
        ann = v * np.sqrt(252)
        ok = M.iloc[t] & s.notna() & v.notna() & (v > 0) & (ann <= VCAP)
        cand = s[ok].sort_values(ascending=False, kind="stable")      # best momentum first, ties by column order
        if len(cand) == 0:
            books.append(pd.Series(0.0, index=mom.columns)); nn.append(0); continue
        lab = secv[cand.index]
        if not na_as_one_sector:
            lab = pd.Series(np.where(lab.values == "NA", "NA::" + cand.index.astype(str), lab.values), index=cand.index)
        within = lab.groupby(lab.values, sort=False).cumcount()          # 0,1,2,... within each sector in rank order
        kept = cand.index[(within.values < SMAX)][:K]
        iv = 1.0 / v[kept]
        w = iv / iv.sum()
        w = cap_weights(w, CAP)
        w = w * min(1.0, len(kept) / K)
        b = pd.Series(0.0, index=mom.columns)
        b[kept] = w.values
        books.append(b); nn.append(len(kept))
    B = pd.DataFrame(np.vstack([x.values for x in books]), index=mom.index, columns=mom.columns)
    return B.rolling(H, min_periods=1).mean(), pd.Series(nn, index=mom.index)

Wm = mod.weights(c, p)
print("module weights built", round(time.time() - t0), "s", flush=True)
Wi, nsel = my_daily_books(c)
print("reimpl built", round(time.time() - t0), "s", flush=True)
d = (Wm.loc["2012":] - Wi.loc["2012":]).abs()
res["reimpl_max_abs_weight_diff_ge2012"] = float(d.to_numpy().max())
res["reimpl_max_abs_weight_diff_all_dates"] = float((Wm - Wi).abs().to_numpy().max())
res["reimpl_rows_with_diff_gt_1e-9"] = int((d.max(axis=1) > 1e-9).sum())
res["n_selected_lt10_days_ge2012"] = int((nsel.loc["2012":] < 10).sum())
res["gross_max"] = float(Wm.sum(axis=1).max()); res["gross_mean_ge2012"] = float(Wm.sum(axis=1).loc["2012":].mean())
res["max_single_weight"] = float(Wm.to_numpy().max())
print({k: v for k, v in res.items()}, flush=True)

# sensitivity: treat unknown-sector (NA) tickers as ONE sector (cap 3 applies to them too)
Wna, _ = my_daily_books(c, na_as_one_sector=True)
res["sens_NA_as_one_sector_max_diff_vs_module"] = float((Wm.loc["2012":] - Wna.loc["2012":]).abs().to_numpy().max())
res["sens_NA_as_one_sector_dev_sharpe"] = float(R.evaluate(R.run(c, Wna), "NA-one-sector", show=False)["dev"]["sharpe"])
# which unknown-sector names are selected by the module (share of weight)?
secs = c["sector"].reindex(Wm.columns).fillna("NA")
res["share_of_weight_in_NA_sector_names_ge2012"] = float(Wm.loc["2012":, (secs == "NA").values].sum(axis=1).mean())
print("NA share", res["share_of_weight_in_NA_sector_names_ge2012"], flush=True)

# ---------------------------------------------------------------- performance
Wb = R.base_weights(c)
res["base_replica_vs_Rbase_max_diff"] = float((mod.weights(c, mod._cfg("base_replica")).loc["2012":] - Wb.loc["2012":]).abs().to_numpy().max())

def sh(x):
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(252))

def devslice(r):
    return r.loc[lib.DEV[0]:lib.DEV[1]].dropna()

runs = {}
for nm, W in (("cand", Wm), ("base", Wb), ("cand_indep", Wi)):
    for tag, kw in (("1x", dict()), ("2x", dict(fee=2 * R.FEE)), ("4x", dict(fee=4 * R.FEE)), ("lag1", dict(lag=1))):
        if nm == "cand_indep" and tag != "1x":
            continue
        r = R.run(c, W, **kw)
        runs[(nm, tag)] = r
        print(nm, tag, "DEV Sh %.4f" % sh(devslice(r)), flush=True)

for nm in ("cand", "base", "cand_indep"):
    st = R.evaluate(runs[(nm, "1x")], nm, show=True)
    res[f"{nm}_eval_1x"] = {k: {m: float(v) for m, v in st[k].items()} for k in st}
    res[f"{nm}_turnover_per_day"] = float(runs[(nm, "1x")].attrs["turnover"].loc["2012":].mean())
    for tag in ("2x", "4x", "lag1"):
        if (nm, tag) in runs:
            res[f"{nm}_dev_sharpe_{tag}"] = sh(devslice(runs[(nm, tag)]))
            res[f"{nm}_hold_sharpe_{tag}"] = sh(runs[(nm, tag)].loc[lib.HOLD[0]:lib.HOLD[1]])

# excluding calendar years 2020 and 2021 from DEV
for nm in ("cand", "base"):
    for tag in ("1x", "2x", "4x", "lag1"):
        r = devslice(runs[(nm, tag)])
        res[f"{nm}_dev_sharpe_ex2020_2021_{tag}"] = sh(r[~r.index.year.isin([2020, 2021])])
r = devslice(runs[("cand", "1x")]); rb = devslice(runs[("base", "1x")])
# year by year
yb = {}
for y in range(2012, 2027):
    a = runs[("cand", "1x")]; b = runs[("base", "1x")]
    xa, xb = a[a.index.year == y].dropna(), b[b.index.year == y].dropna()
    yb[y] = dict(cand_sh=float(xa.mean() / xa.std() * np.sqrt(252)), base_sh=float(xb.mean() / xb.std() * np.sqrt(252)),
                 cand_ret=float((1 + xa).prod() - 1), base_ret=float((1 + xb).prod() - 1))
res["by_year"] = yb
# also excluding only the two worst windows
for nm in ("cand", "base"):
    r = devslice(runs[(nm, "1x")])
    mask = ~(((r.index >= "2020-02-01") & (r.index <= "2020-04-30")) | ((r.index >= "2021-02-01") & (r.index <= "2021-05-31")))
    res[f"{nm}_dev_sharpe_ex_windows_2020Q1-Q2_2021Feb-May"] = sh(r[mask])
# paired block-bootstrap of the DEV Sharpe difference (21-day blocks) - how often cand > base?
rng = np.random.default_rng(1)
a = devslice(runs[("cand", "1x")]).values; b = devslice(runs[("base", "1x")]).reindex(devslice(runs[("cand", "1x")]).index).values
n = len(a); blk = 21; nb = n // blk + 1
wins = 0; diffs = []
for _ in range(2000):
    st_ = rng.integers(0, n - blk, nb)
    idx = (st_[:, None] + np.arange(blk)[None, :]).ravel()[:n]
    sa, sb = a[idx], b[idx]
    diffs.append(sa.mean() / sa.std() - sb.mean() / sb.std())
diffs = np.array(diffs) * np.sqrt(252)
res["dev_sharpe_diff_bootstrap"] = dict(mean=float(diffs.mean()), p_gt0=float((diffs > 0).mean()), q05=float(np.quantile(diffs, .05)), q95=float(np.quantile(diffs, .95)))
res["cand_vs_base_daily_ret_corr_dev"] = float(np.corrcoef(a, b)[0, 1])

# ---------------------------------------------------------------- positive control for the perturbation guard (a deliberately leaky build must be flagged)
def leaky(cc):
    c2 = dict(cc); c2["vol60"] = cc["vol60"].shift(-1)
    return mod.weights(c2, p)
ok, worst = R.check_perturbation(leaky, c, dates=("2021-03-10",), verbose=True)
res["positive_control_leaky_build_flagged"] = (not ok)
res["positive_control_worst"] = worst
json.dump(res, open(OUT, "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
print("done", round(time.time() - t0), "s")
