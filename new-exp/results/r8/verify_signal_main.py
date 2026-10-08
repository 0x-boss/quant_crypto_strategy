import sys, json, time
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import numpy as np, pandas as pd
import r8_common as R
import lib
import r8_f_signal as mod     # used ONLY for the comparison target; the re-implementation below is written from the text description

OUT = {}
p = next(q for q in mod.CONFIGS if q["name"] == "m61_h52")
c = R.load()
idx = c["C"].index
cols = c["C"].columns

# ------------------------------------------------------------------ (3) independent re-implementation (numpy, from the description)
def reimpl(mom, h52, M, k=10, h=5, thr=0.8, ge=True):
    mom = mom.astype(np.float64); h52 = h52.astype(np.float64)
    okh = (h52 >= thr) if ge else (h52 > thr)
    elig = M & np.isfinite(mom) & np.isfinite(h52) & okh
    T, N = mom.shape
    S = np.zeros((T, N))
    nel = np.zeros(T, dtype=int)
    for t in range(T):
        ii = np.flatnonzero(elig[t])
        nel[t] = ii.size
        if ii.size == 0:
            continue
        o = ii[np.argsort(-mom[t, ii], kind="stable")]
        S[t, o[:k]] = 1.0 / k
    cs = np.cumsum(S, axis=0)
    W = np.empty_like(S)
    for t in range(T):
        lo = max(0, t - h + 1)
        prev = cs[lo - 1] if lo > 0 else 0.0
        W[t] = (cs[t] - prev) / (t - lo + 1)
    return W, S, nel

W_mod = mod.weights(c, p).reindex(index=idx, columns=cols)
M = c["M300"].values
W_a, S_a, nel = reimpl(c["mom_6_1"].values, c["high52"].values, M)
sel = idx >= "2012-01-01"
d_all = np.abs(W_a - W_mod.fillna(0).values)
OUT["reimpl_cached_features_max_diff_ge2012"] = float(d_all[sel].max())
OUT["reimpl_cached_features_max_diff_all"] = float(d_all.max())
print("(3a) reimpl using cached mom_6_1/high52 vs module: max|dW| >=2012 =", d_all[sel].max(), " all dates =", d_all.max(), flush=True)

# strict '>' variant (module uses > 0.8; description says >=)
W_gt, _, _ = reimpl(c["mom_6_1"].values, c["high52"].values, M, ge=False)
OUT["reimpl_strict_gt_max_diff"] = float(np.abs(W_gt - W_mod.fillna(0).values)[sel].max())
print("(3a') strict '>' variant diff:", OUT["reimpl_strict_gt_max_diff"])

# (3b) features recomputed from the adjusted close C in float64 (checks the cached features against their textual definition)
C64 = c["C"].astype(np.float64)
mom_b = (C64.shift(21) / C64.shift(126) - 1).values
h52_b = (C64 / C64.rolling(252, min_periods=200).max()).values
W_b, S_b, _ = reimpl(mom_b, h52_b, M)
d_b = np.abs(W_b - W_mod.fillna(0).values)
OUT["reimpl_recomputed_features_max_diff_ge2012"] = float(d_b[sel].max())
OUT["reimpl_recomputed_features_days_with_diff"] = int((d_b[sel].max(axis=1) > 1e-9).sum())
print("(3b) reimpl with features recomputed from C (float64): max|dW| >=2012 =", d_b[sel].max(), " days with diff:", OUT["reimpl_recomputed_features_days_with_diff"], "of", int(sel.sum()), flush=True)
# cached feature vs recomputed feature
fm = np.nanmax(np.abs(c["mom_6_1"].values.astype(np.float64) - mom_b) / (1 + np.abs(mom_b)))
fh = np.nanmax(np.abs(c["high52"].values.astype(np.float64) - h52_b))
print("   cached-vs-recomputed feature max abs rel diff: mom_6_1", fm, " high52", fh)
OUT["feat_maxdiff_mom61"] = float(fm); OUT["feat_maxdiff_high52"] = float(fh)

# sanity: module baseline config == R.base_weights
pb = next(q for q in mod.CONFIGS if q["name"] == "base_mom61")
Wb_mod = mod.weights(c, pb).reindex(index=idx, columns=cols).fillna(0).values
Wb_R = R.base_weights(c).reindex(index=idx, columns=cols).fillna(0).values
OUT["base_mom61_vs_R_base_weights"] = float(np.abs(Wb_mod - Wb_R)[sel].max())
print("module base_mom61 vs R.base_weights:", OUT["base_mom61_vs_R_base_weights"])

# how often does the filter bind?
Sb, _, _ = None, None, None
mom = c["mom_6_1"].where(c["M300"])
rk = mom.rank(axis=1, ascending=False, method="first")
Sbase = ((rk <= 10) & mom.notna()).values
Scand = S_a > 0
diffnames = (Sbase & ~Scand).sum(axis=1)
sd = sel & (np.arange(len(idx)) > 300)
print("filter binding: share of days (>=2012) where >=1 of the baseline top-10 is replaced:", float((diffnames[sel] > 0).mean()),
      " avg names replaced/day:", float(diffnames[sel].mean()), " days with <10 eligible:", int((nel[sel] < 10).sum()))
OUT["filter_bind_share_days"] = float((diffnames[sel] > 0).mean()); OUT["avg_names_replaced"] = float(diffnames[sel].mean())
OUT["days_lt10_eligible"] = int((nel[sel] < 10).sum())
# exact ties at the threshold among selected
hh = c["high52"].values
print("float32 high52 == 0.8 exactly among M300 & finite: ", int(((hh == np.float32(0.8)) & M).sum()))

# ------------------------------------------------------------------ (4) performance
def sharpe(x):
    x = x.dropna(); return float(x.mean() / x.std() * np.sqrt(252))
W_c = W_mod.fillna(0.0)
W_0 = R.base_weights(c)
res = {}
for nm, W in (("cand", W_c), ("base", W_0)):
    for tag, kw in (("1x", dict()), ("2x", dict(fee=2 * R.FEE)), ("4x", dict(fee=4 * R.FEE)), ("lag1", dict(lag=1))):
        t0 = time.time()
        r = R.run(c, W, **kw)
        st = R.evaluate(r, f"{nm}-{tag}", show=(tag == "1x"))
        res[(nm, tag)] = (r, st)
        print(f"{nm:5s} {tag:5s} DEV Sh {st['dev']['sharpe']:.3f} HOLD Sh {st['hold']['sharpe']:.3f} DEV maxDD {st['dev']['maxdd']:.3f} HOLD maxDD {st['hold']['maxdd']:.3f} "
              f"DEV med {st['dev']['med_m']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        OUT[f"{nm}_{tag}"] = dict(dev_sharpe=st["dev"]["sharpe"], hold_sharpe=st["hold"]["sharpe"], dev_maxdd=st["dev"]["maxdd"], hold_maxdd=st["hold"]["maxdd"],
                                  dev_med=st["dev"]["med_m"], turn=float(res[(nm, tag)][0].attrs["turnover"].loc["2012":].mean()))

# ex-2020/2021 on DEV
for nm in ("cand", "base"):
    r = res[(nm, "1x")][0].loc[lib.DEV[0]:lib.DEV[1]]
    ex = r[~r.index.year.isin([2020, 2021])]
    ex21 = r[r.index.year != 2021]
    ex20 = r[r.index.year != 2020]
    OUT[f"{nm}_dev_ex2020_2021"] = sharpe(ex); OUT[f"{nm}_dev_ex2021"] = sharpe(ex21); OUT[f"{nm}_dev_ex2020"] = sharpe(ex20)
    eq = (1 + ex).cumprod(); OUT[f"{nm}_dev_ex2020_2021_maxdd"] = float((eq / eq.cummax() - 1).min())
    print(f"{nm}: DEV Sharpe all {sharpe(r):.3f} | ex2020&2021 {sharpe(ex):.3f} (maxDD {OUT[f'{nm}_dev_ex2020_2021_maxdd']:.3f}) | ex2021 {sharpe(ex21):.3f} | ex2020 {sharpe(ex20):.3f}")
# 2x cost, ex 2020/21
for nm in ("cand", "base"):
    r = res[(nm, "2x")][0].loc[lib.DEV[0]:lib.DEV[1]]
    OUT[f"{nm}_dev_ex2020_2021_2x"] = sharpe(r[~r.index.year.isin([2020, 2021])])
print("2x cost ex2020/21:", OUT["cand_dev_ex2020_2021_2x"], OUT["base_dev_ex2020_2021_2x"])

# per-year table
rc, rb = res[("cand", "1x")][0], res[("base", "1x")][0]
yr = pd.DataFrame({"cand_ret": (1 + rc).groupby(rc.index.year).prod() - 1, "base_ret": (1 + rb).groupby(rb.index.year).prod() - 1,
                   "cand_sh": rc.groupby(rc.index.year).apply(sharpe), "base_sh": rb.groupby(rb.index.year).apply(sharpe)})
yr["d_ret"] = yr.cand_ret - yr.base_ret
pd.set_option("display.width", 200)
print(yr.round(3).to_string())
OUT["years_cand_beats_base_dev"] = int((yr.loc[2012:2021, "d_ret"] > 0).sum()); OUT["years_cand_beats_base_hold"] = int((yr.loc[2022:, "d_ret"] > 0).sum())
OUT["year_table"] = yr.round(4).to_dict(orient="index")

# paired daily-difference stats on DEV (block bootstrap, 20d blocks)
d = (rc - rb).loc[lib.DEV[0]:lib.DEV[1]].dropna().values
rng = np.random.default_rng(1); B = 20; n = len(d); nb = n // B
means = []
for _ in range(4000):
    st = rng.integers(0, n - B, nb)
    means.append(np.mean(np.concatenate([d[s:s + B] for s in st])))
means = np.array(means)
OUT["dev_daily_diff_mean_ann"] = float(d.mean() * 252); OUT["dev_diff_boot_p_le0"] = float((means <= 0).mean())
# Sharpe-diff bootstrap (paired, block)
a = rc.loc[lib.DEV[0]:lib.DEV[1]].dropna().values; b = rb.loc[lib.DEV[0]:lib.DEV[1]].dropna().values
sd_ = []
for _ in range(2000):
    st = rng.integers(0, n - B, nb)
    ii = np.concatenate([np.arange(s, s + B) for s in st])
    sd_.append(a[ii].mean() / a[ii].std() - b[ii].mean() / b[ii].std())
sd_ = np.array(sd_) * np.sqrt(252)
OUT["dev_sharpe_diff_boot_ci"] = [float(np.percentile(sd_, 5)), float(np.percentile(sd_, 50)), float(np.percentile(sd_, 95))]
OUT["dev_sharpe_diff_boot_p_le0"] = float((sd_ <= 0).mean())
print("DEV paired block-bootstrap: ann mean daily diff", OUT["dev_daily_diff_mean_ann"], "P(<=0)", OUT["dev_diff_boot_p_le0"],
      "| Sharpe diff 5/50/95%:", OUT["dev_sharpe_diff_boot_ci"], "P(<=0)", OUT["dev_sharpe_diff_boot_p_le0"])

# DD episodes: candidate vs baseline within the 5 named windows
wins = {"2021-02-12:2021-05-10": ("2021-02-12", "2021-05-10"), "2025-02:2025-04": ("2025-02-01", "2025-04-30"), "2020-02:2020-03": ("2020-02-01", "2020-03-31"),
        "2026-06:2026-07": ("2026-06-01", "2026-07-31"), "2023-08:2023-10": ("2023-08-01", "2023-10-31"), "2018-Q4": ("2018-10-01", "2018-12-31")}
for k, (a_, b_) in wins.items():
    print(f"  episode {k:24s} cand {float((1+rc.loc[a_:b_]).prod()-1):7.3f}  base {float((1+rb.loc[a_:b_]).prod()-1):7.3f}")
    OUT[f"ep_{k}"] = [float((1 + rc.loc[a_:b_]).prod() - 1), float((1 + rb.loc[a_:b_]).prod() - 1)]

json.dump(OUT, open("/home/user/quant_crypto_strategy/new-exp/results/r8/verify_signal_main_out.json", "w"), indent=1, default=float)

# ------------------------------------------------------------------ power check of the perturbation guard: a planted 1-day look-ahead must be flagged
def mutant(cc):
    cc2 = dict(cc); cc2["high52"] = cc["high52"].shift(-1)
    return mod.weights(cc2, p)
print("\nplanted look-ahead (high52.shift(-1)) should be DETECTED:")
ok, w = R.check_perturbation(mutant, c, dates=("2021-03-10", "2018-11-15"))
OUT["mutant_detected"] = (not ok); OUT["mutant_worst"] = w
json.dump(OUT, open("/home/user/quant_crypto_strategy/new-exp/results/r8/verify_signal_main_out.json", "w"), indent=1, default=float)
