import sys, time, json
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import numpy as np, pandas as pd
import r8_common as R
import r8_f_voltarget as mod

t0 = time.time()
c = R.load()
p = next(q for q in mod.CONFIGS if q["name"] == "own10_q65")
print("params:", p, flush=True)
O = c["O"]; idx = O.index; T = len(idx)
out = {}

# ------------------------------------------------------------------ independent re-implementation (written from the text description)
# baseline: each day top-10 of mom_6_1 inside M300, equal weight 1/10, average of the last 5 daily books
mom = c["mom_6_1"].astype(np.float64).where(c["M300"]).values        # T x N
N = mom.shape[1]
sleeve = np.zeros((T, N))
for t in range(T):
    row = mom[t]
    ok = np.where(np.isfinite(row))[0]
    if len(ok) == 0:
        continue
    order = ok[np.argsort(-row[ok], kind="stable")][:10]
    sleeve[t, order] = 0.1
Wb = np.zeros_like(sleeve)
for t in range(T):
    lo = max(0, t - 4)
    Wb[t] = sleeve[lo:t + 1].mean(axis=0)          # min_periods=1 style
Wb_mod = R.base_weights(c).reindex(idx).fillna(0.0).values
print("baseline book: max|mine - R.base_weights| =", np.abs(Wb - Wb_mod).max(), flush=True)
out["base_weight_maxdiff"] = float(np.abs(Wb - Wb_mod).max())

# check the cached feature mom_6_1 == C.shift(21)/C.shift(126)-1  (no future info in the feature)
C64 = c["C"].astype(np.float64)
mom_re = (C64.shift(21) / C64.shift(126) - 1)
dm = (mom_re - c["mom_6_1"].astype(np.float64)).abs().loc["2012":]
print("feature mom_6_1 recompute: max abs diff (2012+) =", float(np.nanmax(dm.values)), " 99.99pct", float(np.nanpercentile(dm.values[np.isfinite(dm.values)], 99.99)), flush=True)
out["mom_feature_maxdiff"] = float(np.nanmax(dm.values))
del mom_re, dm, C64

held_cols = np.where(np.abs(Wb).sum(0) > 0)[0]
Oarr = O.values.astype(np.float64)

def my_backtest(Wt, fee, lag=0):
    """position at open d = target of day d-1-lag; return open d -> open d+1; fee*|w - drifted w| of NAV; (T-1) returns"""
    W = Wt[:, held_cols]
    pos = np.zeros_like(W)
    pos[1 + lag:] = W[:T - 1 - lag]
    assert (pos.sum(1) <= 1 + 1e-9).all()
    ratio = Oarr[1:, held_cols] / Oarr[:-1, held_cols]
    ratio = np.where(np.isfinite(ratio), ratio, 1.0)
    drift = np.zeros(W.shape[1])
    ret = np.zeros(T - 1)
    for d in range(T - 1):
        w = pos[d]
        turn = np.abs(w - drift).sum()
        grow = 1.0 - w.sum() + (w * ratio[d]).sum()
        ret[d] = (1.0 - fee * turn) * grow - 1.0
        drift = w * ratio[d] / grow
    return pd.Series(ret, index=idx[:-1])

r_un = my_backtest(Wb, R.FEE)
r_un_mod = R.run(c, R.base_weights(c))
print("unscaled book returns: max|mine - R.run| =", float((r_un - r_un_mod).abs().max()), flush=True)
out["unscaled_ret_maxdiff"] = float((r_un - r_un_mod).abs().max())

# vol estimate: annualised std (ddof=1) of the last 10 returns known at close t (= returns indexed t-10..t-1)
rk = r_un.reindex(idx).shift(1).values             # return known at close of t
sig = np.full(T, np.nan)
for t in range(10, T):
    win = rk[t - 9:t + 1]
    if np.isfinite(win).all():
        sig[t] = win.std(ddof=1) * np.sqrt(252)
STAR = 0.3012
s = np.where(np.isfinite(sig) & (sig > 0), np.minimum(1.0, STAR / np.where(sig > 0, sig, 1.0)), 1.0)
Wm = Wb * s[:, None]
sig_s = pd.Series(sig, index=idx)

Wmod = mod.apply(c, R.base_weights(c), p)
Wmod_a = Wmod.reindex(idx)
nan_rows = Wmod_a.isna().all(axis=1)
print("module rows that are all-NaN:", list(nan_rows[nan_rows].index.strftime("%Y-%m-%d")), flush=True)
Wmod_v = Wmod_a.fillna(0.0).values
d_all = np.abs(Wm - Wmod_v)
m12 = (idx >= "2012-01-01") & ~nan_rows.values
print("MAX |W_mine - W_module| all dates >= 2012 (excl. NaN final row):", float(d_all[m12].max()), flush=True)
print("MAX |W_mine - W_module| all rows (final NaN row incl.):", float(d_all.max()), flush=True)
out["reimpl_maxdiff_2012plus"] = float(d_all[m12].max())
out["reimpl_maxdiff_allrows"] = float(d_all.max())
out["s_mean_dev"] = float(pd.Series(s, index=idx).loc["2012":"2021"].mean())
out["s_mean_hold"] = float(pd.Series(s, index=idx).loc["2022":].mean())

# ------------------------------------------------------------------ star audit: is 0.3012 the DEV-only 65th pct of the unscaled own10 vol?
dev_sig = sig_s.loc["2012-01-01":"2021-12-31"]
print("DEV q65 of own10 sigma_hat (mine):", dev_sig.quantile(0.65), "  table 0.3012", flush=True)
print("HOLD q65:", sig_s.loc["2022":].quantile(0.65), " HOLD median:", sig_s.loc["2022":].median(), " DEV median:", dev_sig.median(), flush=True)
out["dev_q65_mine"] = float(dev_sig.quantile(0.65))
out["hold_sigma_median"] = float(sig_s.loc["2022":].median())

# ------------------------------------------------------------------ stats helpers
def sharpe(x):
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(252))
def devsl(r):
    return r.loc["2012-01-01":"2021-12-31"]
def ex_years(r, yrs):
    x = devsl(r)
    return x[~x.index.year.isin(yrs)]

res = {}
Wbase_mod = R.base_weights(c)
variants = {"base": Wbase_mod, "cand": Wmod}
for name, W in variants.items():
    for tag, kw in {"fee1": dict(), "fee2": dict(fee=2 * R.FEE), "fee4": dict(fee=4 * R.FEE), "lag1": dict(lag=1)}.items():
        r = R.run(c, W, **kw)
        st = R.evaluate(r, f"{name} {tag}", show=(tag == "fee1"))
        res[(name, tag)] = r
        out[f"{name}_{tag}"] = dict(dev_sh=st["dev"]["sharpe"], hold_sh=st["hold"]["sharpe"], dev_dd=st["dev"]["maxdd"], hold_dd=st["hold"]["maxdd"],
                                    dev_med=st["dev"]["med_m"], hold_med=st["hold"]["med_m"], dev_mean=st["dev"]["mean_m"], hold_mean=st["hold"]["mean_m"],
                                    dev_sh_ex2020_21=sharpe(ex_years(r, [2020, 2021])), dev_sh_ex2020=sharpe(ex_years(r, [2020])),
                                    dev_sh_ex2021=sharpe(ex_years(r, [2021])), turn=float(r.attrs["turnover"].loc["2012":].mean()),
                                    gross=r.attrs["avg_gross"])
        print(f"   {name} {tag}: DEV Sh {st['dev']['sharpe']:.4f} HOLD Sh {st['hold']['sharpe']:.4f} DEV ex2020-21 Sh {out[f'{name}_{tag}']['dev_sh_ex2020_21']:.4f}"
              f" | DEV DD {st['dev']['maxdd']:.3f} HOLD DD {st['hold']['maxdd']:.3f}", flush=True)

# my own backtest of my own weights vs R.run of module weights
for tag, kw in {"fee1": dict(fee=R.FEE), "fee2": dict(fee=2 * R.FEE), "fee4": dict(fee=4 * R.FEE), "lag1": dict(fee=R.FEE, lag=1)}.items():
    rm = my_backtest(Wm, **kw)
    d = float((rm - res[("cand", tag)]).abs().max())
    print(f"   independent backtest of independent weights vs R.run(module) [{tag}]: max|dr| = {d:.2e}  DEV Sh mine {sharpe(devsl(rm)):.4f}", flush=True)
    out[f"cand_{tag}_my_vs_R_ret_maxdiff"] = d

# per-year Sharpe table, candidate vs base
yr = pd.DataFrame({k: res[(k, "fee1")].groupby(res[(k, "fee1")].index.year).apply(lambda x: x.mean() / x.std() * np.sqrt(252)) for k in ("base", "cand")})
yrr = pd.DataFrame({k: res[(k, "fee1")].groupby(res[(k, "fee1")].index.year).apply(lambda x: (1 + x).prod() - 1) for k in ("base", "cand")})
print("\nper-year Sharpe:\n", yr.round(2).to_string()); print("per-year return:\n", yrr.round(3).to_string(), flush=True)
out["per_year_sharpe"] = {int(k): v for k, v in yr.round(3).to_dict("index").items()}
out["per_year_ret"] = {int(k): v for k, v in yrr.round(4).to_dict("index").items()}

# ------------------------------------------------------------------ adversarial extras
# (a) paired block bootstrap of DEV Sharpe difference, 21-day blocks
rb, rc = devsl(res[("base", "fee1")]).values, devsl(res[("cand", "fee1")]).values
n = len(rb); rng = np.random.default_rng(1); L = 21; nb = int(np.ceil(n / L)); diffs = []
for _ in range(3000):
    st_ = rng.integers(0, n, nb)
    ii = ((st_[:, None] + np.arange(L)[None, :]) % n).ravel()[:n]
    a, b = rb[ii], rc[ii]
    diffs.append(b.mean() / b.std() - a.mean() / a.std())
diffs = np.array(diffs) * np.sqrt(252)
print(f"paired block-bootstrap DEV Sharpe diff (cand-base): mean {diffs.mean():.3f} 5-95% [{np.percentile(diffs,5):.3f},{np.percentile(diffs,95):.3f}] P(>0)={np.mean(diffs>0):.3f}", flush=True)
out["boot_dev_sharpe_diff"] = dict(mean=float(diffs.mean()), p5=float(np.percentile(diffs, 5)), p95=float(np.percentile(diffs, 95)), p_gt0=float(np.mean(diffs > 0)))

# (b) causal variant: sigma* = expanding 65th pct of sigma_hat through t (min 504 obs, else s=1)
sg = sig_s.copy()
star_exp = sg.expanding(min_periods=504).quantile(0.65)
s_exp = (star_exp / sg).replace([np.inf, -np.inf], np.nan).clip(upper=1.0).fillna(1.0)
r_exp = my_backtest(Wb * s_exp.values[:, None], R.FEE)
st = R.evaluate(r_exp, "causal expanding-q65 own10", show=True)
print(f"   causal expanding q65: DEV Sh {st['dev']['sharpe']:.4f} HOLD Sh {st['hold']['sharpe']:.4f} DEV ex2020-21 {sharpe(ex_years(r_exp,[2020,2021])):.4f} DEV DD {st['dev']['maxdd']:.3f}", flush=True)
out["causal_exp65"] = dict(dev_sh=st["dev"]["sharpe"], hold_sh=st["hold"]["sharpe"], dev_dd=st["dev"]["maxdd"], hold_dd=st["hold"]["maxdd"], dev_sh_ex2020_21=sharpe(ex_years(r_exp, [2020, 2021])))
r_exp2 = my_backtest(Wb * s_exp.values[:, None], 2 * R.FEE)
out["causal_exp65"]["dev_sh_fee2"] = sharpe(devsl(r_exp2))

# (c) pseudo-OOS inside DEV: star = 65th pct of sigma_hat over 2012-2016 only, applied to 2017-2021 (and vice versa)
for a, b, lab in (("2012-01-01", "2016-12-31", "star from 2012-16 -> eval 2017-21"), ("2017-01-01", "2021-12-31", "star from 2017-21 -> eval 2012-16")):
    stq = float(sig_s.loc[a:b].quantile(0.65))
    sx = np.where(np.isfinite(sig) & (sig > 0), np.minimum(1.0, stq / np.where(sig > 0, sig, 1.0)), 1.0)
    rx = my_backtest(Wb * sx[:, None], R.FEE)
    ev = ("2017-01-01", "2021-12-31") if a.startswith("2012") else ("2012-01-01", "2016-12-31")
    sh_c, sh_b = sharpe(rx.loc[ev[0]:ev[1]]), sharpe(res[("base", "fee1")].loc[ev[0]:ev[1]])
    print(f"   {lab}: star={stq:.4f} cand Sh {sh_c:.3f} vs base {sh_b:.3f}", flush=True)
    out[f"split_{a[:4]}"] = dict(star=stq, cand=sh_c, base=sh_b)

# (d) plateau in the star level (fixed DEV percentiles), my backtest, DEV Sharpe / ex-2020-21 / 2x fee
pl = {}
for q in (50, 55, 60, 65, 70, 75, 80, 90):
    stq = float(dev_sig.quantile(q / 100))
    sx = np.where(np.isfinite(sig) & (sig > 0), np.minimum(1.0, stq / np.where(sig > 0, sig, 1.0)), 1.0)
    rx = my_backtest(Wb * sx[:, None], R.FEE); rx2 = my_backtest(Wb * sx[:, None], 2 * R.FEE)
    pl[q] = dict(star=stq, dev_sh=sharpe(devsl(rx)), dev_sh_ex2020_21=sharpe(ex_years(rx, [2020, 2021])), dev_sh_fee2=sharpe(devsl(rx2)),
                 hold_sh=sharpe(rx.loc["2022":]))
    print(f"   plateau q{q}: star {stq:.4f} DEV Sh {pl[q]['dev_sh']:.3f} ex2020-21 {pl[q]['dev_sh_ex2020_21']:.3f} fee2 {pl[q]['dev_sh_fee2']:.3f} HOLD {pl[q]['hold_sh']:.3f}", flush=True)
out["plateau"] = pl

json.dump(out, open("/home/user/quant_crypto_strategy/new-exp/results/r8/verify_voltarget_own10q65_out.json", "w"), indent=1, default=float)
print("secs", round(time.time() - t0))
