import sys, bisect, time, json
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import numpy as np, pandas as pd
import r8_common as R
import r8_f_voltarget as mod

t0 = time.time()
c = R.load(["O", "mom_6_1", "M300"])
p = next(q for q in mod.CONFIGS if q["name"] == "own20_exp80")
Wb = R.base_weights(c)
Wc = mod.apply(c, Wb, p)

def sh(x):
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(252))

def sh_ex(r, years=(2020, 2021), a="2012-01-01", b="2021-12-31"):
    x = r.loc[a:b]
    x = x[~x.index.year.isin(years)]
    return sh(x)

out = {}
runs = {}
for nm, W in (("base", Wb), ("cand", Wc)):
    for tag, kw in (("fee1", {}), ("fee2", dict(fee=2 * R.FEE)), ("fee4", dict(fee=4 * R.FEE)), ("lag1", dict(lag=1)), ("fee2lag1", dict(fee=2 * R.FEE, lag=1))):
        r = R.run(c, W, **kw)
        runs[(nm, tag)] = r
        st = R.evaluate(r, f"{nm} {tag}", show=(tag == "fee1"))
        out[f"{nm}_{tag}"] = dict(dev_sh=st["dev"]["sharpe"], hold_sh=st["hold"]["sharpe"], dev_dd=st["dev"]["maxdd"], hold_dd=st["hold"]["maxdd"],
                                  dev_med=st["dev"]["med_m"], hold_med=st["hold"]["med_m"],
                                  dev_sh_ex2021=sh_ex(r), dev_sh_2012_19=sh(r.loc["2012":"2019"]),
                                  turn=float(r.attrs["turnover"].loc["2012":].mean()), gross=r.attrs["avg_gross"])
    print(nm, "done", round(time.time() - t0), flush=True)

tab = pd.DataFrame(out).T
pd.set_option("display.width", 250)
print(tab.round(3).to_string())
print()
print("DEV Sharpe improvement (cand - base):")
for tag in ("fee1", "fee2", "fee4", "lag1", "fee2lag1"):
    print(f"  {tag:9s} base {out['base_'+tag]['dev_sh']:.3f} cand {out['cand_'+tag]['dev_sh']:.3f} diff {out['cand_'+tag]['dev_sh']-out['base_'+tag]['dev_sh']:+.3f}"
          f" | excl2020-21: base {out['base_'+tag]['dev_sh_ex2021']:.3f} cand {out['cand_'+tag]['dev_sh_ex2021']:.3f} diff {out['cand_'+tag]['dev_sh_ex2021']-out['base_'+tag]['dev_sh_ex2021']:+.3f}")

# sub-period Sharpe by year, candidate vs baseline (fee1)
rb, rc = runs[("base", "fee1")], runs[("cand", "fee1")]
yr = pd.DataFrame({y: dict(base=sh(rb.loc[str(y)]), cand=sh(rc.loc[str(y)]), base_ret=float((1 + rb.loc[str(y)]).prod() - 1), cand_ret=float((1 + rc.loc[str(y)]).prod() - 1))
                   for y in range(2012, 2027)}).T
print(yr.round(3).to_string())

# paired moving-block bootstrap of DEV Sharpe difference (block 21 days) and excl 2020/21
def boot(a, b, blk=21, n=4000, seed=1):
    a, b = a.values, b.values
    T = len(a)
    nb = int(np.ceil(T / blk))
    rng = np.random.default_rng(seed)
    d = np.empty(n)
    for i in range(n):
        st = rng.integers(0, T - blk + 1, nb)
        idx = (st[:, None] + np.arange(blk)[None]).ravel()[:T]
        x, y = a[idx], b[idx]
        d[i] = y.mean() / y.std() - x.mean() / x.std()
    d *= np.sqrt(252)
    return float(np.percentile(d, 2.5)), float(np.median(d)), float(np.percentile(d, 97.5)), float((d > 0).mean())

for tag in ("fee1", "fee2"):
    a = runs[("base", tag)].loc["2012":"2021"]
    b = runs[("cand", tag)].loc["2012":"2021"]
    print(f"DEV block-bootstrap dSharpe ({tag}) [2.5%, med, 97.5%, P>0]:", np.round(boot(a, b), 3))
    m = ~a.index.year.isin([2020, 2021])
    print(f"DEV excl2020-21 block-bootstrap dSharpe ({tag}):", np.round(boot(a[m], b[m]), 3))

# episode windows
wins = {"2021-02-12..05-10": ("2021-02-12", "2021-05-10"), "2025-02-12..04-04": ("2025-02-12", "2025-04-04"), "2020-02-19..03-18": ("2020-02-19", "2020-03-18"),
        "2026-06-02..07-28": ("2026-06-02", "2026-07-28"), "2023-08-01..10-30": ("2023-08-01", "2023-10-30"), "2018-10-01..12-24": ("2018-10-01", "2018-12-24")}
for k, (a, b) in wins.items():
    print(f"  window {k}: base {((1+rb.loc[a:b]).prod()-1)*100:6.1f}%  cand {((1+rc.loc[a:b]).prod()-1)*100:6.1f}%")

# quirk sensitivity: drop the 235 zero-vol (empty-book) sigma_hat values from the expanding sample
sig = R.lagged(R.run(c, Wb)).rolling(20).std() * np.sqrt(252)
sig_nz = sig.where(sig > 0)
star = sig_nz.expanding(min_periods=504).quantile(0.8)
s2 = (star / sig).replace([np.inf, -np.inf], np.nan).clip(upper=1.0).fillna(1.0)
Wq = Wb.mul(s2, axis=0)
for tag, kw in (("fee1", {}), ("fee2", dict(fee=2 * R.FEE))):
    r = R.run(c, Wq, **kw)
    st = R.evaluate(r, f"cand_dropzero {tag}", show=(tag == "fee1"))
    print(f"  cand_dropzero {tag}: DEV {st['dev']['sharpe']:.3f} HOLD {st['hold']['sharpe']:.3f} DEVdd {st['dev']['maxdd']:.3f} excl2020-21 {sh_ex(r):.3f}")

# constant-scale control (same mean exposure) -> is it timing?
sc = float(Wc.sum(axis=1).loc["2012":"2021"].mean() / Wb.sum(axis=1).loc["2012":"2021"].mean())
rk = R.run(c, Wb * sc)
print(f"constant scale {sc:.3f}: DEV Sharpe {sh(rk.loc['2012':'2021']):.3f} (cand {out['cand_fee1']['dev_sh']:.3f}, base {out['base_fee1']['dev_sh']:.3f})")

json.dump(out, open("/home/user/quant_crypto_strategy/new-exp/results/r8/verify_voltarget_out.json", "w"), indent=1, default=float)
print("secs", round(time.time() - t0))
