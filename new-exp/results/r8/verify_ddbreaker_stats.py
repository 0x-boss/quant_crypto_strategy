import sys, json
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import numpy as np, pandas as pd
import lib
import r8_common as R
import r8_f_ddbreaker as mod

c = R.load()
p = next(q for q in mod.CONFIGS if q["name"] == "w126_t10_c50_d10")
W0 = R.base_weights(c)
W1 = mod.apply(c, W0, p)

def sh(x):
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(252))

def dev(r):
    return r.loc[lib.DEV[0]:lib.DEV[1]]

out = {}
r0 = R.run(c, W0); r1 = R.run(c, W1)
s0 = R.evaluate(r0, "BASELINE"); s1 = R.evaluate(r1, "CANDIDATE")
out["base_dev_sh"], out["base_hold_sh"] = s0["dev"]["sharpe"], s0["hold"]["sharpe"]
out["cand_dev_sh"], out["cand_hold_sh"] = s1["dev"]["sharpe"], s1["hold"]["sharpe"]
for k in ("dev", "hold", "all"):
    for n, s in (("base", s0), ("cand", s1)):
        out[f"{n}_{k}"] = {m: float(s[k][m]) for m in ("sharpe", "med_m", "mean_m", "maxdd", "worst_m", "cagr", "vol")}
out["turnover_base"] = float(r0.attrs["turnover"].loc["2012":].mean()); out["turnover_cand"] = float(r1.attrs["turnover"].loc["2012":].mean())
out["avg_gross_cand"] = r1.attrs["avg_gross"]

# costs and lag
for m in (2, 4):
    a = R.run(c, W0, fee=m * R.FEE); b = R.run(c, W1, fee=m * R.FEE)
    out[f"cost{m}x"] = dict(base_dev=sh(dev(a)), cand_dev=sh(dev(b)), base_hold=sh(a.loc[lib.HOLD[0]:lib.HOLD[1]]), cand_hold=sh(b.loc[lib.HOLD[0]:lib.HOLD[1]]))
a = R.run(c, W0, lag=1); b = R.run(c, W1, lag=1)
out["lag1"] = dict(base_dev=sh(dev(a)), cand_dev=sh(dev(b)), base_hold=sh(a.loc[lib.HOLD[0]:lib.HOLD[1]]), cand_hold=sh(b.loc[lib.HOLD[0]:lib.HOLD[1]]))
# zero cost
a = R.run(c, W0, fee=0.0); b = R.run(c, W1, fee=0.0)
out["cost0"] = dict(base_dev=sh(dev(a)), cand_dev=sh(dev(b)))

# exclude 2020, 2021
d0, d1 = dev(r0), dev(r1)
ex = ~d0.index.year.isin([2020, 2021])
out["ex2020_21"] = dict(base=sh(d0[ex]), cand=sh(d1[ex]))
out["only2020"] = dict(base=sh(d0[d0.index.year == 2020]), cand=sh(d1[d1.index.year == 2020]))
out["only2021"] = dict(base=sh(d0[d0.index.year == 2021]), cand=sh(d1[d1.index.year == 2021]))
# year by year
yy = {}
for y in range(2012, 2027):
    x0, x1 = r0[r0.index.year == y], r1[r1.index.year == y]
    yy[y] = dict(base_sh=sh(x0), cand_sh=sh(x1), base_ret=float((1 + x0).prod() - 1), cand_ret=float((1 + x1).prod() - 1))
out["by_year"] = yy
# also excluding 2020/21 for 2x cost
a = R.run(c, W0, fee=2 * R.FEE); b = R.run(c, W1, fee=2 * R.FEE)
da, db = dev(a), dev(b)
out["ex2020_21_cost2x"] = dict(base=sh(da[~da.index.year.isin([2020, 2021])]), cand=sh(db[~db.index.year.isin([2020, 2021])]))

# episodes
eps = {"2021-02-12:2021-05-10": ("2021-02-12", "2021-05-10"), "2025-02:2025-04": ("2025-02-01", "2025-04-30"), "2020-02:2020-03": ("2020-02-01", "2020-03-31"),
       "2026-06:2026-07": ("2026-06-01", "2026-07-31"), "2023-08:2023-10": ("2023-08-01", "2023-10-31"), "2018-Q4": ("2018-10-01", "2018-12-31")}
ep = {}
for k, (a_, b_) in eps.items():
    ep[k] = dict(base=float((1 + r0.loc[a_:b_]).prod() - 1), cand=float((1 + r1.loc[a_:b_]).prod() - 1))
out["episodes"] = ep
json.dump(out, open("/home/user/quant_crypto_strategy/new-exp/results/r8/verify_ddbreaker_out.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k not in ("by_year",)}, indent=1, default=float))
for y, v in yy.items():
    print(y, {k: round(x, 3) for k, x in v.items()})
