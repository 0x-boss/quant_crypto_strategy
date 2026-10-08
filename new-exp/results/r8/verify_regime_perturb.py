import sys, json
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import numpy as np, pandas as pd
import r8_common as R
import r8_f_regime as mod

c = R.load()
p = next(q for q in mod.CONFIGS if q["name"] == "ph_disp_half80_x75")
print("params", p)
build = lambda cc: mod.apply(cc, R.base_weights(cc), p)

W = build(c)
e = mod.exposure(c, R.base_weights(c), p)
dates = ("2021-03-10", "2020-03-12", "2025-03-20", "2026-06-25", "2018-11-15", "2015-08-24", "2019-08-05", "2024-08-05")
for d in dates:
    print(d, "exposure on d0:", float(e.loc[d]), " #half days up to d0 (>=2012):", int((e.loc["2012":d] == 0.5).sum()))
ok, worst = R.check_perturbation(build, c, dates=dates, tol=1e-9, verbose=True)
print("REAL: ok", ok, "worst", worst)

# sensitivity sanity check (mutant): full-sample percentile of dispersion -> the guard MUST fire
def build_mut(cc):
    Wb = R.base_weights(cc)
    x = mod.disp21(cc)
    pct = x.rank(pct=True)          # FULL SAMPLE rank = look-ahead
    ee, _ = mod.stepwise(pct, 0.80, 0.75)
    return Wb.mul(ee.reindex(Wb.index).fillna(1.0), axis=0)
okm, wm = R.check_perturbation(build_mut, c, dates=dates[:3], tol=1e-9, verbose=False)
print("MUTANT (full-sample pct) flagged as look-ahead:", not okm, "worst", wm)

# truncation invariance of the return series
ok1, d1 = R.check_truncation(lambda cc: R.run(cc, build(cc)), c, T0="2021-03-10")
ok2, d2 = R.check_truncation(lambda cc: R.run(cc, build(cc)), c, T0="2025-03-20")
json.dump(dict(perturb_ok=bool(ok), perturb_worst=worst, mutant_flagged=bool(not okm), mutant_worst=wm, trunc=[[bool(ok1), d1], [bool(ok2), d2]]),
          open("/home/user/quant_crypto_strategy/new-exp/results/r8/verify_regime_perturb_out.json", "w"))
