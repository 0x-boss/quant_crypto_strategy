import sys, json, time
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import numpy as np, pandas as pd
import r8_common as R
import r8_f_signal as mod

p = next(q for q in mod.CONFIGS if q["name"] == "m61_h52")
print("params:", p)
c = R.load()
build = lambda cc: mod.weights(cc, p)

dates = ("2021-03-10", "2020-03-12", "2025-03-20", "2026-06-25", "2018-11-15", "2012-09-12", "2015-08-26", "2023-10-18")
t0 = time.time()
ok, worst = R.check_perturbation(build, c, dates=dates)
print("perturbation:", ok, worst, f"{time.time()-t0:.0f}s")

# truncation test on the weights themselves (stronger / complementary): weights on data cut at T0 must equal full-data weights for rows <= T0
W = build(c)
res = {}
for T0 in ["2013-02-14", "2016-07-08", "2019-06-28", "2020-03-12", "2021-03-10", "2024-05-02", "2026-06-25"]:
    Wc = build(R.truncate(c, T0))
    a = W.loc[:T0]; b = Wc.reindex(a.index)
    d = float((a.fillna(0) - b.fillna(0)).abs().to_numpy().max())
    res[T0] = d
    print("truncation", T0, "max|dW| over ALL rows <= T0 (incl. last):", d, flush=True)
json.dump(dict(perturb_ok=bool(ok), perturb_worst=worst, trunc=res), open("/home/user/quant_crypto_strategy/new-exp/results/r8/verify_signal_perturb_out.json", "w"), indent=1)
