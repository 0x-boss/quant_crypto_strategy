import sys, time
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import r8_common as R
import r8_f_ddbreaker as mod
c = R.load()
p = next(q for q in mod.CONFIGS if q["name"] == "w126_t10_c50_d10")
print(p)
dates = ("2021-03-10", "2020-03-12", "2025-03-20", "2026-06-25", "2018-11-15", "2012-08-09", "2015-09-02", "2023-10-05")
t0 = time.time()
ok, worst = R.check_perturbation(lambda cc: mod.apply(cc, R.base_weights(cc), p), c, dates=dates, tol=1e-9)
print("RESULT", ok, worst, time.time() - t0)
# non-trivial check: weights differ between exposures, i.e. cut is actually active in the pre-d0 windows
W = mod.apply(c, R.base_weights(c), p)
import numpy as np
for d0 in dates:
    pass
