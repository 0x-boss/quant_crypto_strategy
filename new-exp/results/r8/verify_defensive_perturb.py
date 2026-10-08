import sys, time
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import numpy as np, pandas as pd
import r8_common as R
import r8_f_defensive as mod
t0 = time.time()
p = next(q for q in mod.CONFIGS if q["name"] == "trigF_usmv")
c = R.load()
build = lambda cc: mod.apply(cc, R.base_weights(cc), p)
dates = ("2021-03-10", "2020-03-12", "2025-03-20", "2026-06-25", "2018-11-15", "2012-11-07", "2015-08-24", "2024-08-05")
ok, worst = R.check_perturbation(build, c, dates=dates, tol=1e-9, verbose=True)
print("RESULT ok=", ok, "worst=", worst, "time", round(time.time() - t0), "s")
