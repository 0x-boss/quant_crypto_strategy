import sys, time
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import numpy as np, pandas as pd
import r8_common as R
import r8_f_defensive as mod
t0 = time.time()
p = next(q for q in mod.CONFIGS if q["name"] == "trigF_usmv")
c = R.load()
# truncation test on the real config (returns series built only from truncated frames)
for T0 in ("2019-06-28", "2023-03-31"):
    R.check_truncation(lambda cc: R.run(cc, mod.apply(cc, R.base_weights(cc), p)), c, T0=T0)
# mutant: un-lagged own returns (uses r[t] which spans open t -> open t+1): the guard MUST flag it
orig = mod.sleeve_vol20
mod.sleeve_vol20 = lambda cc, W: R.run(cc, W).rolling(20).std() * mod.ANN
ok, worst = R.check_perturbation(lambda cc: mod.apply(cc, R.base_weights(cc), p), c, dates=("2021-03-10", "2018-11-15"), tol=1e-9)
print("MUTANT (un-lagged) guard verdict ok=", ok, "worst", worst, "(expected ok=False)")
mod.sleeve_vol20 = orig
print("time", round(time.time() - t0), "s")
