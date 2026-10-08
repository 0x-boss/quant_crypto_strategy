import sys
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import r8_common as R
import r8_f_voltarget as mod
c = R.load()
p = next(q for q in mod.CONFIGS if q["name"] == "own10_q65")
f = lambda cc: R.run(cc, mod.apply(cc, R.base_weights(cc), p))
for T0 in ("2019-06-28", "2021-03-10", "2025-03-20"):
    R.check_truncation(f, c, T0=T0)
