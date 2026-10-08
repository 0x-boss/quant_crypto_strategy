import sys, time, json
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import numpy as np, pandas as pd
import lib
import r8_common as R
import r8_f_ddbreaker as mod

c = R.load(["O", "C", "M300", "mom_6_1"])
W0 = R.base_weights(c)
expo = pd.read_csv("/home/user/quant_crypto_strategy/new-exp/results/r8/verify_ddbreaker_expo.csv", index_col=0, parse_dates=True).iloc[:, 0]
expo = expo.reindex(W0.index).fillna(1.0)
def sh(x):
    x = x.dropna(); return float(x.mean() / x.std() * np.sqrt(252))
def stats(r):
    d = r.loc[lib.DEV[0]:lib.DEV[1]]
    eq = (1 + d).cumprod(); dd = float((eq / eq.cummax() - 1).min())
    return sh(d), sh(d[~d.index.year.isin([2020, 2021])]), dd, sh(r.loc[lib.HOLD[0]:lib.HOLD[1]])
t0 = time.time()
real = stats(R.run(c, W0.mul(expo, axis=0)))
base = stats(R.run(c, W0))
print("real", real, "base", base, time.time() - t0)
rng = np.random.default_rng(7)
res = []
n = len(expo)
for i in range(40):
    k = int(rng.integers(60, n - 60))
    e2 = pd.Series(np.roll(expo.values, k), index=expo.index)
    res.append(stats(R.run(c, W0.mul(e2, axis=0))) + (k,))
res = np.array(res)
print("circular-shifted exposure placebo (n=40): DEV Sh mean %.3f sd %.3f; frac >= real: %.2f" % (res[:, 0].mean(), res[:, 0].std(), (res[:, 0] >= real[0]).mean()))
print("  DEV ex2020/21 Sh mean %.3f; frac >= real(%.3f): %.2f" % (res[:, 1].mean(), real[1], (res[:, 1] >= real[1]).mean()))
print("  DEV maxDD mean %.3f ; frac better than real(%.3f): %.2f" % (res[:, 2].mean(), real[2], (res[:, 2] >= real[2]).mean()))
print("  HOLD Sh mean %.3f; frac >= real(%.3f): %.2f" % (res[:, 3].mean(), real[3], (res[:, 3] >= real[3]).mean()))
json.dump(dict(real=real, base=base, placebo=res.tolist()), open("/home/user/quant_crypto_strategy/new-exp/results/r8/verify_ddbreaker_placebo.json", "w"))
print(time.time() - t0)
