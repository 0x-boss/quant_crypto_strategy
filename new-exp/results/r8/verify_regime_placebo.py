import sys, json, time
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import numpy as np, pandas as pd
import lib
import r8_common as R
import r8_f_regime as mod

c = R.load()
Wb = R.base_weights(c)
ex = pd.read_csv("/home/user/quant_crypto_strategy/new-exp/results/r8/verify_regime_expo.csv", index_col=0, parse_dates=True).iloc[:, 0]
act = Wb.columns[(Wb.abs().sum() > 0).values]
c2 = dict(c); c2["O"] = c["O"][act]
Wa = Wb[act]

def sh(x):
    x = x.dropna(); return float(x.mean() / x.std() * np.sqrt(252))
D, H = lib.DEV, lib.HOLD
t0 = time.time()
r_real = R.run(c2, Wa.mul(ex, axis=0))
print("real cand DEV/HOLD Sharpe (restricted cols):", sh(r_real.loc[D[0]:D[1]]), sh(r_real.loc[H[0]:H[1]]), "time", time.time() - t0)
r_b = R.run(c2, Wa)
print("base:", sh(r_b.loc[D[0]:D[1]]), sh(r_b.loc[H[0]:H[1]]))
# constant scale placebo (same average gross as DEV candidate)
for k in (0.877, 0.75, 0.5):
    rr = R.run(c2, Wa * k)
    print(f"constant scale {k}: DEV {sh(rr.loc[D[0]:D[1]]):.3f} HOLD {sh(rr.loc[H[0]:H[1]]):.3f}")

# circular-shift placebo of the exposure series (keeps occupancy, spell lengths and switching frequency)
e = ex.loc["2011-01-01":]
vals = e.values
n = len(vals)
rng = np.random.default_rng(7)
res = []
for i in range(150):
    off = int(rng.integers(300, n - 300))
    sh_e = pd.Series(np.roll(vals, off), index=e.index).reindex(Wa.index).fillna(1.0)
    rr = R.run(c2, Wa.mul(sh_e, axis=0))
    d = rr.loc[D[0]:D[1]]
    res.append((sh(d), sh(rr.loc[H[0]:H[1]]), float((sh_e.loc[D[0]:D[1]] == 0.5).mean())))
res = np.array(res)
real_dev = sh(r_real.loc[D[0]:D[1]]); base_dev = sh(r_b.loc[D[0]:D[1]])
print("placebo DEV Sharpe: mean %.3f sd %.3f p95 %.3f max %.3f ; real %.3f ; P(placebo>=real) %.3f ; baseline %.3f" % (res[:, 0].mean(), res[:, 0].std(), np.percentile(res[:, 0], 95), res[:, 0].max(), real_dev, (res[:, 0] >= real_dev).mean(), base_dev))
print("placebo HOLD Sharpe mean %.3f p95 %.3f ; P(>=real hold %.3f) %.3f" % (res[:, 1].mean(), np.percentile(res[:, 1], 95), sh(r_real.loc[H[0]:H[1]]), (res[:, 1] >= sh(r_real.loc[H[0]:H[1]])).mean()))
print("placebo mean DEV half share %.3f (real %.3f)" % (res[:, 2].mean(), (ex.loc[D[0]:D[1]] == 0.5).mean()))
json.dump(dict(real_dev=real_dev, base_dev=base_dev, placebo_mean=float(res[:, 0].mean()), placebo_sd=float(res[:, 0].std()), placebo_p95=float(np.percentile(res[:, 0], 95)),
               p_ge=float((res[:, 0] >= real_dev).mean())), open("/home/user/quant_crypto_strategy/new-exp/results/r8/verify_regime_placebo_out.json", "w"))
