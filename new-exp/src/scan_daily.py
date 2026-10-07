"""Round 1 - Family A: daily cross-sectional rank strategies on the point-in-time universe.

For every signal in the pre-registered menu, k in {10, 25}, hold H in {1, 5, 21} days (staggered sleeves), long-only
equal weight, fee 0.1 %/side.  Every configuration is logged to results/trials.csv (also the bad ones).

  python -I new-exp/src/scan_daily.py [top_n]
"""
import sys
import time

import numpy as np
import pandas as pd

import lib
import signals

top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
P = lib.load_panels()
M = lib.pit_universe(P, top_n=top_n)
F = signals.features(P)
S = signals.build_signal_menu(F)
O = P["O"]
cols = list(M.columns[M.any()])
print(f"universe top_n={top_n}: {len(cols)} tickers ever selected; avg members {M.sum(axis=1).loc['2013':].mean():.0f}", flush=True)
Mc = M[cols]
Oc = O[cols]


def topk_weights(sig, k, h):
    s = sig[cols].where(Mc)
    rank = s.rank(axis=1, ascending=False, method="first")
    sleeve = ((rank <= k) & s.notna()).astype(float) / k
    # only trade when enough names are available
    return sleeve.rolling(h, min_periods=1).mean()


rows = []
t0 = time.time()
for name, sig in S.items():
    for k in (10, 25):
        for h in (1, 5, 21):
            W = topk_weights(sig, k, h)
            r = lib.weights_backtest(W, Oc)
            lib.log_trial("A_daily_rank", name, dict(top_n=top_n, k=k, h=h), r)
            d, o = lib.perf(lib.split(r, "dev")), lib.perf(lib.split(r, "hold"))
            rows.append(dict(signal=name, k=k, h=h, dev_sh=d["sharpe"], dev_med=d["med_m"], dev_mean=d["mean_m"],
                             hold_sh=o["sharpe"], hold_med=o["med_m"], hold_mean=o["mean_m"],
                             turn=r.attrs["turnover"].mean()))
    print(name, f"{time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{lib.RES}/scan_daily_top{top_n}.csv", index=False)
pd.set_option("display.width", 200)
print(df.sort_values("dev_sh", ascending=False).head(25).round(3).to_string())
print("\nBy holdout Sharpe:")
print(df.sort_values("hold_sh", ascending=False).head(15).round(3).to_string())

# benchmarks: equal-weight universe (daily rebalanced) and SPY buy & hold
ew = (Mc.astype(float).div(Mc.sum(axis=1).replace(0, np.nan), axis=0)).fillna(0.0)
lib.print_report(lib.weights_backtest(ew, Oc), f"EW top{top_n} daily rebalanced")
if "SPY" in P["O"]:
    w = pd.DataFrame(0.0, index=O.index, columns=["SPY"])
    w["SPY"] = 1.0
    lib.print_report(lib.weights_backtest(w, O[["SPY"]], fee=0.0), "SPY buy&hold (no fee)")
