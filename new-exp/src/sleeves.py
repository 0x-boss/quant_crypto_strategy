"""Round 5 - build the daily return streams ("sleeves") of the strategies that survived round 1-4 (>= Sharpe 0.6 on both DEV and
HOLDOUT or economically distinct) and save them for blending / overlay tests.   python src/sleeves.py
NOTE (post-hoc disclosure): sleeve choice looks at the round 1-4 tables, i.e. it is conditioned on results. Blend WEIGHTS are
fitted on DEV only and judged on HOLDOUT.
"""
import numpy as np
import pandas as pd

import lib
import signals

P = lib.load_panels()
M3 = lib.pit_universe(P, 300)
M10 = lib.pit_universe(P, 1000)
mkt = P["C"]["SPY"].pct_change()
F = signals.features(P, mkt=mkt)
S = signals.build_signal_menu(F)
C, O = P["C"], P["O"]
atr = F["atr14"]
spy = C["SPY"]
cols = list(M10.columns[M10.any()])


def topk_weights(sig, M, k, h):
    s = sig[cols].where(M[cols])
    rank = s.rank(axis=1, ascending=False, method="first")
    return (((rank <= k) & s.notna()).astype(float) / k).rolling(h, min_periods=1).mean()


out = {}
Oc = O[cols]
out["MOM_res12_1_k25"] = lib.weights_backtest(topk_weights(S["res_mom_12_1"], M3, 25, 5), Oc)
out["MOM_6_1_k10"] = lib.weights_backtest(topk_weights(S["mom_6_1"], M3, 10, 5), Oc)
out["HIVOL_idio_k10_top1000"] = lib.weights_backtest(topk_weights(S["idiovol_hi"], M10, 10, 5), Oc)
out["CLV_high_k10_h21"] = lib.weights_backtest(topk_weights(S["clv_high"], M3, 10, 21), Oc)
out["LOWBETA_k25"] = lib.weights_backtest(topk_weights(S["lowbeta"], M3, 25, 5), Oc)
out["HIGH52_k25_h21"] = lib.weights_backtest(topk_weights(S["high52"], M3, 25, 21), Oc)
# slot-engine sleeves
rk6 = F["mom_6_1"].where(M3).rank(axis=1, pct=True)
hi20 = P["H"].rolling(20).max().shift(1)
up200 = F["sma200"] > 0
gate_on = spy > spy.rolling(200).mean()
r, tr, ex = lib.slots_backtest(P, ((C > hi20) & (F["rvol"] >= 1.5) & up200 & M3), F["mom_6_1"], K=10, max_hold=60, stop_atr=3, trail_atr=3, atr=atr)
out["DONCHIAN_K10"] = r
r, tr, ex = lib.slots_backtest(P, ((rk6 >= 0.9) & (F["ret3"] <= -2 * atr) & M3), -F["ret3"] / atr, K=10, max_hold=5, atr=atr, gate=gate_on)
out["LEADERDIP_K10_gated"] = r
r, tr, ex = lib.slots_backtest(P, ((rk6 >= 0.9) & (C > hi20) & (C > C.rolling(50).mean()) & M3), F["mom_6_1"], K=10, max_hold=80, stop_atr=4, trail_atr=4, atr=atr)
out["LEADERS_TRAIL4_K10"] = r
# benchmark
ew = M3[cols].astype(float).div(M3[cols].sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
out["EW_top300"] = lib.weights_backtest(ew, Oc)
w = pd.DataFrame(0.0, index=O.index, columns=["SPY"]); w["SPY"] = 1.0
out["SPY_bh"] = lib.weights_backtest(w, O[["SPY"]], fee=0.0)

df = pd.DataFrame(out).loc["2012":]
df.to_parquet(f"{lib.RES}/sleeves.parquet")
for k in df.columns:
    lib.print_report(df[k], k)
d = df.loc[lib.DEV[0]:lib.DEV[1]].corr()
print("\nDEV correlation of daily returns:\n", d.round(2).to_string())
