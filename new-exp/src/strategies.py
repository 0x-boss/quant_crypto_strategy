"""Strategy builders used by the final validation (same definitions as rounds 1-7, parameterised for stress tests)."""
import numpy as np
import pandas as pd

import lib


def z(df, M):
    d = df.where(M)
    return d.sub(d.mean(axis=1), axis=0).div(d.std(axis=1), axis=0).clip(-3, 3)


def topk(sig, M, k, h, cols):
    s = sig[cols].where(M[cols])
    rank = s.rank(axis=1, ascending=False, method="first")
    return (((rank <= k) & s.notna()).astype(float) / k).rolling(h, min_periods=1).mean()


def run_w(W, P, cols, fee=lib.FEE, lag=0):
    return lib.weights_backtest(W, P["O"][cols], fee=fee, lag=lag)


def build_all(P, F, S, fee=lib.FEE, lag=0, reselect="M", top_small=300, top_big=1000):
    """Return dict name -> daily return Series for the finalists' building blocks."""
    M3 = lib.pit_universe(P, top_small, reselect=reselect)
    M10 = lib.pit_universe(P, top_big, reselect=reselect)
    cols = list(M10.columns[M10.any()])
    C = P["C"]
    atr = F["atr14"]
    spy = C["SPY"]
    gate_on = (spy > spy.rolling(200).mean())
    out = {}
    out["MOM_res12_1_k25"] = run_w(topk(S["res_mom_12_1"], M3, 25, 5, cols), P, cols, fee, lag)
    out["MOM_6_1_k10"] = run_w(topk(S["mom_6_1"], M3, 10, 5, cols), P, cols, fee, lag)
    out["HIVOL_idio_k10_top1000"] = run_w(topk(S["idiovol_hi"], M10, 10, 5, cols), P, cols, fee, lag)
    out["CLV_high_k10_h21"] = run_w(topk(S["clv_high"], M3, 10, 21, cols), P, cols, fee, lag)
    out["LOWBETA_k25"] = run_w(topk(S["lowbeta"], M3, 25, 5, cols), P, cols, fee, lag)
    out["HIGH52_k25_h21"] = run_w(topk(S["high52"], M3, 25, 21, cols), P, cols, fee, lag)
    rk6 = F["mom_6_1"].where(M3).rank(axis=1, pct=True)
    hi20 = P["H"].rolling(20).max().shift(1)
    up200 = F["sma200"] > 0

    def slots(entry, prio, **kw):
        e = entry.fillna(False) & M3
        if lag:
            e = e.shift(lag).fillna(False)
        r, tr, ex = lib.slots_backtest(P, e, prio, atr=atr, fee=fee, **kw)
        return r

    out["DONCHIAN_K10"] = slots((C > hi20) & (F["rvol"] >= 1.5) & up200, F["mom_6_1"], K=10, max_hold=60, stop_atr=3, trail_atr=3)
    out["LEADERDIP_K10_gated"] = slots((rk6 >= 0.9) & (F["ret3"] <= -2 * atr), -F["ret3"] / atr, K=10, max_hold=5, gate=gate_on)
    out["LEADERS_TRAIL4_K10"] = slots((rk6 >= 0.9) & (C > hi20) & (C > C.rolling(50).mean()), F["mom_6_1"], K=10, max_hold=80, stop_atr=4, trail_atr=4)
    # composite C2 (momentum + residual reversal + 52-week-high), top-1000, k=25, h=21
    zz = {n: z(S[n][cols], M10[cols]) for n in ("res_mom_12_1", "rev_res5", "high52")}
    out["COMP_C2_k25_h21_top1000"] = run_w(topk(zz["res_mom_12_1"] + zz["rev_res5"] + zz["high52"], M10, 25, 21, cols), P, cols, fee, lag)
    return out


BLEND_MEMBERS = ["MOM_res12_1_k25", "MOM_6_1_k10", "HIVOL_idio_k10_top1000", "CLV_high_k10_h21", "LOWBETA_k25", "HIGH52_k25_h21",
                 "DONCHIAN_K10", "LEADERDIP_K10_gated", "LEADERS_TRAIL4_K10"]


def blend(d):
    return pd.DataFrame({k: d[k] for k in BLEND_MEMBERS}).mean(axis=1)
