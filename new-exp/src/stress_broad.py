"""Round 6b - cost realism for the 'too good' broad-pool candidates: add per-side slippage and see where the edge goes."""
import numpy as np, pandas as pd
import lib

top_n = 3000
cols = lib.research_cols(top_n)
P = lib.load_panels(cols=cols, dtype=np.float32)
M = lib.pit_universe(P, top_n=top_n)
ucols = list(M.columns[M.any()])
C, O, V, DV = P["C"][ucols], P["O"][ucols], P["V"][ucols], P["DV"][ucols]
Mc = M[ucols]
r1 = C.pct_change()
sig = {"rev5": -(C / C.shift(5) - 1), "highvol": r1.rolling(60, min_periods=40).std()}
med_dv = DV.rolling(63, min_periods=40).median().shift(1)
rows = []
for name, k, h in [("rev5", 10, 21), ("rev5", 25, 21), ("highvol", 25, 21), ("highvol", 10, 21)]:
    s = sig[name].where(Mc)
    rank = s.rank(axis=1, ascending=False, method="first")
    W = (((rank <= k) & s.notna()).astype(float) / k).rolling(h, min_periods=1).mean()
    mdv = (W.shift(1) * med_dv).sum(axis=1) / W.shift(1).sum(axis=1)
    print(name, k, h, "median $vol of held names (weighted avg, $M):", round(float(mdv.loc['2012':].median()) / 1e6, 1))
    for slip_bps in (0, 10, 25, 50, 100):
        r = lib.weights_backtest(W, O, fee=lib.FEE + slip_bps / 1e4)
        d, o = lib.perf(lib.split(r, "dev")), lib.perf(lib.split(r, "hold"))
        rows.append(dict(sig=name, k=k, h=h, slip_bps_per_side=slip_bps, dev_sh=d["sharpe"], dev_mean=d["mean_m"], dev_med=d["med_m"],
                         hold_sh=o["sharpe"], hold_mean=o["mean_m"], hold_med=o["med_m"]))
df = pd.DataFrame(rows)
df.to_csv(f"{lib.RES}/stress_broad.csv", index=False)
print(df.round(3).to_string())
