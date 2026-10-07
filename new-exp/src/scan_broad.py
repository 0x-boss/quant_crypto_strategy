"""Round 6 - does the picture change in a much broader (less liquid, higher-vol) pool?  Light-weight scan of a few signals on
top_n in (1000, 2000, 3000) names, k in (10, 25), h in (5, 21). Also reports an estimated spread (Corwin-Schultz) of the picks,
because a flat 0.2 % round trip is unrealistic for illiquid names.     python src/scan_broad.py
"""
import sys
import numpy as np, pandas as pd
import lib

rows = []
for top_n in (1000, 2000, 3000):
    cols = lib.research_cols(top_n)
    P = lib.load_panels(cols=cols, dtype=np.float32)
    M = lib.pit_universe(P, top_n=top_n)
    ucols = list(M.columns[M.any()])
    C, O, H, L, V = (P[k][ucols] for k in ("C", "O", "H", "L", "V"))
    Mc = M[ucols]
    r1 = C.pct_change()
    pc = C.shift(1)
    tr = np.maximum(np.maximum(H - L, (H - pc).abs()), (L - pc).abs())
    atr = tr.rolling(14, min_periods=7).mean() / C
    # Corwin-Schultz spread estimate (daily, 21d mean)
    hl = np.log(H / L) ** 2
    beta = hl + hl.shift(1)
    gamma = np.log(np.maximum(H, H.shift(1)) / np.minimum(L, L.shift(1))) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
    cs = (2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))).clip(lower=0).rolling(21, min_periods=10).mean()
    sig = {
        "mom_6_1": C.shift(21) / C.shift(126) - 1,
        "mom_12_1": C.shift(21) / C.shift(252) - 1,
        "rev5": -(C / C.shift(5) - 1),
        "rev1_atr": -r1 / atr,
        "highvol": r1.rolling(60, min_periods=40).std(),
        "high52": C / C.rolling(252, min_periods=200).max(),
    }
    Oc = O
    for name, sg in sig.items():
        for k in (10, 25):
            for h in (5, 21):
                s = sg.where(Mc)
                rank = s.rank(axis=1, ascending=False, method="first")
                W = (((rank <= k) & s.notna()).astype(float) / k).rolling(h, min_periods=1).mean()
                r = lib.weights_backtest(W, Oc)
                lib.log_trial("F_broad", name, dict(top_n=top_n, k=k, h=h), r)
                d, o = lib.perf(lib.split(r, "dev")), lib.perf(lib.split(r, "hold"))
                # average estimated spread of the held names (weights-weighted)
                sp = (W.shift(1) * cs).sum(axis=1) / W.shift(1).sum(axis=1)
                rows.append(dict(top_n=top_n, sig=name, k=k, h=h, dev_sh=d["sharpe"], dev_mean=d["mean_m"], dev_med=d["med_m"],
                                 hold_sh=o["sharpe"], hold_mean=o["mean_m"], hold_med=o["med_m"], est_spread_bps=sp.loc["2012":].mean() * 1e4))
    print(top_n, "done", flush=True)
    del P
df = pd.DataFrame(rows)
df.to_csv(f"{lib.RES}/scan_broad.csv", index=False)
pd.set_option("display.width", 220)
print(df.sort_values("dev_sh", ascending=False).head(25).round(3).to_string())
print(df.sort_values("hold_sh", ascending=False).head(15).round(3).to_string())
