"""Round 5b - blends & overlays of the sleeves. Weights are fitted on DEV only. python src/blend.py"""
import numpy as np, pandas as pd
from scipy.optimize import minimize
import lib

S = pd.read_parquet(f"{lib.RES}/sleeves.parquet")
bench = S[["SPY_bh", "EW_top300"]]
S = S.drop(columns=["SPY_bh", "EW_top300"])
dev = S.loc[lib.DEV[0]:lib.DEV[1]]

def show(r, label):
    lib.print_report(r, label)

# 1. equal weight of all sleeves (capital split, no leverage)
ew = S.mean(axis=1)
show(ew, "EW of 9 sleeves")
# 2. max-Sharpe long-only on DEV
def negsh(w):
    r = dev.values @ w
    return -r.mean() / r.std() * np.sqrt(252)
n = S.shape[1]
res = minimize(negsh, np.ones(n) / n, bounds=[(0, 0.4)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
w = pd.Series(res.x, index=S.columns)
print(w.round(3).to_string())
show(S @ w, "max-Sharpe(DEV) blend")
# 3. inverse-vol (DEV) blend
iv = 1 / dev.std(); iv /= iv.sum()
show(S @ iv, "inverse-vol blend")
# 4. overlays: SPY regime gate / vol-target scale-down on the max-Sharpe blend
import lib as L
P = L.load_panels()
spy = P["C"]["SPY"]
gate = (spy > spy.rolling(200).mean()).shift(1).reindex(S.index).fillna(False)   # info through previous close, trade open
b = S @ w
show(b.where(gate, 0.0), "max-Sharpe blend + SPY>SMA200 gate")
vol = b.rolling(60).std().shift(1) * np.sqrt(252)
scale = (0.30 / vol).clip(upper=1.0).fillna(1.0)
show(b * scale, "blend + vol-target 30% (scale-down only)")
