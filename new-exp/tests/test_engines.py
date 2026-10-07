"""Synthetic sanity tests for the engines (run: python -I new-exp/tests/test_engines.py)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, pandas as pd
import lib

rng = np.random.default_rng(1)
T, N = 1500, 40
idx = pd.bdate_range("2015-01-01", periods=T)
cols = [f"S{i}" for i in range(N)]
ret = rng.normal(0.0003, 0.02, (T, N))
O = pd.DataFrame(100 * np.exp(np.cumsum(ret, 0)), idx, cols)       # open prices = random walk (independent increments)
C = O * np.exp(rng.normal(0, 0.006, (T, N)))
H = np.maximum(O, C) * 1.005
L = np.minimum(O, C) * 0.995

# 1. alignment: the realised return of a position decided at close t is open[t+2]/open[t+1]-1 - a signal that peeks at it must win
fut = O.shift(-2) / O.shift(-1) - 1
rank = fut.rank(axis=1, ascending=False)
Wpeek = ((rank <= 5).astype(float) / 5).fillna(0)
r = lib.weights_backtest(Wpeek, O, fee=0.0)
sh = r.mean() / r.std() * np.sqrt(252)
assert sh > 10, f"peek signal should have huge Sharpe, got {sh}"
# a signal that uses the return one period EARLIER than the traded period must NOT win (same data, shifted back)
past = O.shift(-1) / O - 1  # return from open t to open t+1 : known only at open t+1 -> not known at close t
rank = past.rank(axis=1, ascending=False)
Wlate = ((rank <= 5).astype(float) / 5).fillna(0)
r2 = lib.weights_backtest(Wlate, O, fee=0.0)
sh2 = r2.mean() / r2.std() * np.sqrt(252)
assert abs(sh2) < 1.5, f"lagged-by-one-period signal should be ~0 Sharpe, got {sh2}"

# 2. fee accounting: full turnover each day at fee f costs ~ 2*f per day (sell all + buy all) for alternating books
W = pd.DataFrame(0.0, idx, cols)
W.iloc[::2, 0] = 1.0
W.iloc[1::2, 1] = 1.0
r0 = lib.weights_backtest(W, O, fee=0.0)
r1 = lib.weights_backtest(W, O, fee=0.001)
diff = (r0 - r1).iloc[10:-10].mean()
assert 0.0018 < diff < 0.0022, diff

# 3. leverage guard
Wbad = pd.DataFrame(0.6, idx, cols[:2]).reindex(columns=cols).fillna(0)
try:
    lib.weights_backtest(Wbad, O)
    raise SystemExit("leverage guard failed")
except AssertionError:
    pass

# 4. slots engine: entry at next open, exit after max_hold at open; compare with a manual calc for one trade
P = dict(O=O, H=H, L=L, C=C)
entry = pd.DataFrame(False, idx, cols)
entry.iloc[100, 3] = True            # decided at close of day 100 -> bought at open of day 101
prio = pd.DataFrame(1.0, idx, cols)
rr, tr, ex = lib.slots_backtest(P, entry, prio, K=1, max_hold=3, fee=0.001)
assert len(tr) == 1
t = tr.iloc[0]
assert t.entry == idx[101] and t.exit == idx[104], (t.entry, t.exit)
manual = (O.iloc[104, 3] * 0.999) / (O.iloc[101, 3] / 0.999 * 1.0) - 1
# size*(1-fee)/px shares -> proceeds shares*px_exit*(1-fee); invested=size
manual = (O.iloc[104, 3] / O.iloc[101, 3]) * 0.999 * 0.999 - 1
assert abs(t.ret - manual) < 1e-9, (t.ret, manual)
# equity ends consistent with the trade return (K=1: full capital)
assert abs((1 + rr).prod() - (1 + t.ret)) < 1e-9, ((1 + rr).prod(), 1 + t.ret)

# 5. stop: a gap-down through the stop fills at the open, an intraday touch fills at the stop
C2, O2, H2, L2 = C.copy(), O.copy(), H.copy(), L.copy()
O2.iloc[101, 3] = 100.0; L2.iloc[101, 3] = 99.0; H2.iloc[101, 3] = 101.0; C2.iloc[101, 3] = 100.0
L2.iloc[102, 3] = 90.0; O2.iloc[102, 3] = 100.0; H2.iloc[102, 3] = 100.5; C2.iloc[102, 3] = 92
atr = pd.DataFrame(0.05, idx, cols)   # 5 % ATR, stop_atr=1 -> stop at 95
P2 = dict(O=O2, H=H2, L=L2, C=C2)
rr, tr, ex = lib.slots_backtest(P2, entry, prio, K=1, max_hold=10, stop_atr=1.0, atr=atr, fee=0.0)
assert abs(tr.iloc[0].ret - (95 / 100 - 1)) < 1e-9, tr
O2.iloc[102, 3] = 90.0; L2.iloc[102, 3] = 88.0; H2.iloc[102, 3] = 91.0   # gap through the stop -> fill at open 90
rr, tr, ex = lib.slots_backtest(P2, entry, prio, K=1, max_hold=10, stop_atr=1.0, atr=atr, fee=0.0)
assert abs(tr.iloc[0].ret - (90 / 100 - 1)) < 1e-9, tr
print("engine tests OK")
