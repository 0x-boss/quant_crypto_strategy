"""Truncation-invariance test (the strongest look-ahead detector): features and the point-in-time universe computed on a
panel that ENDS at T0 must equal the same objects computed on the full panel, for every date < T0.
(The last date is excluded: the bad-print cleaner looks one day ahead, documented in README.)
Run: python -I new-exp/tests/test_truncation.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, pandas as pd
import lib, signals

P = lib.load_panels()
T0 = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2019-06-28")
Pt = {k: v.loc[:T0] for k, v in P.items()}
cols = list(P["C"].columns)
mk_full = P["C"].pct_change().mean(axis=1)
Ff, Ft = signals.features(P, mkt=None), signals.features(Pt, mkt=None)
bad = []
for k in Ff:
    a, b = Ff[k].loc[:T0].iloc[:-1], Ft[k].iloc[:-1]
    a, b = a.align(b, join="inner")
    ok = np.allclose(a.values.astype(float), b.values.astype(float), rtol=1e-9, atol=1e-12, equal_nan=True)
    if not ok:
        diff = (a - b).abs().stack().max()
        bad.append((k, diff))
print("features differing:", bad)
Mf, Mt = lib.pit_universe(P, 300), lib.pit_universe(Pt, 300)
a, b = Mf.loc[:T0].iloc[:-1], Mt.iloc[:-1]
nd = int((a.values != b.values).sum())
print("universe membership differences:", nd)
assert not bad and nd == 0, "LOOK-AHEAD DETECTED"
print("truncation test OK")
