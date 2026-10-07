"""Turn the monthly 5-minute parquet files into dense numpy arrays [day, bar(78), ticker] per month, restricted to the
tickers that were members of the point-in-time universe on at least one day of that month (membership uses only
information available before the month: lib.pit_universe).

  python -I new-exp/src/build_arrays.py <top_n> [start YYYY-MM] [end YYYY-MM]

Output data/m5arr/arr_YYYY-MM.npz  with keys: days, tickers, O,H,L,C,V [D,78,T] float32 (NaN = no trade in the bar),
                                         pm_vol, pm_hi, pm_lo, pm_last, ah_vol, ah_last [D,T] float32
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

import lib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "data", "m5")
OUT = os.path.join(ROOT, "data", "m5arr")
os.makedirs(OUT, exist_ok=True)

top_n = int(sys.argv[1])
P = lib.load_panels()
M = lib.pit_universe(P, top_n=top_n)
files = sorted(glob.glob(os.path.join(SRC, "m5_*.parquet")))
lo = sys.argv[2] if len(sys.argv) > 2 else "0000-00"
hi = sys.argv[3] if len(sys.argv) > 3 else "9999-99"
for f in files:
    ym = os.path.basename(f)[3:10]
    if not (lo <= ym <= hi):
        continue
    fo = os.path.join(OUT, f"arr_{ym}.npz")
    if os.path.exists(fo):
        continue
    mo = M.loc[ym]
    keep = list(mo.columns[mo.any().values])
    m5 = pd.read_parquet(f)
    m5 = m5[m5.ticker.isin(keep)]
    ex = pd.read_parquet(os.path.join(SRC, f"ext_{ym}.parquet"))
    ex = ex[ex.ticker.isin(keep)]
    days = np.sort(m5.ts.dt.normalize().unique())
    tick = np.array(sorted(m5.ticker.unique()))
    di = np.searchsorted(days, m5.ts.dt.normalize().values)
    ti = np.searchsorted(tick, m5.ticker.values)
    mod = m5.ts.dt.hour.values * 60 + m5.ts.dt.minute.values
    bi = (mod - 570) // 5
    D, T = len(days), len(tick)
    arrs = {}
    for k in "ohlcv":
        a = np.full((D, 78, T), np.nan, dtype=np.float32)
        a[di, bi, ti] = m5[k].values.astype(np.float32)
        arrs[k.upper()] = a
    e = {}
    edi = np.searchsorted(days, ex.day.values)
    ok = (edi < D) & (days[np.minimum(edi, D - 1)] == ex.day.values)
    eti = np.searchsorted(tick, ex.ticker.values)
    ok &= (eti < T) & (tick[np.minimum(eti, T - 1)] == ex.ticker.values)
    for k in ["pm_vol", "pm_hi", "pm_lo", "pm_last", "ah_vol", "ah_last"]:
        a = np.full((D, T), np.nan, dtype=np.float32)
        a[edi[ok], eti[ok]] = ex[k].values[ok].astype(np.float32)
        e[k] = a
    np.savez_compressed(fo, days=days.astype("datetime64[ns]"), tickers=tick, **arrs, **e)
    print(ym, D, T, flush=True)
