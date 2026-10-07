"""Assemble the monthly 5-minute arrays into ONE compact panel restricted, per day, to the point-in-time top-N names.

  python src/intraday_panel.py <top_n>          ->  data/ipanel_top<N>.npz

Arrays are [day, slot, bar]; slot s of day d holds ticker `tickers[slot_t[d, s]]` (-1 = empty).  Prices are RAW and used only
as within-day ratios.  Bars: 78 five-minute bars, bar 0 = 09:30-09:35 ... bar 77 = 15:55-16:00.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

import lib

top_n = int(sys.argv[1])
P = lib.load_panels()
M = lib.pit_universe(P, top_n=top_n)
tick_all = np.array(M.columns)
tidx = {t: i for i, t in enumerate(tick_all)}
S = top_n
files = sorted(glob.glob(os.path.join(lib.DATA, "m5arr", "arr_*.npz")))
days_l, slot_l = [], []
Oa, Ha, La, Ca, Va, PMa, AHa = [], [], [], [], [], [], []
for f in files:
    z = np.load(f, allow_pickle=True)
    days = pd.DatetimeIndex(z["days"])
    tk = z["tickers"]
    gi = np.array([tidx[t] for t in tk])
    for di, day in enumerate(days):
        if day not in M.index:
            continue
        mem = M.loc[day].values[gi]          # membership of this month's tickers on this day
        sel = np.where(mem)[0]
        if len(sel) == 0:
            continue
        sel = sel[:S]
        n = len(sel)
        slot = np.full(S, -1, dtype=np.int32)
        slot[:n] = gi[sel]

        def pack(a):  # a: [D, 78, T]
            out = np.full((S, 78), np.nan, dtype=np.float32)
            out[:n] = a[di][:, sel].T
            return out

        Oa.append(pack(z["O"])); Ha.append(pack(z["H"])); La.append(pack(z["L"])); Ca.append(pack(z["C"])); Va.append(pack(z["V"]))
        pm = np.full(S, np.nan, dtype=np.float32); pm[:n] = z["pm_vol"][di][sel]
        ah = np.full(S, np.nan, dtype=np.float32); ah[:n] = z["ah_vol"][di][sel]
        PMa.append(pm); AHa.append(ah)
        days_l.append(day); slot_l.append(slot)
    print(os.path.basename(f), len(days_l), flush=True)
np.savez(os.path.join(lib.DATA, f"ipanel_top{top_n}.npz"), days=np.array(days_l, dtype="datetime64[ns]"), tickers=tick_all,
         slot_t=np.stack(slot_l), O=np.stack(Oa), H=np.stack(Ha), L=np.stack(La), C=np.stack(Ca), V=np.stack(Va),
         pm_vol=np.stack(PMa), ah_vol=np.stack(AHa))
print("saved", len(days_l))
