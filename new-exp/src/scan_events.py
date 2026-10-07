"""Round 2 - Family B event study: forward open->open returns after pre-registered close-of-day events, net of 0.2 % RT.

  python src/scan_events.py [top_n]
Prints, for DEV and HOLDOUT separately: n events, mean gross, mean net-of-cost, win-rate, day-clustered t-stat.
"""
import sys

import numpy as np
import pandas as pd

import events
import lib
import signals

top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
P = lib.load_panels()
M = lib.pit_universe(P, top_n=top_n)
mkt = P["C"]["SPY"].pct_change()
F = signals.features(P, mkt=mkt)
C, H, L, V = P["C"], P["H"], P["L"], P["V"]
atr = F["atr14"]
fwd = events.fwd_returns(P, hs=(1, 2, 3, 5, 10, 20))
rk6 = F["mom_6_1"].where(M).rank(axis=1, pct=True)
hi20 = H.rolling(20).max().shift(1)

E = {
    "B1_rsi2<10_up200": (F["rsi2"] < 10) & (F["sma200"] > 0),
    "B1b_rsi2<5": (F["rsi2"] < 5),
    "B2_gapdown3atr_rvol2": (F["gap"] <= -3 * atr) & (F["rvol"] >= 2),
    "B3_gapup2atr_rvol3_hold": (F["gap"] >= 2 * atr) & (F["rvol"] >= 3) & (C >= P["O"]),
    "B4_donchian20_rvol1.5": (C > hi20) & (F["rvol"] >= 1.5),
    "B5_new52wh_rvol1.5": (F["high52"] >= 0.999) & (F["rvol"] >= 1.5),
    "B6_leader_dip": (rk6 >= 0.9) & (F["ret3"] <= -2 * atr),
    "B7_bigdown_vol": (F["ret1"] <= -3 * atr) & (F["rvol"] >= 2),
    "B8_bigup_vol": (F["ret1"] >= 3 * atr) & (F["rvol"] >= 2),
    "B9_10dlow_up200": (C <= C.rolling(10).min()) & (F["sma200"] > 0),
    "B10_gapdown_closehigh": (F["gap"] <= -2 * atr) & (F["clv"] >= 0.7),
    "B11_3down_up200": ((F["ret1"] < 0) & (F["ret1"].shift(1) < 0) & (F["ret1"].shift(2) < 0)) & (F["sma200"] > 0),
}

pd.set_option("display.width", 220)
for name, mask in E.items():
    mask = mask.fillna(False) & M
    for part, (a, b) in dict(dev=lib.DEV, hold=lib.HOLD).items():
        m = mask.loc[a:b]
        f = {h: v.loc[a:b] for h, v in fwd.items()}
        res = events.study(m, f, label=name)
        if res.empty:
            continue
        res["part"] = part
        res["per_year"] = res["n"] / ((pd.Timestamp(b if part == "dev" else "2026-10-07") - pd.Timestamp(a)).days / 365.25)
        print(res[["label", "part", "h", "n", "per_year", "mean", "net_mean", "win", "t_day"]].round(4).to_string(index=False, header=(part == "dev")), flush=True)
        for _, r in res.iterrows():
            lib.log_trial("B_event_study", name, dict(top_n=top_n, h=int(r.h), part=part), pd.Series(dtype=float), note=f"net={r.net_mean:.4f}")
