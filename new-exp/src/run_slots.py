"""Round 4 - Family B/E: event / trend swing trades with explicit exits through the K-slot engine (no leverage).

  python src/run_slots.py [top_n]
Primary parameters were fixed in PREREGISTRATION.md; variants (K, gate, stop) are plateau checks and all are logged.
"""
import sys
import time

import numpy as np
import pandas as pd

import lib
import signals

top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
P = lib.load_panels()
M = lib.pit_universe(P, top_n=top_n)
mkt = P["C"]["SPY"].pct_change()
F = signals.features(P, mkt=mkt)
C, O = P["C"], P["O"]
atr = F["atr14"]
spy = P["C"]["SPY"]
gate_on = (spy > spy.rolling(200).mean())
rk6 = F["mom_6_1"].where(M).rank(axis=1, pct=True)
hi20 = P["H"].rolling(20).max().shift(1)
sma5 = C.rolling(5).mean()
rows = []


def run(name, entry, prio, K=10, exit_sig=None, max_hold=5, stop=0.0, trail=0.0, tp=0.0, gate=None, note=""):
    entry = (entry.fillna(False) & M)
    r, tr, ex = lib.slots_backtest(P, entry, prio, K=K, exit_sig=exit_sig, max_hold=max_hold, stop_atr=stop, trail_atr=trail,
                                   tp_atr=tp, atr=atr, gate=gate)
    cfg = dict(top_n=top_n, K=K, max_hold=max_hold, stop=stop, trail=trail, tp=tp, gate=gate is not None)
    lib.log_trial("B_slots", name, cfg, r, note)
    d, h = lib.perf(lib.split(r, "dev")), lib.perf(lib.split(r, "hold"))
    rows.append(dict(name=name, K=K, hold=max_hold, stop=stop, trail=trail, gate=gate is not None, ntr=len(tr),
                     avg_tr=tr.ret.mean() if len(tr) else np.nan, expo=ex.loc["2012":].mean(),
                     dev_sh=d["sharpe"], dev_med=d["med_m"], dev_mean=d["mean_m"], hold_sh=h["sharpe"], hold_med=h["med_m"],
                     hold_mean=h["mean_m"], dev_dd=d.get("maxdd"), hold_dd=h.get("maxdd")))
    return r


t0 = time.time()
up200 = F["sma200"] > 0
confs = {
    "rsi2<10 up200 exit>sma5": dict(entry=(F["rsi2"] < 10) & up200, prio=-F["rsi2"] + F["vol60"], exit_sig=C > sma5, max_hold=5),
    "gapup2atr rvol3": dict(entry=(F["gap"] >= 2 * atr) & (F["rvol"] >= 3) & (C >= O), prio=F["gap"] / atr, max_hold=10),
    "donchian20 rvol1.5 trail3": dict(entry=(C > hi20) & (F["rvol"] >= 1.5) & up200, prio=F["mom_6_1"], max_hold=60, stop=3.0, trail=3.0),
    "52wh rvol1.5 trail3": dict(entry=(F["high52"] >= 0.999) & (F["rvol"] >= 1.5), prio=F["mom_6_1"], max_hold=60, stop=3.0, trail=3.0),
    "leader dip": dict(entry=(rk6 >= 0.9) & (F["ret3"] <= -2 * atr), prio=-F["ret3"] / atr, max_hold=5),
    "leaders 20d-high trail4": dict(entry=(rk6 >= 0.9) & (C > hi20) & (C > C.rolling(50).mean()), prio=F["mom_6_1"], max_hold=80, stop=4.0, trail=4.0),
    "bigup rvol2 hold10": dict(entry=(F["ret1"] >= 3 * atr) & (F["rvol"] >= 2), prio=F["ret1"] / atr, max_hold=10),
}
for name, c in confs.items():
    for K in (5, 10, 20):
        for gate in (None, gate_on):
            run(name, K=K, gate=gate, **c)
    print(name, f"{time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{lib.RES}/slots_top{top_n}.csv", index=False)
pd.set_option("display.width", 250)
print(df.sort_values("dev_sh", ascending=False).head(30).round(3).to_string())
print("\nBy holdout:")
print(df.sort_values("hold_sh", ascending=False).head(15).round(3).to_string())
