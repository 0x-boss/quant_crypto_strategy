"""Gross (before-cost) mean entry->close return of the top-10 names by each causal intraday signal, per entry time, DEV vs HOLDOUT,
next to the 20 bp round-trip cost.   python src/intraday_gross.py  ->  results/intraday_gross_edge.csv"""
import numpy as np
import pandas as pd

src = open(__file__.replace("intraday_gross.py", "intraday_explore.py")).read()
exec(compile(src[:src.index("results = []")], "ie", "exec"))

rows = []
entries = {"09:35": 1, "10:00": 6, "10:30": 12, "11:30": 24, "13:30": 48, "14:30": 60, "15:30": 72}
sp = np.searchsorted(days, pd.Timestamp("2022-01-01"))
for ename, k in entries.items():
    ent, ex = OP[:, :, k], CL[:, :, 77]
    ret_so = ent / OP[:, :, 0] - 1
    rv = rvol_k(k)
    base = np.where(valid & np.isfinite(ent) & np.isfinite(ex), ex / ent - 1, np.nan)
    sigs = {"momentum since open": ret_so, "reversal since open": -ret_so, "gap up": gap, "gap down": -gap, "rel. volume": rv,
            "ATR (volatility)": atr, "gap x rel.vol": gap * np.minimum(rv, 10)}
    for sn, sg in sigs.items():
        ok = np.isfinite(sg) & np.isfinite(base)
        order = np.argsort(-np.where(ok, sg, -np.inf), axis=1)[:, :10]
        r = np.take_along_axis(base, order, axis=1)
        okk = np.take_along_axis(ok, order, axis=1)
        for part, sl in (("dev", slice(0, sp)), ("hold", slice(sp, None))):
            rr = np.where(okk[sl], r[sl], np.nan)
            rows.append(dict(entry=ename, signal=sn, part=part, gross_bps=np.nanmean(rr) * 1e4, unconditional_bps=np.nanmean(base[sl]) * 1e4))
df = pd.DataFrame(rows)
df.to_csv(f"{lib.RES}/intraday_gross_edge.csv", index=False)
print(df.groupby("part").gross_bps.describe().round(1))
print("max gross bps:", df.gross_bps.max().round(1), " cost = 20 bps")
