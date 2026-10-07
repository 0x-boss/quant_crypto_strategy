"""Round 3b - pre-registered C4/C5: buy near the close (bar 77 open = 15:55), sell next day's open.  Gross and net edge of top-10 by
causal signals computed from bars < 77.   python src/intraday_overnight.py"""
import sys
import numpy as np, pandas as pd
s = open(__file__.replace("intraday_overnight.py", "intraday_explore.py")).read()
exec(compile(s[:s.index("results = []")], "ie", "exec"))

nxt = np.full((D, S), np.nan, dtype=np.float32)       # next open / this 15:55 open - 1
for d in range(D - 1):
    if (days[d + 1] - days[d]).days > 5:
        continue
    pos = {t: j for j, t in enumerate(slot[d + 1]) if t >= 0}
    for sl in range(S):
        t = slot[d, sl]
        if t >= 0 and t in pos and np.isfinite(OP[d, sl, 77]):
            nxt[d, sl] = OP[d + 1, pos[t], 0] / OP[d, sl, 77] - 1
k = 77
ent = OP[:, :, k]
ret_so = ent / OP[:, :, 0] - 1
hi_so = np.nanmax(np.where(np.isfinite(Hf[:, :, :k]), Hf[:, :, :k], -np.inf), axis=2)
lo_so = np.nanmin(np.where(np.isfinite(Lf[:, :, :k]), Lf[:, :, :k], np.inf), axis=2)
clv = (CL[:, :, k - 1] - lo_so) / np.where(hi_so - lo_so > 0, hi_so - lo_so, np.nan)
rv = rvol_k(k)
lastbar = CL[:, :, 76] / OP[:, :, 71] - 1             # last half hour return (bars 71..76)
sg = {"ret_so": ret_so, "rev_so": -ret_so, "clv_hi": clv, "clv_lo": -clv, "rvol": rv, "rvol_up": np.where(ret_so > 0, rv, np.nan),
      "last30_mom": lastbar, "last30_rev": -lastbar, "gap": gap, "vol_atr": atr, "ret_so_atr": ret_so / atr}
sp = np.searchsorted(days, pd.Timestamp("2022-01-01"))
print("unconditional mean overnight (bps): dev %.1f hold %.1f" % (np.nanmean(nxt[:sp]) * 1e4, np.nanmean(nxt[sp:]) * 1e4))
rows = []
for n, x in sg.items():
    r, _ = day_portfolio(x, ent, np.where(np.isfinite(nxt), ent * (1 + nxt), np.nan))
    lib.log_trial("C_intraday", f"{n}@1555->nextopen", dict(ntop=NTOP), r)
    ok = np.isfinite(x) & np.isfinite(nxt)
    s_ = np.where(ok, x, -np.inf); order = np.argsort(-s_, axis=1)[:, :10]
    g = np.take_along_axis(nxt, order, axis=1); okk = np.take_along_axis(ok, order, axis=1)
    g = np.where(okk, g, np.nan)
    d, h = lib.perf(r.loc["2016":"2021"]), lib.perf(r.loc["2022":])
    rows.append(dict(sig=n, gross_bps_dev=np.nanmean(g[:sp]) * 1e4, gross_bps_hold=np.nanmean(g[sp:]) * 1e4, net_sh_dev=d["sharpe"], net_sh_hold=h["sharpe"]))
print(pd.DataFrame(rows).round(2).to_string())
