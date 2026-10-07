"""Round 3 - Family C scan: cross-sectional intraday signals on the PIT top-300 names (5-minute bars, 2016-01..2026-03).

For each entry time k (bar index) and each causal signal (uses bars < k and daily data through the previous close),
buy the top-N (N=10) names at the OPEN of bar k, exit at the day's close (bar 77 close)  -- or, for k=72, hold overnight to
the next open ("on").  Net of 0.2 % round trip.  Output: Sharpe / monthly stats per DEV (2016-2021) and HOLDOUT (2022-2026-03).
"""
import sys
import time

import numpy as np
import pandas as pd

import lib
import signals

NTOP = 10
Z = np.load(f"{lib.DATA}/ipanel_top300.npz", allow_pickle=True)
days = pd.DatetimeIndex(Z["days"])
tick = Z["tickers"]
slot = Z["slot_t"]                       # [D,S]
D, S = slot.shape
valid = slot >= 0
O, H, L, C, V = (Z[k] for k in "OHLCV")
print("loaded", O.shape, flush=True)
# forward-fill closes along the bar axis, rebuild opens
CL = C.copy()
for b in range(1, 78):
    m = ~np.isfinite(CL[:, :, b])
    CL[:, :, b][m] = CL[:, :, b - 1][m]
OP = np.where(np.isfinite(O), O, np.concatenate([CL[:, :, :1], CL[:, :, :-1]], axis=2))
OP[:, :, 0] = np.where(np.isfinite(O[:, :, 0]), O[:, :, 0], CL[:, :, 0])
Hf = np.where(np.isfinite(H), H, CL)
Lf = np.where(np.isfinite(L), L, CL)
Vz = np.nan_to_num(V)

P = lib.load_panels()
tcol = {t: i for i, t in enumerate(P["C"].columns)}
didx = P["C"].index.get_indexer(days)
cols_idx = np.array([tcol.get(t, -1) for t in tick])
F = signals.features(P, mkt=P["C"]["SPY"].pct_change())


def lookup(df, shift=0):
    """daily DataFrame -> [D,S] array of the value of the slot's ticker on (day - shift)."""
    v = df.values.astype(np.float32)
    di = didx - shift
    ti = np.where(valid, slot, 0)
    ci = cols_idx[ti]
    out = v[np.clip(di, 0, None)[:, None], np.where(ci >= 0, ci, 0)]
    out[(ci < 0) | (di < 0)[:, None] | ~valid] = np.nan
    return out


gap = lookup(F["gap"])                       # daily open / prev close - 1   (known at the open)
atr = lookup(F["atr14"], 1)                  # through previous close
vol_prev = lookup(P["V"], 1)
pc_adj = lookup(P["C"], 1)
# next-day open (for overnight holds): next day's first-bar open of the same ticker
tmat = {}


def next_open():
    out = np.full((D, S), np.nan, dtype=np.float32)
    for d in range(D - 1):
        if (days[d + 1] - days[d]).days > 5:
            continue
        m = {t: j for j, t in enumerate(slot[d + 1]) if t >= 0}
        for s in range(S):
            t = slot[d, s]
            if t >= 0 and t in m:
                out[d, s] = OP[d + 1, m[t], 0] / CL[d, s, 77] - 1 if np.isfinite(CL[d, s, 77]) else np.nan
    return out


# ticker-day history of "volume up to bar k" for relative-volume signals
Ttot = len(tick)


def rvol_k(k, win=20):
    cum = Vz[:, :, :k].sum(axis=2).astype(np.float64)
    cum[~valid] = np.nan
    td = np.full((D, Ttot), np.nan)
    rows = np.repeat(np.arange(D)[:, None], S, 1)
    td[rows[valid], slot[valid]] = cum[valid]
    ref = pd.DataFrame(td).rolling(win, min_periods=10).mean().shift(1).values
    out = np.full((D, S), np.nan)
    out[valid] = cum[valid] / ref[rows[valid], slot[valid]]
    return out


def day_portfolio(sig, entry_px, exit_px, ntop=NTOP, fee=lib.FEE):
    ok = np.isfinite(sig) & np.isfinite(entry_px) & np.isfinite(exit_px) & valid
    s = np.where(ok, sig, -np.inf)
    order = np.argsort(-s, axis=1)[:, :ntop]
    r = np.take_along_axis(exit_px / entry_px - 1, order, axis=1)
    okk = np.take_along_axis(ok, order, axis=1)
    r = np.where(okk, r - 2 * fee, np.nan)
    cnt = okk.sum(1)
    daily = np.nansum(r, axis=1) / ntop            # equal 1/ntop slots; empty slots = cash
    daily[cnt < ntop // 2] = 0.0
    return pd.Series(daily, index=days), r


results = []
entries = {"0935": 1, "1000": 6, "1030": 12, "1130": 24, "1330": 48, "1430": 60, "1530": 72}
t0 = time.time()
for ename, k in entries.items():
    ent = OP[:, :, k]
    ex_close = CL[:, :, 77]
    ret_so = ent / OP[:, :, 0] - 1                                     # return since the open (bars < k)
    hi_so = np.nanmax(np.where(np.isfinite(Hf[:, :, :k]), Hf[:, :, :k], -np.inf), axis=2)
    lo_so = np.nanmin(np.where(np.isfinite(Lf[:, :, :k]), Lf[:, :, :k], np.inf), axis=2)
    clv_so = (CL[:, :, k - 1] - lo_so) / np.where(hi_so - lo_so > 0, hi_so - lo_so, np.nan)
    rv = rvol_k(k)
    atr_abs = atr
    S_ = {
        "ret_since_open": ret_so,
        "rev_since_open": -ret_so,
        "ret_so_atr": ret_so / atr,
        "rev_so_atr": -ret_so / atr,
        "gap": gap,
        "rev_gap": -gap,
        "gap_x_rvol": gap * np.minimum(rv, 10),
        "rvol_up": np.where(ret_so > 0, rv, np.nan),
        "rvol": rv,
        "rvol_down_rev": np.where(ret_so < 0, rv, np.nan),
        "clv_hi": clv_so,
        "clv_lo": -clv_so,
        "gap_up_hold": np.where((gap > 0) & (ret_so > 0), gap + ret_so, np.nan),
        "orb_break": np.where(ent >= hi_so, rv, np.nan),
        "vol_atr": atr,
    }
    for sn, sg in S_.items():
        r, _ = day_portfolio(sg, ent, ex_close)
        lib.log_trial("C_intraday", f"{sn}@{ename}->close", dict(ntop=NTOP), r)
        d, h = lib.perf(r.loc["2016":"2021"]), lib.perf(r.loc["2022":])
        results.append(dict(entry=ename, sig=sn, exit="close", dev_sh=d["sharpe"], dev_mean=d["mean_m"], hold_sh=h["sharpe"],
                            hold_mean=h["mean_m"], gross_per_trade=None))
    print(ename, f"{time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(results)
df.to_csv(f"{lib.RES}/scan_intraday.csv", index=False)
pd.set_option("display.width", 200)
print(df.sort_values("dev_sh", ascending=False).head(30).round(3).to_string())
print(df.sort_values("hold_sh", ascending=False).head(15).round(3).to_string())
