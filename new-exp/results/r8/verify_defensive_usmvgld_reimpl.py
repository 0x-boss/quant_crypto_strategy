"""Independent re-implementation of trigF_usmv_gld from the TEXT description only:
   own-vol scale-down  s = min(1, 0.2501 / vol20),  vol20 = 20-day annualised std of the strategy's own (unscaled baseline) returns
   known at the close (i.e. lagged one day),  W = s * W_baseline  +  (1-s) * 50/50 {USMV, GLD}; ETF weight 0 before listing (cash fallback).
Own baseline book, own backtest loop, own rolling std.  Compared with the module for all dates >= 2012."""
import sys
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import numpy as np, pandas as pd
import r8_common as R          # only for load() and the module/engine cross-checks
import r8_f_defensive as mod

c = R.load(["O", "C", "mom_6_1", "M300"])
O = c["O"].astype(np.float64)
Cc = c["C"].astype(np.float64)

# ---------- own baseline: top-10 of PIT top-300 by 6-1 momentum, equal weight, mean of last 5 daily books ----------
sig = c["mom_6_1"].where(c["M300"])
rk = sig.rank(axis=1, ascending=False, method="first")
pick = ((rk <= 10) & sig.notna()).astype(np.float64) / 10.0
W0 = pick.rolling(5, min_periods=1).mean()
print("baseline book vs R.base_weights max diff:", float((W0 - R.base_weights(c)).abs().to_numpy().max()))
stock_cols = list(W0.columns)


# ---------- own backtest: decision at close d-1-lag executed at open d, held to open d+1, fee on traded notional ----------
def bt(Wdf, fee=R.FEE, lag=0):
    used = [x for x in Wdf.columns if float(Wdf[x].abs().sum()) > 0]
    Wv = Wdf[used].reindex(O.index).fillna(0.0).to_numpy(np.float64)
    Ov = O[used].to_numpy(np.float64)
    T, N = Wv.shape
    pos = np.zeros_like(Wv)
    pos[1 + lag:] = Wv[: T - 1 - lag]
    ratio = Ov[1:] / Ov[:-1]
    ratio[~np.isfinite(ratio)] = 1.0
    hold = np.zeros(N)
    ret = np.zeros(T - 1)
    for d in range(T - 1):
        w = pos[d]
        turn = np.abs(w - hold).sum()
        g = 1.0 - w.sum() + (w * ratio[d]).sum()
        ret[d] = (1.0 - fee * turn) * g - 1.0
        hold = w * ratio[d] / g
    return pd.Series(ret, index=O.index[:-1])


r0 = bt(W0)
r0_lib = R.run(c, W0)
print("own backtest vs R.run (baseline) max |dr|:", float((r0 - r0_lib).abs().max()))

# ---------- own 20d vol of lagged returns ----------
rl = np.concatenate([[np.nan], r0.to_numpy()])[: len(O)]            # value at t = return indexed t-1 (known at close t); length T
rl = np.concatenate([rl, [np.nan]])[: len(O)]
x = np.full(len(O), np.nan)
win = np.lib.stride_tricks.sliding_window_view(rl, 20)               # win[i] = rl[i..i+19] -> window ending at t=i+19
sd = np.where(np.isfinite(win).all(axis=1), win.std(axis=1, ddof=1), np.nan)
x[19:] = sd * np.sqrt(252)
vol = pd.Series(x, index=O.index)

# audit of the hard-coded sigma*: DEV-only median of vol (should be ~0.2501)
print("my DEV median of vol20:", float(vol.loc["2012-01-01":"2021-12-31"].median()), " module STAR:", mod.STAR_DEV_MEDIAN)
print("fraction of DEV days with s<1:", float((vol.loc["2012":"2021"] > 0.2501).mean()), "HOLD:", float((vol.loc["2022":] > 0.2501).mean()))

s = (0.2501 / vol).where(np.isfinite(0.2501 / vol), np.nan).clip(upper=1.0).fillna(1.0)

# ---------- ETF listing (first date with both a close and an open) and gap check ----------
W = W0.mul(s, axis=0).copy()
for t in ("USMV", "GLD"):
    first = max(Cc[t].first_valid_index(), O[t].first_valid_index())
    gaps = int((Cc[t].loc[first:].isna() | O[t].loc[first:].isna()).sum())
    print(t, "first valid", first.date(), "NaN gaps after listing:", gaps)
    on = pd.Series((O.index >= first).astype(float), index=O.index)
    W[t] = (1.0 - s) * 0.5 * on

print("max gross:", float(W.sum(axis=1).max()), " min weight:", float(W.min().min()))

# ---------- compare with the module ----------
p = next(q for q in mod.CONFIGS if q["name"] == "trigF_usmv_gld")
Wm = mod.apply(c, R.base_weights(c), p)
sel = W.index >= "2012-01-01"
cols = W.columns.union(Wm.columns)
a = W.loc[sel].reindex(columns=cols).fillna(0.0)
b = Wm.reindex(W.index).loc[sel].reindex(columns=cols).fillna(0.0)
d = (a - b).abs()
print("MAX ABS WEIGHT DIFF (>=2012, all columns):", float(d.to_numpy().max()))
a2 = W.reindex(columns=cols).fillna(0.0); b2 = Wm.reindex(W.index).reindex(columns=cols).fillna(0.0)
print("MAX ABS WEIGHT DIFF (all dates):", float((a2 - b2).abs().to_numpy().max()))
print("worst column:", d.max().idxmax(), " worst date:", d.max(axis=1).idxmax())
print("ETF-only max diff:", float(d[["USMV", "GLD"]].to_numpy().max()), " stock-only max diff:", float(d.drop(columns=["USMV", "GLD"]).to_numpy().max()))

last = W.index[-1]
d2 = d.drop(index=last) if last in d.index else d
print("last row", last.date(), "diff on it:", float(d.loc[last].max()), "| module last-row gross:", float(Wm.reindex(W.index).loc[last].fillna(0).sum()), "NaN count in module last row:", int(Wm.reindex(W.index).loc[last].isna().sum()), "| mine last-row gross:", float(W.loc[last].sum()))
print("MAX ABS WEIGHT DIFF (>=2012, EXCLUDING last untradable row):", float(d2.to_numpy().max()))
print("worst date excluding last row:", d2.max(axis=1).idxmax(), "second-to-last row diff:", float(d.iloc[-2].max()))
# ---------- returns from my own engine on my own weights ----------
r = bt(W)
rm = R.run(c, Wm)
print("own engine(own W) vs R.run(module W) max |dr|:", float((r - rm).abs().max()))
for nm, rr in (("mine", r), ("module", rm)):
    dev = rr.loc["2012-01-01":"2021-12-31"]; hold = rr.loc["2022-01-01":]
    print(nm, "DEV Sharpe %.4f HOLD Sharpe %.4f" % (dev.mean() / dev.std() * np.sqrt(252), hold.mean() / hold.std() * np.sqrt(252)))
