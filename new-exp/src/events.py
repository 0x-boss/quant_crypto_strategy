"""Event-study helpers: what is the average forward open-to-open return after a (close-t) event, net of costs?"""
import numpy as np
import pandas as pd

import lib


def fwd_returns(P, hs=(1, 2, 3, 5, 10, 20)):
    """Forward open->open returns for a trade entered at the open of t+1 and closed at the open of t+1+h.
    Row t is the decision day (close t)."""
    O = P["O"]
    out = {}
    for h in hs:
        out[h] = O.shift(-(1 + h)) / O.shift(-1) - 1
    return out


def study(mask, fwd, M=None, label="", cost=2 * lib.FEE, by_year=False, dev_only=False, min_n=200):
    """mask: bool DataFrame (decision day x ticker). Returns DataFrame by horizon: n, mean, median, win, t, net mean."""
    m = mask if M is None else (mask & M)
    if dev_only:
        m = m.loc[: lib.DEV[1]]
    rows = []
    for h, f in fwd.items():
        ff = f.reindex_like(m)
        ok = m.values & np.isfinite(ff.values)
        x = ff.values[ok]
        if len(x) < min_n:
            continue
        # events on the same day are correlated: aggregate per day first, then t-stat across days
        per_day = pd.Series(np.where(ok, ff.values, np.nan).mean(axis=1) if False else np.nanmean(np.where(ok, ff.values, np.nan), axis=1),
                            index=m.index).dropna()
        t = per_day.mean() / (per_day.std() / np.sqrt(len(per_day))) if per_day.std() > 0 else np.nan
        rows.append(dict(label=label, h=h, n=len(x), days=len(per_day), mean=x.mean(), median=np.median(x),
                         win=(x > cost).mean(), net_mean=x.mean() - cost, t_day=t))
    return pd.DataFrame(rows)


def study_by_year(mask, fwd, h, M=None, cost=2 * lib.FEE):
    m = mask if M is None else (mask & M)
    f = fwd[h].reindex_like(m)
    v = np.where(m.values & np.isfinite(f.values), f.values, np.nan)
    df = pd.DataFrame(v, index=m.index)
    g = df.stack().groupby(level=0).agg(["mean", "count"])
    g["yr"] = g.index.year
    out = g.groupby("yr").apply(lambda d: pd.Series(dict(n=d["count"].sum(), net=(d["mean"] * d["count"]).sum() / d["count"].sum() - cost)))
    return out
