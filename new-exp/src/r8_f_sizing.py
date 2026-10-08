"""Round-8 family SIZING: can position weights / concentration cut the idiosyncratic blow-ups of the 6-1 momentum sleeve?

Everything is built from scratch by weights(c, p) (selection-changing configs need it).  The daily book is
    candidates   = PIT universe mask (M300, or the M1000-minus-M300 ring) & finite 6-1 momentum & [vol60_ann <= vcap] & [finite risk measure]
    ranking      = 6-1 momentum, best first (ties -> column order, same as the baseline's rank(method="first"))
    sector cap   = walk down the ranking, skip a name if `smax` names of its sector are already taken (unknown-sector tickers are never capped)
    top-k        = first k survivors
    weights      = equal  or  inverse risk (1/vol20, 1/vol60, 1/atr14), then iterative water-filling per-name cap `cap`
    stagger      = average of the last 5 daily books (identical to r8_common.base_weights)
Every ingredient on row t is a feature of the close of day t (vol20, vol60, atr14, mom_6_1, masks) -> causal; the runner's perturbation guard checks it.
Gross exposure <= 1 by construction (weights are normalised to 1 per daily book, a per-name cap can only make a book smaller, never larger).

Parameter priors (each is a prior from the literature / the task statement, none is searched):
  inverse-vol weights  : risk-parity / inverse-volatility sizing, Moskowitz-Ooi-Pedersen 2012, Barroso-Santa-Clara 2015 (vol-managed momentum);
                         vol20 ~ 1 month (short, reactive), vol60 ~ 3 months (the library's standard), atr14 ~ Wilder (1978) range-based vol
  per-name cap         : 15 % / 20 % / 25 % = 1.5x / 2x / 2.5x the equal weight of a 10-name book (typical UCITS-style concentration limits)
  K                    : 15 / 20 / 30 names, Jegadeesh-Titman style diversification (their portfolios were deciles, i.e. >> 10 names)
  vol cap              : 80 / 100 / 120 % annualised vol60 - lottery-stock / idiosyncratic-volatility puzzle (Ang et al. 2006, Bali et al. 2011 MAX);
                         held names median ~48 %, 75th pct ~68 %, so 80 % is a mild trim, 120 % trims only the extreme tail
  sector cap           : max 2 / 3 names per sector - industry momentum clusters (Moskowitz-Grinblatt 1999), momentum crashes are sector-concentrated
  universe mix         : 30 % / 50 % of the book from the mid-cap ring (top-1000 minus top-300), momentum is stronger / noisier in smaller caps
                         (Hong-Lim-Stein 2000); and a pure top-1000 book
"""
import numpy as np
import pandas as pd

import r8_common as R

FAMILY = "sizing"
ANN = np.sqrt(252.0)


def _cfg(name, **kw):
    d = dict(name=name, k=10, risk=None, cap=None, vcap=None, smax=None, sleeves=None, h=5)
    d.update(kw)
    return d


CONFIGS = [
    # --- sanity: must reproduce the baseline exactly (logged, expected score = -inf because it is the baseline)
    _cfg("base_replica_k10_eq"),
    # --- A. inverse-vol weights among the 10 selected names, with a per-name cap
    _cfg("iv_vol20_cap25", risk="vol20", cap=0.25), _cfg("iv_vol20_cap20", risk="vol20", cap=0.20), _cfg("iv_vol20_cap15", risk="vol20", cap=0.15),
    _cfg("iv_vol60_cap25", risk="vol60", cap=0.25), _cfg("iv_vol60_cap20", risk="vol60", cap=0.20), _cfg("iv_vol60_cap15", risk="vol60", cap=0.15),
    _cfg("iv_atr14_cap25", risk="atr14", cap=0.25), _cfg("iv_atr14_cap20", risk="atr14", cap=0.20), _cfg("iv_atr14_cap15", risk="atr14", cap=0.15),
    # --- B. more names, equal weight (same 6-1 ranking)
    _cfg("eq_k15", k=15), _cfg("eq_k20", k=20), _cfg("eq_k30", k=30),
    # --- C. exclude names whose annualised vol60 is above a cap (replaced by the next-ranked name), equal weight, K=10
    _cfg("eq_k10_vcap80", vcap=0.80), _cfg("eq_k10_vcap100", vcap=1.00), _cfg("eq_k10_vcap120", vcap=1.20),
    # --- D. sector cap, equal weight, K=10
    _cfg("eq_k10_sec3", smax=3), _cfg("eq_k10_sec2", smax=2),
    # --- E. universe mixing: sleeves = [(universe mask, mask to exclude, k, share of the book)]
    _cfg("mix_m300_70_ring_30", sleeves=[("M300", None, 10, 0.7), ("M1000", "M300", 10, 0.3)]),
    _cfg("mix_m300_50_ring_50", sleeves=[("M300", None, 10, 0.5), ("M1000", "M300", 10, 0.5)]),
    _cfg("eq_k10_m1000", sleeves=[("M1000", None, 10, 1.0)]),
    # --- F. combinations of the building blocks (pre-registered, prior-driven: diversify + inverse-vol + trim the lottery tail)
    _cfg("k15_iv60_cap20", k=15, risk="vol60", cap=0.20),
    _cfg("k20_iv60_cap15", k=20, risk="vol60", cap=0.15),
    _cfg("k15_iv60_cap20_vcap100", k=15, risk="vol60", cap=0.20, vcap=1.00),
    _cfg("k10_iv60_cap20_vcap100_sec3", k=10, risk="vol60", cap=0.20, vcap=1.00, smax=3),
    _cfg("k20_eq_vcap100_sec3", k=20, vcap=1.00, smax=3),
]

# --- POST-HOC configs (post_hoc=True): appended AFTER the first run of the 26 pre-registered configs above (run-1 log: logs/r8_sizing_run1_prereg.log).
# They were added only because (i) in run 1 the vol cap was the only ingredient that ever scored, and the tightest tested cap (80 %) was the best
# single ingredient, i.e. the winner sat at the edge of the grid -> two tighter caps (70 %, 60 %) to see whether the plateau has an interior optimum;
# (ii) three decompositions of the run-1 winner to see which ingredient matters (mechanism).  Selection still uses the DEV-only objective; no HOLD
# column was used to choose them.  They are reported separately and flagged "post-hoc".
CONFIGS += [
    _cfg("ph_eq_k10_vcap70", vcap=0.70, post_hoc=True),
    _cfg("ph_eq_k10_vcap60", vcap=0.60, post_hoc=True),
    _cfg("ph_eq_k10_vcap100_sec3", vcap=1.00, smax=3, post_hoc=True),
    _cfg("ph_k10_iv60_cap20_vcap100", k=10, risk="vol60", cap=0.20, vcap=1.00, post_hoc=True),
    _cfg("ph_k10_iv60_cap20_vcap80_sec3", k=10, risk="vol60", cap=0.20, vcap=0.80, smax=3, post_hoc=True),
]


def _sector_codes(c, cols):
    """Integer sector code per column; unknown sector ('NA' / missing) -> a unique code per ticker (so it is never capped)."""
    sec = c["sector"].reindex(cols).fillna("NA").astype(str).values
    codes = pd.factorize(sec)[0].astype(np.int64)
    na = (sec == "NA")
    codes[na] = codes.max() + 1 + np.arange(int(na.sum()))
    return codes


def _water_fill(w, cap):
    """w >= 0 summing to 1; iteratively clip at `cap` and redistribute the excess pro rata over the unclipped names.  If the cap is infeasible
    (n*cap < 1) every name gets `cap` and the book is simply under-invested (gross < 1, never > 1)."""
    n = len(w)
    if cap * n <= 1.0 + 1e-12:
        return np.full(n, min(cap, 1.0 / n))
    w = w.copy()
    for _ in range(50):
        over = w > cap + 1e-13
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = w < cap - 1e-13
        w[free] += excess * w[free] / w[free].sum()
    return w


def _daily_book(c, mask, excl, k, risk, cap, vcap, smax):
    """Dense (T, N) float64 array of the unstaggered daily target books (row t uses data through close t only)."""
    S = c["mom_6_1"].values.astype(np.float64)
    ok = c[mask].values.copy()
    if excl is not None:
        ok &= ~c[excl].values
    ok &= np.isfinite(S)
    if vcap is not None:
        v60 = c["vol60"].values.astype(np.float64) * ANN
        ok &= np.isfinite(v60) & (v60 <= vcap)
    if risk is not None:
        rv = c[risk].values.astype(np.float64)
        ok &= np.isfinite(rv) & (rv > 0)
    else:
        rv = None
    codes = _sector_codes(c, c["mom_6_1"].columns) if smax is not None else None
    T, N = S.shape
    B = np.zeros((T, N))
    for t in range(T):
        idx = np.flatnonzero(ok[t])
        if idx.size == 0:
            continue
        order = idx[np.argsort(-S[t, idx], kind="stable")]
        if smax is not None:
            taken, sel = {}, []
            for j in order:
                s = codes[j]
                if taken.get(s, 0) < smax:
                    taken[s] = taken.get(s, 0) + 1
                    sel.append(j)
                    if len(sel) == k:
                        break
            sel = np.array(sel, dtype=np.int64)
        else:
            sel = order[:k]
        n = sel.size
        if n == 0:
            continue
        if rv is None:
            w = np.full(n, 1.0 / k)               # baseline convention: 1/k each, so a short book (< k names) is under-invested, not levered up
            if cap is not None:
                w = np.minimum(w, cap)
        else:
            w = 1.0 / rv[t, sel]                  # inverse-risk weights, normalised to 1 over the n selected names
            w = w / w.sum()
            if cap is not None:
                w = _water_fill(w, cap)
            w = w * min(1.0, n / k)               # fewer than k candidates: invest n/k of the book, like the baseline
        B[t, sel] = w
    return B


def weights(c, p):
    """Staggered book (average of the last p['h'] daily books) for config p.  Returns DataFrame (date x ticker)."""
    sleeves = p["sleeves"] or [("M300", None, p["k"], 1.0)]
    tot = None
    for mask, excl, k, share in sleeves:
        B = _daily_book(c, mask, excl, k, p["risk"], p["cap"], p["vcap"], p["smax"]) * share
        tot = B if tot is None else tot + B
    idx, cols = c["mom_6_1"].index, c["mom_6_1"].columns
    return pd.DataFrame(tot, index=idx, columns=cols).rolling(p["h"], min_periods=1).mean()
