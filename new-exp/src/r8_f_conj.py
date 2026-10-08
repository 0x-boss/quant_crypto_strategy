"""Round-8c family CONJ (written by the lead after reading the finished families' reports; the DESIGN is therefore post-hoc, the grid below is fixed
before it is run and every config is logged).

Why: the finished overlays (vol-target, dispersion half-cut, DD breaker) all cut the book in HIGH-VOL states, and high-vol months in a bull market are
exactly the >5 % months, so they pull the HOLD median under 5 %.  A momentum UNWIND, however, is high vol AND the strategy is below its recent peak;
a melt-up is high vol AND the strategy is at new highs.  The conjunction  [vol state high] AND [lagged equity >= x % below its 63-day peak]  should keep
the book full in melt-ups and cut it in unwinds.  Thresholds use ROLLING 504-day percentiles (not expanding: the expanding percentile drifted up
after 2021 in the regime/voltarget families and left the book half-cut for most of HOLD).

  flag_t  = (volpct_t >= q) & (dd63_t <= -x)             volpct = rolling-504d percentile rank of the vol measure (needs >= 252 obs, else 0.5)
  s_t     = cut if flag in the last `lock` days (inclusive) else 1       (lock = 1: pure state; 5: stay cut for a week after the last trigger)
vol measure: 'own20' = 20-day vol of the unscaled book's lagged returns (Barroso & Santa-Clara 2015); 'disp21' = cross-sectional IQR/1.349 of 21-day
returns in the PIT top-300 (Stivers & Sun 2010).  dd63 = lagged equity of the unscaled book / its 63-day max - 1 (CPPI-style, Grossman & Zhou 1993).
"""
import numpy as np
import pandas as pd

import r8_common as R
import r8_f_voltarget as VT

FAMILY = "conj"
ANN = np.sqrt(252)

CONFIGS = []
for src in ("own20", "disp21"):
    for q in (0.70, 0.80):
        for x in (0.05, 0.10):
            for cut in (0.5, 0.25):
                CONFIGS.append(dict(name=f"{src}_q{int(q*100)}_dd{int(x*100)}_c{int(cut*100)}", src=src, q=q, x=x, cut=cut, lock=1))
for q in (0.70, 0.80):                                         # a week of persistence for the own-vol version
    for x in (0.05, 0.10):
        CONFIGS.append(dict(name=f"own20_q{int(q*100)}_dd{int(x*100)}_c50_lock5", src="own20", q=q, x=x, cut=0.5, lock=5))
for src in ("own20", "disp21"):                                # references: vol state alone with the same ROLLING percentile (no drawdown condition)
    for q in (0.70, 0.80):
        CONFIGS.append(dict(name=f"{src}_q{int(q*100)}_novolonly_c50", src=src, q=q, x=0.0, cut=0.5, lock=1))
for q in (0.70, 0.80):                                         # references: drawdown alone (no vol condition)
    for x in (0.05, 0.10):
        CONFIGS.append(dict(name=f"ddonly_dd{int(x*100)}_q{int(q*100)}_c50", src="none", q=q, x=x, cut=0.5, lock=1))


def roll_pct(x, win=504, minp=252):
    """percentile rank of x[t] among x[t-win+1..t] (inclusive of t; past data only). 0.5 until minp observations exist."""
    a = x.values.astype(np.float64)
    out = np.full(len(a), 0.5)
    for t in range(len(a)):
        lo = max(0, t - win + 1)
        w = a[lo:t + 1]
        w = w[np.isfinite(w)]
        if len(w) >= minp and np.isfinite(a[t]):
            out[t] = (w <= a[t]).mean()
    return pd.Series(out, index=x.index)


def apply(c, W, p):
    r = R.run(c, W)
    rl = R.lagged(r)
    eq = (1.0 + rl.fillna(0.0)).cumprod()
    dd63 = eq / eq.rolling(63, min_periods=20).max() - 1.0
    if p["src"] == "own20":
        vp = roll_pct(rl.rolling(20).std() * ANN)
    elif p["src"] == "disp21":
        vp = roll_pct(VT.disp21(c))
    else:
        vp = pd.Series(1.0, index=W.index)
    vol_hi = (vp >= p["q"]) if p["src"] != "none" else pd.Series(True, index=W.index)
    dd_on = (dd63 <= -p["x"]) if p["x"] > 0 else pd.Series(True, index=W.index)
    flag = (vol_hi.reindex(W.index).fillna(False) & dd_on.reindex(W.index).fillna(False)).astype(float)
    flag = flag.rolling(p["lock"], min_periods=1).max() > 0
    s = pd.Series(np.where(flag.values, p["cut"], 1.0), index=W.index)
    return W.mul(s, axis=0)
