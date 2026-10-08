"""Round-8 family DDBREAKER: drawdown circuit breakers on the strategy's OWN lagged equity curve.

Mechanism: eq_t = cumprod(1 + lagged(r_unscaled))  (r_unscaled = open-to-open net return of the UNSCALED baseline book, shifted one day
so that only returns known at the close of t enter eq_t).  The breaker watches dd_t = eq_t / max(eq over the last `win` days since the last
re-entry) - 1.  If dd_t <= -thr the book is cut to `cut` x (0.5 or 0.0) of its weights; a re-risk rule returns it to 100 %.
  re = 'high'   : eq_t makes a new `win`-day high                                   (rule a)
  re = 'half'   : half of the peak-to-trough drawdown (peak = rolling max at the trigger, trough = running min since) is recovered (b)
  re = 'd<N>'   : N trading days after the cut (N = 5/10/21)                         (c)
  re = 'mkt'    : market filter benign = SPY > SMA50 and VIX < VIX3M (contango), after >= 5 days in the cut state    (d)
  re = 'ladder' : asymmetric re-entry: half recovery -> half-way between cut and 1, new `win`-day high -> 100 %; new trough -> back to cut
After ANY re-entry the high-water mark is reset to the re-entry date (the drawdown clock restarts), otherwise a half-recovered book would
re-trip the same threshold on the next day (ping-pong).  The trigger uses the unscaled baseline equity (not the scaled one), so the state
machine has no feedback loop and the cut is a pure function of past, known baseline returns.
LOOK-AHEAD: eq uses R.lagged(r); SPY/VIX are same-day closes; every rolling window ends at t.  All of it is recomputed from `c` inside
apply() so the perturbation guard sees it.
"""
import hashlib

import numpy as np
import pandas as pd

import r8_common as R

FAMILY = "ddbreaker"

# --- priors (no searching) ---------------------------------------------------------------------------------------------------------------
# thr   : 10/15/20 % drawdown triggers - practitioner stop-loss/CPPI bands (Grossman & Zhou 1993 drawdown control; Kaminski & Lo 2014 "When do
#         stop-loss rules stop losses?" use 10-20 % bands).
# win   : 126 / 252 d peak look-back = 6 / 12 months (the horizon of the momentum signal itself / the usual 12-month trend window).
# cut   : 50 % (half-risk) or 0 % (go to cash); Daniel & Moskowitz (2016) show momentum crashes are followed by violent rebounds -> expect whipsaw.
# half  : recover half of the drawdown (classic "50 % retracement" re-entry); d5/d10/d21 = 1 week / 2 weeks / 1 month; mkt = Simon & Campasano
#         (2014) VIX term-structure contango + trend filter; ladder = asymmetric (slow) re-entry.
CONFIGS = []
for cut in (0.5, 0.0):
    for win in (126, 252):
        for thr in (0.10, 0.15, 0.20):
            for re in ("half", "d10"):
                CONFIGS.append(dict(name=f"w{win}_t{int(thr*100)}_c{int(cut*100)}_{re}", win=win, thr=thr, cut=cut, re=re))
for cut in (0.5, 0.0):                                     # centre point (win 252, thr 15 %): the other re-risk rules
    for re in ("high", "d5", "d21", "mkt"):
        CONFIGS.append(dict(name=f"w252_t15_c{int(cut*100)}_{re}", win=252, thr=0.15, cut=cut, re=re))
CONFIGS.append(dict(name="w252_t15_c0_ladder", win=252, thr=0.15, cut=0.0, re="ladder"))
# --- POST-HOC (added after seeing the 33 pre-registered results; plateau probes around the DEV winner w126_t10_c50_d10, NOT used for selection) ---
CONFIGS += [
    dict(name="PH_w126_t10_c50_d5", win=126, thr=0.10, cut=0.5, re="d5"),        # neighbour of d10 on the re-risk axis
    dict(name="PH_w126_t10_c50_d21", win=126, thr=0.10, cut=0.5, re="d21"),      # neighbour of d10 on the re-risk axis
    dict(name="PH_w126_t10_c75_d10", win=126, thr=0.10, cut=0.75, re="d10"),     # cut-depth neighbour (25 % cut)
    dict(name="PH_w126_t125_c50_d10", win=126, thr=0.125, cut=0.5, re="d10"),    # threshold mid-point between the 10 % and 15 % grid points
    dict(name="PH_w126_t10_c25_d10", win=126, thr=0.10, cut=0.25, re="d10"),     # cut-depth neighbour (75 % cut)
]

_CACHE = {}


def _base_equity(c, W):
    """eq of the unscaled baseline book from lagged returns (exact-content cache: a perturbed c / W hashes differently)."""
    h = hashlib.blake2b(digest_size=16)
    h.update(np.ascontiguousarray(W.values).tobytes())
    h.update(np.ascontiguousarray(c["O"].values).tobytes())
    key = h.hexdigest()
    if key not in _CACHE:
        _CACHE.clear()
        r = R.run(c, W)
        rl = R.lagged(r).fillna(0.0)                       # return known at the close of t (indexed t-1)
        _CACHE[key] = (1.0 + rl).cumprod()
    return _CACHE[key]


def breaker_exposure(eq, benign, win, thr, cut, re, n_days=0):
    """Sequential state machine -> exposure s_t in [0,1] for each row t, decided from eq[:t+1] only.  eq: Series, benign: bool array."""
    e = eq.values.astype(np.float64)
    T = len(e)
    s = np.ones(T)
    on = True
    t_r = 0                    # last re-entry index (high-water mark reset)
    t_cut = peak = trough = 0.0
    stage = 0                  # ladder stage within the cut state: 0 = cut, 1 = half-way
    mid = 0.5 * (1.0 + cut)
    for t in range(T):
        lo = max(t - win + 1, 0)
        if on:
            ref = e[max(lo, t_r):t + 1].max()
            if e[t] / ref - 1.0 <= -thr:
                on, t_cut, peak, trough, stage = False, t, ref, e[t], 0
                s[t] = cut
            continue
        trough = min(trough, e[t])
        wmax = e[lo:t + 1].max()
        lvl = cut
        back = False
        if re == "high":
            back = e[t] >= wmax
        elif re == "half":
            back = e[t] >= trough + 0.5 * (peak - trough)
        elif re[0] == "d":
            back = (t - t_cut) >= int(re[1:])
        elif re == "mkt":
            back = (t - t_cut) >= 5 and bool(benign[t])
        elif re == "ladder":
            if stage == 0 and e[t] >= trough + 0.5 * (peak - trough):
                stage = 1
            elif stage == 1:
                if e[t] >= wmax:
                    back = True
                elif e[t] <= trough:
                    stage = 0
            lvl = mid if stage == 1 else cut
        if back:
            on, t_r = True, t
            s[t] = 1.0
        else:
            s[t] = lvl
    return pd.Series(s, index=eq.index)


def apply(c, W, p):
    eq = _base_equity(c, W)
    spy = c["C"]["SPY"].astype(np.float64)
    benign = ((spy > spy.rolling(50).mean()) & (c["vix"]["^VIX"] < c["vix"]["^VIX3M"])).reindex(eq.index).fillna(False).values
    s = breaker_exposure(eq, benign, p["win"], p["thr"], p["cut"], p["re"])
    return W.mul(s.reindex(W.index).fillna(1.0), axis=0)
