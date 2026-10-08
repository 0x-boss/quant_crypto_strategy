"""Round-8 family REGIME: factor-aware regime filters.  The momentum book W (row t, decided at the close of t) is multiplied by a stepwise
exposure flag e[t] in {1, 0.5, 0} (cash earns 0, no leverage).  Each flag is a small state machine WITH HYSTERESIS (separate enter / exit
thresholds) so it does not whipsaw.  The -53 % DEV drawdown (Feb-May 2021) happened while SPY was UP, so no flag is a plain SPY trend gate: the
triggers describe the state of the MOMENTUM FACTOR itself and of the names we hold.

Indicators and the idea/literature behind each (thresholds are round numbers from these priors, NOT searched; trigger frequencies were only
sanity-checked on DEV, never performance):
  wml   : daily return of a top-minus-bottom DECILE (30 vs 30 names) 6-1 momentum portfolio of the PIT top-300, ranks of day t-1 applied to the
          returns of day t (so the factor return of day t is known at the close of t).  Trailing 21 / 63-day compounded return = "factor
          momentum" (Ehsani & Linnainmaa 2022 JF; Gupta & Kelly 2019 RFS: factor returns are positively autocorrelated, so a losing factor keeps
          losing) and trailing vol of the factor (Barroso & Santa-Clara 2015 JFE; Daniel & Moskowitz 2016 JFE: momentum risk is predictable).
  breadth: share of the HELD names (W>0, weighted) / of the PIT top-300 whose close is above its SMA20 / SMA50.  A collapse of breadth inside the
          held basket = the winners are being sold in unison (momentum unwind); universe breadth = Zweig/Whaley-style breadth thrust washout.
  disp  : cross-sectional dispersion (IQR/1.349) of the 21-day returns of the top-300 (Stivers & Sun 2010 JFQA: high dispersion -> weak momentum),
          scored as its EXPANDING percentile (strictly causal, scale free).
  vix   : VIX level (20 = long-run median, 30 = stress regime, 40 = panic: Whaley 2000 "investor fear gauge" conventions) and the VIX/VIX3M term
          structure (backwardation, ratio > 1, = acute stress: Simon & Campasano 2014; Johnson 2017 JFQA).
  dm    : Daniel & Moskowitz (2016) bear-market-rebound rule, ex-ante version: SPY trailing 24-month (504d) return < 0 together with elevated
          market vol (126-day realized vol of SPY).  The paper's contemporaneous "rebound" dummy is not observable, so only the state part is used.
  shock : 5-day return of the basket of held names (weighted mean of ret5 over W>0): a fast momentum crash trigger (momentum crashes are weeks
          long, DM 2016 / Barroso-SC).
  vote/comp: several of the above combined (count of active flags -> exposure; or average of expanding-percentile stress scores).

LOOK-AHEAD: every series is a rolling / expanding function of data through the close of t (ranks use .shift(1) before being applied to day-t
returns; expanding percentiles use values <= t only), the flag multiplies row t of W.  The perturbation guard in the runner re-verifies.
"""
import bisect

import numpy as np
import pandas as pd

import r8_common as R

FAMILY = "regime"
ANN = np.sqrt(252)

# --------------------------------------------------------------------------------------------------------------------------------------------
# feature builders (all causal)
# --------------------------------------------------------------------------------------------------------------------------------------------
_CACHE = {"obj": None, "feat": {}}          # universe-level features, cached for the LAST dict `c` (kept referenced, so id() cannot be recycled)


def _store(c):
    if _CACHE["obj"] is not c:
        _CACHE["obj"], _CACHE["feat"] = c, {}
    return _CACHE["feat"]


def wml_daily(c):
    """Daily return of long top-decile / short bottom-decile 6-1 momentum inside the PIT top-300.  Ranks known at the close of t-1 are applied to
    the close-to-close returns of day t  ->  the series value at t is known at the close of t."""
    f = _store(c)
    if "wml" not in f:
        s = c["mom_6_1"].astype(np.float64).where(c["M300"])
        pr = s.rank(axis=1, pct=True)
        top = (pr > 0.9).shift(1, fill_value=False)
        bot = (pr <= 0.1).shift(1, fill_value=False)
        ret = c["ret1"].astype(np.float64)
        f["wml"] = ret.where(top).mean(axis=1) - ret.where(bot).mean(axis=1)
    return f["wml"]


def wml_trail(c, n):
    """Compounded trailing n-day return of the long-short factor, through t."""
    f = _store(c)
    k = f"wml_trail{n}"
    if k not in f:
        w = wml_daily(c).fillna(0.0)
        f[k] = (1 + w).rolling(n).apply(np.prod, raw=True) - 1
    return f[k]


def wml_vol(c, n):
    f = _store(c)
    k = f"wml_vol{n}"
    if k not in f:
        f[k] = wml_daily(c).rolling(n).std() * ANN
    return f[k]


def disp21(c):
    f = _store(c)
    if "disp21" not in f:
        x = c["ret21"].astype(np.float64).where(c["M300"])
        q = x.quantile([0.25, 0.75], axis=1).T
        f["disp21"] = (q[0.75] - q[0.25]) / 1.349
    return f["disp21"]


def univ_breadth(c, n):
    """Share of the PIT top-300 with close > SMA_n (n in 20, 50) at the close of t."""
    f = _store(c)
    k = f"ub{n}"
    if k not in f:
        m = c["M300"]
        sm = c[f"sma{n}"]                               # cache feature sma_n = C / SMA_n - 1  -> above the average  <=>  sma_n > 0
        f[k] = (sm > 0).where(m & sm.notna()).astype(float).where(m & sm.notna()).mean(axis=1)
    return f[k]


def held_breadth(c, W, n):
    """Weighted share of held names (weights W[t]) with close > SMA_n at the close of t."""
    above = (c[f"sma{n}"] > 0).reindex(W.index).astype(np.float64)       # sma_n = C / SMA_n - 1
    cols = W.columns
    g = W.sum(axis=1).replace(0, np.nan)
    return (W * above[cols]).sum(axis=1) / g


def held_ret5(c, W):
    """Weighted mean 5-day return (through t) of the held names."""
    r5 = c["ret5"].astype(np.float64).reindex(W.index)[W.columns].fillna(0.0)
    g = W.sum(axis=1).replace(0, np.nan)
    return (W * r5).sum(axis=1) / g


def spy_bear_vol(c):
    """(SPY 504-day return, SPY 126-day realised vol annualised) at the close of t."""
    f = _store(c)
    if "spy" not in f:
        spy = c["C"]["SPY"].astype(np.float64)
        f["spy"] = (spy / spy.shift(504) - 1, c["ret1"]["SPY"].astype(np.float64).rolling(126).std() * ANN)
    return f["spy"]


def exp_pct(x, minp=504):
    """Expanding percentile rank of x[t] among x[..t] (inclusive); NaN until `minp` observations.  Strictly causal."""
    xv = x.values.astype(np.float64)
    out = np.full(len(xv), np.nan)
    srt, n = [], 0
    for i, v in enumerate(xv):
        if np.isnan(v):
            continue
        bisect.insort(srt, v)
        n += 1
        if n >= minp:
            out[i] = bisect.bisect_right(srt, v) / n
    return pd.Series(out, index=x.index)


def stepwise(x, enter_half, exit_half, enter_zero=None, exit_zero=None):
    """Hysteresis state machine on a STRESS score x (higher = worse).  Level 0 = fully invested, 1 = half, 2 = flat.
    Up: x >= enter_zero -> 2 ; x >= enter_half -> 1.   Down: from 2 back to 1 when x < exit_zero (to 0 if also < exit_half); from 1 to 0 when
    x < exit_half.  NaN keeps the previous level.  Returns exposure multiplier in {1, .5, 0}."""
    xv = x.values.astype(np.float64)
    lvl = np.zeros(len(xv))
    cur = 0
    for i, v in enumerate(xv):
        if not np.isnan(v):
            if enter_zero is not None and v >= enter_zero:
                cur = 2
            elif cur < 1 and v >= enter_half:
                cur = 1
            elif cur == 2 and v < exit_zero:
                cur = 1 if v >= exit_half else 0
            elif cur == 1 and v < exit_half:
                cur = 0
        lvl[i] = cur
    return pd.Series(1.0 - 0.5 * lvl, index=x.index), pd.Series(lvl, index=x.index)


def flag_bool(x, enter, exit_):
    """Boolean hysteresis flag (on when x >= enter, off when x < exit_)."""
    e, lvl = stepwise(x, enter, exit_)
    return lvl > 0


def held_shock_z(c, W):
    """Held-basket 5-day return in units of its own trailing 252-day std (daily-sampled 5-day returns, through t)."""
    h5 = held_ret5(c, W)
    return -h5 / h5.rolling(252, min_periods=126).std()


# --------------------------------------------------------------------------------------------------------------------------------------------
# indicator table: name -> STRESS score (higher = worse), every entry a function of data through the close of t only
# --------------------------------------------------------------------------------------------------------------------------------------------
def _pct(c, key, fn):
    f = _store(c)
    if key not in f:
        f[key] = exp_pct(fn())
    return f[key]


def score(c, W, ind):
    v = c["vix"]
    if ind in ("wml21", "wml63"):
        return -wml_trail(c, int(ind[3:]))                                   # factor momentum: trailing LOSS of the long-short factor
    if ind in ("wmlvol21", "wmlvol63"):
        n = int(ind[6:])
        return _pct(c, "p_" + ind, lambda: wml_vol(c, n))                    # expanding percentile of the factor's own realised vol
    if ind == "disp":
        return _pct(c, "p_disp", lambda: disp21(c))                          # expanding percentile of cross-sectional dispersion
    if ind in ("hb50", "hb20"):
        return 1.0 - held_breadth(c, W, int(ind[2:]))                        # share of held names BELOW the average
    if ind in ("ub50", "ub20"):
        return 1.0 - univ_breadth(c, int(ind[2:]))
    if ind == "vix":
        return v["^VIX"].astype(np.float64)
    if ind == "vts":
        return (v["^VIX"] / v["^VIX3M"]).astype(np.float64)                  # > 1 = backwardation
    if ind == "dm":                                                          # SPY 24m return < 0  AND  elevated 126d market vol -> score = vol
        b, vol = spy_bear_vol(c)
        return vol.where(b < 0, 0.0).where(b.notna())
    if ind == "dmbear":
        return -spy_bear_vol(c)[0]
    if ind == "shock":
        return -held_ret5(c, W)
    if ind == "shockz":
        return held_shock_z(c, W)
    if ind == "vixp":
        return _pct(c, "p_vix", lambda: v["^VIX"].astype(np.float64))
    if ind == "vtsp":
        return _pct(c, "p_vts", lambda: (v["^VIX"] / v["^VIX3M"]).astype(np.float64))
    if ind == "wmlneg":                                                      # expanding percentile of the factor's trailing LOSS (for composites)
        return _pct(c, "p_wmlneg", lambda: -wml_trail(c, 21))
    if ind == "hbp":
        return exp_pct(1.0 - held_breadth(c, W, 50))
    raise KeyError(ind)


# named boolean flags (indicator, enter, exit) used by the vote configs
FLAGS = {
    "wml21": ("wml21", 0.08, 0.04),     # factor lost >= 8 % (about 1.2 sd of its 21d return) in 21 days; off again above -4 %
    "hb50": ("hb50", 0.65, 0.45),       # <= 35 % of held names above SMA50; off again when > 55 % are
    "shock": ("shock", 0.08, 0.03),     # held basket -8 % in 5 days; off above -3 %
    "wmlvol": ("wmlvol21", 0.85, 0.75), # factor vol in its top 15 % of history; off below the 75th percentile
    "disp": ("disp", 0.85, 0.75),       # cross-sectional dispersion in its top 15 % of history
    "vix": ("vix", 25.0, 20.0),         # VIX >= 25, off below 20
    "vts": ("vts", 1.00, 0.95),         # VIX / VIX3M >= 1 (backwardation), off below 0.95
    "disp90": ("disp", 0.90, 0.80),     # (post-hoc, used only by ph_or_disp_shock) dispersion in its top 10 % of history
}


def _flag(c, W, name):
    ind, enter, exit_ = FLAGS[name]
    return flag_bool(score(c, W, ind), enter, exit_).astype(float)


def exposure(c, W, p):
    kind = p["kind"]
    if kind == "single":
        x = score(c, W, p["ind"])
        e, _ = stepwise(x, p["eh"], p["xh"], p.get("ez"), p.get("xz"))
        return e
    if kind == "vote":                                   # count of active flags: >= n_half -> 0.5 ; >= n_zero -> 0
        n = sum(_flag(c, W, f) for f in p["flags"])
        e = pd.Series(1.0, index=W.index)
        e[n >= p["n_half"]] = 0.5
        if p.get("n_zero"):
            e[n >= p["n_zero"]] = 0.0
        return e
    if kind == "comp":                                   # average of expanding-percentile stress scores, then hysteresis
        x = sum(score(c, W, i) for i in p["inds"]) / len(p["inds"])
        e, _ = stepwise(x, p["eh"], p["xh"], p.get("ez"), p.get("xz"))
        return e
    raise KeyError(kind)


def apply(c, W, p):
    e = exposure(c, W, p).reindex(W.index).fillna(1.0)
    return W.mul(e, axis=0)


# --------------------------------------------------------------------------------------------------------------------------------------------
# PRE-REGISTERED GRID (written before any backtest was run).  eh/xh = enter/exit for the HALF (0.5) level, ez/xz = enter/exit for the ZERO level,
# in the unit of the stress score of the indicator (see score()).  Thresholds are round numbers from the priors in the module docstring.
# --------------------------------------------------------------------------------------------------------------------------------------------
def S(name, ind, eh, xh, ez=None, xz=None):
    return dict(name=name, kind="single", ind=ind, eh=eh, xh=xh, ez=ez, xz=xz)


CONFIGS = [
    # (i) momentum-factor state --------------------------------------------------------------------------------------------------------------
    S("wml21_step", "wml21", 0.08, 0.04, 0.15, 0.08),           # trailing 21d factor return < -8 % half, < -15 % flat  (factor momentum, E&L 2022)
    S("wml21_half", "wml21", 0.08, 0.04),
    S("wml63_step", "wml63", 0.10, 0.05, 0.20, 0.10),           # 63d factor return < -10 % half, < -20 % flat
    S("wml63_half", "wml63", 0.10, 0.05),
    S("wmlvol21_step", "wmlvol21", 0.80, 0.70, 0.95, 0.85),     # factor 21d vol above its expanding 80th / 95th percentile (Barroso-SC)
    S("wmlvol63_step", "wmlvol63", 0.80, 0.70, 0.95, 0.85),
    # (ii) breadth ----------------------------------------------------------------------------------------------------------------------------
    S("hb50_step", "hb50", 0.65, 0.45, 0.85, 0.65),             # held-name share above SMA50 <= 35 % half, <= 15 % flat; back when > 55 % / 35 %
    S("hb20_step", "hb20", 0.75, 0.55, 0.95, 0.75),             # held-name share above SMA20 <= 25 % half, <= 5 % flat
    S("ub50_step", "ub50", 0.60, 0.45, 0.80, 0.60),             # universe share above SMA50 <= 40 % half, <= 20 % flat
    S("ub20_step", "ub20", 0.70, 0.55, 0.85, 0.70),             # universe share above SMA20 <= 30 % half, <= 15 % flat
    # (iii) dispersion --------------------------------------------------------------------------------------------------------------------------
    S("disp_step", "disp", 0.80, 0.70, 0.95, 0.85),             # 21d-return dispersion above its expanding 80th / 95th pct (Stivers-Sun 2010)
    S("disp_half90", "disp", 0.90, 0.80),
    # (iv) VIX level / term structure ---------------------------------------------------------------------------------------------------------
    S("vix_a", "vix", 25.0, 20.0, 35.0, 28.0),                  # VIX >= 25 half, >= 35 flat (Whaley 2000 fear-gauge conventions)
    S("vix_b", "vix", 30.0, 22.0, 40.0, 30.0),
    S("vts_a", "vts", 1.00, 0.95, 1.10, 1.00),                  # VIX/VIX3M backwardation: >= 1.00 half, >= 1.10 flat
    S("vts_b", "vts", 0.95, 0.90),                              # earlier warning, half only
    # (v) Daniel-Moskowitz bear-market-rebound state (ex ante) -------------------------------------------------------------------------------
    S("dm_half", "dm", 0.20, 0.17),                             # SPY 24m return < 0 and 126d SPY vol >= 20 % (market ~16 % normal)
    S("dm_step", "dm", 0.20, 0.17, 0.30, 0.25),                 # ... flat when vol >= 30 %
    S("dm_bear", "dmbear", 0.0, -0.05),                         # SPY 24m return < 0 alone (half until it recovers above +5 %)
    # (vi) held-names 5-day return shock ------------------------------------------------------------------------------------------------------
    S("shock_step", "shock", 0.08, 0.03, 0.14, 0.06),           # held basket 5d return < -8 % half, < -14 % flat
    S("shock_half", "shock", 0.08, 0.03),
    S("shockz_step", "shockz", 2.0, 1.0, 3.0, 1.5),             # same in units of its own trailing 252d std
    # votes: several flags, exposure by number of active flags ---------------------------------------------------------------------------------
    dict(name="vote_fac3", kind="vote", flags=["wml21", "hb50", "shock"], n_half=2, n_zero=3),
    dict(name="vote_fac4", kind="vote", flags=["wml21", "hb50", "shock", "wmlvol"], n_half=2, n_zero=3),
    dict(name="vote_all7", kind="vote", flags=["wml21", "hb50", "shock", "wmlvol", "disp", "vix", "vts"], n_half=2, n_zero=4),
    dict(name="vote_mkt3", kind="vote", flags=["disp", "vix", "vts"], n_half=1, n_zero=2),
    dict(name="vote_fac3_half", kind="vote", flags=["wml21", "hb50", "shock"], n_half=2),
    # composites of expanding-percentile scores -----------------------------------------------------------------------------------------------
    dict(name="comp4_step", kind="comp", inds=["wmlvol21", "disp", "wmlneg", "hbp"], eh=0.70, xh=0.60, ez=0.85, xz=0.75),
    dict(name="comp4_half", kind="comp", inds=["wmlvol21", "disp", "wmlneg", "hbp"], eh=0.75, xh=0.65),
    dict(name="comp_mkt", kind="comp", inds=["vixp", "vtsp", "disp"], eh=0.75, xh=0.65, ez=0.90, xz=0.80),
    dict(name="comp_fac", kind="comp", inds=["wmlvol21", "wmlneg", "hbp"], eh=0.75, xh=0.65, ez=0.90, xz=0.80),
    # ---- POST-HOC additions (appended after the first 32 configs were run and ALL of them scored -inf on the DEV objective; names start with ph_).
    # Motivation (DEV-only diagnostic, see report): the deep "flat" level is followed by V-shaped rebounds (next-day Sharpe of the unscaled book is
    # >> 0.78 in the flat states) so the zero level destroys return; the half level of dispersion / vol / breadth states is where the information
    # is.  So: HALF-ONLY counterparts (no zero level) of the indicators whose half states looked informative on DEV, plus one union vote.
    S("ph_disp_half80", "disp", 0.80, 0.70),                    # = half level of disp_step, no flat level
    S("ph_disp_half85", "disp", 0.85, 0.75),                    # between ph_disp_half80 and disp_half90 (plateau check)
    S("ph_hb50_half", "hb50", 0.65, 0.45),                      # half level of hb50_step, no flat level
    S("ph_wmlvol21_half", "wmlvol21", 0.80, 0.70),
    S("ph_vts_a_half", "vts", 1.00, 0.95),
    dict(name="ph_comp_fac_half", kind="comp", inds=["wmlvol21", "wmlneg", "hbp"], eh=0.75, xh=0.65),
    dict(name="ph_vote_fac4_half", kind="vote", flags=["wml21", "hb50", "shock", "wmlvol"], n_half=2),
    dict(name="ph_or_disp_shock", kind="vote", flags=["disp90", "shock"], n_half=1, n_zero=2),   # either flag -> half, both -> flat
    # ---- POST-HOC batch 2: PLATEAU neighbours of ph_disp_half80 (the only family member that passed the DEV objective), added for the plateau report
    # only.  They test the enter percentile (0.75) and the hysteresis width (exit 0.75 = narrow, 0.60 = wide) around (0.80 / 0.70).
    S("ph_disp_half75", "disp", 0.75, 0.65),
    S("ph_disp_half80_x75", "disp", 0.80, 0.75),
    S("ph_disp_half80_x60", "disp", 0.80, 0.60),
]
assert len({p["name"] for p in CONFIGS}) == len(CONFIGS)


if __name__ == "__main__":                 # occupancy audit on DEV ONLY (share of days at half / flat) - a sanity check of trigger frequency
    c = R.load()
    W = R.base_weights(c)
    dev = slice("2012-01-01", "2021-12-31")
    for p in CONFIGS:
        e = exposure(c, W, p).loc[dev]
        print(f"{p['name']:18s} half {np.mean(e == 0.5)*100:5.1f}%  flat {np.mean(e == 0.0)*100:5.1f}%  switches/yr {(e.diff().abs() > 0).sum()/10:5.1f}")
