"""Round-8 family SIGNAL: better stock SELECTION inside the same PIT top-300 universe (M300), top-k equal weight, h-day stagger.

Everything is built from the cache frames in `c` only (no own-return feedback, nothing shifted into the future): row t uses data through the
close of t.  The book is built from scratch with weights(c, p) (same construction as R.base_weights: top-k by score, 1/k each, average of the
last h daily books), so the baseline itself is config `base_mom61` (sanity row: must reproduce the baseline exactly).

Config keys
  score : which cross-sectional ranking score picks the top-k (see _score)
  pre   : list of hard filters applied BEFORE ranking (a failing name is excluded and the next-ranked name replaces it)
  cash  : list of filters applied AFTER the top-k pick (a failing name's 1/k slot stays in cash)
  k, h  : number of names, stagger length (average of the last h daily books)
Priors (no search): k=10,h=5 is the baseline; k in {8,15,20} / h in {3,10} are the pre-registered neighbours (breadth vs concentration, turnover vs lag).
"""
import numpy as np
import pandas as pd
import r8_common as R

FAMILY = "signal"

# ---- filters (all evaluated on the close of day t) -------------------------------------------------------------------------------------------
# trend      : close > SMA50 and close > SMA200            (Faber 2007 / Antonacci style trend-quality, c['sma50'], c['sma200'] are close/SMA-1)
# h52_80     : close >= 80 % of the 52-week high            (George & Hwang 2004 52-week-high momentum)
# abs        : own 6-1 return > 0                           (Antonacci 2014 dual / absolute momentum; slot stays in cash otherwise)
# ext25/ext40: close not more than 25 % / 40 % above SMA50 (avoid over-extended parabolic names; ~1.5-2.5 sigma of a monthly move of a 60 % vol stock)
PRE = {
    "trend": lambda c: (c["sma50"] > 0) & (c["sma200"] > 0),
    "h52_80": lambda c: c["high52"] > 0.8,
    "ext25": lambda c: c["sma50"] <= 0.25,
    "ext40": lambda c: c["sma50"] <= 0.40,
}
CASH = {
    "abs": lambda c: c["mom_6_1"] > 0,
}


def _pct(df, m):
    """cross-sectional percentile rank (0..1) among the PIT universe names, row by row (uses row t only)."""
    return df.where(m).rank(axis=1, pct=True)


def _fip(c):
    """Frog-in-the-pan smoothness (Da, Gurun & Warachka 2014): share of POSITIVE daily returns over the same 105-day formation window as 6-1
    momentum (days t-125 .. t-21; the shift(21) skips the most recent month).  Higher = smoother, more 'continuous' information arrival."""
    r = c["ret1"]
    pos = (r > 0).astype(np.float32).where(r.notna())
    return pos.shift(21).rolling(105, min_periods=80).mean()


def _score(c, kind, m):
    mom = c["mom_6_1"]
    vol = c["vol60"].where(c["vol60"] > 0)
    if kind == "mom61":                         # baseline: Jegadeesh & Titman 1993
        s = mom
    elif kind == "riskadj":                     # return / risk momentum (Barroso & Santa-Clara 2015; Sharpe-style momentum): 6-1 return over 60d vol
        s = mom / vol
    elif kind == "resmom":                      # residual (market-neutral) 12-1 momentum: Blitz, Huij & Martens 2011
        s = c["res_mom_12_1"]
    elif kind == "resadj":                      # residual momentum scaled by residual vol (Blitz et al. divide by the residual std)
        s = c["res_mom_12_1"] / c["idio_vol60"].where(c["idio_vol60"] > 0)
    elif kind == "b_m61_m121":                  # rank blend of two horizons (Novy-Marx 2012 / Asness: intermediate-horizon blends)
        s = (_pct(mom, m) + _pct(c["mom_12_1"], m)) / 2
    elif kind == "b_m61_res":                   # raw 6-1 + residual 12-1 rank blend
        s = (_pct(mom, m) + _pct(c["res_mom_12_1"], m)) / 2
    elif kind == "b_radj_resadj":               # risk-adjusted 6-1 + risk-adjusted residual 12-1
        s = (_pct(mom / vol, m) + _pct(c["res_mom_12_1"] / c["idio_vol60"].where(c["idio_vol60"] > 0), m)) / 2
    elif kind == "b_m61_fip":                   # 6-1 + smoothness (FIP) rank blend
        s = (_pct(mom, m) + _pct(_fip(c), m)) / 2
    elif kind == "b_radj_fip":
        s = (_pct(mom / vol, m) + _pct(_fip(c), m)) / 2
    elif kind == "fip30":                       # Da et al.: among the top-30 (top decile of 300) momentum names take the SMOOTHEST k
        s0 = mom.where(m)
        cand = s0.rank(axis=1, ascending=False, method="first") <= 30
        s = (_fip(c) + 1e-4 * _pct(mom, m)).where(cand)   # tiny momentum term only breaks ties
    else:
        raise ValueError(kind)
    return s.where(m)


def _cfg(name, score, k=10, h=5, pre=(), cash=()):
    return dict(name=name, score=score, k=k, h=h, pre=list(pre), cash=list(cash))


CONFIGS = [
    # --- block A: one recipe at a time, k=10, h=5 -------------------------------------------------------------------------------------------
    _cfg("base_mom61", "mom61"),                                   # sanity: must equal the baseline
    _cfg("riskadj", "riskadj"),
    _cfg("resmom", "resmom"),
    _cfg("resmom_adj", "resadj"),
    _cfg("blend_m61_m121", "b_m61_m121"),
    _cfg("blend_m61_res", "b_m61_res"),
    _cfg("blend_radj_resadj", "b_radj_resadj"),
    _cfg("m61_trend", "mom61", pre=["trend"]),
    _cfg("m61_h52", "mom61", pre=["h52_80"]),
    _cfg("m61_abscash", "mom61", cash=["abs"]),
    _cfg("m61_fip30", "fip30"),
    _cfg("blend_m61_fip", "b_m61_fip"),
    _cfg("m61_ext25", "mom61", pre=["ext25"]),
    _cfg("m61_ext40", "mom61", pre=["ext40"]),
    _cfg("riskadj_trend", "riskadj", pre=["trend"]),
    _cfg("riskadj_h52", "riskadj", pre=["h52_80"]),
    _cfg("riskadj_ext25", "riskadj", pre=["ext25"]),
    _cfg("blend_radj_fip", "b_radj_fip"),
    _cfg("riskadj_trend_h52", "riskadj", pre=["trend", "h52_80"]),
    _cfg("blend_m61_m121_trend", "b_m61_m121", pre=["trend"]),
    _cfg("blend_radj_resadj_trend", "b_radj_resadj", pre=["trend"]),
    # --- block B: breadth (k) and stagger (h) around the risk-adjusted recipe -------------------------------------------------------------------
    _cfg("riskadj_k8", "riskadj", k=8),
    _cfg("riskadj_k15", "riskadj", k=15),
    _cfg("riskadj_k20", "riskadj", k=20),
    _cfg("riskadj_h3", "riskadj", h=3),
    _cfg("riskadj_h10", "riskadj", h=10),
    _cfg("riskadj_k15_h10", "riskadj", k=15, h=10),
    _cfg("riskadj_k20_h10", "riskadj", k=20, h=10),
    # --- block C: breadth / stagger around risk-adjusted + trend ------------------------------------------------------------------------------
    _cfg("riskadj_trend_k15", "riskadj", k=15, pre=["trend"]),
    _cfg("riskadj_trend_k20", "riskadj", k=20, pre=["trend"]),
    _cfg("riskadj_trend_h10", "riskadj", h=10, pre=["trend"]),
    _cfg("riskadj_trend_k15_h10", "riskadj", k=15, h=10, pre=["trend"]),
    # --- block D: baseline signal with wider book (does breadth alone cut the drawdown?) ----------------------------------------------------
    _cfg("m61_k15", "mom61", k=15),
    _cfg("m61_k20", "mom61", k=20),
]


def weights(c, p):
    m = c["M300"]
    s = _score(c, p["score"], m)
    for f in p["pre"]:
        s = s.where(PRE[f](c))
    rank = s.rank(axis=1, ascending=False, method="first")
    sleeve = ((rank <= p["k"]) & s.notna()).astype(float) / p["k"]
    for f in p["cash"]:
        sleeve = sleeve.where(CASH[f](c), 0.0)           # slot stays in cash (gross < 1)
    return sleeve.rolling(p["h"], min_periods=1).mean()
