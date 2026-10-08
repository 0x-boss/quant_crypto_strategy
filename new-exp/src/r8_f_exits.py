"""Round-8 family EXITS: position-level exits implemented as filters on the target weights of the 6-1 momentum sleeve.

For every (date t, name i) a stop condition cond[t,i] is computed from data through the CLOSE of t only (C, atr14, sma20, sma50, ret5 on row t,
rolling windows ending at t).  A name is "blocked" on row t when cond[t,i] is true (state re-entry) or, with a lock of N days, when it was stopped
out (cond true while the unfiltered book would hold it, W0[t,i] > 0) on any of the last N rows t-N+1..t (a stopped name re-enters only after N days).
Two replacement styles:
  cash : W = W0 * (1 - blocked)            the freed weight sits in cash (return 0, no interest), nothing is bought instead
  fill : the five daily books that make up the 5-day average are RE-RANKED today over the unblocked names only, i.e. a stopped name is replaced by
         the next-best 6-1 momentum name of the PIT top-300 universe (book j = top-10 of mom_6_1 shifted j rows, restricted to names unblocked at t).
         With no name blocked this reproduces r8_common.base_weights exactly (config sanity_replica_fill).
Gross exposure <= 1 by construction (cash: only removes weight; fill: each of the five books has <= k names at 1/k).  Everything on row t uses data through t.

Stop rules (parameters are priors, none is searched):
  atr{m}   : close < max(close, 20d) - m*ATR14   (chandelier / trailing ATR stop, LeBeau; Kaufman 1995 'Smarter Trading').  m = 2 tight, 3 the textbook
             default (Le Beau uses 3), 4 wide.  c['atr14'] is ATR/close so the stop is C < rollmax20 * (1 - m*atr14).  Held names: median ATR 4 %, so
             m = 2/3/4 means a stop ~8/12/16 % under the 20-day closing high.  20 days = one trading month (Donchian / Turtle exit horizon, Dennis).
  sma20/50 : close < SMA20 / SMA50 (Brock-Lakonishok-LeBaron 1992 moving-average rules; Faber 2007 trend exit; 20 d fast, 50 d medium).
  shock{X} : 5-day return < -X * ATR14 * sqrt(5).  ATR is ~1.3 x the daily close-to-close sigma, so X = 1.0 / 1.5 / 2.0 ~ 1.3 / 2 / 2.6 sigma weekly
             losses (the usual 2-sigma outlier convention); note the short-term-reversal literature (Jegadeesh 1990, Lehmann 1990) warns that
             selling last week's losers fights the 1-week reversal - this is exactly what the family tests.
Replacement style: Han-Zhou-Zhu 2016 'Taming momentum crashes: a simple stop-loss strategy' and Kaminski-Lo 2014 'When do stop-loss rules stop losses?' study
  stops with the proceeds in cash; re-ranking and refilling is the usual practitioner variant that keeps the book invested.
Re-entry lock N = 5 / 10 trading days: 5 = one week = the stagger length of the book, 10 = two weeks (cool-off period against whipsaw; Kaminski-Lo: the
  value of a stop depends on whether the stopped asset re-trends or mean-reverts, the lock lets the data answer).
"""
import numpy as np
import pandas as pd

import r8_common as R

FAMILY = "exits"
K, H, SIG, MASK = 10, 5, "mom_6_1", "M300"      # the baseline MOM_6_1_k10 sleeve (top-10, stagger 5) - NOT varied in this family


def _cfg(style, stop, par, lock=0, name=None):
    nm = name or f"{style}_{stop}{'' if par is None else par}_{'state' if lock == 0 else f'lock{lock}'}"
    return dict(name=nm, style=style, stop=stop, par=par, lock=lock)


_STOPS = [("atr", 2), ("atr", 3), ("atr", 4), ("sma", 20), ("sma", 50), ("shock", 1.0), ("shock", 1.5), ("shock", 2.0)]
CONFIGS = (
    [_cfg("fill", None, None, name="sanity_replica_fill")]                                        # must equal the baseline (score -inf by construction)
    + [_cfg(style, s, par) for style in ("cash", "fill") for s, par in _STOPS]                    # 16: 8 stop rules x 2 replacement styles, state re-entry
    + [_cfg(style, s, par, lock) for style in ("cash", "fill") for s, par in (("atr", 3), ("sma", 20), ("shock", 1.5)) for lock in (5, 10)]  # 12 lock variants
)


def _cond(c, stop, par):
    """Boolean frame, True where the stop condition holds at the close of t (NaN inputs -> False)."""
    if stop is None:
        return pd.DataFrame(False, index=c["C"].index, columns=c["C"].columns)
    if stop == "atr":
        C = c["C"].astype(np.float64)
        hh = C.rolling(20, min_periods=20).max()                   # trailing 20-day closing high, includes today
        return C < hh * (1.0 - par * c["atr14"].astype(np.float64))
    if stop == "sma":
        return c[f"sma{par}"].astype(np.float64) < 0.0            # sma{n} feature = C / SMA_n - 1
    if stop == "shock":
        return c["ret5"].astype(np.float64) < -par * c["atr14"].astype(np.float64) * np.sqrt(5.0)
    raise ValueError(stop)


def _blocked(cond, W0, lock):
    if lock <= 0:
        return cond
    stopped = cond & (W0.reindex_like(cond).fillna(0.0) > 0)       # stop event of a name the unfiltered book would hold
    recent = stopped.astype(np.float32).rolling(lock, min_periods=1).max() > 0     # stopped within the last `lock` rows (incl. today)
    return cond | recent


def _fill_weights(c, block, cols):
    """Average of the last H daily books, each re-ranked today over the names not blocked today (top-K, equal weight)."""
    s = c[SIG][cols].where(c[MASK][cols])
    ok = ~block[cols]
    acc = np.zeros(s.shape, dtype=np.float64)       # float64: lib.weights_backtest asserts gross <= 1 + 1e-9
    for j in range(H):
        sj = s.shift(j).where(ok)
        rank = sj.rank(axis=1, ascending=False, method="first")
        acc += (((rank <= K) & sj.notna()).to_numpy().astype(np.float64)) / K
    return pd.DataFrame(acc / H, index=s.index, columns=cols)


def weights(c, p):
    W0 = R.base_weights(c, k=K, h=H, sig=SIG, mask=MASK)
    cond = _cond(c, p["stop"], p["par"])
    block = _blocked(cond, W0, p["lock"]).reindex(index=W0.index, columns=W0.columns).fillna(False).astype(bool)
    if p["style"] == "cash":
        return W0.mul((~block).astype(np.float32))
    cols = c[MASK].columns[c[MASK].any().to_numpy()]               # names that are ever in the PIT top-300 (the rest can never be held)
    return _fill_weights(c, block, cols)
