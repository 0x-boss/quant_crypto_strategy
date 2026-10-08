"""Round-8 family DEFENSIVE: what does the capital freed by a risk-reducing rule do?  (gross <= 1, long-only, spot, no leverage)

Three mechanisms, all built on the unscaled baseline book W0 = MOM_6_1 top-10 / PIT top-300 / stagger 5 (stocks only):
  (T) TRIGGER  W = s*W0 + (1-s)*D   with s[t] = min(1, sigma*/sigma_hat[t]),  sigma_hat = 20-day vol of R.lagged(R.run(c, W0)) (own-vol scale-down
      of the finished voltarget family, window 20d),  sigma* = DEV (2012-2021) MEDIAN of sigma_hat  ->  the book is scaled on ~50 % of DEV days.
      D = the destination of the freed fraction (1-s):  cash (the baseline treatment, earns 0), BIL, GLD, TLT, USMV, SPLV, IEF, 50/50 GLD+TLT, 50/50 USMV+GLD.
        mode "fixed": sigma* = 0.2501, the DEV median of sigma_hat, HARD-CODED (audit: python3 src/r8_f_defensive.py re-derives it from DEV data only).
                      Caveat (voltarget report): strategy vol in HOLD is ~1.7x DEV, so a fixed DEV level keeps the book de-risked most of HOLD.
        mode "exp"  : sigma*[t] = expanding median of sigma_hat[..t] (min 504 live days, before that s = 1): strictly causal, adapts to the vol regime.
  (M) STATIC MIX  W = (1-m)*W0 + m*D   (m = 10/20/30 %, rebalanced to target daily; the momentum sleeve is not scaled by any trigger).
  (V) INVERSE-VOL DYNAMIC MIX of the momentum sleeve and one defensive leg: w_def = vm/(vm+vd), vm = 20-day vol of R.lagged(sleeve returns),
      vd = 20-day close-to-close vol of the defensive leg through t; optionally capped (w_def <= 30 %).

Priors / literature (no parameter was searched):
  * Barroso & Santa-Clara (2015 JFE), Daniel & Moskowitz (2016 JFE): own-vol timing of momentum; window 20d and the DEV-median level are fixed by the task / the
    voltarget family, not tuned here.
  * Baur & Lucey (2010), Baur & McDermott (2010): gold as a safe haven in equity stress; Ilmanen (2003) / Campbell-Viceira: nominal Treasuries as flight-to-quality
    hedge (TLT = long, IEF = intermediate duration); Baker-Bradley-Wurgler (2011), Frazzini-Pedersen (2014): low-vol equity (USMV/SPLV) earns a high risk-adjusted
    return; BIL = 1-3 month T-bills (risk-free carry).  50/50 pairs are the equal-weight priors (no weight search).
  * Asness-Frazzini-Pedersen (2012) risk parity: weights proportional to 1/vol.  Mix sizes 10/20/30 % are the usual strategic-allocation round numbers
    (the "90/10, 80/20, 70/30" of the task).  The 30 % cap on the inverse-vol defensive weight is the 70/30 mix as an upper bound.

BIL CARRY WARNING: prices in c["O"] are total-return adjusted, so BIL earns its T-bill yield (DEV 2012-21 average ~0.7 %/yr, HOLD 2022-26 ~4 %/yr).  That is a
risk-free CARRY, not alpha.  The "cash" twin of every trigger config (trigF_cash / trigE_cash) is the carry-free reference: the difference between
trigF_bil and trigF_cash is the carry (plus 0.1 % fee on the BIL leg).  SGOV starts 2020-06 -> unusable in DEV, not registered.

ETF START DATES: SPLV 2011-05-05, USMV 2011-10-20, BIL/GLD/TLT/IEF/SHY since 2008.  Weight of an ETF is multiplied by avail[t] = (close[t] and open[t] exist), so
before listing the weight is 0 and the freed capital simply stays in cash.  (All ETFs exist during the whole DEV window 2012-2021.)

Look-ahead: sigma_hat uses R.lagged(R.run(c, W0)); expanding median uses data <= t; defensive vols use closes <= t; availability uses closes <= t.
All ETF columns are asserted to be empty in the stock-only baseline book before the defensive weights are written into them.
"""
import numpy as np
import pandas as pd

import r8_common as R

FAMILY = "defensive"
ANN = np.sqrt(252)
STAR_DEV_MEDIAN = 0.2501      # DEV (2012-2021) median of the 20d own vol of the unscaled baseline (= voltarget STAR["own20"][50]); audited in __main__
EXP_MIN = 504                 # expanding median needs 2 years of live sigma_hat

DEST = {                      # destination key -> [(ticker, share)]
    "cash":     [],
    "bil":      [("BIL", 1.0)],
    "gld":      [("GLD", 1.0)],
    "tlt":      [("TLT", 1.0)],
    "usmv":     [("USMV", 1.0)],
    "splv":     [("SPLV", 1.0)],
    "ief":      [("IEF", 1.0)],
    "gld_tlt":  [("GLD", 0.5), ("TLT", 0.5)],
    "usmv_gld": [("USMV", 0.5), ("GLD", 0.5)],
}

CONFIGS = []
# (T) trigger, sigma* = hard-coded DEV median, freed fraction to D                                           (9 configs)
for _d in ("cash", "bil", "gld", "tlt", "usmv", "splv", "ief", "gld_tlt", "usmv_gld"):
    CONFIGS.append(dict(name=f"trigF_{_d}", kind="trig", mode="fixed", star=STAR_DEV_MEDIAN, dest=_d))
# (T) trigger, sigma* = causal expanding median                                                               (5 configs)
for _d in ("cash", "gld", "tlt", "usmv", "gld_tlt"):
    CONFIGS.append(dict(name=f"trigE_{_d}", kind="trig", mode="exp", star=None, dest=_d))
# (M) fixed strategic mixes                                                                                   (13 configs)
for _d in ("bil", "gld", "tlt", "usmv", "splv", "ief", "gld_tlt"):
    CONFIGS.append(dict(name=f"mix80_{_d}", kind="mix", m=0.20, dest=_d))
for _m in (0.10, 0.30):
    for _d in ("gld", "tlt", "usmv"):
        CONFIGS.append(dict(name=f"mix{int(round((1 - _m) * 100))}_{_d}", kind="mix", m=_m, dest=_d))
# (V) inverse-vol dynamic mixes using lagged sleeve returns                                                   (6 configs)
for _d in ("gld_tlt", "usmv"):
    CONFIGS.append(dict(name=f"ivol_{_d}", kind="ivol", cap=None, dest=_d))
for _d in ("gld", "tlt", "usmv", "gld_tlt"):
    CONFIGS.append(dict(name=f"ivolcap30_{_d}", kind="ivol", cap=0.30, dest=_d))
# ---- POST-HOC configs (added AFTER the 33 pre-registered ones were run and seen; plateau / mechanism diagnostics only, never used to select; flagged posthoc=True) ----
CONFIGS += [
    dict(name="ph_trigE_usmv_gld", kind="trig", mode="exp", star=None, dest="usmv_gld", posthoc=True),            # winner with the causal expanding sigma*
    dict(name="ph_trigF65_usmv_gld", kind="trig", mode="fixed", star=0.3075, dest="usmv_gld", posthoc=True),      # sigma* = DEV 65th pct (voltarget STAR own20 q65)
    dict(name="ph_trigF80_usmv_gld", kind="trig", mode="fixed", star=0.4042, dest="usmv_gld", posthoc=True),      # sigma* = DEV 80th pct (voltarget STAR own20 q80)
    dict(name="ph_const17_usmv_gld", kind="mix", m=0.17, dest="usmv_gld", posthoc=True),                           # no timing: constant 17 % = 1 - avg DEV stock exposure (0.832) of the winner
]


def sleeve_vol20(c, W):
    """20-day annualised vol of the strategy's own returns known at the close of t (R.lagged)."""
    return R.lagged(R.run(c, W)).rolling(20).std() * ANN


def avail(c, tkr, idx):
    """1.0 on days when the ETF has a close and an open (known at the close of t), else 0.0 -> weight 0, freed capital stays in cash."""
    ok = c["C"][tkr].reindex(idx).notna() & c["O"][tkr].reindex(idx).notna()
    return ok.astype(float)


def leg_vol20(c, key, idx):
    """20-day annualised vol of the defensive leg (share-weighted daily return of its ETFs), close-to-close through t."""
    ret = None
    for tkr, sh in DEST[key]:
        r = c["C"][tkr].astype(np.float64).pct_change(fill_method=None).reindex(idx) * sh
        ret = r if ret is None else ret + r
    return ret.rolling(20).std() * ANN


def put_defensive(c, W2, key, amount):
    """Write amount[t]*share*avail[t] into the ETF columns of W2 (in place).  amount: Series (weight of the defensive leg on day t)."""
    for tkr, sh in DEST[key]:
        assert tkr in W2.columns, tkr
        assert float(W2[tkr].abs().sum()) == 0.0, f"{tkr} already has weight in the stock book"
        W2[tkr] = (amount * sh * avail(c, tkr, W2.index)).astype(np.float64)


def trig_scale(c, W, p):
    sig = sleeve_vol20(c, W)
    if p["mode"] == "fixed":
        star = p["star"]
    else:
        star = sig.where(sig > 0).expanding(min_periods=EXP_MIN).median()     # live (non-zero-vol) days only, causal
    s = (star / sig).replace([np.inf, -np.inf], np.nan).clip(upper=1.0).fillna(1.0)
    return s.clip(lower=0.0, upper=1.0)


def apply(c, W, p):
    kind = p["kind"]
    if kind == "trig":
        s = trig_scale(c, W, p)
        W2 = W.mul(s, axis=0)
        amount = 1.0 - s
    elif kind == "mix":
        W2 = W * (1.0 - p["m"])
        amount = pd.Series(p["m"], index=W.index)
    elif kind == "ivol":
        vm = sleeve_vol20(c, W)
        vd = leg_vol20(c, p["dest"], W.index)
        wd = (vm / (vm + vd)).where((vm > 0) & (vd > 0), 0.0).fillna(0.0)    # inverse-vol share of the defensive leg (0 if either vol is unknown)
        if p["cap"] is not None:
            wd = wd.clip(upper=p["cap"])
        W2 = W.mul(1.0 - wd, axis=0)
        amount = wd
    else:
        raise ValueError(kind)
    if DEST[p["dest"]]:
        W2 = W2.copy()
        put_defensive(c, W2, p["dest"], amount)
    return W2


if __name__ == "__main__":      # audit of the hard-coded DEV median sigma* and of ETF start dates (DEV window only for the statistic)
    c = R.load(["O", "C", "mom_6_1", "M300"])
    W = R.base_weights(c)
    sg = sleeve_vol20(c, W).loc["2012-01-01":"2021-12-31"]
    q = float(sg.median())
    print("DEV median of 20d own vol:", round(q, 4), "hard-coded", STAR_DEV_MEDIAN, "OK" if abs(q - STAR_DEV_MEDIAN) < 6e-5 else "MISMATCH")
    for t in ("BIL", "GLD", "TLT", "USMV", "SPLV", "IEF"):
        print(t, "first close", c["C"][t].first_valid_index().date())
