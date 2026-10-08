"""Round-8 family VOLTARGET: scale the momentum book by s[t] = min(1, sigma_star / sigma_hat[t])   (s <= 1 ALWAYS: no leverage, cash earns 0).

Literature / priors (no parameter was searched):
  * Barroso & Santa-Clara (2015, JFE) "Momentum has its moments": momentum's risk is predictable from its own trailing realised vol; scaling the
    WML by (target vol / trailing vol) lifts the Sharpe and removes most of the crash risk.  Their window is 6 months of daily returns; the task
    fixes the short windows 10/20/40/60 d (plus EWMA) because the crashes here are fast.
  * Daniel & Moskowitz (2016, JFE) "Momentum crashes": crashes happen in high-vol "panic" states -> a high-vol state is the trigger.
  * Moreira & Muir (2017, JF) "Volatility-managed portfolios": s = c/sigma_hat^2-ish, capped here at 1 (no leverage), c set by a level rule.
  * RiskMetrics (1996): EWMA vol with lambda = 0.94 (daily) and the slower 0.97.
  * Stivers & Sun (2010, JFQA): high cross-sectional return dispersion is followed by a low momentum premium -> dispersion trigger
    (robust: inter-quartile range / 1.349 of the 21-day returns of the PIT top-300 universe).
  * "held" = trailing vol of the equal-weight return of the names held TODAY (ex-ante portfolio vol of today's book over the last n days),
    a Barroso-style estimator that needs no backtest of the strategy.

sigma_star: no leverage means sigma_star must sit in the body of the sigma_hat distribution.  FIXED configs use the 50th / 65th / 80th percentile of the
DEV (2012-2021) distribution of sigma_hat of the unscaled baseline book (so the book is scaled on ~50 / 35 / 20 % of DEV days).  Those
percentiles were computed ONCE (see `python3 src/r8_f_voltarget.py` which re-derives them as an audit) and are hard-coded in STAR below: they use
no HOLD data.  Caveat: strategy vol in HOLD is ~1.7x the DEV vol (median sigma_hat 0.43-0.48 vs 0.25), so a fixed DEV level leaves the book
permanently de-risked in HOLD; the six EXPANDING configs set sigma_star[t] = q-th percentile of sigma_hat[1..t] (strictly causal, adapts to the vol regime).

Look-ahead: sigma_hat from the unscaled book uses R.lagged(R.run(c, W)); held/dispersion use closes through t only; expanding quantile uses data <= t.
"""
import numpy as np
import pandas as pd

import r8_common as R

FAMILY = "voltarget"
ANN = np.sqrt(252)

# DEV (2012-2021) percentiles of sigma_hat (annualised vol; dispersion in return units) of the UNSCALED baseline book, hard-coded, see __main__ audit.
STAR = {
    "own10":    {50: 0.2528, 65: 0.3012, 80: 0.4048},   # 10-day vol of lagged strategy returns
    "own20":    {50: 0.2501, 65: 0.3075, 80: 0.4042},   # 20-day
    "own40":    {50: 0.2529, 65: 0.3104, 80: 0.3858},   # 40-day
    "own60":    {50: 0.2597, 65: 0.3051, 80: 0.4358},   # 60-day
    "ewma94":   {50: 0.2537, 65: 0.3052, 80: 0.4104},   # RiskMetrics lambda 0.94
    "ewma97":   {50: 0.2627, 65: 0.2992, 80: 0.4383},   # lambda 0.97
    "held20":   {50: 0.2447, 65: 0.2932, 80: 0.3700},   # 20-day vol of the EW return of today's held names
    "disp21":   {50: 0.0564, 65: 0.0624, 80: 0.0711},   # cross-sectional IQR/1.349 of 21-day returns, PIT top-300
}

CONFIGS = []
for _k in ("own10", "own20", "own40", "own60", "ewma94", "ewma97"):          # strategy's own vol, DEV-quantile level   (18 configs)
    for _q in (50, 65, 80):
        CONFIGS.append(dict(name=f"{_k}_q{_q}", kind=_k, q=_q, star=STAR[_k][_q], mode="fixed"))
for _k in ("held20", "disp21"):                                               # held-names vol / dispersion triggers      (6 configs)
    for _q in (50, 65, 80):
        CONFIGS.append(dict(name=f"{_k}_q{_q}", kind=_k, q=_q, star=STAR[_k][_q], mode="fixed"))
for _k in ("own20", "own60"):                                                 # causal expanding-percentile level         (6 configs)
    for _q in (50, 65, 80):
        CONFIGS.append(dict(name=f"{_k}_exp{_q}", kind=_k, q=_q, star=None, mode="expanding"))

EXP_MIN = 504   # expanding percentile needs 2 years of sigma_hat history, before that s = 1


def held_vol(c, W, n):
    """Annualised vol over the last n days (through t) of the equal-weight(=W[t]/sum) return of the names held in row t."""
    ret = c["C"].pct_change(fill_method=None).reindex(W.index).astype(np.float64)
    cols = W.columns[(W.abs().sum() > 0).values]
    Rv = ret[cols].fillna(0.0).values
    Wv = W[cols].fillna(0.0).values
    gross = Wv.sum(1)
    out = np.full(len(Wv), np.nan)
    for t in range(n, len(Wv)):
        if gross[t] > 0:
            out[t] = (Rv[t - n + 1:t + 1] @ (Wv[t] / gross[t])).std(ddof=1) * ANN
    return pd.Series(out, index=W.index)


def disp21(c):
    """Cross-sectional robust std (IQR/1.349) of 21-day returns inside the PIT top-300 universe, known at the close of t."""
    x = c["ret21"].astype(np.float64).where(c["M300"])
    q = x.quantile([0.25, 0.75], axis=1).T
    return (q[0.75] - q[0.25]) / 1.349


def sigma_hat(c, W, kind):
    if kind.startswith("own"):
        rl = R.lagged(R.run(c, W))                       # returns known at the close of t
        return rl.rolling(int(kind[3:])).std() * ANN
    if kind.startswith("ewma"):
        rl = R.lagged(R.run(c, W))
        lam = {"ewma94": 0.94, "ewma97": 0.97}[kind]
        return rl.ewm(alpha=1 - lam, min_periods=20).std() * ANN
    if kind == "held20":
        return held_vol(c, W, 20)
    if kind == "disp21":
        return disp21(c)
    raise ValueError(kind)


def scale(c, W, p):
    """The exposure series s[t] in (0,1]."""
    sig = sigma_hat(c, W, p["kind"])
    if p["mode"] == "fixed":
        star = p["star"]
    else:
        star = sig.expanding(min_periods=EXP_MIN).quantile(p["q"] / 100.0)
    s = (star / sig).replace([np.inf, -np.inf], np.nan).clip(upper=1.0).fillna(1.0)
    return s.clip(lower=0.0, upper=1.0)


def apply(c, W, p):
    return W.mul(scale(c, W, p), axis=0)


if __name__ == "__main__":      # audit of the hard-coded STAR table (DEV window only)
    c = R.load(["O", "C", "mom_6_1", "M300", "ret21"])
    W = R.base_weights(c)
    for k in STAR:
        sg = sigma_hat(c, W, k).loc["2012-01-01":"2021-12-31"]
        qs = sg.quantile([0.5, 0.65, 0.8]).values
        print(k, np.round(qs, 4), "table", list(STAR[k].values()), "OK" if np.allclose(qs, list(STAR[k].values()), atol=6e-5) else "MISMATCH")
