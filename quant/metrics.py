"""Performance statistics for daily return series (crypto trades 365 days / year)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ANN = 365


def equity(r: pd.Series) -> pd.Series:
    return (1 + r.fillna(0)).cumprod()


def max_drawdown(r: pd.Series) -> float:
    eq = equity(r)
    return float((eq / eq.cummax() - 1).min())


def drawdown_series(r: pd.Series) -> pd.Series:
    eq = equity(r)
    return eq / eq.cummax() - 1


def summary(r: pd.Series, rf: float = 0.0) -> dict:
    r = r.dropna()
    n = len(r)
    if n < 30:
        return {}
    yrs = n / ANN
    eq = (1 + r).prod()
    cagr = eq ** (1 / yrs) - 1 if eq > 0 else -1.0
    vol = r.std(ddof=1) * np.sqrt(ANN)
    sharpe = (r.mean() - rf / ANN) / r.std(ddof=1) * np.sqrt(ANN) if r.std() > 0 else np.nan
    dn = r[r < 0].std(ddof=1) * np.sqrt(ANN)
    sortino = r.mean() * ANN / dn if dn and dn > 0 else np.nan
    mdd = max_drawdown(r)
    dd = drawdown_series(r)
    ulcer = float(np.sqrt((dd ** 2).mean()))
    return dict(cagr=cagr, vol=vol, sharpe=sharpe, sortino=sortino, maxdd=mdd,
                calmar=cagr / abs(mdd) if mdd < 0 else np.nan, ulcer=ulcer,
                skew=float(stats.skew(r)), kurt=float(stats.kurtosis(r)),
                hit=float((r > 0).mean()), worst_day=float(r.min()), best_day=float(r.max()),
                years=yrs, total=eq - 1)


def yearly(r: pd.Series) -> pd.DataFrame:
    out = {}
    for y, g in r.groupby(r.index.year):
        s = summary(g) if len(g) > 60 else {}
        if s:
            out[y] = dict(ret=s["total"], sharpe=s["sharpe"], maxdd=s["maxdd"], vol=s["vol"])
        else:
            out[y] = dict(ret=(1 + g).prod() - 1, sharpe=np.nan, maxdd=max_drawdown(g), vol=g.std() * np.sqrt(ANN))
    return pd.DataFrame(out).T


def probabilistic_sharpe(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR (Bailey & Lopez de Prado): P(true Sharpe > benchmark) accounting for skew / kurtosis."""
    r = r.dropna()
    n = len(r)
    sr = r.mean() / r.std(ddof=1)  # per-period
    g3, g4 = stats.skew(r), stats.kurtosis(r, fisher=False)
    sb = sr_benchmark / np.sqrt(ANN)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return float(stats.norm.cdf((sr - sb) * np.sqrt(n - 1) / denom))


def deflated_sharpe(r: pd.Series, n_trials: int, trial_sr_var: float | None = None) -> float:
    """Deflated Sharpe ratio: PSR against the expected max Sharpe of ``n_trials`` independent trials.

    ``trial_sr_var`` is the variance of annualised Sharpes across trials (default: 1.0, a generous
    value for correlated crypto-strategy variants).
    """
    if n_trials <= 1:
        return probabilistic_sharpe(r, 0.0)
    v = 1.0 if trial_sr_var is None else trial_sr_var
    emc = 0.5772156649
    z = stats.norm.ppf
    sr0 = np.sqrt(v) * ((1 - emc) * z(1 - 1 / n_trials) + emc * z(1 - 1 / (n_trials * np.e)))
    return probabilistic_sharpe(r, sr0)


def block_bootstrap_sharpe(r: pd.Series, n: int = 2000, block: int = 20, seed: int = 0) -> np.ndarray:
    """Stationary-ish block bootstrap of the annualised Sharpe."""
    rng = np.random.default_rng(seed)
    x = r.dropna().values
    T = len(x)
    nb = int(np.ceil(T / block))
    out = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, T - block, nb)
        s = np.concatenate([x[j:j + block] for j in starts])[:T]
        out[i] = s.mean() / s.std(ddof=1) * np.sqrt(ANN)
    return out
