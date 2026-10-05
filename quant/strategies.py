"""Strategy building blocks: trend signals, inverse-vol sizing, portfolio volatility targeting."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*keyword-only.*")

ANN = 365


def ewma_vol(P: pd.DataFrame, span: int = 30) -> pd.DataFrame:
    """Annualised EWMA volatility of daily log returns (information up to and including day t)."""
    lr = np.log(P).diff()
    return lr.ewm(span=span, min_periods=max(10, span // 2)).std() * np.sqrt(ANN)


def trend_signal(P: pd.DataFrame, lookbacks=(20, 50, 100, 200), kind: str = "sma") -> pd.DataFrame:
    """Ensemble trend score in [0, 1]: fraction of lookbacks voting 'up'.

    kind='sma'  : price above its L-day simple moving average
    kind='mom'  : L-day return > 0
    """
    votes = []
    for L in lookbacks:
        if kind == "sma":
            v = (P > P.rolling(L, min_periods=L).mean()).astype(float)
            v = v.where(P.rolling(L, min_periods=L).mean().notna())
        elif kind == "mom":
            v = (P / P.shift(L) > 1.0).astype(float).where(P.shift(L).notna())
        else:
            raise ValueError(kind)
        votes.append(v)
    return sum(votes) / len(votes)


def vol_target_scale(unscaled_ret: pd.Series, target_vol: float, span: int | tuple = 30, max_lev: float = 2.0,
                     min_lev: float = 0.0) -> pd.Series:
    """Leverage factor for day t+1 from realised vol of the unscaled strategy up to day t.

    ``span`` may be a tuple of EWMA spans; the *largest* vol estimate is used (reacts fast to vol spikes,
    decays slowly)."""
    spans = (span,) if isinstance(span, int) else tuple(span)
    rv = pd.concat([unscaled_ret.ewm(span=s, min_periods=10).std() * np.sqrt(ANN) for s in spans], axis=1).max(axis=1)
    lev = (target_vol / rv).clip(lower=min_lev, upper=max_lev)
    return lev.fillna(0.0)


def inverse_vol_trend_weights(P: pd.DataFrame, U: pd.DataFrame, lookbacks=(20, 50, 100, 200), kind: str = "sma",
                              vol_span: int = 30, max_weight: float = 0.35) -> pd.DataFrame:
    """Unscaled long-only weights: signal x inverse-vol share among universe members (sum<=1)."""
    sig = trend_signal(P, lookbacks, kind).where(U, 0.0).fillna(0.0)
    vol = ewma_vol(P, vol_span).where(U)
    iv = (1.0 / vol).where(U).fillna(0.0)
    share = iv.div(iv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    w = (sig * share).clip(upper=max_weight)
    return w


# ----- Carver-style EWMAC continuous forecasts (canonical speeds / forecast scalars, not tuned) -----
EWMAC_SPEEDS = ((8, 32, 10.6), (16, 64, 7.5), (32, 128, 5.3), (64, 256, 3.75))


def ewmac_forecast(P: pd.DataFrame, speeds=EWMAC_SPEEDS, vol_span: int = 30, cap: float = 20.0) -> pd.DataFrame:
    """Average capped EWMAC forecast; +-10 is 'average conviction', result in [-cap, cap]."""
    dvol = np.log(P).diff().ewm(span=vol_span, min_periods=10).std()          # daily return vol
    fcs = []
    for fast, slow, scalar in speeds:
        raw = (P.ewm(span=fast, min_periods=fast).mean() - P.ewm(span=slow, min_periods=slow).mean()) / (P * dvol)
        fcs.append((raw * scalar).clip(-cap, cap))
    return sum(fcs) / len(fcs)


# ----- configurable end-to-end strategy -----
def dd_overlay(ret: pd.Series, start: float = 0.05, floor_at: float = 0.20, floor: float = 0.3) -> pd.Series:
    """Risk-scale factor for day t+1 from the strategy's own drawdown at close of day t (path dependent,
    uses only past strategy returns).  1.0 above -start drawdown, linearly down to ``floor`` at -floor_at."""
    eq = (1 + ret.fillna(0)).cumprod()
    dd = (eq / eq.cummax() - 1).clip(upper=0)
    x = ((-dd - start) / (floor_at - start)).clip(0, 1)
    return (1 - x * (1 - floor)).fillna(1.0)


def trend_forecast(A: pd.DataFrame, signal: str = "mix", lookbacks=(14, 30, 60, 90, 180)) -> pd.DataFrame:
    """Own-trend score in [0, 1] for each column of the price frame ``A``."""
    if signal in ("mom", "sma"):
        return trend_signal(A, lookbacks, signal)
    if signal == "ewmac":
        return (ewmac_forecast(A).clip(lower=0) / 10.0).clip(upper=1.0)
    if signal == "mix":  # average of three indicator families -> robust to any single definition
        return (trend_signal(A, lookbacks, "mom") + trend_signal(A, (20, 50, 100, 200), "sma")
                + (ewmac_forecast(A).clip(lower=0) / 10.0).clip(upper=1.0)) / 3.0
    raise ValueError(signal)


def build_weights(P: pd.DataFrame, R: pd.DataFrame, U: pd.DataFrame, *, signal: str = "mix",
                  lookbacks=(14, 30, 60, 90, 180), gate_sma: int | None = 200, asset_vol_tgt: float = 0.45,
                  n_min: int = 2, max_weight: float = 1.0, target_vol: float | None = None, max_lev: float = 2.0,
                  port_span: int | tuple = 30, overlay: dict | None = None, engine_kwargs=None) -> pd.DataFrame:
    """Final target weights (date x asset; fractions of NAV).

    U           : boolean membership matrix (point-in-time tradable universe)
    gate_sma    : an asset is only held while its price is above its ``gate_sma``-day SMA (None = off)
    asset_vol_tgt: per-asset inverse-vol sizing, w_i = trend_i * tgt / vol_i / max(N_t, n_min)
    target_vol  : portfolio-level realised-vol targeting (None = off)
    overlay     : dict(start=, floor_at=, floor=) enables the drawdown risk overlay
    """
    from .engine import run_backtest

    vol = ewma_vol(P, 30)
    f = trend_forecast(P, signal, lookbacks)
    if gate_sma:
        f = f * (P > P.rolling(gate_sma, min_periods=gate_sma).mean()).astype(float)
    f = f.where(U, 0.0).fillna(0.0)
    n = U.sum(axis=1).clip(lower=n_min)
    W = (f * asset_vol_tgt / vol).div(n, axis=0).where(U).fillna(0.0).clip(upper=max_weight)
    kw = engine_kwargs or {}
    if target_vol is not None:
        r0 = run_backtest(W, R, **kw)
        W = W.mul(vol_target_scale(r0, target_vol, port_span, max_lev), axis=0)
    if overlay:
        r1 = run_backtest(W, R, **kw)
        W = W.mul(dd_overlay(r1, **overlay), axis=0)
    return W


def build_weights_bagged(P: pd.DataFrame, R: pd.DataFrame, U: pd.DataFrame, *, gates=(100, 150, 200),
                         signals=("mix",), vol_spans=(20, 30, 60), asset_vol_tgt: float = 0.45, n_min: int = 2,
                         target_vol: float | None = 0.35, max_lev: float = 1.5, port_span=30,
                         max_gross: float | None = 2.0, engine_kwargs=None) -> pd.DataFrame:
    """Parameter-bagged version of :func:`build_weights`: average of the pre-scaling weights over the grid
    (gate length x signal family x asset-vol span), followed by one portfolio-level vol target.

    Averaging over the plateau instead of choosing the 'best' cell is the anti-overfitting device.
    ``max_lev`` caps the vol-target multiplier; ``max_gross`` is a hard cap on gross exposure (x NAV)."""
    from .engine import run_backtest

    n = U.sum(axis=1).clip(lower=n_min)
    acc = None
    cnt = 0
    for g in gates:
        for sg in signals:
            f = trend_forecast(P, sg)
            if g:
                f = f * (P > P.rolling(g, min_periods=g).mean()).astype(float)
            f = f.where(U, 0.0).fillna(0.0)
            for vs in vol_spans:
                vol = ewma_vol(P, vs)
                w = (f * asset_vol_tgt / vol).div(n, axis=0).where(U).fillna(0.0).clip(upper=1.0)
                acc = w if acc is None else acc + w
                cnt += 1
    W = acc / cnt
    if target_vol is not None:
        r0 = run_backtest(W, R, **(engine_kwargs or {}))
        W = W.mul(vol_target_scale(r0, target_vol, port_span, max_lev), axis=0)
    if max_gross is not None:  # hard cap on gross exposure (x NAV): scale a row down proportionally if breached
        gross = W.abs().sum(axis=1)
        W = W.mul((max_gross / gross).clip(upper=1.0).fillna(1.0), axis=0)
    return W
