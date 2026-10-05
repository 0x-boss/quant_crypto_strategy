"""Vectorised, cost-aware daily portfolio backtester.

Timing convention (no look-ahead)
---------------------------------
* ``W.loc[t]`` is the target portfolio (fractions of NAV) decided at the close of day ``t`` using only
  data up to and including day ``t``.
* It is held over day ``t+1`` and earns ``R.loc[t+1]`` = ``P[t+1]/P[t]-1``.
* ``lag`` adds extra days of execution delay (``lag=1`` -> trade one full day after the signal), used
  as a stress test.

Costs
-----
* Trading cost = one-way ``fee_bps`` (exchange fee + half-spread + impact) times traded notional,
  computed against the *drifted* previous weights.  BTC / ETH get a tighter tier.
* ``band`` > 0 adds a no-trade band (in fractions of NAV) to cut churn from tiny rebalances.
* Financing: ``fin_rate`` p.a. is charged on borrowed notional (gross exposure above 100 % of NAV) and
  on short notional (perp funding / borrow); uninvested cash earns ``cash_rate`` (default 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TIER1 = {"btc", "eth"}


def fee_vector(columns, tier1_bps: float = 10.0, other_bps: float = 25.0) -> np.ndarray:
    return np.array([tier1_bps if c in TIER1 else other_bps for c in columns]) / 1e4


def run_backtest(W: pd.DataFrame, R: pd.DataFrame, tier1_bps: float = 10.0, other_bps: float = 25.0,
                 lag: int = 0, fin_rate: float = 0.10, cash_rate: float = 0.0, band: float = 0.0,
                 return_details: bool = False):
    W = W.reindex(R.index).fillna(0.0)
    cols = list(R.columns)
    W = W[cols]
    Wl = W.shift(1 + lag).fillna(0.0).values          # weights actually held over day t
    Rv = R.fillna(0.0).values
    fee = fee_vector(cols, tier1_bps, other_bps)
    T, N = Wl.shape
    ret = np.zeros(T)
    cost = np.zeros(T)
    turn = np.zeros(T)
    fin = np.zeros(T)
    prev = np.zeros(N)                                 # drifted holdings at start of day (fractions of NAV)
    for t in range(T):
        tgt = Wl[t]
        if band > 0.0:
            # no-trade band: keep the drifted holding unless the target is more than ``band`` (fraction of
            # NAV) away; always trade fully to/from zero so exits are never delayed.
            keep = (np.abs(tgt - prev) <= band) & (tgt != 0.0)
            tgt = np.where(keep, prev, tgt)
        dtrade = np.abs(tgt - prev)
        c = float(dtrade @ fee)
        gross_long = tgt[tgt > 0].sum()
        gross_short = -tgt[tgt < 0].sum()
        borrow = max(gross_long + gross_short - 1.0, 0.0)
        f = borrow * fin_rate / 365.0 - max(1.0 - gross_long - gross_short, 0.0) * cash_rate / 365.0
        pr = float(tgt @ Rv[t])
        net = pr - c - f
        ret[t] = net
        cost[t] = c
        turn[t] = dtrade.sum()
        fin[t] = f
        # drift to next open
        grown = tgt * (1.0 + Rv[t])
        denom = 1.0 + net
        prev = grown / denom if denom > 0 else np.zeros(N)
    out = pd.Series(ret, index=R.index, name="ret")
    if return_details:
        det = pd.DataFrame({"ret": ret, "cost": cost, "turnover": turn, "financing": fin,
                            "gross": np.abs(Wl).sum(1), "net": Wl.sum(1)}, index=R.index)
        return out, det
    return out
