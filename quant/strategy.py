"""The final strategy: **TrendCore** - a volatility-targeted, regime-gated trend-following book on BTC + ETH.

Design (every choice is a canonical / a-priori value; nothing is optimised on returns):

1. Universe     : the two largest liquid crypto assets (BTC, ETH).  Broader alt universes, long/short,
                  cross-sectional factors, ML models, on-chain features were all tested and rejected
                  (see ``research/`` and the README) - they did not add out-of-sample value.
2. Signal       : per asset, trend score in [0, 1] = average of three indicator families
                    * momentum votes     : sign of 14/30/60/90/180-day return
                    * moving-average votes: price > SMA 20/50/100/200
                    * EWMAC (Carver)     : 8/32, 16/64, 32/128, 64/256 EMA crossovers, vol-normalised
3. Regime gate  : asset is only held while price > its SMA(g); bagged over g in {100, 150, 200}.
4. Sizing       : inverse-volatility (EWMA span s in {20, 30, 60}, bagged) -> equal risk per asset.
5. Portfolio    : ex-ante realised-vol targeting (35 % annualised), vol-scalar capped at 1.5, gross <= 2x NAV.
6. Execution    : daily at the 00:00-UTC close, 5 % NAV no-trade band, long-only, no shorting.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import pandas as pd

from .data import load_panels
from .engine import run_backtest
from .strategies import build_weights_bagged


@dataclass
class Config:
    assets: tuple = ("btc", "eth")
    gates: tuple = (100, 150, 200)
    vol_spans: tuple = (20, 30, 60)
    signals: tuple = ("mix",)
    asset_vol_tgt: float = 0.45
    target_vol: float = 0.35
    max_lev: float = 1.5
    max_gross: float = 2.0         # hard cap on gross exposure, x NAV
    port_span: int = 30
    # execution / cost model
    tier1_bps: float = 10.0       # one-way fee + slippage for BTC / ETH
    other_bps: float = 25.0
    fin_rate: float = 0.10        # p.a. on borrowed notional above 1x NAV (perp funding / margin interest)
    cash_rate: float = 0.0        # idle cash earns nothing (conservative)
    band: float = 0.05            # no-trade band, fraction of NAV
    lag: int = 0                  # extra execution delay in days (stress test)
    start: str = "2016-01-01"

    def engine_kwargs(self):
        return dict(tier1_bps=self.tier1_bps, other_bps=self.other_bps, fin_rate=self.fin_rate,
                    cash_rate=self.cash_rate, band=self.band, lag=self.lag)

    def to_dict(self):
        return asdict(self)


def _universe(P: pd.DataFrame, assets) -> pd.DataFrame:
    U = pd.DataFrame(False, index=P.index, columns=P.columns)
    U[list(assets)] = P[list(assets)].notna()
    return U


def target_weights(P: pd.DataFrame, R: pd.DataFrame, cfg: Config = Config()) -> pd.DataFrame:
    """Target weights decided at each day's close (fractions of NAV) - what a live system would send."""
    U = _universe(P, cfg.assets)
    return build_weights_bagged(P, R, U, gates=cfg.gates, signals=cfg.signals, vol_spans=cfg.vol_spans,
                                asset_vol_tgt=cfg.asset_vol_tgt, n_min=2,
                                target_vol=cfg.target_vol, max_lev=cfg.max_lev, max_gross=cfg.max_gross, port_span=cfg.port_span,
                                engine_kwargs=dict(tier1_bps=cfg.tier1_bps, other_bps=cfg.other_bps,
                                                   fin_rate=cfg.fin_rate, cash_rate=cfg.cash_rate, band=cfg.band))


def run(cfg: Config = Config(), details: bool = False):
    P, _ = load_panels(include=tuple(cfg.assets))
    R = P.pct_change(fill_method=None)
    W = target_weights(P, R, cfg)
    out = run_backtest(W, R, return_details=details, **cfg.engine_kwargs())
    if details:
        r, det = out
        return r.loc[cfg.start:], det.loc[cfg.start:], W.loc[cfg.start:], P, R
    return out.loc[cfg.start:]
