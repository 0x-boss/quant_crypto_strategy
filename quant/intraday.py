"""TrendCore-I: TrendCore on 6-hour bars with hourly realised-vol sizing, plus a delta-neutral carry sleeve on idle capital.

Differences from the daily TrendCore (everything else - signal families, gates, bagging, vol target, caps - is identical,
with every look-back scaled by the number of bars per day, k):
  * decisions every 24/k hours (default k=4 -> 6h) instead of once a day,
  * volatility = EWMA of the sum of squared HOURLY log returns inside each bar (realised vol) instead of close-to-close,
  * open-interest crowding overlay: exposure x (1 - 0.5*clip(z,0,2)/2), z = 365d z-score of BTC futures open interest,
  * optional carry: capital not used by the trend book (1 - min(gross, 1)) earns BTC/ETH funding+basis carry (long spot / short perp).
Data: Binance spot 1h closes (data.binance.vision), history from 2019-09 -> effective start 2020-03 (200-day warm-up).
"""
from __future__ import annotations
import os
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from . import strategies as S
from .engine import run_backtest
from .carry import carry_returns

D = os.path.join(os.path.dirname(__file__), "..", "data", "intraday")


@dataclass
class IConfig:
    assets: tuple = ("btc", "eth")
    k: int = 4                      # bars per day (decisions every 24/k hours)
    rv_vol: bool = True             # hourly realised-vol sizing
    gates: tuple = (100, 150, 200)
    vol_spans: tuple = (20, 30, 60)
    asset_vol_tgt: float = 0.45
    target_vol: float = 0.40        # risk dial (Sharpe is flat across 0.25-0.45); 0.40 chosen to meet a 50 % CAGR target
    max_lev: float = 1.5
    max_gross: float = 2.0
    tier1_bps: float = 10.0
    other_bps: float = 25.0        # one-way cost for assets outside BTC/ETH (e.g. SOL)
    fin_rate: float = 0.10
    band: float = 0.05
    lag_bars: int = 0
    carry: bool = True
    carry_margin: float = 0.15
    oi_strength: float = 0.5        # de-risk when BTC open interest is high vs its own year (0 = off)
    start: str = "2020-03-01"

    def to_dict(self):
        return asdict(self)


def hourly_prices(assets=("btc", "eth")) -> pd.DataFrame:
    return pd.concat({a: pd.read_parquet(f"{D}/{a.upper()}USDT_spot_1h.parquet")["c"] for a in assets}, axis=1)


def oi_multiplier(index: pd.DatetimeIndex, strength: float) -> pd.Series:
    """1 - strength * clip(z, 0, 2) / 2 with z = 365-day z-score of log BTC open interest (USD), known at the previous day's close.
    No overlay (1.0) before the z-score exists (OI data starts 2020-09)."""
    oi = np.log(pd.read_parquet(f"{D}/BTCUSDT_oi_daily.parquet")["sum_open_interest_value"])
    z = ((oi - oi.rolling(365).mean()) / oi.rolling(365).std()).shift(1)
    zb = z.reindex(index, method="ffill")
    return (1 - strength * zb.clip(0, 2) / 2).fillna(1.0)


def target_weights_bars(px1h: pd.DataFrame, cfg: IConfig):
    k = cfg.k
    old_ann = S.ANN
    S.ANN = 365 * k
    try:
        px1h = px1h[list(cfg.assets)]
        P = px1h.resample(f"{24 // k}h").last().dropna(how="all")
        R = P.pct_change()
        ek = dict(band=cfg.band, tier1_bps=cfg.tier1_bps, other_bps=cfg.other_bps, fin_rate=cfg.fin_rate / k)
        if cfg.rv_vol:
            rv2 = (np.log(px1h).diff() ** 2).resample(f"{24 // k}h").sum().reindex(P.index)
            vol_fn = lambda PP, span=30: (rv2.ewm(span=span, min_periods=10).mean() ** 0.5) * np.sqrt(365 * k)
        else:
            vol_fn = S.ewma_vol
        mom = S.trend_signal(P, tuple(int(L * k) for L in (14, 30, 60, 90, 180)), "mom")
        sma = S.trend_signal(P, tuple(int(L * k) for L in (20, 50, 100, 200)), "sma")
        sp = tuple((int(f * k), int(s * k), sc) for f, s, sc in S.EWMAC_SPEEDS)
        ew = (S.ewmac_forecast(P, sp, vol_span=30 * k).clip(lower=0) / 10).clip(upper=1.0)
        f0 = (mom + sma + ew) / 3
        acc, cnt = None, 0
        for g in cfg.gates:
            f = (f0 * (P > P.rolling(g * k).mean())).fillna(0.0)
            for vs in cfg.vol_spans:
                w = (f * cfg.asset_vol_tgt / vol_fn(P, vs * k)).div(P.notna().sum(axis=1).clip(lower=1), axis=0).fillna(0.0).clip(upper=1.0)
                acc = w if acc is None else acc + w
                cnt += 1
        W = acc / cnt
        r0 = run_backtest(W, R, **ek)
        W = W.mul(S.vol_target_scale(r0, cfg.target_vol, 30 * k, cfg.max_lev), axis=0)
        if cfg.oi_strength > 0:
            W = W.mul(oi_multiplier(W.index, cfg.oi_strength), axis=0)
        gross = W.abs().sum(axis=1)
        W = W.mul((cfg.max_gross / gross).clip(upper=1.0).fillna(1.0), axis=0)
        return P, R, W, ek
    finally:
        S.ANN = old_ann


def run(cfg: IConfig = IConfig(), px1h: pd.DataFrame | None = None):
    """Daily return series (net of costs) of TrendCore-I; also returns components."""
    px1h = hourly_prices(cfg.assets) if px1h is None else px1h
    P, R, W, ek = target_weights_bars(px1h, cfg)
    r, det = run_backtest(W, R, lag=cfg.lag_bars, return_details=True, **ek)
    daily = lambda x: (1 + x).groupby(x.index.date).prod() - 1
    tr = daily(r)
    tr.index = pd.to_datetime(tr.index)
    gross_d = det["gross"].groupby(det.index.date).mean()
    gross_d.index = pd.to_datetime(gross_d.index)
    out = {"trend": tr, "gross": gross_d}
    if cfg.carry:
        car = pd.concat([carry_returns(a.upper() + "USDT", margin=cfg.carry_margin) for a in cfg.assets], axis=1).mean(axis=1)
        car = car.reindex(tr.index).fillna(0.0)
        out["carry"] = car
        out["total"] = tr + (1 - gross_d.clip(upper=1.0)).reindex(tr.index).fillna(1.0) * car
    else:
        out["total"] = tr
    return {k_: v.loc[cfg.start:] for k_, v in out.items()}
