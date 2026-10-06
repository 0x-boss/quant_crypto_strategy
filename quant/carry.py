"""Delta-neutral funding / basis carry (long spot, short USDT-perp) from Binance hourly archive data.

Daily P&L per 1.0 of hedged notional (hedge re-set every day at 00:00 UTC):
    funding received  +  (spot return - perp return)  -  trading costs
Capital = notional * (1 + margin) because the perp leg needs collateral (default 15 %).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(__file__), "..", "data", "intraday")
SPOT_BPS, PERP_BPS = 10.0, 4.5          # taker fee-equivalent one-way costs


def load(sym: str):
    sp = pd.read_parquet(f"{D}/{sym}_spot_1h.parquet")["c"]
    pf = pd.read_parquet(f"{D}/{sym}_perp_1h.parquet")["c"]
    fu = pd.read_parquet(f"{D}/{sym}_funding.parquet")["fundingRate"]
    return sp, pf, fu


def daily_carry(sym: str):
    """Return DataFrame(date): funding, basis_pnl (per 1.0 notional, always-on, before trading costs), funding_ann."""
    sp, pf, fu = load(sym)
    s = sp.resample("1D").last()                       # price at end of day (00:00 UTC next day)
    f = pf.resample("1D").last()
    j = pd.concat([s, f], axis=1, keys=["s", "f"]).dropna()
    basis = j.s.pct_change() - j.f.pct_change()
    fund = fu.resample("1D").sum().reindex(j.index).fillna(0.0)
    out = pd.DataFrame({"funding": fund, "basis": basis}).dropna()
    out["fund_ma30"] = out.funding.rolling(30).mean() * 365     # trailing annualised funding, known at close of day t
    return out


def carry_returns(sym: str, margin: float = 0.15, rule: str = "always", thr: float = 0.05, tier_bps=(SPOT_BPS, PERP_BPS)):
    d = daily_carry(sym)
    on = pd.Series(1.0, index=d.index) if rule == "always" else (d.fund_ma30 > thr).astype(float)
    on_held = on.shift(1).fillna(0.0)                  # decided at close t-1, earns day t
    gross = on_held * (d.funding + d.basis)
    turn = on_held.diff().abs().fillna(on_held.abs())  # entering / leaving costs both legs
    cost = turn * 2 * sum(tier_bps) / 1e4              # buy spot + sell perp on entry, reverse on exit (each leg one-way)
    cost = turn * (tier_bps[0] + tier_bps[1]) / 1e4 + 0 * cost
    ret = (gross - cost) / (1 + margin)
    return ret.rename(sym)
