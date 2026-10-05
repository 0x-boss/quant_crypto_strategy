"""Data loading and point-in-time universe construction.

Source: CoinMetrics community data (daily PriceUSD, reported spot volume).  Everything here is
*point in time*: a coin only enters the universe on a date if its history and liquidity, measured
using data up to (and including) that date, qualify it.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Stablecoins, wrapped/pegged duplicates, tokenised gold, chain-specific duplicate listings.
EXCLUDE = set(
    """usdt_omni usdt usdt_eth usdt_trx usdt_avaxc tusd tusd_eth tusd_trx gusd usdc usdc_eth usdc_trx
    usdc_avaxc pax busd husd dai sai usdk usdd_eth fdusd_eth pyusd_eth usde_eth susde_eth crvusd_eth
    usdm_eth sdai_eth eurc_eth frax_eth lusd_eth wbtc hbtc renbtc weth wnxm bnb_eth avaxp avaxx
    flow_native flow_evm leo_eos paxg xaut pol_eth nxm""".split()
)

# price series that trade under another base ticker in the volume file
ALIAS = {"avaxc": "avax", "vet_eth": "vet", "shib_eth": "shib", "matic_eth": "matic", "zil_eth": "zil",
         "icx_eth": "icx", "qtum_eth": "qtum", "leo_eth": "leo", "lrc_eth": "lrc", "ae_eth": "ae"}


def load_panels(start: str | None = None, end: str | None = None):
    """Return (price, volume) DataFrames (date x asset) restricted to tradable candidates."""
    P = pd.read_parquet(os.path.join(DATA_DIR, "panel_price.parquet"))
    V = pd.read_parquet(os.path.join(DATA_DIR, "panel_volume.parquet"))
    cols = [c for c in P.columns if c not in EXCLUDE]
    P = P[cols]
    Vm = pd.DataFrame(index=P.index, columns=cols, dtype="float64")
    for c in cols:
        src = ALIAS.get(c, c)
        if src in V.columns:
            Vm[c] = V[src].reindex(P.index)
    P = P.loc[start:end]
    Vm = Vm.loc[start:end]
    return P, Vm


def liquid_universe(P: pd.DataFrame, V: pd.DataFrame, top_n: int = 15, min_adv: float = 5e6,
                    min_hist: int = 180, adv_window: int = 90, rebalance: str = "ME") -> pd.DataFrame:
    """Boolean membership matrix, refreshed monthly from information available at month end.

    A coin qualifies at a snapshot if it has >= ``min_hist`` days of price history and median
    reported spot volume over the last ``adv_window`` days >= ``min_adv`` USD; the ``top_n`` by that
    median volume are kept and held in the universe until the next snapshot.
    """
    hist = P.notna().cumsum()
    adv = V.rolling(adv_window, min_periods=int(adv_window * 0.6)).median()
    ok = (hist >= min_hist) & (adv >= min_adv) & P.notna()
    score = adv.where(ok)
    snap_idx = score.resample(rebalance).last().index  # month-end labels
    member = pd.DataFrame(False, index=P.index, columns=P.columns)
    snaps = []
    for d in snap_idx:
        # last available trading date <= d
        loc = P.index.searchsorted(d, side="right") - 1
        if loc < 0:
            continue
        row = score.iloc[loc].dropna().nlargest(top_n)
        snaps.append((P.index[loc], set(row.index)))
    for i, (d, names) in enumerate(snaps):
        nxt = snaps[i + 1][0] if i + 1 < len(snaps) else P.index[-1] + pd.Timedelta(days=1)
        # membership applies from the day AFTER the snapshot date
        mask = (P.index > d) & (P.index <= nxt)
        if names:
            member.loc[mask, list(names)] = True
    return member & P.notna()
