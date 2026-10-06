"""Exploratory extension of round F part 1a (added after seeing that the 0.40 portfolio brake helps): finer grid of the two risk knobs.
ASSET_VOL = per-asset risk budget (official 0.45); target_vol = portfolio own-vol brake (official 0.40, scalar capped at 1.0, gross <= 1).
Descriptive only: it maps the CAGR / drawdown frontier of the SAME signals; Sharpe is ~flat (1.32-1.45) over the whole grid."""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
from quant import metrics as M
import trendcore_spot as T
px = pd.concat({a: pd.read_parquet(f'data/intraday/{a.upper()}USDT_spot_1h.parquet')['c'] for a in ('btc', 'eth')}, axis=1, sort=True)
oi = pd.read_parquet('data/intraday/BTCUSDT_oi_daily.parquet')['sum_open_interest_value']
def to_daily(rb):
    d = (1 + rb).groupby(rb.index.date).prod() - 1; d.index = pd.to_datetime(d.index); return d
rows = []
for av in (0.30, 0.45, 0.60, 0.80, 1.00, 1.50):
    for tv in (0.20, 0.25, 0.30, 0.35, 0.40, 0.60):
        T.ASSET_VOL = av
        try: W, R = T.target_weights(px, oi, target_vol=tv)
        finally: T.ASSET_VOL = 0.45
        rb, gb = T._backtest(W, R); r = to_daily(rb).loc['2018-03-01':]
        s = M.summary(r); s2 = M.summary(r.loc['2023-01-01':]); s3 = M.summary(r.loc['2022-01-01':]); s1 = M.summary(r.loc[:'2022-12-31'])
        rows.append(dict(asset_vol=av, target_vol=tv, cagr=s['cagr'], sharpe=s['sharpe'], maxdd=s['maxdd'], calmar=s['calmar'], cagr_1822=s1['cagr'], sharpe_1822=s1['sharpe'], dd_1822=s1['maxdd'],
                         cagr_2023=s2['cagr'], sharpe_2023=s2['sharpe'], dd_2023=s2['maxdd'], cagr_2022=s3['cagr'], sharpe_2022=s3['sharpe'], avg_exposure=float(gb.mean())))
d = pd.DataFrame(rows); d.to_csv('results/dial_grid.csv', index=False)
print(d.round(3).to_string(index=False))
print(f"\nSharpe over the grid (2018-03+): min {d.sharpe.min():.2f}, median {d.sharpe.median():.2f}, max {d.sharpe.max():.2f}; official point {d[(d.asset_vol==0.45)&(d.target_vol==0.40)].sharpe.iloc[0]:.2f}")
