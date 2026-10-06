"""SPOT-ONLY, LONG-ONLY, NO LEVERAGE: gross exposure <= 1.0 of capital, no carry, no perps, no financing.  Idle cash earns 0."""
import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from quant.strategy import Config, run as run_daily
from quant.intraday import IConfig, run as run_intra
from quant import metrics as M
def line(name,r):
    out=[]
    for lab,a in [('start','2016-01-01' if 'daily' in name else '2020-03-01'),('2020-03+','2020-03-01'),('2022+','2022-01-01'),('fwd OOS','2026-05-24')]:
        x=r.loc[a:]
        if len(x)<60: continue
        s=M.summary(x); out.append(f"{lab}: CAGR {s['cagr']*100:5.1f}% Sh {s['sharpe']:4.2f} DD {s['maxdd']*100:6.1f}%")
    print(f'{name:44s}'+' | '.join(out))
print('--- DAILY TrendCore, spot-only (CoinMetrics data, 2016-2026; extended file not used)')
for tv in [0.25,0.40,0.60]:
    line(f'daily spot-only TV {tv}',run_daily(Config(target_vol=tv,max_lev=1.0,max_gross=1.0,fin_rate=0.0)))
print('--- TrendCore-I (6h bars, hourly RV), spot-only, no carry; OI overlay = signal only')
for oi in [0.0,0.5]:
    for tv in [0.25,0.40,0.60]:
        o=run_intra(IConfig(target_vol=tv,max_lev=1.0,max_gross=1.0,fin_rate=0.0,carry=False,oi_strength=oi))
        line(f'6h spot-only OI {oi} TV {tv}',o['total'])
o=run_intra(IConfig(target_vol=0.40,max_lev=1.0,max_gross=1.0,fin_rate=0.0,carry=False,oi_strength=0.5))
print('avg gross exposure 2022+: %.2f, max %.2f, time flat %.0f%%'%(o['gross'].loc['2022':].mean(),o['gross'].max(),(o['gross']<0.05).mean()*100))
