"""Exploration 2: diversified long-only trend portfolio, vol-targeted. Reports dev (2017-21) vs holdout (2022-26)."""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from quant.data import load_panels, liquid_universe
from quant.engine import run_backtest
from quant.strategies import *
from quant import metrics as M
P,V=load_panels(); R=P.pct_change(fill_method=None)
periods={'full':('2017-01-01','2026-05-24'),'dev 17-21':('2017-01-01','2021-12-31'),'hold 22-26':('2022-01-01','2026-05-24')}
def show(name,r):
    row=[]
    for k,(a,b) in periods.items():
        s=M.summary(r.loc[a:b]); row.append(f"{k}: CAGR {s['cagr']*100:6.1f}% Sh {s['sharpe']:4.2f} DD {s['maxdd']*100:6.1f}% vol {s['vol']*100:4.1f}")
    print(f"{name:34s}"+' | '.join(row))
for top_n in [10,20]:
    U=liquid_universe(P,V,top_n=top_n,min_adv=5e6,min_hist=180)
    for kind,lb in [('sma',(20,50,100,200)),('mom',(14,30,60,90,180)),('sma',(50,100,150))]:
        W0=inverse_vol_trend_weights(P,U,lb,kind)
        r0=run_backtest(W0,R,fin_rate=0.1)
        show(f'top{top_n} {kind}{lb} unlevered',r0)
        for tv,ml in [(0.35,2.0),(0.5,2.5)]:
            lev=vol_target_scale(r0,tv,30,ml)
            W=W0.mul(lev,axis=0)
            show(f'   volT {tv} maxlev {ml}',run_backtest(W,R,fin_rate=0.1))
