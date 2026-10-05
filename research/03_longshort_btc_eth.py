"""Exploration 3: long/short trend on BTC and ETH (perps allow shorting). Costs incl. financing on shorts."""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from quant.data import load_panels
from quant.engine import run_backtest
from quant.strategies import *
from quant import metrics as M
P,V=load_panels(); R=P.pct_change(fill_method=None)
print((P['btc'].resample('YE').last().pct_change()*100).round(0).loc['2015':].to_dict())
periods={'full':('2017-01-01','2026-05-24'),'dev 17-21':('2017-01-01','2021-12-31'),'hold 22-26':('2022-01-01','2026-05-24')}
def show(name,r):
    row=[]
    for k,(a,b) in periods.items():
        s=M.summary(r.loc[a:b]); row.append(f"{k}: CAGR {s['cagr']*100:6.1f}% Sh {s['sharpe']:4.2f} DD {s['maxdd']*100:6.1f}% vol {s['vol']*100:4.1f}")
    print(f"{name:34s}"+' | '.join(row))
for assets in (['btc'],['btc','eth']):
    A=P[assets]
    for kind,lb in [('sma',(20,50,100,200)),('mom',(14,30,60,90,180)),('sma',(50,100,150)),('mom',(30,60,90))]:
        up=trend_signal(A,lb,kind)          # [0,1]
        sig=2*up-1                           # [-1,1] long/short
        vol=ewma_vol(A,30)
        w0=(sig/vol).where(vol.notna(),0.0)
        w0=w0.div(len(assets))
        W0=pd.DataFrame(0.0,index=P.index,columns=P.columns)
        for a in assets: W0[a]=w0[a]
        # scale to 40% portfolio vol (weights = sig * 0.4/vol_i /n)  -> raw
        W=W0*0.40
        W=W.clip(-2,2)
        show(f'{"+".join(assets)} L/S {kind}{lb}',run_backtest(W,R,fin_rate=0.10))
        # long-only version for comparison
        Wl=W.clip(lower=0)
        show(f'{"+".join(assets)} L/O {kind}{lb}',run_backtest(Wl,R,fin_rate=0.10))
