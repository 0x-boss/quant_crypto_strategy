import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from quant.data import load_panels, liquid_universe
from quant.engine import run_backtest
from quant import metrics as M
pd.set_option('display.width',200)
P,V=load_panels()
R=P.pct_change(fill_method=None)
periods={'full 2016-26':('2016-01-01','2026-05-24'),'dev 2016-21':('2016-01-01','2021-12-31'),'hold 2022-26':('2022-01-01','2026-05-24')}
def show(name,r):
    row=[]
    for k,(a,b) in periods.items():
        s=M.summary(r.loc[a:b])
        row.append(f"{k}: CAGR {s['cagr']*100:6.1f}% Sh {s['sharpe']:4.2f} DD {s['maxdd']*100:6.1f}%")
    print(f"{name:28s}"+' | '.join(row))
for a in ['btc','eth']:
    W=pd.DataFrame(0.0,index=P.index,columns=P.columns); W[a]=1.0
    show(a+' buy&hold',run_backtest(W,R))
# BTC trend: price > SMA(L)
for L in [20,50,100,150,200]:
    for a in ['btc']:
        sig=(P[a]>P[a].rolling(L).mean()).astype(float)
        W=pd.DataFrame(0.0,index=P.index,columns=P.columns); W[a]=sig
        show(f'{a} P>SMA{L}',run_backtest(W,R))
for L in [10,20,40,80,160]:
    sig=(P['btc']/P['btc'].shift(L)>1).astype(float)
    W=pd.DataFrame(0.0,index=P.index,columns=P.columns); W['btc']=sig
    show(f'btc TSMOM{L}',run_backtest(W,R))
