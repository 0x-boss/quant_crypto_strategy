import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from quant.data import load_panels
from quant.engine import run_backtest
from quant.strategies import *
from quant import metrics as M
P,V=load_panels(); R=P.pct_change(fill_method=None)
blocks={'17-19':('2017-01-01','2019-12-31'),'20-21':('2020-01-01','2021-12-31'),'22-23':('2022-01-01','2023-12-31'),'24-26':('2024-01-01','2026-05-24')}
def show(name,r):
    f=M.summary(r.loc['2017':]); s=' '.join(f"{k}:{M.summary(r.loc[a:b])['sharpe']:4.2f}" for k,(a,b) in blocks.items())
    print(f"{name:40s} CAGR {f['cagr']*100:5.1f}% Sh {f['sharpe']:4.2f} DD {f['maxdd']*100:6.1f}% vol {f['vol']*100:4.1f} Calmar {f['calmar']:4.2f} | {s}")
assets=['btc','eth']
U=pd.DataFrame(False,index=P.index,columns=P.columns); U[assets]=P[assets].notna()
A=P[assets]; vol=ewma_vol(A,30)
f=trend_forecast(A,'mix')*(A>A.rolling(200).mean())
for p in [0.5,1,1.5,2,3]:
    fp=f**p
    W=pd.DataFrame(0.0,index=P.index,columns=P.columns); W[assets]=(fp*0.45/vol/2).fillna(0).clip(upper=1.0)
    r0=run_backtest(W,R); lev=vol_target_scale(r0,0.30,30,2.0); show(f'f^{p} gate200 portTV30',run_backtest(W.mul(lev,axis=0),R))
