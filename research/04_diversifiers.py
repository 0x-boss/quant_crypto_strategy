"""Exploration 4: candidate sleeves, each evaluated stand-alone on full / dev / holdout, plus correlation.
 S1 trend: BTC+ETH long-only ensemble, inverse-vol sized
 S2 relative trend: ETH vs BTC (dollar-neutral) on ratio trend
 S3 breadth-gated: scale S1 exposure by fraction of top-20 coins above 50d SMA
"""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from quant.data import load_panels, liquid_universe
from quant.engine import run_backtest
from quant.strategies import *
from quant import metrics as M
P,V=load_panels(); R=P.pct_change(fill_method=None)
U=liquid_universe(P,V,top_n=20,min_adv=5e6,min_hist=180)
periods={'full':('2017-01-01','2026-05-24'),'dev 17-21':('2017-01-01','2021-12-31'),'hold 22-26':('2022-01-01','2026-05-24')}
def show(name,r):
    row=[]
    for k,(a,b) in periods.items():
        s=M.summary(r.loc[a:b]); row.append(f"{k}: CAGR {s['cagr']*100:6.1f}% Sh {s['sharpe']:4.2f} DD {s['maxdd']*100:6.1f}% vol {s['vol']*100:4.1f}")
    print(f"{name:30s}"+' | '.join(row))
LB=(14,30,60,90,180)
A=P[['btc','eth']]
up=trend_signal(A,LB,'mom'); vol=ewma_vol(A,30)
W1=pd.DataFrame(0.0,index=P.index,columns=P.columns)
W1[['btc','eth']]=(up*0.30/vol/2).fillna(0).clip(upper=1.0)
r1=run_backtest(W1,R); show('S1 trend BTC+ETH',r1)
# S2: ETH/BTC ratio trend, dollar neutral, vol-targeted to 20%
ratio=(P['eth']/P['btc']).to_frame('ratio')
s=2*trend_signal(ratio,LB,'mom')['ratio']-1
rv=np.log(ratio['ratio']).diff().ewm(span=30).std()*np.sqrt(365)
k=(0.20/rv).clip(upper=2.0)
W2=pd.DataFrame(0.0,index=P.index,columns=P.columns)
W2['eth']=(s*k).fillna(0); W2['btc']=-(s*k).fillna(0)
r2=run_backtest(W2,R); show('S2 ETH/BTC rel trend',r2)
# S2b: long-only version: hold the stronger of the two? (skip)
# S3 breadth gating
sma50=P.rolling(50).mean()
breadth=((P>sma50)&U).sum(axis=1)/U.sum(axis=1).replace(0,np.nan)
for lo in [0.3]:
    g=((breadth-0.2)/(0.6-0.2)).clip(0,1)   # 0 when <=20% above, 1 when >=60%
    W3=W1.mul(g.fillna(0),axis=0)
    show('S3 S1 x breadth gate',run_backtest(W3,R))
    W3b=W1.mul(0.5+0.5*g.fillna(0),axis=0)
    show('S3b S1 x (0.5+0.5 gate)',run_backtest(W3b,R))
print(pd.concat([r1,r2],axis=1,keys=['S1','S2']).loc['2017':].corr().round(2))
