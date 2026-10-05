import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from quant.data import load_panels
from quant.engine import run_backtest
from quant.strategies import *
from quant import metrics as M
P,V=load_panels(); R=P.pct_change(fill_method=None)
assets=['btc','eth']; A=P[assets]
U=pd.DataFrame(False,index=P.index,columns=P.columns); U[assets]=P[assets].notna()
mix=trend_forecast(A,'mix')
def run_variant(riskw=(0.5,0.5),eth_rel=None,tv=0.35):
    n=U.sum(axis=1).clip(lower=2); acc=None; cnt=0
    rw=pd.Series(riskw,index=assets)*2           # relative to equal
    for g in (100,150,200):
        gate=(A>A.rolling(g).mean()).astype(float)
        if eth_rel:
            ratio=P['eth']/P['btc']; gate['eth']=gate['eth']*(ratio>ratio.rolling(eth_rel).mean()).astype(float)
        f=(mix*gate).reindex(columns=P.columns).fillna(0.0).where(U,0.0)
        for vs in (20,30,60):
            v=ewma_vol(P,vs); w=(f*0.45/v).div(n,axis=0).where(U).fillna(0.0)
            for a in assets: w[a]=w[a]*rw[a]
            w=w.clip(upper=1.0); acc=w if acc is None else acc+w; cnt+=1
    W=acc/cnt; ek=dict(band=0.05)
    r0=run_backtest(W,R,**ek); W=W.mul(vol_target_scale(r0,tv,30,1.5),axis=0)
    g=W.abs().sum(axis=1); W=W.mul((2.0/g).clip(upper=1.0).fillna(1.0),axis=0)
    return run_backtest(W,R,**ek)
def show(name,r):
    out=[]
    for a,b in [('2018-01-01','2021-12-31'),('2022-01-01','2026-05-23'),('2016-01-01','2026-05-23')]:
        s=M.summary(r.loc[a:b]); out.append(f"{a[:4]}-{b[2:4]}: CAGR {s['cagr']*100:5.1f}% Sh {s['sharpe']:4.2f} DD {s['maxdd']*100:6.1f}%")
    print(f"{name:34s}"+' | '.join(out))
for rw in [(0.5,0.5),(0.6,0.4),(0.67,0.33),(0.75,0.25),(1.0,0.0)]:
    show(f'risk weights BTC:ETH {rw}',run_variant(rw))
for L in [30,50,100,200]:
    show(f'ETH needs ETH/BTC>SMA{L}',run_variant((0.5,0.5),L))
