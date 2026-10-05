"""Round 2 (user asked to fix 2022-26).  Diagnostics, judged on BOTH eras (2017-21 and 2022-26).
 (a) forward returns by 'stretch' = distance above SMA200 in vol units
 (b) stricter regime gates inside the existing TrendCore pipeline"""
import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from quant.data import load_panels
from quant.engine import run_backtest
from quant.strategies import *
from quant import metrics as M
P,V=load_panels(); R=P.pct_change(fill_method=None)
assets=['btc','eth']; A=P[assets]
lr=np.log(A).diff(); vol=lr.ewm(span=30).std()
sma200=A.rolling(200).mean()
stretch=np.log(A/sma200)/(vol*np.sqrt(60))
print('--- (a) forward 30d log-return by stretch bucket, only when price>SMA200; mean (bp) / n   [2017-21 | 2022-26]')
for a in assets:
    f30=np.log(A[a].shift(-30)/A[a])
    d=pd.DataFrame({'s':stretch[a],'f':f30}).dropna(); d=d[(A[a]>sma200[a]).reindex(d.index)]
    d['b']=pd.cut(d['s'],[0,0.5,1,1.5,2,99])
    for lab,(lo,hi) in {'17-21':('2017','2021'),'22-26':('2022','2026')}.items():
        t=d.loc[lo:hi].groupby('b',observed=True)['f'].agg(['mean','count']); print(a,lab,[(str(i),round(r['mean']*1e4),int(r['count'])) for i,r in t.iterrows()])
U=pd.DataFrame(False,index=P.index,columns=P.columns); U[assets]=P[assets].notna()
base=trend_forecast(A,'mix')
def run_variant(gatefun,tv=0.35):
    n=U.sum(axis=1).clip(lower=2); acc=None; cnt=0
    for g in (100,150,200):
        gate=gatefun(g)
        f=(base*gate).reindex(columns=P.columns).fillna(0.0).where(U,0.0)
        for vs in (20,30,60):
            v=ewma_vol(P,vs); w=(f*0.45/v).div(n,axis=0).where(U).fillna(0.0).clip(upper=1.0)
            acc=w if acc is None else acc+w; cnt+=1
    W=acc/cnt; ek=dict(band=0.05)
    r0=run_backtest(W,R,**ek); W=W.mul(vol_target_scale(r0,tv,30,1.5),axis=0)
    g=W.abs().sum(axis=1); W=W.mul((2.0/g).clip(upper=1.0).fillna(1.0),axis=0)
    return run_backtest(W,R,**ek)
def show(name,r):
    out=[]
    for a,b in [('2018-01-01','2021-12-31'),('2022-01-01','2026-05-23'),('2016-01-01','2026-05-23')]:
        s=M.summary(r.loc[a:b]); out.append(f"{a[:4]}-{b[2:4]}: CAGR {s['cagr']*100:5.1f}% Sh {s['sharpe']:4.2f} DD {s['maxdd']*100:6.1f}%")
    print(f"{name:40s}"+' | '.join(out))
print('--- (b) gate variants')
base_gate=lambda g:(A>A.rolling(g).mean()).astype(float)
show('baseline (P>SMA g)',run_variant(base_gate))
show('P>SMA g and SMA g rising(20d)',run_variant(lambda g:((A>A.rolling(g).mean())&(A.rolling(g).mean()>A.rolling(g).mean().shift(20))).astype(float)))
show('P>SMA g and SMA50>SMA g',run_variant(lambda g:((A>A.rolling(g).mean())&(A.rolling(50).mean()>A.rolling(g).mean())).astype(float)))
show('P>SMA g and P>SMA50',run_variant(lambda g:((A>A.rolling(g).mean())&(A>A.rolling(50).mean())).astype(float)))
show('P>SMA g and 365d mom>0',run_variant(lambda g:((A>A.rolling(g).mean())&(A>A.shift(365))).astype(float)))
show('P>SMA g, stretch trim (z>2 -> x0.5)',run_variant(lambda g:((A>A.rolling(g).mean()).astype(float)*np.where(stretch>2,0.5,1.0))))
