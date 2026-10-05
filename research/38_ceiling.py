"""Hindsight ceiling: best 2022-26 Sharpe over a broad grid of single long-only trend rules on BTC+ETH (vol-targeted 35%, 10bp)."""
import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, itertools
from quant.data import load_panels
from quant.engine import run_backtest
from quant.strategies import *
from quant import metrics as M
P,V=load_panels(); R=P.pct_change(fill_method=None)
assets=['btc','eth']; A=P[assets]; vol=ewma_vol(A,30)
def pipeline(f,tv=0.35):
    W=pd.DataFrame(0.0,index=P.index,columns=P.columns); W[assets]=(f*0.45/vol/2).fillna(0).clip(upper=1.0)
    r0=run_backtest(W,R,band=0.05); W=W.mul(vol_target_scale(r0,tv,30,1.5),axis=0)
    g=W.abs().sum(axis=1); W=W.mul((2.0/g).clip(upper=1.0).fillna(1.0),axis=0)
    return run_backtest(W,R,band=0.05)
rules={}
for L in [10,20,30,50,75,100,150,200,250]: rules[f'sma{L}']=(A>A.rolling(L).mean()).astype(float)
for L in [5,10,14,21,30,45,60,90,120,180,250,365]: rules[f'mom{L}']=(A/A.shift(L)>1).astype(float)
for fast,slow in [(5,20),(8,32),(10,50),(16,64),(20,100),(32,128),(50,200),(64,256)]:
    rules[f'ema{fast}/{slow}']=(A.ewm(span=fast).mean()>A.ewm(span=slow).mean()).astype(float)
for L in [20,40,60,100,150,200]:
    hi=A.rolling(L).max().shift(1); lo=A.rolling(L//2).min().shift(1)
    st=pd.DataFrame(0.0,index=A.index,columns=A.columns); s=pd.Series(0.0,index=A.columns)
    for d in A.index:
        up=A.loc[d]>hi.loc[d]; dn=A.loc[d]<lo.loc[d]
        s=s.where(~up,1.0).where(~dn,0.0); st.loc[d]=s
    rules[f'donch{L}']=st
res=[]
for name,f in rules.items():
    for gate in [None,100,200]:
        ff=f if gate is None else f*(A>A.rolling(gate).mean())
        r=pipeline(ff)
        a=M.summary(r.loc['2022-01-01':'2026-05-23']); b=M.summary(r.loc['2018-01-01':'2021-12-31'])
        res.append(dict(rule=name,gate=gate,sh22=a['sharpe'],cagr22=a['cagr'],dd22=a['maxdd'],sh18=b['sharpe']))
D=pd.DataFrame(res).sort_values('sh22',ascending=False)
print(len(D),'rules; 2022-26 Sharpe: max %.2f, 90th pct %.2f, median %.2f, min %.2f'%(D.sh22.max(),D.sh22.quantile(.9),D.sh22.median(),D.sh22.min()))
print('corr between 2018-21 and 2022-26 Sharpe across rules: %.2f'%D[['sh22','sh18']].corr().iloc[0,1])
print(D.head(12).round(2).to_string(index=False))
print('current TrendCore: 2022-26 Sharpe 0.78')
D.to_csv('results/ceiling_grid.csv',index=False)
