"""Round 2: trend-quality (slope t-stat) family, chandelier exit, BTC-gated ETH.  Both-eras rule."""
import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from quant.data import load_panels
from quant.engine import run_backtest
from quant.strategies import *
from quant import metrics as M
P,V=load_panels(); R=P.pct_change(fill_method=None)
assets=['btc','eth']; A=P[assets]; lA=np.log(A)
U=pd.DataFrame(False,index=P.index,columns=P.columns); U[assets]=P[assets].notna()
def slope_t(L):
    x=np.arange(L); xm=x-x.mean(); sxx=(xm**2).sum()
    def f(col):
        y=col.values; out=np.full(len(y),np.nan)
        for i in range(L-1,len(y)):
            yy=y[i-L+1:i+1]
            if np.isnan(yy).any(): continue
            b=(xm*(yy-yy.mean())).sum()/sxx; res=yy-yy.mean()-b*xm; s=np.sqrt((res**2).sum()/(L-2)/sxx)
            out[i]=b/s if s>0 else 0
        return pd.Series(out,index=col.index)
    return lA.apply(f)
T={L:slope_t(L) for L in (30,60,120,250)}
qual=sum(((T[L]/2.0).clip(0,1)) for L in T)/len(T)     # t-stat 2 -> full score
mix=trend_forecast(A,'mix')
def chandelier(k,N):
    lr=lA.diff(); vol=lr.ewm(span=30).std()
    hi=A.rolling(N).max(); stop=hi*np.exp(-k*vol*np.sqrt(10))
    return (A>stop).astype(float)
def run_variant(score,gate_fn,tv=0.35):
    n=U.sum(axis=1).clip(lower=2); acc=None; cnt=0
    for g in (100,150,200):
        f=(score*gate_fn(g)).reindex(columns=P.columns).fillna(0.0).where(U,0.0)
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
g1=lambda g:(A>A.rolling(g).mean()).astype(float)
g2=lambda g:((A>A.rolling(g).mean())&(A>A.rolling(50).mean())).astype(float)
show('mix + gate(SMA g)',run_variant(mix,g1))
show('mix + gate(SMA g & SMA50)  [cand]',run_variant(mix,g2))
show('t-stat quality + gate2',run_variant(qual,g2))
show('(mix+quality)/2 + gate2',run_variant((mix+qual)/2,g2))
for k,N in [(3,60),(4,60),(3,100)]:
    ch=chandelier(k,N)
    show(f'mix + gate2 + chandelier k{k} N{N}',run_variant(mix,lambda g:g2(g)*ch))
# ETH only when BTC gate is on
def g3(g):
    base=g2(g).copy(); btc_on=base['btc']; base['eth']=base['eth']*btc_on; return base
show('mix + gate2 + ETH needs BTC gate',run_variant(mix,g3))
