"""Shock circuit-breaker: after a vol-adjusted down-day (z < -thr) cut exposure for h days. Both eras."""
import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from quant.data import load_panels
from quant.engine import run_backtest
from quant.strategies import *
from quant import metrics as M
P,V=load_panels(); R=P.pct_change(fill_method=None)
assets=['btc','eth']; A=P[assets]; lr=np.log(A).diff()
vol=lr.ewm(span=30,min_periods=15).std().shift(1); z=lr/vol
print('--- clean event study BTC+ETH (non-overlapping events: first shock day only)')
for thr in [2.0,2.5,3.0]:
    for era,(a,b) in {'2017-21':('2017-01-01','2021-12-31'),'2022-26':('2022-01-01','2026-05-23')}.items():
        out=[]
        for h in [1,3,5,10]:
            fwd=(np.log(A.shift(-h)/A)/(vol*np.sqrt(h)))
            ev=((z<-thr)&~(z<-thr).shift(1,fill_value=False)).loc[a:b]
            x=fwd.loc[a:b].where(ev).stack(future_stack=False).dropna() if False else fwd.loc[a:b][ev].stack().dropna()
            out.append(f'h{h}: n={len(x)} mean {x.mean():+.2f}sd t={x.mean()/(x.std()/np.sqrt(len(x))):+.1f}')
        print(f'thr {thr} {era}: '+' | '.join(out))
U=pd.DataFrame(False,index=P.index,columns=P.columns); U[assets]=P[assets].notna()
base=trend_forecast(A,'mix')
def run_variant(extra_mult=None,second_gate=True,tv=0.35):
    n=U.sum(axis=1).clip(lower=2); acc=None; cnt=0
    for g in (100,150,200):
        gate=(A>A.rolling(g).mean()).astype(float)
        if second_gate: gate=gate*(A>A.rolling(50).mean())
        f=(base*gate)
        if extra_mult is not None: f=f*extra_mult
        f=f.reindex(columns=P.columns).fillna(0.0).where(U,0.0)
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
    print(f"{name:36s}"+' | '.join(out))
print('--- overlay on strategy (with P>SMA50 second gate)')
show('no breaker, 2nd gate',run_variant())
for thr in [2.0,2.5,3.0]:
    for h in [3,5,10]:
        shock=(z<-thr).astype(float)
        cut=1.0-(shock.rolling(h).max().shift(0).fillna(0))     # exposure 0 for h days starting the day of shock (decided at close of shock day)
        show(f'breaker z<-{thr} for {h}d',run_variant(cut))
