"""Walk-forward adaptive tilt across the 13 sub-signals of the 'mix' ensemble.
 w_k(t) = (1-s)/K + s * softmax-free positive-part of trailing Sharpe, shrunk toward equal (s in {0, .5, 1}).
Only information up to day t is used (trailing window of sub-sleeve P&L)."""
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
# sub-signals, each in [0,1]
subs={}
for L in (14,30,60,90,180): subs[f'mom{L}']=trend_signal(A,(L,),'mom')
for L in (20,50,100,200): subs[f'sma{L}']=trend_signal(A,(L,),'sma')
for fast,slow,sc in EWMAC_SPEEDS: subs[f'ew{fast}']=(ewmac_forecast(A,((fast,slow,sc),)).clip(lower=0)/10).clip(upper=1.0)
K=len(subs)
gate=(A>A.rolling(150).mean()).astype(float)
vol=ewma_vol(A,30)
def sleeve_ret(f):
    W=pd.DataFrame(0.0,index=P.index,columns=P.columns)
    W[assets]=((f*gate)*0.45/vol/2).fillna(0).clip(upper=1.0)
    return run_backtest(W,R,band=0.05), W
sl={k:sleeve_ret(f) for k,f in subs.items()}
SR=pd.DataFrame({k:v[0] for k,v in sl.items()})
def tilt_weights(s,win=730):
    mu=SR.rolling(win,min_periods=250).mean(); sd=SR.rolling(win,min_periods=250).std()
    sh=(mu/sd*np.sqrt(365)).clip(lower=0).fillna(0.0)
    adapt=sh.div(sh.sum(axis=1).replace(0,np.nan),axis=0).fillna(1.0/K)
    return (1-s)/K+s*adapt
def combine(s,win=730,tv=0.35):
    a=tilt_weights(s,win)
    Wsum=None
    for k in subs:
        Wk=sl[k][1]; contrib=Wk.mul(a[k].shift(0),axis=0)    # a[k] at close t (uses P&L through t) -> weight for day t+1
        Wsum=contrib if Wsum is None else Wsum+contrib
    ek=dict(band=0.05); r0=run_backtest(Wsum,R,**ek)
    W=Wsum.mul(vol_target_scale(r0,tv,30,1.5),axis=0); g=W.abs().sum(axis=1); W=W.mul((2.0/g).clip(upper=1.0).fillna(1.0),axis=0)
    return run_backtest(W,R,**ek)
def show(name,r):
    out=[]
    for a,b in [('2018-01-01','2021-12-31'),('2022-01-01','2026-05-23'),('2018-01-01','2026-05-23')]:
        s=M.summary(r.loc[a:b]); out.append(f"{a[:4]}-{b[2:4]}: CAGR {s['cagr']*100:5.1f}% Sh {s['sharpe']:4.2f} DD {s['maxdd']*100:6.1f}%")
    print(f"{name:34s}"+' | '.join(out))
for s in [0.0,0.5,1.0]:
    for win in [365,730]:
        if s==0.0 and win==730: continue
        show(f'adaptive tilt s={s} win={win}',combine(s,win))
