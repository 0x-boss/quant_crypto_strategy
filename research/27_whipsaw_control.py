"""Whipsaw control: EWMA-smoothed trend score, and Schmitt-trigger hysteresis on the score."""
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
base=trend_forecast(A,'mix')
def hysteresis(f,hi,lo):
    out=pd.DataFrame(0.0,index=f.index,columns=f.columns); st=pd.Series(0.0,index=f.columns)
    for i,(d,row) in enumerate(f.iterrows()):
        on=(row>=hi)|((st>0)&(row>lo)); st=on.astype(float); out.iloc[i]=np.where(on,row,0.0)
    return out
def run_variant(fa):
    # replicate build_weights_bagged but with a custom score passed through the gate bagging
    n=U.sum(axis=1).clip(lower=2); acc=None; cnt=0
    for g in (100,150,200):
        f=fa*(P[assets]>P[assets].rolling(g).mean()); f=f.reindex(columns=P.columns).fillna(0.0).where(U,0.0)
        for vs in (20,30,60):
            vol=ewma_vol(P,vs); w=(f*0.45/vol).div(n,axis=0).where(U).fillna(0.0).clip(upper=1.0)
            acc=w if acc is None else acc+w; cnt+=1
    W=acc/cnt; ek=dict(band=0.05)
    r0=run_backtest(W,R,**ek); W=W.mul(vol_target_scale(r0,0.35,30,1.5),axis=0)
    g=W.abs().sum(axis=1); W=W.mul((2.0/g).clip(upper=1.0).fillna(1.0),axis=0)
    return run_backtest(W,R,**ek)
def show(name,r):
    out=[]
    for a in ['2016-01-01','2018-01-01','2022-01-01']:
        s=M.summary(r.loc[a:]); out.append(f"{a[:4]}+: CAGR {s['cagr']*100:5.1f}% Sh {s['sharpe']:4.2f} DD {s['maxdd']*100:6.1f}%")
    print(f"{name:34s}"+' | '.join(out))
show('baseline',run_variant(base))
for k in [3,7,14]:
    show(f'EWMA-smoothed score span {k}',run_variant(base.ewm(span=k).mean()))
for hi,lo in [(0.6,0.3),(0.7,0.4),(0.5,0.2)]:
    show(f'hysteresis enter>={hi} exit<={lo}',run_variant(hysteresis(base,hi,lo)))
