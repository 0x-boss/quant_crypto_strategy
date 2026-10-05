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
def run_variant(gate_fn,tv=0.35,score=mix):
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
    print(f"{name:42s}"+' | '.join(out))
def mk(fastL=None,btc_confirm_eth=False,eth_confirm_btc=False):
    def gate(g):
        base=(A>A.rolling(g).mean()).astype(float)
        if fastL: base=base*(A>A.rolling(fastL).mean())
        out=base.copy()
        if btc_confirm_eth: out['eth']=base['eth']*base['btc']
        if eth_confirm_btc: out['btc']=base['btc']*base['eth']
        return out
    return gate
for fl in [None,20,30,50,75]:
    show(f'fast gate SMA{fl}, ETH needs BTC',run_variant(mk(fl,True)))
show('SMA50, BTC needs ETH',run_variant(mk(50,False,True)))
show('SMA50, mutual confirmation',run_variant(mk(50,True,True)))
show('no fast gate, ETH needs BTC',run_variant(mk(None,True)))
