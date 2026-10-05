"""Gated bear-regime short sleeve (symmetric to the long gate), BTC+ETH.  Both-eras rule.
Short funding/borrow charged at 10%/yr on short notional via an extra cost term."""
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
mix=trend_forecast(A,'mix'); bear=1.0-mix
def build(short_mult,short_gate_fast=50,tv=0.35,short_fund=0.10):
    n=U.sum(axis=1).clip(lower=2); accL=None; accS=None; cnt=0
    for g in (100,150,200):
        lg=(A>A.rolling(g).mean()).astype(float)
        sg=(A<A.rolling(g).mean()).astype(float)
        if short_gate_fast: sg=sg*(A<A.rolling(short_gate_fast).mean())
        fl=(mix*lg).reindex(columns=P.columns).fillna(0.0).where(U,0.0)
        fs=(bear*sg).reindex(columns=P.columns).fillna(0.0).where(U,0.0)
        for vs in (20,30,60):
            v=ewma_vol(P,vs)
            wl=(fl*0.45/v).div(n,axis=0).where(U).fillna(0.0).clip(upper=1.0)
            ws=(fs*0.45/v).div(n,axis=0).where(U).fillna(0.0).clip(upper=1.0)
            accL=wl if accL is None else accL+wl; accS=ws if accS is None else accS+ws; cnt+=1
    W=accL/cnt-short_mult*accS/cnt
    ek=dict(band=0.05)
    r0=run_backtest(W,R,**ek); W=W.mul(vol_target_scale(r0,tv,30,1.5),axis=0)
    g=W.abs().sum(axis=1); W=W.mul((2.0/g).clip(upper=1.0).fillna(1.0),axis=0)
    return W
def pnl(W,fund=0.10):
    r,det=run_backtest(W,R,band=0.05,return_details=True)
    short_notional=(-W.where(W<0,0.0)).sum(axis=1).shift(1).reindex(r.index).fillna(0.0)
    return r-short_notional*fund/365.0
def show(name,r):
    out=[]
    for a,b in [('2018-01-01','2021-12-31'),('2022-01-01','2026-05-23'),('2016-01-01','2026-05-23')]:
        s=M.summary(r.loc[a:b]); out.append(f"{a[:4]}-{b[2:4]}: CAGR {s['cagr']*100:5.1f}% Sh {s['sharpe']:4.2f} DD {s['maxdd']*100:6.1f}%")
    print(f"{name:40s}"+' | '.join(out))
show('long-only (current TrendCore)',pnl(build(0.0)))
for sm in [0.5,1.0]:
    for fast in [None,50]:
        show(f'gated short x{sm}, short fast gate {fast}',pnl(build(sm,fast)))
# year by year for the best-looking
W=build(1.0,50); r=pnl(W); r0=pnl(build(0.0))
print(pd.concat([M.yearly(r0)['ret'],M.yearly(r)['ret']],axis=1,keys=['long-only','L+gated short']).round(3).T.to_string())
