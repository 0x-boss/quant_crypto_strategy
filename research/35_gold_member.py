import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from quant.data import load_panels, DATA_DIR
from quant.engine import run_backtest
from quant.strategies import *
from quant import metrics as M
import os
P0=pd.read_parquet(os.path.join(DATA_DIR,'panel_price.parquet')); V0=pd.read_parquet(os.path.join(DATA_DIR,'panel_volume.parquet'))
print('PAXG ADV (median daily reported spot volume, $M) by year:', (V0['paxg'].groupby(V0.index.year).median()/1e6).round(1).loc[2020:].to_dict())
print('XAUT ADV:', (V0['xaut'].groupby(V0.index.year).median()/1e6).round(1).loc[2020:].to_dict())
P=P0[['btc','eth','paxg','xaut']]; R=P.pct_change(fill_method=None)
def run_universe(cols,tv=0.35,start='2020-06-01'):
    A=P[cols]; U=A.notna()
    n=U.sum(axis=1).clip(lower=len(cols)); acc=None; cnt=0
    mix=trend_forecast(A,'mix')
    for g in (100,150,200):
        f=(mix*(A>A.rolling(g).mean())).fillna(0.0).where(U,0.0)
        for vs in (20,30,60):
            v=ewma_vol(A,vs); w=(f*0.45/v).div(n,axis=0).where(U).fillna(0.0).clip(upper=1.0)
            acc=w if acc is None else acc+w; cnt+=1
    W=acc/cnt
    # engine tier: treat gold as 'other' (25bp) -> conservative
    ek=dict(band=0.05)
    r0=run_backtest(W,R[cols],**ek); W=W.mul(vol_target_scale(r0,tv,30,1.5),axis=0)
    g=W.abs().sum(axis=1); W=W.mul((2.0/g).clip(upper=1.0).fillna(1.0),axis=0)
    return run_backtest(W,R[cols],**ek).loc[start:]
def show(name,r):
    out=[]
    for a,b in [('2020-06-01','2021-12-31'),('2022-01-01','2026-05-23'),('2020-06-01','2026-05-23')]:
        s=M.summary(r.loc[a:b]); out.append(f"{a[:4]}-{b[2:4]}: CAGR {s['cagr']*100:5.1f}% Sh {s['sharpe']:4.2f} DD {s['maxdd']*100:6.1f}%")
    print(f"{name:34s}"+' | '.join(out))
show('BTC+ETH',run_universe(['btc','eth']))
show('BTC+ETH+PAXG (equal risk)',run_universe(['btc','eth','paxg']))
show('BTC+ETH+XAUT (equal risk)',run_universe(['btc','eth','xaut']))
show('BTC+ETH+PAXG+XAUT',run_universe(['btc','eth','paxg','xaut']))
print(M.yearly(run_universe(['btc','eth'])).loc[2020:,'sharpe'].round(2).to_dict()); print(M.yearly(run_universe(['btc','eth','paxg'])).loc[2020:,'sharpe'].round(2).to_dict())
