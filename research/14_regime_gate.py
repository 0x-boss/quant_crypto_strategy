import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from quant.data import load_panels
from quant.engine import run_backtest
from quant.strategies import *
from quant import metrics as M
P,V=load_panels(); R=P.pct_change(fill_method=None)
blocks={'17-19':('2017-01-01','2019-12-31'),'20-21':('2020-01-01','2021-12-31'),'22-23':('2022-01-01','2023-12-31'),'24-26':('2024-01-01','2026-05-24')}
def show(name,r):
    f=M.summary(r.loc['2017':]); s=' '.join(f"{k}:{M.summary(r.loc[a:b])['sharpe']:4.2f}" for k,(a,b) in blocks.items())
    print(f"{name:40s} CAGR {f['cagr']*100:5.1f}% Sh {f['sharpe']:4.2f} DD {f['maxdd']*100:6.1f}% vol {f['vol']*100:4.1f} Calmar {f['calmar']:4.2f} | {s}")
assets=['btc','eth']
A=P[assets]; vol=ewma_vol(A,30)
fmix=(trend_signal(A,(14,30,60,90,180),'mom')+trend_signal(A,(20,50,100,200),'sma')+(ewmac_forecast(A).clip(lower=0)/10).clip(upper=1))/3
def mk(f,tv=0.30):
    W=pd.DataFrame(0.0,index=P.index,columns=P.columns); W[assets]=(f*tv/vol/2).fillna(0).clip(upper=1.0); return W
show('mix (no gate)',run_backtest(mk(fmix),R))
for L in [100,150,200]:
    gate=(A>A.rolling(L).mean()).astype(float)
    show(f'mix x gate P>SMA{L} (own)',run_backtest(mk(fmix*gate),R))
    # soft gate
    show(f'mix x (0.4+0.6 gate{L})',run_backtest(mk(fmix*(0.4+0.6*gate)),R))
# gate on BTC only (market regime) for both assets
for L in [100,200]:
    g=(P['btc']>P['btc'].rolling(L).mean()).astype(float)
    show(f'mix x BTC-regime gate SMA{L}',run_backtest(mk(fmix.mul(g,axis=0)),R))
# slow-only below 200d: if own price < SMA200 use only the slow half of signals
