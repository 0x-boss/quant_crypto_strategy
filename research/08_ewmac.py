import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from quant.data import load_panels
from quant.engine import run_backtest
from quant.strategies import *
from quant import metrics as M
P,V=load_panels(); R=P.pct_change(fill_method=None)
periods={'full':('2017-01-01','2026-05-24'),'dev 17-21':('2017-01-01','2021-12-31'),'hold 22-26':('2022-01-01','2026-05-24')}
def show(name,r):
    row=[]
    for k,(a,b) in periods.items():
        s=M.summary(r.loc[a:b]); row.append(f"{k}: CAGR {s['cagr']*100:6.1f}% Sh {s['sharpe']:4.2f} DD {s['maxdd']*100:6.1f}% vol {s['vol']*100:4.1f}")
    print(f"{name:34s}"+' | '.join(row))
for assets in (['btc'],['btc','eth'],['btc','eth','bnb','xrp','ltc','ada','doge','link','trx']):
    A=P[assets]; vol=ewma_vol(A,30)
    fc=ewmac_forecast(A)                       # [-20,20]
    for mode,sig in [('L/O',fc.clip(lower=0)/10),('L/S',fc/10)]:
        n=len(assets)
        w=(sig*(0.25/vol)/n).clip(-1,1)         # each asset targets 25%/sqrt-ish vol share; no further scaling
        W=pd.DataFrame(0.0,index=P.index,columns=P.columns); W[assets]=w.fillna(0)
        show(f'{len(assets)} assets EWMAC {mode}',run_backtest(W,R))
