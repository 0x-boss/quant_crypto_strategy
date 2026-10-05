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
    print(f"{name:44s} CAGR {f['cagr']*100:5.1f}% Sh {f['sharpe']:4.2f} DD {f['maxdd']*100:6.1f}% vol {f['vol']*100:4.1f} Calmar {f['calmar']:4.2f} | {s}")
U=pd.DataFrame(False,index=P.index,columns=P.columns); U[['btc','eth']]=P[['btc','eth']].notna()
show('single: gate200 span30 TV.35 ml1.5',run_backtest(build_weights(P,R,U,n_min=2,target_vol=0.35,max_lev=1.5),R))
for sg in [('mix',),('mom','sma','ewmac')]:
    for tv,ml in [(0.35,1.5),(0.40,1.5),(0.35,2.0)]:
        W=build_weights_bagged(P,R,U,signals=sg,target_vol=tv,max_lev=ml); show(f'bagged {sg} TV {tv} ml {ml}',run_backtest(W,R))
