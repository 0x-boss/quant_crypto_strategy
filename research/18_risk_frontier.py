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
    print(f"{name:40s} CAGR {f['cagr']*100:5.1f}% Sh {f['sharpe']:4.2f} DD {f['maxdd']*100:6.1f}% vol {f['vol']*100:4.1f} Calmar {f['calmar']:4.2f} worst {f['worst_day']*100:5.1f}% | {s}")
U=pd.DataFrame(False,index=P.index,columns=P.columns); U[['btc','eth']]=P[['btc','eth']].notna()
ovs={'none':None,'mild':dict(start=0.07,floor_at=0.25,floor=0.4),'mod':dict(start=0.05,floor_at=0.20,floor=0.3)}
for tv in [0.30,0.35,0.40,0.45]:
    for on,ov in ovs.items():
        W=build_weights(P,R,U,n_min=2,target_vol=tv,max_lev=2.0,overlay=ov)
        show(f'TV {tv:.2f} overlay {on}',run_backtest(W,R))
