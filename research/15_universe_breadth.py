import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from quant.data import load_panels, liquid_universe
from quant.engine import run_backtest
from quant.strategies import *
from quant import metrics as M
P,V=load_panels(); R=P.pct_change(fill_method=None)
blocks={'17-19':('2017-01-01','2019-12-31'),'20-21':('2020-01-01','2021-12-31'),'22-23':('2022-01-01','2023-12-31'),'24-26':('2024-01-01','2026-05-24')}
def show(name,r):
    f=M.summary(r.loc['2017':]); s=' '.join(f"{k}:{M.summary(r.loc[a:b])['sharpe']:4.2f}" for k,(a,b) in blocks.items())
    print(f"{name:34s} CAGR {f['cagr']*100:5.1f}% Sh {f['sharpe']:4.2f} DD {f['maxdd']*100:6.1f}% vol {f['vol']*100:4.1f} Calmar {f['calmar']:4.2f} | {s}")
def fixed(assets):
    U=pd.DataFrame(False,index=P.index,columns=P.columns); U[assets]=P[assets].notna(); return U
unis={'BTC':fixed(['btc']),'BTC+ETH':fixed(['btc','eth']),
      'top5 liquid':liquid_universe(P,V,top_n=5,min_adv=5e6,min_hist=180),
      'top10 liquid':liquid_universe(P,V,top_n=10,min_adv=5e6,min_hist=180),
      'top20 liquid':liquid_universe(P,V,top_n=20,min_adv=5e6,min_hist=180)}
for name,U in unis.items():
    n_min=len(U.columns[U.any()]) if name in('BTC','BTC+ETH') else 5
    W=build_weights(P,R,U,n_min={'BTC':1,'BTC+ETH':2}.get(name,5),target_vol=0.30,max_lev=2.0)
    show(name+' (gate200, portTV30)',run_backtest(W,R))
