import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from quant.data import load_panels
from quant.engine import run_backtest
from quant.strategies import *
from quant import metrics as M
P,V=load_panels(); R=P.pct_change(fill_method=None)
U=pd.DataFrame(False,index=P.index,columns=P.columns); U[['btc','eth']]=P[['btc','eth']].notna()
W=build_weights_bagged(P,R,U,target_vol=0.35,max_lev=1.5)
for band in [0,0.02,0.05,0.10]:
    r,det=run_backtest(W,R,band=band,return_details=True); s=M.summary(r.loc['2017':])
    d=det.loc['2017':]
    print(f"band {band:4.2f}: CAGR {s['cagr']*100:5.1f}% Sh {s['sharpe']:4.2f} DD {s['maxdd']*100:6.1f}% turnover/yr {d['turnover'].sum()/len(d)*365:5.1f}x cost/yr {d['cost'].sum()/len(d)*365*100:4.2f}%")
