"""Pre-sample test: identical BTC-only rule on 2011-2015 (never examined during development), 30bp costs."""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from quant.data import load_panels
from quant.engine import run_backtest
from quant.strategies import build_weights_bagged
from quant import metrics as M
P,V=load_panels(); R=P.pct_change(fill_method=None)
U=pd.DataFrame(False,index=P.index,columns=P.columns); U['btc']=P['btc'].notna()
for bps in [10,30,60]:
    W=build_weights_bagged(P,R,U,n_min=1,target_vol=0.35,max_lev=1.5,engine_kwargs=dict(tier1_bps=bps,band=0.05))
    r=run_backtest(W,R,tier1_bps=bps,band=0.05)
    for a,b in [('2012-01-01','2015-12-31'),('2012-01-01','2013-12-31'),('2014-01-01','2015-12-31'),('2013-01-01','2015-12-31')]:
        s=M.summary(r.loc[a:b]); bh=M.summary(R['btc'].loc[a:b])
        print(f"cost {bps}bp {a[:4]}-{b[:4]}: CAGR {s['cagr']*100:6.1f}% Sh {s['sharpe']:4.2f} DD {s['maxdd']*100:6.1f}% | BTC B&H: CAGR {bh['cagr']*100:6.1f}% Sh {bh['sharpe']:4.2f} DD {bh['maxdd']*100:6.1f}%")
