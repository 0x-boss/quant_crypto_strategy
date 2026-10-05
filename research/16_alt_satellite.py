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
    print(f"{name:40s} CAGR {f['cagr']*100:5.1f}% Sh {f['sharpe']:4.2f} DD {f['maxdd']*100:6.1f}% vol {f['vol']*100:4.1f} Calmar {f['calmar']:4.2f} | {s}")
U=pd.DataFrame(False,index=P.index,columns=P.columns); U[['btc','eth']]=P[['btc','eth']].notna()
core=build_weights(P,R,U,n_min=2,target_vol=None)           # unscaled core (per-asset 45%/2)
rcore=run_backtest(core,R)
Ul=liquid_universe(P,V,top_n=12,min_adv=5e6,min_hist=180)
Ualt=Ul.copy(); Ualt[['btc','eth']]=False
f=trend_forecast(P,'mix'); f=f*(P>P.rolling(200).mean()); 
rel=np.log(P/P.shift(90)).sub(np.log(P['btc']/P['btc'].shift(90)),axis=0)
strong=(f>=0.8)&(rel>0)&Ualt
vol=ewma_vol(P,30)
for budget in [0.0,0.15,0.30]:
    Wa=(strong*(0.45/vol)).div(Ualt.sum(axis=1).clip(lower=5),axis=0).fillna(0).clip(upper=0.15)*budget/0.15*0.15  # per-name cap 15%
    Wa=Wa.where(strong,0.0)
    W=core+Wa
    r0=run_backtest(W,R)
    lev=vol_target_scale(r0,0.30,30,2.0); show(f'core + alt satellite (budget {budget})',run_backtest(W.mul(lev,axis=0),R))
