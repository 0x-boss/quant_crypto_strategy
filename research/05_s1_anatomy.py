import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from quant.data import load_panels
from quant.engine import run_backtest
from quant.strategies import *
from quant import metrics as M
pd.set_option('display.width',200)
P,V=load_panels(); R=P.pct_change(fill_method=None)
LB=(14,30,60,90,180)
A=P[['btc','eth']]
up=trend_signal(A,LB,'mom'); vol=ewma_vol(A,30)
W1=pd.DataFrame(0.0,index=P.index,columns=P.columns)
W1[['btc','eth']]=(up*0.30/vol/2).fillna(0).clip(upper=1.0)
r1,det=run_backtest(W1,R,return_details=True)
print(M.yearly(r1.loc['2017':]).round(3))
print('btc B&H yearly'); print(M.yearly(R['btc'].loc['2017':]).round(3))
print('avg gross exposure by year'); print(det['gross'].groupby(det.index.year).mean().round(2).to_dict())
print('avg turnover/day, annual cost drag by year'); print((det['turnover'].groupby(det.index.year).mean()).round(3).to_dict(), (det['cost'].groupby(det.index.year).sum()).round(3).to_dict())
# monthly returns 2022+
m=(1+r1).resample('ME').prod()-1
print((m.loc['2022':]*100).round(1).to_string())
