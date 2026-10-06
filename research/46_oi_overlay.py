import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from quant.intraday import IConfig, hourly_prices, target_weights_bars
from quant.engine import run_backtest
from quant.carry import carry_returns
from quant import metrics as M
px=hourly_prices(); cfg=IConfig(); P,R,W,ek=target_weights_bars(px,cfg)
m=pd.read_parquet('data/intraday/BTCUSDT_metrics_5m.parquet')
oi=np.log(m.sum_open_interest_value.resample('1D').last()); z=(oi-oi.rolling(365).mean())/oi.rolling(365).std()
zl=np.log(m.sum_open_interest_value.resample('1D').last()/px['btc'].resample('1D').last().reindex(oi.index))   # OI in coin terms (leverage per unit price)
zl=(zl-zl.rolling(365).mean())/zl.rolling(365).std()
car=pd.concat([carry_returns('BTCUSDT'),carry_returns('ETHUSDT')],axis=1).mean(axis=1)
def total(Wx):
    rr,dd=run_backtest(Wx,R,return_details=True,**ek); dly=lambda x:(1+x).groupby(x.index.date).prod()-1
    t=dly(rr); t.index=pd.to_datetime(t.index); g=dd['gross'].groupby(dd.index.date).mean(); g.index=pd.to_datetime(g.index)
    return t+(1-g.clip(upper=1.0)).reindex(t.index).fillna(1.0)*car.reindex(t.index).fillna(0.0)
def show(name,r):
    out=[]
    for lab,a,b in [('21-09..22','2021-09-01','2022-12-31'),('2023+','2023-01-01','2026-10-05'),('21-09+','2021-09-01','2026-10-05')]:
        s=M.summary(r.loc[a:b]); out.append(f"{lab}: CAGR {s['cagr']*100:5.1f}% Sh {s['sharpe']:4.2f} DD {s['maxdd']*100:6.1f}%")
    print(f'{name:34s}'+' | '.join(out))
show('baseline TrendCore-I+carry',total(W))
for lab,zz in [('OI value z',z),('OI coin-level z',zl)]:
    zb=zz.shift(1).reindex(W.index,method='ffill')       # known at previous day's close
    for s in [0.25,0.5,0.75,1.0]:
        mult=(1-s*zb.clip(0,2)/2).fillna(1.0)
        show(f'{lab}, strength {s}',total(W.mul(mult,axis=0)))
