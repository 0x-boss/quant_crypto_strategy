"""TrendCore extended to 2026-10-05 with Binance spot closes (chained returns), plus BTC/ETH carry on idle capital."""
import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from quant.data import load_panels
from quant.engine import run_backtest
from quant.strategy import Config, target_weights
from quant.carry import carry_returns
from quant import metrics as M
P,_=load_panels(); 
# extend BTC/ETH after CM end with Binance spot daily closes (chain returns)
ext={}
for a,s in [('btc','BTCUSDT'),('eth','ETHUSDT')]:
    sp=pd.read_parquet(f'data/intraday/{s}_spot_1h.parquet')['c'].resample('1D').last()
    last=P[a].last_valid_index(); r=sp.pct_change().loc[last+pd.Timedelta(days=1):]
    new=P[a].dropna().iloc[-1]*(1+r).cumprod()
    ext[a]=pd.concat([P[a].dropna(),new])
PE=pd.DataFrame(ext); PE=PE[PE.index<'2026-10-06']
print('extended to',PE.index[-1].date(),'rows',len(PE))
R=PE.pct_change(fill_method=None)
cfg=Config()
W=target_weights(PE,R,cfg)
tr,det=run_backtest(W,R,return_details=True,**cfg.engine_kwargs())
car=pd.concat([carry_returns('BTCUSDT'),carry_returns('ETHUSDT')],axis=1).mean(axis=1).reindex(tr.index).fillna(0.0)
idle=(1-det['gross'].clip(upper=1.0))
comb=tr+idle*car
def row(name,r):
    out=[]
    for lab,a,b in [('2020-03..2026-10','2020-03-01','2026-10-05'),('2022+','2022-01-01','2026-10-05'),('NEW OOS 2026-05-24..10-05','2026-05-24','2026-10-05')]:
        s=M.summary(r.loc[a:b]) or {}
        if s: out.append(f"{lab}: CAGR {s['cagr']*100:6.1f}% Sh {s['sharpe']:5.2f} DD {s['maxdd']*100:6.1f}%")
    print(f'{name:28s}'+' | '.join(out))
row('TrendCore',tr); row('carry BTC/ETH (per capital)',car); row('Trend + carry on idle cash',comb)
print('corr(trend, carry) daily:',round(tr.loc['2020-03':].corr(car.loc['2020-03':]),3),'avg idle capital 2022+:',round(idle.loc['2022':].mean(),2))
print('BTC buy&hold OOS window return %.1f%%, TrendCore %.1f%%'%(((1+R['btc'].loc['2026-05-24':]).prod()-1)*100,((1+tr.loc['2026-05-24':]).prod()-1)*100))
pd.DataFrame({'trend':tr,'carry':car,'combined':comb,'gross':det['gross']}).to_csv('results/trend_carry_daily.csv')
