import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from quant.intraday import IConfig, run, hourly_prices
from quant import metrics as M
W={'2021-03+':('2021-03-01','2026-10-05'),'2022+':('2022-01-01','2026-10-05'),'2023+':('2023-01-01','2026-10-05'),'fwd OOS':('2026-05-24','2026-10-05')}
def line(name,r):
    out=[]
    for lab,(a,b) in W.items():
        s=M.summary(r.loc[a:b]); out.append(f"{lab}: {s['cagr']*100:6.1f}% Sh {s['sharpe']:5.2f} DD {s['maxdd']*100:6.1f}%")
    print(f'{name:26s}'+' | '.join(out))
px=hourly_prices(('btc','eth','sol')).resample('1D').last().pct_change()
print('--- buy & hold (same windows)')
for a in ['btc','eth','sol']:
    line(a.upper()+' B&H',px[a].fillna(0))
print('--- TrendCore-I (6h bars, hourly RV, OI overlay, carry on idle cash)')
res={}
for name,assets in [('BTC',('btc',)),('ETH',('eth',)),('SOL',('sol',)),('ETH+SOL',('eth','sol')),('BTC+ETH',('btc','eth')),('BTC+ETH+SOL',('btc','eth','sol'))]:
    o=run(IConfig(assets=assets,start='2021-03-01')); res[name]=o; line(name,o['total'])
print('--- trend leg only (no carry)')
for name,assets in [('ETH+SOL',('eth','sol')),('BTC+ETH+SOL',('btc','eth','sol')),('SOL',('sol',))]:
    line(name+' trend only',run(IConfig(assets=assets,start='2021-03-01',carry=False))['total'])
print('--- calendar years (total return %)')
yy=pd.DataFrame({k:v['total'].groupby(v['total'].index.year).apply(lambda g:(1+g).prod()-1)*100 for k,v in res.items()}).round(1)
yy['SOL B&H']=(px['sol'].fillna(0)['2021-03-01':].groupby(px['2021-03-01':].index.year).apply(lambda g:(1+g).prod()-1)*100).round(1)
yy['ETH B&H']=(px['eth']['2021-03-01':].groupby(px['2021-03-01':].index.year).apply(lambda g:(1+g).prod()-1)*100).round(1)
print(yy.to_string())
print('avg gross exposure 2022+:',{k:round(v['gross'].loc['2022':].mean(),2) for k,v in res.items()})
