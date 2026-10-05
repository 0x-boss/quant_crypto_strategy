import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from quant.strategies import *
from quant.engine import run_backtest
from quant import metrics as M
P=pd.read_parquet('data/panel_price.parquet'); V=pd.read_parquet('data/panel_volume.parquet')
R=P.pct_change(fill_method=None)
print('paxg first',P['paxg'].first_valid_index(),'xaut',P['xaut'].first_valid_index())
print('paxg vol', 'paxg' in V.columns, 'xaut' in V.columns)
for a in ['paxg','xaut']:
    x=P[a].dropna(); print(a, (x.resample('YE').last().pct_change()*100).round(1).to_dict())
print('corr daily ret paxg vs btc', R[['paxg','btc','eth']].loc['2021':].corr().round(2).to_dict())
periods={'full':('2020-06-01','2026-05-24'),'dev 20-21':('2020-06-01','2021-12-31'),'hold 22-26':('2022-01-01','2026-05-24')}
def show(name,r):
    row=[]
    for k,(a,b) in periods.items():
        s=M.summary(r.loc[a:b]); row.append(f"{k}: CAGR {s['cagr']*100:6.1f}% Sh {s['sharpe']:4.2f} DD {s['maxdd']*100:6.1f}% vol {s['vol']*100:4.1f}")
    print(f"{name:30s}"+' | '.join(row))
LB=(14,30,60,90,180)
def sleeve(assets,tv=0.30):
    A=P[assets]; up=trend_signal(A,LB,'mom'); vol=ewma_vol(A,30)
    W=pd.DataFrame(0.0,index=P.index,columns=P.columns)
    W[assets]=(up*tv/vol/len(assets)).fillna(0).clip(upper=1.0)
    return W
for name,assets in [('BTC+ETH',['btc','eth']),('PAXG',['paxg']),('XAUT',['xaut'])]:
    show(name,run_backtest(sleeve(assets),R))
W=sleeve(['btc','eth'])+sleeve(['paxg'],0.15)*1.0
show('BTC+ETH + 0.5x PAXG sleeve',run_backtest(W,R))
rb=run_backtest(sleeve(['btc','eth']),R); rg=run_backtest(sleeve(['paxg']),R)
print('corr of sleeves 2021+', pd.concat([rb,rg],axis=1).loc['2021':].corr().iloc[0,1].round(2))
