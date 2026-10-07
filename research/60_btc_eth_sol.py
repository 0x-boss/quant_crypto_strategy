"""User request (post-hoc, NOT pre-registered): hold BTC+ETH+SOL, signal computed on BTC only, each asset sized by its own volatility.
Spot-only, gross<=1.  SOL is a known survivor -> hindsight warning.  Compared with BTC-only, BTC+ETH (BTC signal) and the official BTC+ETH (own signals)."""
import sys; sys.path.insert(0,'.')
import importlib, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from quant import metrics as M
from quant.intraday import hourly_prices
import trendcore_spot as T
s51=importlib.import_module('research.51_btc_signal_swap') if False else None
exec(open('research/51_btc_signal_swap.py').read().split("if __name__=='__main__':")[0])
oi=pd.read_parquet('data/intraday/BTCUSDT_oi_daily.parquet')['sum_open_interest_value']
px=hourly_prices(('btc','eth','sol')).sort_index()
k=4; P=px.resample('6h').last().dropna(how='all'); rv2=(np.log(px).diff()**2).resample('6h').sum().reindex(P.index)
V={}
for name,tg,sz in [('BTC only (BTC signal)',['btc'],'own'),('BTC+ETH (BTC signal, own vol)',['btc','eth'],'own'),
                   ('BTC+ETH+SOL (BTC signal, own vol)',['btc','eth','sol'],'own'),('BTC+ETH+SOL (BTC signal, BTC vol)',['btc','eth','sol'],'swap')]:
    r,W=swap_strategy(P,rv2,k,tg,sz,oi); V[name]=r
    if name.startswith('BTC+ETH+SOL (BTC signal, own'): Wf=W
# official strategy (own signals) on the same window
rb,_=T.backtest(px[['btc','eth']],oi,start='2018-03-01'); V['official BTC+ETH (own signals)']=rb
W1=(ERA1[0],'2022-12-31'); 
wins={'2020-09..22':ERA1,'2023-26':ERA2,'full 2020-09+':(ERA1[0],ERA2[1]),'2022+':('2022-01-01',ERA2[1])}
print(f"{'':36s}"+' | '.join(f'{k:>26s}' for k in wins))
for n,r in V.items():
    o=[]
    for lab,(a,b) in wins.items():
        s=M.summary(r.loc[a:b]); o.append(f"{s['cagr']*100:5.1f}% {s['sharpe']:4.2f} {s['maxdd']*100:6.1f}%")
    print(f'{n:36s}'+' | '.join(f'{x:>26s}' for x in o))
y=pd.DataFrame({n:r.loc['2020-09-01':].groupby(r.loc['2020-09-01':].index.year).apply(lambda x:(1+x).prod()-1)*100 for n,r in V.items()}); print('\ncalendar years %'); print(y.round(1).to_string())
print('\navg gross exposure (3-asset):',round(Wf.sum(axis=1).loc['2020-09-01':].mean(),2),' avg weights:',Wf.loc['2020-09-01':].mean().round(3).to_dict())
pd.DataFrame(V).to_csv('results/btc_eth_sol_6h.csv')
