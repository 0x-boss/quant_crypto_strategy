import sys, warnings, importlib.util
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0,'.')
from quant import metrics as M
from quant.intraday import hourly_prices, spot_only, run
spec=importlib.util.spec_from_file_location('c51','research/51_btc_signal_swap.py'); c51=importlib.util.module_from_spec(spec); spec.loader.exec_module(c51)
oi=pd.read_parquet('data/intraday/BTCUSDT_oi_daily.parquet')['sum_open_interest_value']
px=hourly_prices(('btc','eth','sol')).sort_index(); P=px.resample('6h').last().dropna(how='all'); rv2=(np.log(px).diff()**2).resample('6h').sum().reindex(P.index)
S={}
S['BTC only (ref)'],_=c51.swap_strategy(P,rv2,4,['btc'],'own',oi)
S['BTC+SOL, BTC signal, own-vol']= c51.swap_strategy(P,rv2,4,['btc','sol'],'own',oi)[0]
S['BTC+SOL, BTC signal, BTC-vol (swap)']= c51.swap_strategy(P,rv2,4,['btc','sol'],'swap',oi)[0]
r=run(spot_only(assets=('btc','sol'),start='2020-09-01'),px[['btc','sol']])['total']; S['BTC+SOL, each on own signal']=r
r=run(spot_only(assets=('btc','eth'),start='2020-09-01'),px[['btc','eth']])['total']; S['BTC+ETH, own signals (final)']=r
for k,v in S.items():
    out=[]
    for lab,(a,b) in {'2020-09..22':('2020-09-01','2022-12-31'),'2023-26':('2023-01-01','2026-10-05'),'full':('2020-09-01','2026-10-05'),'2022+':('2022-01-01','2026-10-05')}.items():
        s=M.summary(v.loc[a:b]); out.append(f"{lab}: {s['cagr']*100:5.1f}% Sh {s['sharpe']:4.2f} DD {s['maxdd']*100:6.1f}%")
    print(f'{k:38s}'+' | '.join(out))
pd.DataFrame(S).to_csv('results/btc_sol.csv')
