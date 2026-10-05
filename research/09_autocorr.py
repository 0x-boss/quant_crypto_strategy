import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from quant.data import load_panels, liquid_universe
from quant.engine import run_backtest
from quant.strategies import *
from quant import metrics as M
P,V=load_panels(); R=P.pct_change(fill_method=None)
U=liquid_universe(P,V,top_n=20,min_adv=5e6,min_hist=180)
lr=np.log(P).diff()
ew=(lr.where(U)).mean(axis=1)           # equal-weight liquid-universe log return
btc=lr['btc']; eth=lr['eth']
def ac(x,lag,a,b):
    x=x.loc[a:b]; return x.autocorr(lag)
for name,x in [('BTC',btc),('ETH',eth),('EW top20',ew)]:
    for a,b in [('2017-01-01','2019-12-31'),('2020-01-01','2022-12-31'),('2023-01-01','2026-05-24')]:
        print(f'{name:9s} {a[:4]}-{b[:4]}  AC lag1..5:', ' '.join(f'{ac(x,l,a,b):+.3f}' for l in range(1,6)))
# long/flat on short TSMOM for the EW market via BTC+ETH
periods={'full':('2017-01-01','2026-05-24'),'dev 17-21':('2017-01-01','2021-12-31'),'hold 22-26':('2022-01-01','2026-05-24')}
def show(name,r):
    row=[]
    for k,(a,b) in periods.items():
        s=M.summary(r.loc[a:b]); row.append(f"{k}: CAGR {s['cagr']*100:6.1f}% Sh {s['sharpe']:4.2f} DD {s['maxdd']*100:6.1f}%")
    print(f"{name:34s}"+' | '.join(row))
for L in [1,2,3,5]:
    for mode in ['L/O','L/S']:
        sig=np.sign(np.log(P[['btc','eth']]/P[['btc','eth']].shift(L)))
        if mode=='L/O': sig=sig.clip(lower=0)
        W=pd.DataFrame(0.0,index=P.index,columns=P.columns); W[['btc','eth']]=sig*0.5
        show(f'BTC+ETH TSMOM{L} {mode} (no vol scaling)',run_backtest(W,R))
