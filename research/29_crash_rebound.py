"""Event study: forward returns after large vol-adjusted daily drops, BTC/ETH and pooled liquid top-20. Two eras."""
import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from quant.data import load_panels, liquid_universe
P,V=load_panels(); lr=np.log(P).diff()
U=liquid_universe(P,V,top_n=20,min_adv=5e6,min_hist=180)
vol=lr.ewm(span=30,min_periods=15).std().shift(1)      # vol known BEFORE the shock day
z=lr/vol
def study(cols,label,thr):
    rows=[]
    for era,(a,b) in {'2017-21':('2017-01-01','2021-12-31'),'2022-26':('2022-01-01','2026-05-23')}.items():
        res={}
        for h in [1,2,3,5,7]:
            fwd=np.log(P.shift(-h)/P)
            ev=(z<-thr)&U if label.startswith('pool') else (z<-thr)
            ev=ev[cols].loc[a:b]; f=fwd[cols].loc[a:b]
            x=f.where(ev).stack()
            # standardise by vol*sqrt(h) so assets are comparable
            sd=(vol[cols].loc[a:b]*np.sqrt(h)).where(ev).stack()
            xs=(x/sd)
            res[h]=(len(x), x.mean()*1e4, xs.mean(), xs.mean()/ (xs.std()/np.sqrt(max(len(xs)/3,1))) )
        rows.append((era,res))
    print(f'== {label}  shock z< -{thr}')
    for era,res in rows:
        print(f'  {era}: '+' | '.join(f'h{h}: n={r[0]} mean {r[1]:+.0f}bp  t={r[3]:+.1f}' for h,r in res.items()))
for thr in [2.0,3.0]:
    study(['btc','eth'],'BTC+ETH',thr)
    study(list(P.columns),'pool top20',thr)
