import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from quant.data import load_panels, liquid_universe
from quant.strategies import *
from quant import metrics as M
P,V=load_panels(); R=P.pct_change(fill_method=None)
U=liquid_universe(P,V,top_n=20,min_adv=5e6,min_hist=180)
lr=np.log(P).diff()
def tercile(score,q):
    s=score.where(U); rk=s.rank(axis=1,pct=True)
    lw=(rk>=1-q).astype(float); sw=(rk<=q).astype(float)
    lw=lw.div(lw.sum(axis=1).replace(0,np.nan),axis=0); sw=sw.div(sw.sum(axis=1).replace(0,np.nan),axis=0)
    return (lw-sw).fillna(0)*0.5
for name,sc in [('rev1',-lr),('rev3',-np.log(P/P.shift(3))),('rev7',-np.log(P/P.shift(7)))]:
    w=tercile(sc,1/3)
    g=(w.shift(1)*R.fillna(0)).sum(axis=1); to=w.diff().abs().sum(axis=1)
    for a,b in [('2017-01-01','2021-12-31'),('2022-01-01','2026-05-24')]:
        gg=g.loc[a:b]; print(f'{name} {a[:4]}-{b[:4]} GROSS Sharpe {gg.mean()/gg.std()*np.sqrt(365):5.2f}  gross ret/yr {gg.mean()*365*100:6.1f}%  turnover/day {to.loc[a:b].mean():.2f}  breakeven cost bp/turnover {gg.mean()/to.loc[a:b].mean()*1e4:5.1f}')
