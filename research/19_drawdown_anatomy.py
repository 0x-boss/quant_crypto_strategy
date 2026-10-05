import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from quant.data import load_panels
from quant.engine import run_backtest
from quant.strategies import *
from quant import metrics as M
pd.set_option('display.width',200)
P,V=load_panels(); R=P.pct_change(fill_method=None)
U=pd.DataFrame(False,index=P.index,columns=P.columns); U[['btc','eth']]=P[['btc','eth']].notna()
W=build_weights(P,R,U,n_min=2,target_vol=0.35,max_lev=2.0)
r,det=run_backtest(W,R,return_details=True)
r=r.loc['2016':]; det=det.loc['2016':]
dd=M.drawdown_series(r)
# top drawdown episodes
ep=[]; inn=False
for d,v in dd.items():
    if v<0 and not inn: start=d; inn=True; trough=d; tv=v
    elif v<0 and inn:
        if v<tv: tv=v; trough=d
    elif v==0 and inn:
        ep.append((start,trough,d,tv)); inn=False
if inn: ep.append((start,trough,dd.index[-1],tv))
ep=sorted(ep,key=lambda x:x[3])[:6]
for s,t,e,v in ep:
    print(f'DD {v*100:6.1f}%  start {s.date()} trough {t.date()} end {e.date()}  gross at trough {det.loc[t,"gross"]:.2f}  btc move start->trough {(P.btc[t]/P.btc[s]-1)*100:6.1f}%')
print('gross exposure distribution:'); print(det['gross'].describe().round(2).to_dict())
print('days at gross>1.5:',(det['gross']>1.5).mean().round(3),' time flat (gross<0.05):',(det['gross']<0.05).mean().round(3))
print('financing total %/yr', (det['financing'].sum()/len(det)*365*100).round(2),' cost %/yr',(det['cost'].sum()/len(det)*365*100).round(2))
