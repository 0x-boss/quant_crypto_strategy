import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from quant.strategy import Config, run
from quant import metrics as M
pd.set_option('display.width',220)
r,det,W,P,R=run(Config(),details=True)
w=(W[['btc','eth']].sum(axis=1)).reindex(r.index)
held=w.shift(1)>0.02     # position held over day t
# episodes of consecutive held days
ep=[];start=None
for d,h in held.items():
    if h and start is None: start=d
    if (not h) and start is not None:
        ep.append((start,prev)); start=None
    prev=d
if start is not None: ep.append((start,prev))
rows=[]
for s,e in ep:
    seg=r.loc[s:e]; rows.append(dict(start=s.date(),end=e.date(),days=len(seg),ret=(1+seg).prod()-1,btc=(1+R['btc'].loc[s:e]).prod()-1,avgw=w.shift(1).loc[s:e].mean()))
E=pd.DataFrame(rows)
E['yr']=pd.to_datetime(E['start']).dt.year
e22=E[pd.to_datetime(E['start'])>='2022-01-01']
print(e22.round(3).to_string())
for lab,sub in [('2017-21',E[(E.yr>=2017)&(E.yr<=2021)]),('2022-26',e22)]:
    win=sub[sub.ret>0]; los=sub[sub.ret<=0]
    print(f"{lab}: episodes {len(sub)}, win rate {len(win)/len(sub):.0%}, avg win {win.ret.mean()*100:.1f}% avg loss {los.ret.mean()*100:.1f}%, avg days win {win.days.mean():.0f}/loss {los.days.mean():.0f}, sum ret {sub.ret.sum()*100:.0f}%")
