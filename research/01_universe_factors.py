"""Exploration: how many liquid assets are available per year, and do simple cross-sectional factors
(dollar-neutral, equal-weight top/bottom tercile) carry any robust signal?  Only canonical horizons."""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from quant.data import load_panels, liquid_universe
from quant import metrics as M
pd.set_option('display.width',220)
P,V=load_panels()
R=P.pct_change(fill_method=None)
U=liquid_universe(P,V,top_n=20,min_adv=5e6,min_hist=180)
print('members/yr', U.sum(1).groupby(U.index.year).mean().round(1).to_dict())
print('ever members', sorted(U.columns[U.any()]))
def xs_factor(score, U, q=1/3, rebal=1):
    """long top q / short bottom q of score among members; returns daily dollar-neutral (1/2 long,1/2 short) return"""
    s=score.where(U)
    rk=s.rank(axis=1,pct=True)
    lw=(rk>=1-q).astype(float); sw=(rk<=q).astype(float)
    lw=lw.div(lw.sum(1).replace(0,np.nan),axis=0); sw=sw.div(sw.sum(1).replace(0,np.nan),axis=0)
    w=(lw-sw).fillna(0)*0.5
    return w
facs={}
lr=np.log(P).diff()
for L in [7,14,30,60,90,180]:
    facs[f'mom{L}']=np.log(P/P.shift(L))
for L in [1,3,5]:
    facs[f'rev{L}']=-np.log(P/P.shift(L))
facs['lowvol30']=-lr.rolling(30).std()
facs['mom30_skip7']=np.log(P.shift(7)/P.shift(37))
facs['mom90_skip7']=np.log(P.shift(7)/P.shift(97))
for k,sc in facs.items():
    w=xs_factor(sc,U)
    # factor return: weights decided at t earn R[t+1]
    fr=(w.shift(1)*R.fillna(0)).sum(1)
    to=(w.diff().abs().sum(1))
    fr_net=fr-to*0.0020   # 20bp per unit traded
    out=[]
    for a,b in [('2017-01-01','2021-12-31'),('2022-01-01','2026-05-24')]:
        s=M.summary(fr_net.loc[a:b]); out.append(f"{a[:4]}-{b[:4]} Sh {s['sharpe']:5.2f} ret/yr {s['cagr']*100:6.1f}% DD {s['maxdd']*100:6.1f}% to {to.loc[a:b].mean():.2f}")
    print(f'{k:12s}',' | '.join(out))
