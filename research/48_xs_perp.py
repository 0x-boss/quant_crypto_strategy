"""Run the 6 PRE-REGISTERED candidates (research/PREREGISTRATION.md). Nothing else is run on this data."""
import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from quant.xs_perp import load_panels, run_sleeve
from quant import metrics as M
C,QV,F=load_panels()
tc=pd.read_csv('results/intraday_daily.csv',index_col=0,parse_dates=True)
E1=('2020-12-01','2022-12-31'); E2=('2023-01-01','2026-10-05')
res={}
print(f"{'cand':8s} | era1 20-22: CAGR  Sh | era2 23-26: CAGR  Sh | full Sh  t-stat | funding%/yr  cost%/yr  avg gross | corr(trend) corr(BTC)")
btc=C['BTCUSDT'].pct_change()
for n in ['MOM30','MOM90','REV7','LOWVOL','FUND14','MULTI']:
    o=run_sleeve(C,QV,F,n); r=o['ret']; res[n]=o
    s1=M.summary(r.loc[E1[0]:E1[1]]); s2=M.summary(r.loc[E2[0]:E2[1]]); sf=M.summary(r)
    t=r.mean()/r.std()*np.sqrt(len(r))
    cor=pd.concat([r,tc['total']],axis=1).dropna().corr().iloc[0,1]; cb=pd.concat([r,btc],axis=1).dropna().corr().iloc[0,1]
    print(f"{n:8s} | {s1['cagr']*100:7.1f}% {s1['sharpe']:5.2f} | {s2['cagr']*100:7.1f}% {s2['sharpe']:5.2f} | {sf['sharpe']:5.2f} {t/np.sqrt(365):5.2f}*sqrt(yrs)->t={r.mean()/(r.std()/np.sqrt(len(r))):4.1f} | {o['funding'].mean()*365*100:6.1f} {o['cost'].mean()*365*100:6.1f} {o['gross'].mean():4.2f} | {cor:+.2f} {cb:+.2f}")
pd.DataFrame({k:v['ret'] for k,v in res.items()}).to_csv('results/xs_perp_sleeves.csv')
