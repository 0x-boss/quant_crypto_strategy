import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from quant.xs_perp import load_panels, run_tsmom
from quant import metrics as M
C,QV,F=load_panels(); tc=pd.read_csv('results/intraday_daily.csv',index_col=0,parse_dates=True)
E1=('2020-12-01','2022-12-31'); E2=('2023-01-01','2026-10-05'); btc=C['BTCUSDT'].pct_change()
res={}
for n,ls in [('B1 TSMOM-LS',True),('B2 TSMOM-LO',False)]:
    o=run_tsmom(C,QV,F,long_short=ls); r=o['ret']; res[n]=r
    s1=M.summary(r.loc[E1[0]:E1[1]]); s2=M.summary(r.loc[E2[0]:E2[1]]); sf=M.summary(r)
    print(f"{n}: era1 CAGR {s1['cagr']*100:6.1f}% Sh {s1['sharpe']:5.2f} DD {s1['maxdd']*100:6.1f}% | era2 CAGR {s2['cagr']*100:6.1f}% Sh {s2['sharpe']:5.2f} DD {s2['maxdd']*100:6.1f}% | full Sh {sf['sharpe']:5.2f} t={r.mean()/(r.std()/np.sqrt(len(r))):4.1f} | corr(trend) {pd.concat([r,tc['total']],axis=1).dropna().corr().iloc[0,1]:+.2f} corr(BTC) {pd.concat([r,btc],axis=1).dropna().corr().iloc[0,1]:+.2f} | funding {o['funding'].mean()*36500:5.1f}%/yr cost {o['cost'].mean()*36500:4.1f}%/yr")
pd.DataFrame(res).to_csv('results/tsmom_perp_sleeves.csv')
