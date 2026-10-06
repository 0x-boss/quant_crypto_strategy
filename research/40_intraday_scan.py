"""Hourly scan on BTC/ETH perp: stable edges across eras 2020-22 vs 2023-26?  (t-stats are for non-overlapping-ish samples)"""
import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
D='data/intraday/'
ERA={'20-22':('2020-01-01','2022-12-31'),'23-26':('2023-01-01','2026-10-05')}
def tstat(x): x=x.dropna(); return x.mean()/(x.std()/np.sqrt(len(x))) if len(x)>30 else np.nan
for sym in ['BTCUSDT','ETHUSDT']:
    k=pd.read_parquet(D+f'{sym}_perp_1h.parquet'); c=k['c']; r=np.log(c).diff()
    print('=====',sym)
    # a) hour-of-day mean return (bp) t-stats, both eras
    hod=r.groupby(r.index.hour)
    for era,(a,b) in ERA.items():
        x=r.loc[a:b]; g=x.groupby(x.index.hour).apply(tstat)
        print(f'  hour-of-day t-stats {era}:',' '.join(f'{h}:{g[h]:+.1f}' for h in range(24) if abs(g[h])>2.0) or 'none |t|>2')
    # b) intraday momentum: return over hours 0-? predicts next hours (same UTC day)
    d=pd.DataFrame({'r':r}); d['day']=d.index.date; d['h']=d.index.hour
    for (h0,h1) in [(0,8),(0,12),(8,16),(16,23)]:
        pass
    # generic: past k-hour return vs next-hour return (rank corr) by era
    for k_ in [1,3,6,12,24]:
        past=np.log(c).diff(k_); nxt=r.shift(-1)
        out=[]
        for era,(a,b) in ERA.items():
            df=pd.concat([past,nxt],axis=1,keys=['p','n']).loc[a:b].dropna()
            ic=df.corr(method='spearman').iloc[0,1]; out.append(f'{era}: IC {ic:+.3f} (t~{ic*np.sqrt(len(df)):+.1f})')
        print(f'  past {k_:2d}h -> next 1h:',' | '.join(out))
    # c) large-move reversal/continuation: |ret_1h|>3 sigma -> next 1,3,6h
    vol=r.rolling(24*7).std().shift(1); z=r/vol
    for era,(a,b) in ERA.items():
        res=[]
        for h in [1,3,6]:
            f=np.log(c.shift(-h)/c)
            ev_dn=(z<-3).loc[a:b]; ev_up=(z>3).loc[a:b]
            res.append(f'h{h}: after drop {f.loc[a:b][ev_dn].mean()*1e4:+.1f}bp (n={ev_dn.sum()}) after jump {f.loc[a:b][ev_up].mean()*1e4:+.1f}bp (n={ev_up.sum()})')
        print(f'  3-sigma hourly moves {era}:',' | '.join(res))
