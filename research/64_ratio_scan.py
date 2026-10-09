"""Descriptive scan (not a trial): does the ETH/BTC ratio mean-revert (relative value) or continue?  Daily CoinMetrics prices 2016-2026.05."""
import sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
import numpy as np, pandas as pd
CM='/home/user/coinmetrics/data/csv/'
def cm(a): return pd.read_csv(CM+f'{a}.csv',parse_dates=['time']).set_index('time')['PriceUSD'].dropna()
b,e=cm('btc'),cm('eth'); y=np.log(e/b).dropna(); y=y.loc['2016-01-01':'2026-05-23']
eras={'2016-17':('2016-01-01','2017-12-31'),'2018-22':('2018-01-01','2022-12-31'),'2023-26':('2023-01-01','2026-05-23')}
print('ETH/BTC log-ratio: excess forward change after displacement (bp, t); events de-clustered; cost of a full BTC<->ETH switch ~ 20 bp')
for n in (10,20,60,120):
    z=((y-y.rolling(n).mean())/y.rolling(n).std())
    for h in (1,5,10,20):
        f=(y.shift(-h)-y)
        for side,cond in (('cheap ETH (z<=-2)',z<=-2),('rich ETH (z>=2)',z>=2)):
            out=[]
            for en,(lo,hi) in {'all':('2016-01-01','2026-05-23'),**eras}.items():
                m=(y.index>=lo)&(y.index<=hi); ev=np.where(m&cond.values&f.notna().values)[0]; keep=[];last=-99
                for i in ev:
                    if i-last>=max(n//2,h): keep.append(i); last=i
                base=f[m&f.notna().values]; 
                if len(keep)<8: out.append(f'{en}: n/a'); continue
                ex=f.iloc[keep].mean()-base.mean(); t=ex/(base.std()/np.sqrt(len(keep)))
                out.append(f'{en}: {ex*1e4:+6.0f}bp (t {t:+4.1f}, n={len(keep)})')
            if side.startswith('cheap'): sign=+1
            print(f'n={n:3d} h={h:2d} {side:18s} '+' | '.join(out))
