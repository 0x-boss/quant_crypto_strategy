import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
from quant import metrics as M
import trendcore_spot as T
px=pd.concat({a:pd.read_parquet(f'data/intraday/{a.upper()}USDT_spot_1h.parquet')['c'] for a in ('btc','eth')},axis=1,sort=True)
oi=pd.read_parquet('data/intraday/BTCUSDT_oi_daily.parquet')['sum_open_interest_value']
r,W=T.backtest(px,oi,start='2018-03-01')
expo=W.sum(axis=1).groupby(W.index.date).last(); expo.index=pd.to_datetime(expo.index)       # exposure at day close (decided, held next day)
expo=expo.reindex(r.index).ffill()
btc=px['btc'].resample('1D').last().pct_change().reindex(r.index).fillna(0)
# ---------------- Q1
m=pd.DataFrame({'ret':(1+r).resample('ME').prod()-1,'btc':(1+btc).resample('ME').prod()-1})
start_exp=expo.shift(1).resample('ME').first(); avg_exp=expo.resample('ME').mean()
m['start_exp']=start_exp.reindex(m.index); m['avg_exp']=avg_exp.reindex(m.index); m=m.dropna()
m['bucket']=pd.cut(m.start_exp,[-1,0.05,0.5,1.01],labels=['idle at start (<5%)','partly in (5-50%)','mostly in (>50%)'])
g=m.groupby('bucket',observed=True)
print('--- Q1: month result by exposure at the START of the month (2018-03..2026-10)')
print(pd.DataFrame({'months':g.size(),'mean ret %':g.ret.mean()*100,'median %':g.ret.median()*100,'loss months %':g.ret.apply(lambda x:(x<-0.005).mean()*100),'worst %':g.ret.min()*100,'avg BTC %':g.btc.mean()*100,'avg exposure':g.avg_exp.mean()}).round(1).to_string())
idle=(m.avg_exp<0.05); pred=(m.start_exp<0.05)
print(f"months that were idle (avg exposure <5%): {idle.sum()} of {len(m)}; start-of-month exposure<5% flagged {pred.sum()}; correctly flagged {(idle&pred).sum()}, missed {(idle&~pred).sum()}, false alarms {(~idle&pred).sum()}")
bad=m[(m.ret<-0.03)]
print(f"loss months worse than -3%: {len(bad)} | exposure at start >50%: {(bad.start_exp>0.5).sum()} | partly in: {((bad.start_exp>0.05)&(bad.start_exp<=0.5)).sum()} | idle: {(bad.start_exp<=0.05).sum()}")
under=m[m.ret<m.btc]
print(f"months strategy < BTC: {len(under)} of {len(m)} ({len(under)/len(m)*100:.0f}%); in big BTC-up months (BTC>+10%): strategy < BTC in {((m.ret<m.btc)&(m.btc>0.10)).sum()} of {(m.btc>0.10).sum()}")
# ---------------- Q2
pg=pd.read_parquet('data/intraday/PAXGUSDT_spot_1d.parquet')['c'].reindex(r.index); pgr=pg.pct_change().fillna(0.0)
trend_ok=(pg>pg.rolling(200,min_periods=200).mean()).fillna(False).astype(float)
def overlay(gate):
    idle_w=(1-expo.clip(upper=1.0)).where(pg.notna(),0.0)*gate
    w_prev=0.0; out=np.zeros(len(r)); wl=idle_w.shift(1).fillna(0.0).values; rv=pgr.values
    for t in range(len(r)):
        tgt=wl[t]
        if tgt!=0.0 and abs(tgt-w_prev)<=0.05: tgt=w_prev
        out[t]=tgt*rv[t]-abs(tgt-w_prev)*25e-4; w_prev=tgt*(1+rv[t])
    return r+pd.Series(out,index=r.index)
V={'G0 baseline (idle = cash)':r,'G1 idle -> PAXG always':overlay(1.0),'G2 idle -> PAXG if PAXG > SMA200':overlay(trend_ok)}
E1=('2020-09-01','2022-12-31'); E2=('2023-01-01','2026-10-05')
print('\n--- Q2: gold on idle capital (spot PAXG); PAXG median volume $1.6-3.9M/day until 2025 (thin!)')
for k,v in V.items():
    o=[]
    for lab,(a,b) in {'era1':E1,'era2':E2,'full 2020-09+':(E1[0],E2[1])}.items():
        s=M.summary(v.loc[a:b]); o.append(f"{lab}: CAGR {s['cagr']*100:5.1f}% Sh {s['sharpe']:4.2f} DD {s['maxdd']*100:6.1f}%")
    print(f'{k:34s}'+' | '.join(o))
gb=M.summary(pgr.loc[E1[0]:E2[1]]); print(f"PAXG buy&hold same window: CAGR {gb['cagr']*100:.1f}% Sharpe {gb['sharpe']:.2f} DD {gb['maxdd']*100:.1f}%")
y=pd.DataFrame({k:v.loc['2020-09-01':].groupby(v.loc['2020-09-01':].index.year).apply(lambda x:(1+x).prod()-1)*100 for k,v in V.items()}); y['PAXG B&H']=pgr.loc['2020-09-01':].groupby(pgr.loc['2020-09-01':].index.year).apply(lambda x:(1+x).prod()-1)*100
print(y.round(1).to_string())
