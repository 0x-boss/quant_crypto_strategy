"""Hourly reversal strategy on BTC/ETH perp.  Position decided at close of bar t, earns bar t+1 return.
Reports gross Sharpe, turnover and breakeven cost (bp per unit traded) by era."""
import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
D='data/intraday/'
ERA={'20-22':('2020-01-01','2022-12-31'),'23-26':('2023-01-01','2026-10-05')}
def run(sym,lookback,thr,cap=1.0,hold=1):
    c=pd.read_parquet(D+f'{sym}_perp_1h.parquet')['c']; r=np.log(c).diff(); vol=r.rolling(24*7).std().shift(1)
    z=np.log(c).diff(lookback)/(vol*np.sqrt(lookback))
    pos=(-z).where(z.abs()>thr,0.0).clip(-cap,cap)
    if hold>1: pos=pos.rolling(hold,min_periods=1).mean()
    pos=pos/ (cap)                                       # in [-1,1] of 'unit risk'
    # scale to constant risk: position * (target hourly vol / vol)
    tgt=0.20/np.sqrt(24*365)
    w=(pos*tgt/vol).clip(-3,3)
    gross=(w.shift(1)*r)                                  # exposure fixed during bar t+1
    to=w.diff().abs()
    return gross,to
def sh(x): x=x.dropna(); return x.mean()/x.std()*np.sqrt(24*365)
print('sym lookback thr hold | era: grossSh  turnover/day  breakeven bp | netSh @2bp  @4bp  @8bp')
for sym in ['BTCUSDT','ETHUSDT']:
    for lb,thr,hold in [(1,0.0,1),(3,0.0,1),(6,0.0,1),(3,1.0,1),(3,1.5,1),(3,2.0,1),(6,1.5,2),(3,1.5,3)]:
        g,to=run(sym,lb,thr,hold=hold); line=[]
        for era,(a,b) in ERA.items():
            gg,tt=g.loc[a:b],to.loc[a:b]
            be=gg.sum()/tt.sum()*1e4
            nets=[sh(gg-tt*bp/1e4) for bp in (2,4,8)]
            line.append(f"{era}: {sh(gg):5.2f} {tt.mean()*24:6.1f}x {be:5.1f}bp | "+' '.join(f'{n:5.2f}' for n in nets))
        print(f'{sym[:3]} {lb}h thr{thr} hold{hold}  '+'  ||  '.join(line))
