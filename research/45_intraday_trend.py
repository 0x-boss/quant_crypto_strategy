"""Same TrendCore logic on k-bars-per-day sampling (all lookbacks scaled by k) using Binance perp 1h closes, 2020-03..2026-10."""
import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import quant.strategies as S
from quant.engine import run_backtest
from quant import metrics as M
D='data/intraday/'
px1h=pd.concat({a:pd.read_parquet(D+f'{s}_spot_1h.parquet')['c'] for a,s in [('btc','BTCUSDT'),('eth','ETHUSDT')]},axis=1).loc['2019-09-01':]
def system(k,tv=0.35,rv_hourly=False):
    S.ANN=365*k
    orig_vol=S.ewma_vol
    P=px1h.resample(f'{24//k}h').last().dropna(); R=P.pct_change()
    U=P.notna()
    def mixk(A):
        mom=S.trend_signal(A,tuple(int(L*k) for L in (14,30,60,90,180)),'mom'); sma=S.trend_signal(A,tuple(int(L*k) for L in (20,50,100,200)),'sma')
        sp=tuple((int(f*k),int(s*k),sc) for f,s,sc in S.EWMAC_SPEEDS)
        ew=(S.ewmac_forecast(A,sp,vol_span=30*k).clip(lower=0)/10).clip(upper=1.0)
        return (mom+sma+ew)/3
    if rv_hourly:
        rv2=(np.log(px1h).diff()**2).resample(f'{24//k}h').sum().reindex(P.index)
        def rvvol(PP,span=30): return (rv2.ewm(span=span,min_periods=10).mean()**0.5)*np.sqrt(365*k)
        S.ewma_vol=rvvol
    f0=mixk(P); n=2; acc=None; cnt=0
    for g in (100,150,200):
        f=(f0*(P>P.rolling(g*k).mean())).fillna(0.0)
        for vs in (20,30,60):
            v=S.ewma_vol(P,vs*k); w=(f*0.45/v).div(n).fillna(0.0).clip(upper=1.0); acc=w if acc is None else acc+w; cnt+=1
    W=acc/cnt; ek=dict(band=0.05,fin_rate=0.10/k)
    r0=run_backtest(W,R,**ek); lev=S.vol_target_scale(r0,tv,30*k,1.5); W=W.mul(lev,axis=0)
    g=W.abs().sum(axis=1); W=W.mul((2.0/g).clip(upper=1.0).fillna(1.0),axis=0)
    r=run_backtest(W,R,**ek); S.ANN=365; S.ewma_vol=orig_vol
    return (1+r).groupby(r.index.date).prod()-1
res={}
for k in [1,4]:
    for rv in [False,True]:
        r=system(k,rv_hourly=rv); r.index=pd.to_datetime(r.index); res[(k,rv)]=r
        out=[]
        for lab,(a,b) in {'2020-03..26':('2020-03-01','2026-10-05'),'2022+':('2022-01-01','2026-10-05'),'OOS 26-05-24+':('2026-05-24','2026-10-05')}.items():
            s_=M.summary(r.loc[a:b]); out.append(f"{lab}: CAGR {s_['cagr']*100:5.1f}% Sh {s_['sharpe']:4.2f} DD {s_['maxdd']*100:6.1f}%")
        print(f"{24//k:2d}h bars, {'hourly-RV vol' if rv else 'close-to-close vol'} | "+' | '.join(out))
