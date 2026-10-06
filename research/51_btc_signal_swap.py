"""Round C (see PREREGISTRATION.md): BTC-derived signal, higher-beta holding.  Spot-only, long-only, gross<=1."""
import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from quant import metrics as M
from quant.intraday import hourly_prices
import trendcore_spot as T
ERA1=('2020-09-01','2022-12-31'); ERA2=('2023-01-01','2026-10-05')
COST={'btc':10.0,'eth':10.0,'sol':25.0,'basket':25.0}

def bt(W,R,costs,band=T.BAND):
    Wl=W.shift(1).fillna(0.0).values; Rv=R.fillna(0.0).values; cvec=np.array([costs[c] for c in W.columns])/1e4
    ret=np.zeros(len(Wl)); prev=np.zeros(Wl.shape[1])
    for t in range(len(Wl)):
        tgt=np.where((np.abs(Wl[t]-prev)<=band)&(Wl[t]!=0.0),prev,Wl[t]); c=float((np.abs(tgt-prev)*cvec).sum())
        ret[t]=float(tgt@Rv[t])-c; prev=tgt*(1+Rv[t])/(1+ret[t])
    return pd.Series(ret,index=R.index)

def swap_strategy(P, rv2, k, targets, sizing, oi, tv=0.40):
    """P: price frame (bars) incl. 'btc' + targets; rv2: realised variance per bar for all columns."""
    ann=365*k
    # BTC trend score with k bars/day (copy of trendcore_spot logic, parameterised by k)
    Pb=P[['btc']]
    mom=sum((Pb/Pb.shift(L*k)>1.0).astype(float).where(Pb.shift(L*k).notna()) for L in T.MOM_LB)/len(T.MOM_LB)
    sma=0
    for L in T.SMA_LB:
        m=Pb.rolling(L*k,min_periods=L*k).mean(); sma=sma+(Pb>m).astype(float).where(m.notna())
    sma=sma/len(T.SMA_LB)
    dv=np.log(Pb).diff().ewm(span=30*k,min_periods=10).std(); fc=0
    for f,s,sc in T.EWMAC:
        raw=(Pb.ewm(span=f*k,min_periods=f*k).mean()-Pb.ewm(span=s*k,min_periods=s*k).mean())/(Pb*dv); fc=fc+(raw*sc).clip(-20,20)
    ew=((fc/len(T.EWMAC)).clip(lower=0)/10.0).clip(upper=1.0)
    score=((mom+sma+ew)/3.0)['btc']
    vol=lambda col,span:(rv2[col].ewm(span=span*k,min_periods=10).mean()**0.5)*np.sqrt(ann)
    R=P[targets].pct_change(); n=len(targets); acc=0; cnt=0
    for g in T.GATES:
        f=(score*(P['btc']>P['btc'].rolling(g*k).mean())).fillna(0.0)
        for vs in T.VOL_SPANS:
            w=pd.DataFrame({a:(f*T.ASSET_VOL/vol('btc' if sizing=='swap' else a,vs)).fillna(0.0).clip(upper=1.0)/n for a in targets}); acc=acc+w; cnt+=1
    W=acc/cnt; costs={a:COST.get(a,25.0) for a in targets}
    r0=bt(W,R,costs); rv=r0.ewm(span=30*k,min_periods=10).std()*np.sqrt(ann)
    W=W.mul((tv/rv).clip(upper=1.0).fillna(0.0),axis=0)
    if oi is not None:
        z=((np.log(oi)-np.log(oi).rolling(365).mean())/np.log(oi).rolling(365).std()).shift(1)
        W=W.mul((1-T.OI_STRENGTH*z.reindex(W.index,method='ffill').clip(0,2)/2).fillna(1.0),axis=0)
    g=W.sum(axis=1); W=W.mul((1.0/g).clip(upper=1.0).fillna(1.0),axis=0)
    r=bt(W,R,costs); d=(1+r).groupby(r.index.date).prod()-1; d.index=pd.to_datetime(d.index)
    return d, W

def report(name,r,base=None):
    out=[]
    for lab,(a,b) in {'era1':ERA1,'era2':ERA2,'full 2020-09+':(ERA1[0],ERA2[1])}.items():
        s=M.summary(r.loc[a:b]); out.append(f"{lab}: CAGR {s['cagr']*100:6.1f}% Sh {s['sharpe']:5.2f} DD {s['maxdd']*100:6.1f}%")
    print(f'{name:34s}'+' | '.join(out))

if __name__=='__main__':
    oi=pd.read_parquet('data/intraday/BTCUSDT_oi_daily.parquet')['sum_open_interest_value']
    px=hourly_prices(('btc','eth','sol')).sort_index()
    k=4; P=px.resample('6h').last().dropna(how='all'); rv2=(np.log(px).diff()**2).resample('6h').sum().reindex(P.index)
    res={}
    print('--- 6h framework (SOL = known survivor: hindsight warning)')
    for name,tg,sz in [('C0 BTC (baseline)',['btc'],'own'),('C1 ETH own-vol',['eth'],'own'),('C2 ETH BTC-vol swap',['eth'],'swap'),
                       ('C3 SOL own-vol',['sol'],'own'),('C4 SOL BTC-vol swap',['sol'],'swap'),
                       ('C5 ETH+SOL own-vol',['eth','sol'],'own'),('C6 ETH+SOL BTC-vol swap',['eth','sol'],'swap')]:
        r,W=swap_strategy(P,rv2,k,tg,sz,oi); res[name]=r; report(name,r)
    pd.DataFrame(res).to_csv('results/btc_signal_swap_6h.csv')
