"""Exploration 6: do a short, pre-specified list of on-chain/volume features predict forward BTC / ETH returns
with the same sign in both 2017-21 and 2022-26?  (8 features x 2 assets x 2 horizons = 32 tests: treat as trials.)"""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from scipy import stats
src='/home/user/coinmetrics/data/csv/'
def load(a,cols): 
    d=pd.read_csv(src+a+'.csv',parse_dates=['time']).set_index('time'); return d[[c for c in cols if c in d]]
P=pd.read_parquet('data/panel_price.parquet'); 
st=load('usdt',['SplyCur'])['SplyCur'].add(load('usdc',['SplyCur'])['SplyCur'],fill_value=0)
feats={}
for a in ['btc','eth']:
    d=load(a,['FlowInExUSD','FlowOutExUSD','SplyExNtv','SplyExUSD','CapMVRVCur','AdrActCnt','TxCnt','volume_reported_spot_usd_1d','CapMrktCurUSD'])
    f=pd.DataFrame(index=d.index)
    f['stable_growth30']=np.log(st/st.shift(30)).reindex(d.index)
    if 'FlowOutExUSD' in d:
        f['netflow30']=((d['FlowOutExUSD']-d['FlowInExUSD']).rolling(30).sum()/d['CapMrktCurUSD'])
    if 'SplyExNtv' in d: f['exsupply_chg30']=-np.log(d['SplyExNtv']/d['SplyExNtv'].shift(30))
    f['mvrv_z']=(d['CapMVRVCur']-d['CapMVRVCur'].rolling(365).mean())/d['CapMVRVCur'].rolling(365).std()
    f['adr_growth']=np.log(d['AdrActCnt'].rolling(7).mean()/d['AdrActCnt'].rolling(90).mean())
    f['tx_growth']=np.log(d['TxCnt'].rolling(7).mean()/d['TxCnt'].rolling(90).mean())
    v=d['volume_reported_spot_usd_1d']; f['vol_surge']=np.log(v.rolling(7).mean()/v.rolling(90).mean())
    f['mom30']=np.log(P[a]/P[a].shift(30)).reindex(d.index)
    feats[a]=f
for h in [7,14]:
    print(f'--- horizon {h}d (non-overlapping samples every {h}d)')
    for a in ['btc','eth']:
        fwd=np.log(P[a].shift(-h)/P[a])
        f=feats[a]
        for c in f.columns:
            out=[]
            for lo,hi in [('2017-01-01','2021-12-31'),('2022-01-01','2026-04-30')]:
                x=f[c].loc[lo:hi].iloc[::h]; y=fwd.loc[lo:hi].iloc[::h]
                m=x.notna()&y.notna()
                if m.sum()<30: out.append('  n/a'); continue
                ic,p=stats.spearmanr(x[m],y[m]); out.append(f'IC {ic:+.2f} (p={p:.2f}, n={m.sum()})')
            print(f'{a} {c:16s}', ' | '.join(out))
