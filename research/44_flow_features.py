"""New-information features from the hourly archive: order flow (taker buy imbalance), perp premium, funding, hourly RV/skew.
Daily features known at 00:00 UTC; IC vs forward 1/3/7-day returns of spot, two eras. Rolling z-scores (365d) - no look-ahead."""
import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats
D='data/intraday/'
ERA={'20-22':('2020-06-01','2022-12-31'),'23-26':('2023-01-01','2026-09-25')}
def feats(sym):
    sp=pd.read_parquet(D+f'{sym}_spot_1h.parquet'); pf=pd.read_parquet(D+f'{sym}_perp_1h.parquet'); fu=pd.read_parquet(D+f'{sym}_funding.parquet')['fundingRate']
    d=pd.DataFrame(index=sp.resample('1D').last().index)
    d['px']=sp.c.resample('1D').last()
    d['flow_spot']=(2*sp.tb-sp.v).resample('1D').sum()/sp.v.resample('1D').sum()
    d['flow_perp']=(2*pf.tb-pf.v).resample('1D').sum()/pf.v.resample('1D').sum()
    prem=(pf.c/sp.c-1).reindex(sp.index); d['premium']=prem.resample('1D').mean()
    d['funding3']=fu.resample('1D').sum().rolling(3).mean()
    d['perp_spot_vol']=np.log(pf.qv.resample('1D').sum()/sp.qv.resample('1D').sum())
    r=np.log(sp.c).diff()
    d['rv']=np.log((r**2).resample('1D').sum()); d['skew']=r.resample('1D').apply(lambda x: x.skew())
    d['jump']=np.log((r.abs()).resample('1D').max()/np.sqrt((r**2).resample('1D').mean()))
    return d
def z(x,w=365): return (x-x.rolling(w).mean())/x.rolling(w).std()
for sym in ['BTCUSDT','ETHUSDT']:
    d=feats(sym); lp=np.log(d.px)
    print('=====',sym,'  rank-IC (t) of rolling-z feature vs forward return')
    for f in ['flow_spot','flow_perp','premium','funding3','perp_spot_vol','rv','skew','jump']:
        x=z(d[f]); out=[]
        for h in [1,3,7]:
            fwd=lp.shift(-h)-lp; res=[]
            for era,(a,b) in ERA.items():
                df=pd.concat([x,fwd],axis=1,keys=['x','y']).loc[a:b].dropna()
                if h>1: df=df.iloc[::h]
                ic,p=stats.spearmanr(df.x,df.y); res.append(f'{ic:+.2f}({ic*np.sqrt(len(df)):+.1f})')
            out.append(f'h{h}: '+' | '.join(res))
        print(f'  {f:14s} '+'   '.join(out))
