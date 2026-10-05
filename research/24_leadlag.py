import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from quant.data import load_panels, liquid_universe
from quant.strategies import *
P,V=load_panels(); R=np.log(P).diff()
U=liquid_universe(P,V,top_n=20,min_adv=5e6,min_hist=180)
alts=U.copy(); alts[['btc','eth']]=False
altret=R.where(alts).mean(axis=1)
btc=R['btc']; eth=R['eth']
def reg(y,xs,lo,hi):
    d=pd.concat([y]+xs,axis=1).loc[lo:hi].dropna(); 
    X=np.column_stack([np.ones(len(d))]+[d.iloc[:,i].values for i in range(1,d.shape[1])]); Y=d.iloc[:,0].values
    b=np.linalg.lstsq(X,Y,rcond=None)[0]; res=Y-X@b; s2=res@res/(len(Y)-X.shape[1]); cov=s2*np.linalg.inv(X.T@X); t=b/np.sqrt(np.diag(cov))
    return b[1:],t[1:],len(Y)
for lo,hi in [('2018-01-01','2020-06-30'),('2020-07-01','2022-12-31'),('2023-01-01','2026-05-23')]:
    for name,y in [('BTC',btc),('ETH',eth)]:
        b,t,n=reg(y.shift(-1),[altret,btc,eth],lo,hi)
        print(f'{lo[:4]}-{hi[:4]} next-day {name} ~ alt_t, btc_t, eth_t : beta', np.round(b,3),'t',np.round(t,2),'n',n)
