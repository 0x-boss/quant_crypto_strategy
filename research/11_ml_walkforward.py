"""Exploration 11: pooled LightGBM / ridge on liquid-universe features, strict expanding-window walk-forward.
Hyper-parameters fixed a priori (no tuning).  Reports OOS IC and OOS portfolio statistics vs. trend baseline."""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import Ridge
from quant.data import load_panels, liquid_universe
from quant.engine import run_backtest
from quant.strategies import *
from quant import metrics as M
P,V=load_panels(); R=P.pct_change(fill_method=None)
U=liquid_universe(P,V,top_n=20,min_adv=5e6,min_hist=180)
lr=np.log(P).diff(); vol=lr.ewm(span=30,min_periods=15).std()          # daily vol
H=7
feat={}
for k in [1,3,7,14,30,60,90,180]:
    feat[f'r{k}']=np.log(P/P.shift(k))/(vol*np.sqrt(k))
feat['vr_7_30']=np.log(lr.rolling(7).std()/lr.rolling(30).std())
feat['vr_30_90']=np.log(lr.rolling(30).std()/lr.rolling(90).std())
feat['dd_90']=np.log(P/P.rolling(90).max())/vol
feat['up_90']=np.log(P/P.rolling(90).min())/vol
feat['dd_200']=np.log(P/P.rolling(200).max())/vol
feat['volsurge']=np.log(V.rolling(7).mean()/V.rolling(90).mean())
btc30=feat['r30']['btc']; breadth=((P>P.rolling(50).mean())&U).sum(axis=1)/U.sum(axis=1).replace(0,np.nan)
ew7=lr.where(U).mean(axis=1).rolling(7).sum()
rel30=np.log(P/P.shift(30)).sub(np.log(P['btc']/P['btc'].shift(30)),axis=0)
feat['rel30_btc']=rel30/(vol*np.sqrt(30))
X=pd.concat({k:v.where(U) for k,v in feat.items()},axis=1).stack(future_stack=True)
X=X.dropna(how='all')
mk=pd.DataFrame({'btc_r30':btc30,'breadth':breadth,'ew7':ew7})
X=X.join(mk,on=X.index.get_level_values(0).name or 'date') if False else X
dates=X.index.get_level_values(0)
for c in mk.columns: X[c]=mk[c].reindex(dates).values
y=(np.log(P.shift(-H)/P)/(vol*np.sqrt(H))).where(U).stack(future_stack=True).reindex(X.index)
y=y.clip(-4,4)
m=X.notna().all(axis=1)&y.notna()
Xf=X[m]; yf=y[m]; dts=Xf.index.get_level_values(0)
print('rows',len(Xf),'features',Xf.shape[1])
preds=pd.Series(np.nan,index=Xf.index)
tests=pd.date_range('2019-01-01','2026-06-30',freq='6MS')
for a,b in zip(tests[:-1],tests[1:]):
    tr=dts<=(a-pd.Timedelta(days=H+3)); te=(dts>=a)&(dts<b)
    if te.sum()==0 or tr.sum()<3000: continue
    mdl=lgb.LGBMRegressor(n_estimators=250,learning_rate=0.03,num_leaves=8,min_child_samples=300,subsample=0.7,subsample_freq=1,
                          colsample_bytree=0.7,reg_lambda=10.0,verbose=-1,random_state=0)
    mdl.fit(Xf[tr],yf[tr]); preds[te]=mdl.predict(Xf[te])
    rg=Ridge(alpha=100.0).fit(Xf[tr],yf[tr])
pr=preds.dropna()
print('OOS rows',len(pr))
# IC
dfp=pd.DataFrame({'p':pr,'y':yf.reindex(pr.index)})
ic=dfp.groupby(level=0).apply(lambda g: g['p'].corr(g['y'],method='spearman') if len(g)>=5 else np.nan)
for a,b in [('2019-01-01','2021-12-31'),('2022-01-01','2026-05-24')]:
    x=ic.loc[a:b].dropna(); print(f'cross-sectional rank IC {a[:4]}-{b[:4]}: mean {x.mean():+.3f}  t {x.mean()/x.std()*np.sqrt(len(x)/H):.2f}')
print('pooled corr', dfp.corr().iloc[0,1])
# portfolio: long-only, w_i = relu(pred) * (1/vol_i) scaled; and normalised
Pw=preds.unstack().reindex(index=P.index,columns=P.columns)
sig=Pw.clip(lower=0)
volA=vol*np.sqrt(365)
w0=(sig*(0.25/volA)/10).fillna(0)
w0=w0.clip(upper=0.3)
r=run_backtest(w0,R)
periods={'oos 19-26':('2019-01-01','2026-05-24'),'19-21':('2019-01-01','2021-12-31'),'22-26':('2022-01-01','2026-05-24')}
def show(name,r):
    row=[]
    for k,(a,b) in periods.items():
        s=M.summary(r.loc[a:b]); row.append(f"{k}: CAGR {s['cagr']*100:6.1f}% Sh {s['sharpe']:4.2f} DD {s['maxdd']*100:6.1f}% vol {s['vol']*100:4.1f}")
    print(f"{name:26s}"+' | '.join(row))
show('ML long-only (relu)',r)
# market-neutral version: long top tercile, short bottom tercile by prediction
rk=Pw.rank(axis=1,pct=True)
lw=(rk>=2/3).astype(float); sw=(rk<=1/3).astype(float)
lw=lw.div(lw.sum(axis=1).replace(0,np.nan),axis=0); sw=sw.div(sw.sum(axis=1).replace(0,np.nan),axis=0)
wl=((lw-sw).fillna(0))*0.5
rl=run_backtest(wl,R)
show('ML dollar-neutral tercile',rl)
g=(wl.shift(1)*R.fillna(0)).sum(axis=1)
show('  (gross of costs)',g)
# baseline trend on same OOS span
A=P[['btc','eth']]; up=trend_signal(A,(14,30,60,90,180),'mom'); v2=ewma_vol(A,30)
W1=pd.DataFrame(0.0,index=P.index,columns=P.columns); W1[['btc','eth']]=(up*0.30/v2/2).fillna(0).clip(upper=1.0)
show('baseline trend BTC+ETH',run_backtest(W1,R))
