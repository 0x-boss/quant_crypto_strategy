import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from quant.strategies import *
P=pd.read_parquet('data/panel_price.parquet')
pd.set_option('display.width',200)
for a in ['btc','eth']:
    A=P[[a]]; up=trend_signal(A,(14,30,60,90,180),'mom')[a]
    lr=np.log(P[a]).diff()
    r3=np.log(P[a]/P[a].shift(3)); vol=lr.ewm(span=30).std()
    z3=r3/(vol*np.sqrt(3))
    nxt1=lr.shift(-1); nxt3=np.log(P[a].shift(-3)/P[a])
    df=pd.DataFrame({'up':up,'z3':z3,'n1':nxt1,'n3':nxt3,'vol':vol}).dropna()
    df['regime']=pd.cut(df['up'],[-0.01,0.2,0.6,1.01],labels=['bear(<=.2)','mixed','bull(>.6)'])
    df['shock']=pd.cut(df['z3'],[-99,-1.5,-0.5,0.5,1.5,99],labels=['z<-1.5','-1.5..-.5','flat','.5..1.5','>1.5'])
    for per,(lo,hi) in {'2017-21':('2017-01-01','2021-12-31'),'2022-26':('2022-01-01','2026-04-30')}.items():
        d=df.loc[lo:hi]
        t=d.groupby(['regime','shock'],observed=True).apply(lambda g: pd.Series({'n':len(g),'mean_n3_bp':g['n3'].mean()*1e4,'t':g['n3'].mean()/(g['n3'].std()/np.sqrt(max(len(g)/3,1)))}))
        print(a,per); print(t.round(1).unstack(0).to_string()); print()
