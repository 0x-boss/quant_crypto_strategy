"""Round-2 summary: graded variants of TrendCore for the weak 2022-26 window -> results/round2_variants.csv/.png"""
import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from quant.strategy import Config, run
from quant import metrics as M
V={'A  core (BTC+ETH)':Config(),
   'B  core + 4% idle-cash yield (data-grounded)':Config(cash_rate=0.04),
   'C  core + tokenised gold member (hindsight-flavoured)':Config(assets=('btc','eth','paxg')),
   'D  core + gold + 4% idle-cash yield':Config(assets=('btc','eth','paxg'),cash_rate=0.04)}
R={k:run(c) for k,c in V.items()}
rows=[]
for k,r in R.items():
    for lab,a in [('2020-06+','2020-06-01'),('2022+','2022-01-01')]:
        s=M.summary(r.loc[a:]); rows.append(dict(variant=k,window=lab,cagr=s['cagr'],sharpe=s['sharpe'],sortino=s['sortino'],maxdd=s['maxdd'],vol=s['vol']))
D=pd.DataFrame(rows); D.to_csv('results/round2_variants.csv',index=False)
print(D.pivot(index='variant',columns='window',values=['cagr','sharpe','maxdd']).round(3).to_string())
yr=pd.DataFrame({k[:1]:M.yearly(r.loc['2020-06-01':])['ret'] for k,r in R.items()}).loc[2021:]
print((yr*100).round(1).T.to_string())
btc=pd.read_parquet('data/panel_price.parquet')['btc'].pct_change().loc['2022-01-01':]
fig,ax=plt.subplots(1,2,figsize=(13,4.3))
for k,r in R.items(): ax[0].plot(M.equity(r.loc['2022-01-01':]),label=k[:1]+' '+k[3:30],lw=1.5)
ax[0].plot(M.equity(btc),color='#f2a900',lw=1,ls='--',label='BTC buy&hold'); ax[0].legend(fontsize=7); ax[0].set_title('Growth of $1 since 2022-01'); ax[0].grid(alpha=.25)
for k,r in R.items(): ax[1].plot((r.loc['2022-01-01':].rolling(365).mean()/r.loc['2022-01-01':].rolling(365).std()*np.sqrt(365)),lw=1.2,label=k[:1])
ax[1].axhline(1.5,color='k',ls='--',lw=.8); ax[1].axhline(0,color='k',lw=.5); ax[1].set_title('Rolling 1y Sharpe (dashed = 1.5 target)'); ax[1].legend(); ax[1].grid(alpha=.25)
plt.tight_layout(); plt.savefig('results/round2_variants.png',dpi=130)
