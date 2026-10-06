import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from quant.carry import daily_carry, SPOT_BPS, PERP_BPS
from quant import metrics as M
SYMS=["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","LINKUSDT","LTCUSDT","AVAXUSDT","TRXUSDT","DOTUSDT"]
C={s:daily_carry(s) for s in SYMS}
F=pd.DataFrame({s:C[s].fund_ma30 for s in SYMS}); G=pd.DataFrame({s:C[s].funding+C[s].basis for s in SYMS})
def sleeve(topk,thr,margin=0.15,maxw=0.5):
    # decided at close t from trailing-30d funding; earns day t+1
    rk=F.where(F>thr).rank(axis=1,ascending=False)
    sel=(rk<=topk).astype(float); w=sel.div(sel.sum(axis=1).replace(0,np.nan),axis=0).fillna(0.0).clip(upper=maxw)
    wl=w.shift(1).fillna(0.0)
    gross=(wl*G).sum(axis=1)
    turn=wl.diff().abs().sum(axis=1)
    cost=turn*(SPOT_BPS+PERP_BPS)/1e4
    exposure=wl.sum(axis=1)
    return ((gross-cost)/(1+margin)).loc['2020-03-01':],exposure
print('top-k, funding threshold (ann.)  | 2020-21 CAGR/vol/Sh | 2022-26 CAGR/vol/Sh/DD | avg capital deployed 22-26')
for k,thr in [(1,0.05),(3,0.05),(3,0.10),(5,0.05),(5,0.10),(12,0.03)]:
    r,ex=sleeve(k,thr); out=[]
    for a,b in [('2020-03-01','2021-12-31'),('2022-01-01','2026-09-30')]:
        s=M.summary(r.loc[a:b]); out.append(f"{s['cagr']*100:5.1f}% {s['vol']*100:4.1f}% {s['sharpe']:5.1f}"+(f" {s['maxdd']*100:5.1f}%" if a.startswith('2022') else ''))
    print(f'k={k:2d} thr={thr:.2f} | '+' | '.join(out)+f" | {ex.loc['2022':].mean():.2f}")
