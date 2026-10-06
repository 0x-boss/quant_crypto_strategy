"""C7: BTC signal -> hold the top-3 non-BTC symbols by trailing volume (point-in-time, delisted coins included). Daily bars."""
import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from quant import metrics as M
from quant.xs_perp import load_panels, universe
from quant.intraday import hourly_prices
import importlib; S=importlib.import_module('research.51_btc_signal_swap') if False else None
sys.path.insert(0,'research')
import importlib.util
spec=importlib.util.spec_from_file_location('c51','research/51_btc_signal_swap.py'); c51=importlib.util.module_from_spec(spec); spec.loader.exec_module(c51)
C,QV,F=load_panels()
U=universe(C,QV,top_n=20).copy(); U['BTCUSDT']=False
adv=QV.rolling(30,min_periods=20).median().where(U)
R=C.pct_change(fill_method=None)
# weekly (Sunday-close) top-3 membership, held until next Sunday
idx=C.index; members=pd.DataFrame(0.0,index=idx,columns=C.columns); cur=None; turn=pd.Series(0.0,index=idx)
for t,d in enumerate(idx):
    if d.weekday()==6:
        top=adv.iloc[t].dropna().nlargest(3).index
        new=pd.Series(0.0,index=C.columns); new[top]=1/3
        if cur is not None: turn.iloc[t]=(new-cur).abs().sum()/2
        cur=new
    if cur is not None: members.iloc[t]=cur
held=members.shift(1)                                    # membership decided at close t, earns t+1
basket_ret=(held*R.fillna(0.0)).sum(axis=1)-turn.shift(1).fillna(0.0)*25e-4
basket_ret=basket_ret.loc['2020-06-01':]
print('typical members (share of weeks held, top 10):'); print((members.loc['2020-09':]>0).mean().sort_values(ascending=False).head(10).round(2).to_dict())
dead=[c for c in members.columns if (members[c]>0).any() and C[c].last_valid_index()<pd.Timestamp('2026-09-01')]; print('delisted/dead coins that were held at some point:',dead)
# daily framework prices for btc, eth, sol (Binance spot daily closes) and the basket as a synthetic price series
px=hourly_prices(('btc','eth','sol')).sort_index(); Pd=px.resample('1D').last()
Pd['basket']=(1+basket_ret.reindex(Pd.index).fillna(0.0)).cumprod()
rv2=(np.log(Pd).diff()**2)
oi=pd.read_parquet('data/intraday/BTCUSDT_oi_daily.parquet')['sum_open_interest_value']

print('--- BTC + survivor-corrected alt basket (50/50), BTC signal, daily framework')
res={}
for name,tg in [('BTC only',['btc']),('BTC+SOL (survivor!)',['btc','sol']),('BTC+top-3 alts basket (corrected)',['btc','basket'])]:
    r,W=c51.swap_strategy(Pd,rv2,1,tg,'own',oi); res[name]=r; c51.report(name,r)
    s_=M.summary(r.loc['2022-01-01':]); print(f"   2022+: CAGR {s_['cagr']*100:5.1f}% Sh {s_['sharpe']:4.2f} DD {s_['maxdd']*100:6.1f}%")
