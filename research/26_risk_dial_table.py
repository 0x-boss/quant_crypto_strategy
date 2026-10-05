import sys; sys.path.insert(0,'.')
import pandas as pd, json
from quant.strategy import Config, run
from quant import metrics as M
rows=[]
for tv in [0.15,0.20,0.25,0.30,0.35,0.40,0.50]:
    r=run(Config(target_vol=tv))
    a=M.summary(r); b=M.summary(r.loc['2018-01-01':]); c=M.summary(r.loc['2022-01-01':])
    rows.append(dict(target_vol=tv,cagr=a['cagr'],sharpe=a['sharpe'],maxdd=a['maxdd'],cagr_2018=b['cagr'],sharpe_2018=b['sharpe'],maxdd_2018=b['maxdd'],cagr_2022=c['cagr'],sharpe_2022=c['sharpe'],maxdd_2022=c['maxdd']))
    print(f"TV {tv:.2f}: 2016+ CAGR {a['cagr']*100:5.1f}% Sh {a['sharpe']:4.2f} DD {a['maxdd']*100:6.1f}% | 2018+ CAGR {b['cagr']*100:5.1f}% Sh {b['sharpe']:4.2f} DD {b['maxdd']*100:6.1f}% | 2022+ CAGR {c['cagr']*100:5.1f}% Sh {c['sharpe']:4.2f} DD {c['maxdd']*100:6.1f}%")
pd.DataFrame(rows).to_csv('results/risk_dial.csv',index=False)
