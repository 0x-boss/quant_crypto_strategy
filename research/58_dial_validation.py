"""Validation of the spot-only exposure dial (descriptive follow-up to round F, part 1a): the per-asset risk budget ASSET_VOL.
No new signal, no selection: the same strategy at 3 risk budgets, stressed the way the official version was."""
import sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
from quant import metrics as M
import trendcore_spot as T

px = pd.concat({a: pd.read_parquet(f'data/intraday/{a.upper()}USDT_spot_1h.parquet')['c'] for a in ('btc', 'eth')}, axis=1, sort=True)
oi = pd.read_parquet('data/intraday/BTCUSDT_oi_daily.parquet')['sum_open_interest_value']
START = '2018-03-01'
def to_daily(rb):
    d = (1 + rb).groupby(rb.index.date).prod() - 1; d.index = pd.to_datetime(d.index); return d
def build(av):
    T.ASSET_VOL = av
    try: W, R = T.target_weights(px, oi)
    finally: T.ASSET_VOL = 0.45
    return W, R
def S(r, a, b): return M.summary(r.loc[a:b])
WIN = {'full 2018-03+': (START, '2026-10-05'), '2020-03+': ('2020-03-01', '2026-10-05'), '2022+': ('2022-01-01', '2026-10-05'), '2023+': ('2023-01-01', '2026-10-05'), 'fwd 26-05-24+': ('2026-05-24', '2026-10-05')}
OUT = {}
yrs = {}
for av in (0.45, 0.60, 0.80):
    W, R = build(av)
    rb, gb = T._backtest(W, R); r = to_daily(rb).loc[START:]
    turn = (W.shift(1).fillna(0).diff().abs().sum(axis=1)).sum() / (len(W) / (365 * 4))      # approx one-way turnover per year (before the no-trade band)
    print(f'\n===== ASSET_VOL {av:.2f}  (avg exposure {gb.mean():.2f}, flat {np.mean(gb < 0.05)*100:.0f}% of bars)')
    o = {}
    for lab, (a, b) in WIN.items():
        s = S(r, a, b); o[lab] = dict(cagr=s['cagr'], sharpe=s['sharpe'], maxdd=s['maxdd'], sortino=s['sortino'])
        print(f"  {lab:14s} CAGR {s['cagr']*100:5.1f}%  Sharpe {s['sharpe']:4.2f}  Sortino {s['sortino']:4.2f}  maxDD {s['maxdd']*100:6.1f}%")
    for cb in (20.0, 40.0):
        r2 = to_daily(T._backtest(W, R, cost_bps=cb)[0]).loc[START:]; s = S(r2, START, '2026-10-05')
        print(f"  costs {cb/10:.0f}x ({cb:.0f} bp): CAGR {s['cagr']*100:5.1f}%  Sharpe {s['sharpe']:4.2f}  maxDD {s['maxdd']*100:6.1f}%"); o[f'cost{cb:.0f}'] = dict(cagr=s['cagr'], sharpe=s['sharpe'], maxdd=s['maxdd'])
    r3 = to_daily(T._backtest(W, R, lag=4)[0]).loc[START:]; s = S(r3, START, '2026-10-05')
    print(f"  +24h execution lag: CAGR {s['cagr']*100:5.1f}%  Sharpe {s['sharpe']:4.2f}  maxDD {s['maxdd']*100:6.1f}%"); o['lag24h'] = dict(cagr=s['cagr'], sharpe=s['sharpe'], maxdd=s['maxdd'])
    bs = M.block_bootstrap_sharpe(r.loc['2020-03-01':], n=2000, block=20, seed=1); bs22 = M.block_bootstrap_sharpe(r.loc['2022-01-01':], n=2000, block=20, seed=1)
    print(f"  bootstrap 90% CI Sharpe: 2020-03+ [{np.percentile(bs,5):.2f}, {np.percentile(bs,95):.2f}]   2022+ [{np.percentile(bs22,5):.2f}, {np.percentile(bs22,95):.2f}]")
    m = (1 + r).resample('ME').prod() - 1
    print(f"  worst month {m.min()*100:.1f}% ({m.idxmin().strftime('%Y-%m')}), worst 30d {((1+r).rolling(30).apply(np.prod, raw=True)-1).min()*100:.1f}%, months < -5%: {(m < -0.05).sum()} of {len(m)}, one-way turnover/yr ~ {turn:.1f}x (before band)")
    o['worst_month'] = float(m.min()); o['avg_exposure'] = float(gb.mean())
    OUT[f'{av:.2f}'] = o
    yrs[f'{av:.2f}'] = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    r.to_csv(f'results/dial_daily_{int(av*100):03d}.csv', header=['r'])
print('\ncalendar-year returns %:'); print(pd.DataFrame(yrs).round(1).to_string())
json.dump(OUT, open('results/dial_validation.json', 'w'), indent=1, default=float)
