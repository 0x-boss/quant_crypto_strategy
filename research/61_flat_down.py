"""Round G (pre-registered in research/PREREGISTRATION.md BEFORE this script was run): what can be paired with trend when the market is flat or falling?
Spot-only, long-only, sleeve only uses capital the main strategy leaves idle.  G1a/G1b dip-buy in FLAT/DOWN regimes, G2 value (MVRV < 1).
Definition fixed before any result: a G2 'episode' = contiguous run of >= 5 days with MVRV < 1 (shorter blips are traded but not counted)."""
import sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
from quant import metrics as M
import trendcore_spot as T

CM = '/home/user/coinmetrics/data/csv/'
COST = 10.0 / 1e4
BUDGET = 0.25
ERAS = {'pre 2012-17': ('2012-01-01', '2017-12-31'), '2018-22': ('2018-01-01', '2022-12-31'), '2023-26': ('2023-01-01', '2026-05-23')}
OUT = {}

def load(a):
    d = pd.read_csv(CM + f'{a}.csv', parse_dates=['time']).set_index('time')
    return d['PriceUSD'].dropna().loc[:'2026-05-23'], d['CapMVRVCur'].dropna().loc[:'2026-05-23']

def regimes(P):
    off = P < P.rolling(150).mean()
    r90 = P / P.shift(90) - 1
    flat = off & (r90.abs() < 0.20)
    down = off & (r90 <= -0.20)
    return off.fillna(False), flat.fillna(False), down.fillna(False), r90

def dip_signal(P, regime):
    ma, sd = P.rolling(20).mean(), P.rolling(20).std()
    z = ((P - ma) / sd).values; rg = regime.values
    sig = np.zeros(len(P)); inp = False; days = 0
    for t in range(len(P)):
        if np.isnan(z[t]): continue
        if inp:
            days += 1
            if z[t] >= 0 or days >= 10 or not rg[t]: inp = False
        elif rg[t] and z[t] < -2:
            inp = True; days = 0
        sig[t] = 1.0 if inp else 0.0
    return pd.Series(sig, index=P.index)                                           # decided at the close of day t

def trades_from(sig, r, cost=COST):
    pos = sig.shift(1).fillna(0.0)                                                  # held over day t+1  (r stamped t+1 is the return of day t+1)
    rows = []; run = None
    for t, v in pos.items():
        if v == 1.0 and run is None: run = [t, 1.0, 0]
        if v == 1.0: run[1] *= (1 + r.loc[t]); run[2] += 1
        if (v == 0.0 or t == pos.index[-1]) and run is not None:
            rows.append(dict(entry=run[0], days=run[2], ret=run[1] - 1 - 2 * cost)); run = None
    return pd.DataFrame(rows)

def tstat(x): return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan

def sleeve_series(pos, r, w):
    wv = (pos * w).astype(float)
    return wv * r - wv.diff().abs().fillna(wv.abs()) * COST

# ---------------------------------------------------------------- main strategy (official spot, ASSET_VOL 0.45) daily return + start-of-day exposure
px = pd.concat({a: pd.read_parquet(f'data/intraday/{a.upper()}USDT_spot_1h.parquet')['c'] for a in ('btc', 'eth')}, axis=1, sort=True)
oi = pd.read_parquet('data/intraday/BTCUSDT_oi_daily.parquet')['sum_open_interest_value']
Wm, Rm = T.target_weights(px, oi); rb, gb = T._backtest(Wm, Rm)
main = (1 + rb).groupby(rb.index.date).prod() - 1; main.index = pd.to_datetime(main.index); main = main.loc['2018-03-01':'2026-05-23']
e0 = gb.groupby(gb.index.date).first(); e0.index = pd.to_datetime(e0.index); e0 = e0.reindex(main.index)
emean = gb.groupby(gb.index.date).mean(); emean.index = pd.to_datetime(emean.index); emean = emean.reindex(main.index)

# ---------------------------------------------------------------- D1 diagnostics
print('================ D1: what happens when the strategy is not making money? (BTC regimes; strategy = official spot)')
Pb, Mb = load('btc'); rb_ = Pb.pct_change(); off, flat, down, r90 = regimes(Pb)
# the regime of day t is known at the close of t-1 (a same-day classification would leak the day's own move into the label)
on = (~off).shift(1).fillna(False).astype(bool); flat = flat.shift(1).fillna(False).astype(bool); down = down.shift(1).fillna(False).astype(bool)
def reg_table(r, mask, label_idx):
    rows = []
    for lab, m in (('ON  (close > SMA150)', on), ('FLAT (below SMA150, |90d| < 20%)', flat), ('DOWN (below SMA150, 90d <= -20%)', down)):
        x = r[m.reindex(r.index).fillna(False)]
        rows.append(dict(regime=lab, days=len(x), share=len(x) / len(r), btc_cum=(1 + x).prod() - 1, btc_ann=x.mean() * 365, btc_sharpe=x.mean() / x.std() * np.sqrt(365) if x.std() > 0 else np.nan))
    return pd.DataFrame(rows)
for lab, (a, b) in (('BTC 2012-2026.05', ('2012-01-01', '2026-05-23')), ('2018-03..2026.05 (strategy window)', ('2018-03-01', '2026-05-23'))):
    t = reg_table(rb_.loc[a:b], None, None); print(f'\n{lab}')
    if lab.startswith('2018'):
        mr = []
        for lab2, m in (('ON  (close > SMA150)', on), ('FLAT (below SMA150, |90d| < 20%)', flat), ('DOWN (below SMA150, 90d <= -20%)', down)):
            mm = m.reindex(main.index).fillna(False); x = main[mm]
            mr.append(dict(strategy_cum=(1 + x).prod() - 1, strategy_ann=x.mean() * 365, avg_exposure=emean[mm].mean(), pct_days_invested=(emean[mm] > 0.05).mean()))
        t = pd.concat([t, pd.DataFrame(mr)], axis=1)
    pd.options.display.width = 250
    print(t.round(3).to_string(index=False))
    OUT['D1_' + lab[:3]] = t.round(4).to_dict('records')

# ---------------------------------------------------------------- G1 / G2
def evaluate(asset, P, Mv):
    r = P.pct_change(); off, flat, down, r90 = regimes(P)
    res = {}
    sigs = {'G1a dip-buy FLAT': dip_signal(P, flat), 'G1b dip-buy DOWN': dip_signal(P, down), 'G2 value MVRV<1': (Mv.reindex(P.index) < 1.0).astype(float).where(Mv.reindex(P.index).notna(), 0.0)}
    for name, sig in sigs.items():
        tr = trades_from(sig, r); tr = tr[tr.entry >= ERAS['pre 2012-17'][0]]
        info = dict(n=len(tr))
        if name.startswith('G2'):
            ep = tr[tr.days >= 5]
        else: ep = tr
        info['episodes'] = len(ep)
        info['mean'] = float(ep.ret.mean()) if len(ep) else np.nan; info['pos_share'] = float((ep.ret > 0).mean()) if len(ep) else np.nan; info['t'] = float(tstat(ep.ret)) if len(ep) else np.nan
        per = {}
        for era, (a, b) in ERAS.items():
            e = ep[(ep.entry >= a) & (ep.entry <= b)]
            per[era] = dict(n=len(e), mean=float(e.ret.mean()) if len(e) else np.nan, pos=float((e.ret > 0).mean()) if len(e) else np.nan)
        info['eras'] = per
        # sensitivity: pre-sample at 30 bp
        tr30 = trades_from(sig, r, cost=30e-4); tr30 = tr30[(tr30.entry >= ERAS['pre 2012-17'][0]) & (tr30.entry <= ERAS['pre 2012-17'][1])]
        if name.startswith('G2'): tr30 = tr30[tr30.days >= 5]
        info['pre_30bp_mean'] = float(tr30.ret.mean()) if len(tr30) else np.nan
        res[name] = (info, sig, ep)
    return res

print('\n================ G1a / G1b / G2 (trade-level evidence; net of 10 bp per side)')
RES = {}
for asset in ('btc', 'eth'):
    P, Mv = load(asset); RES[asset] = evaluate(asset, P, Mv)
    print(f'\n--- {asset.upper()}')
    for name, (info, sig, ep) in RES[asset].items():
        print(f"{name:20s} trades/episodes {info['episodes']:3d} | mean net {info['mean']*100:6.2f}% | positive {info['pos_share']*100:4.0f}% | t {info['t']:5.2f} | pre-sample at 30bp mean {info['pre_30bp_mean']*100:6.2f}%")
        for era, d in info['eras'].items():
            print(f"      {era:12s} n={d['n']:3d} mean {d['mean']*100:6.2f}% pos {d['pos']*100:4.0f}%" if d['n'] else f"      {era:12s} n=  0")
        OUT.setdefault(asset, {})[name] = {k: v for k, v in info.items()}

# G2 episode list
print('\n--- G2 episodes (>= 5 days with MVRV < 1), net trade return while held')
for asset in ('btc', 'eth'):
    ep = RES[asset]['G2 value MVRV<1'][2]; P = load(asset)[0]
    print(asset.upper())
    for _, x in ep.iterrows():
        fwd = P.shift(-365).loc[x.entry] / P.loc[x.entry] - 1 if x.entry in P.index else np.nan
        print(f"   {x.entry.date()}  {int(x.days):4d} days  net {x.ret*100:7.1f}%   (BTC/ETH 365d forward from entry {fwd*100:6.1f}%)" if not np.isnan(fwd) else f"   {x.entry.date()}  {int(x.days):4d} days  net {x.ret*100:7.1f}%")

# ---------------------------------------------------------------- combination with main strategy
print('\n================ Combination with the official spot strategy (2018-03..2026-05-23; sleeve weight = min(25%, 1 - start-of-day main exposure))')
def stats(r, a, b): return M.summary(r.loc[a:b])
E1 = ('2018-03-01', '2022-12-31'); E2 = ('2023-01-01', '2026-05-23'); FULL = ('2018-03-01', '2026-05-23')
base = stats(main, *FULL); b1 = stats(main, *E1); b2 = stats(main, *E2)
print(f"{'main only':34s} full {base['cagr']*100:5.1f}% / {base['sharpe']:4.2f} / {base['maxdd']*100:6.1f}% | era1 {b1['sharpe']:4.2f} | era2 {b2['sharpe']:4.2f}")
ACC = {}
for asset in ('btc', 'eth'):
    P = load(asset)[0]; r = P.pct_change().reindex(main.index)
    for name, (info, sig, ep) in RES[asset].items():
        pos = sig.shift(1).reindex(main.index).fillna(0.0)
        w = pos * np.minimum(BUDGET, (1.0 - e0).clip(lower=0.0))
        comb = main + (w * r - w.diff().abs().fillna(w.abs()) * COST)
        c = stats(comb, *FULL); c1 = stats(comb, *E1); c2 = stats(comb, *E2)
        print(f"{asset.upper()+' + '+name:34s} full {c['cagr']*100:5.1f}% / {c['sharpe']:4.2f} / {c['maxdd']*100:6.1f}% | era1 {c1['sharpe']:4.2f} | era2 {c2['sharpe']:4.2f} | days active {(w>0).mean()*100:4.1f}% | max combined gross {(e0 + w).max():.2f}")
        acc = dict(mean_pos=bool(info['mean'] > 0), pos_share_ok=bool((info['pos_share'] >= 0.75) if name.startswith('G2') else (info['t'] > 2)),
                   eras_pos=bool(all((d['mean'] > 0) for d in info['eras'].values() if (d['n'] >= (1 if name.startswith('G2') else 5)))),
                   sharpe_eras=bool(c1['sharpe'] >= b1['sharpe'] - 0.02 and c2['sharpe'] >= b2['sharpe'] - 0.02), dd_ok=bool(c['maxdd'] >= base['maxdd'] - 0.02))
        acc['ACCEPT'] = all(acc.values()); ACC[f'{asset}:{name}'] = acc
        OUT.setdefault('combo', {})[f'{asset}:{name}'] = dict(cagr=c['cagr'], sharpe=c['sharpe'], maxdd=c['maxdd'], sharpe_e1=c1['sharpe'], sharpe_e2=c2['sharpe'])
print('\n--- pre-registered acceptance')
for k, v in ACC.items(): print(f'{k:32s}', {kk: vv for kk, vv in v.items()})
OUT['acceptance'] = ACC; OUT['main'] = dict(cagr=base['cagr'], sharpe=base['sharpe'], maxdd=base['maxdd'], sharpe_e1=b1['sharpe'], sharpe_e2=b2['sharpe'])
json.dump(OUT, open('results/flat_down.json', 'w'), indent=1, default=float)
print('\nsaved results/flat_down.json')
