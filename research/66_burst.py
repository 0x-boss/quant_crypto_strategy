"""Round K (pre-registered in research/PREREGISTRATION.md BEFORE this script was run): continuation after bursts (72 h z >= 2 -> hold 72 h).
K1 only while OFF (last completed daily close < SMA150), K2 all regimes.  BTC primary / ETH replication / BTC 2012-17 daily replication / plateau neighbours / random-entry control."""
import sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
from quant import metrics as M
import trendcore_spot as T

CM = '/home/user/coinmetrics/data/csv/'
COST = 10e-4; WSL = 0.25; SIG_REF = 0.024
rng = np.random.default_rng(66)
OUT = {}
hist = {a: pd.read_parquet(f'data/intraday/{a.upper()}USDT_spot_1h.parquet')['c'] for a in ('btc', 'eth')}
def cm_daily(a): return pd.read_csv(CM + f'{a}.csv', parse_dates=['time']).set_index('time')['PriceUSD'].dropna()
def daily_px(a):
    d = hist[a].resample('1D').last().dropna(); cm = cm_daily(a).loc[:'2017-12-31']
    return pd.concat([cm, d.loc['2018-01-01':] / (d.loc['2017-12-31'] / cm.loc['2017-12-31'])])
def known(daily, idx_h):
    s = daily.copy(); s.index = s.index + pd.Timedelta(days=1)
    return s.reindex(idx_h + pd.Timedelta(hours=1), method='ffill').values
def tstat(x): return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan

px2 = pd.concat({a: hist[a] for a in ('btc', 'eth')}, axis=1, sort=True)
oi = pd.read_parquet('data/intraday/BTCUSDT_oi_daily.parquet')['sum_open_interest_value']
Wm, Rm = T.target_weights(px2, oi); rb, gb = T._backtest(Wm, Rm)
main = (1 + rb).groupby(rb.index.date).prod() - 1; main.index = pd.to_datetime(main.index); main = main.loc['2018-03-01':'2026-10-05']
e0 = gb.groupby(gb.index.date).first(); e0.index = pd.to_datetime(e0.index); e0 = e0.reindex(main.index)
E1 = ('2018-03-01', '2022-12-31'); E2 = ('2023-01-01', '2026-10-05'); FULL = ('2018-03-01', '2026-10-05')
mb = {k: M.summary(main.loc[a:b]) for k, (a, b) in (('E1', E1), ('E2', E2), ('full', FULL))}
print(f"main strategy 2018-03..2026-10: {mb['full']['cagr']*100:.1f}% / {mb['full']['sharpe']:.2f} / {mb['full']['maxdd']*100:.1f}%  | E1 Sharpe {mb['E1']['sharpe']:.2f} | E2 Sharpe {mb['E2']['sharpe']:.2f}")

def burst_trades(P, window, thr, hold, regime=None, start='2018-01-01'):
    lp = np.log(P.values); r1 = np.log(P).diff(); sig = r1.ewm(span=720, min_periods=240).std().shift(1).values; N = len(P)
    z = np.full(N, np.nan); z[window:] = (lp[window:] - lp[:-window]) / (sig[window:] * np.sqrt(window))
    t0 = int(np.searchsorted(P.index.values, np.datetime64(start))); trades = []; t = max(window, t0)
    while t < N - hold - 1:
        if np.isfinite(z[t]) and z[t] >= thr and (regime is None or regime[t]): trades.append((t, t + hold)); t += hold
        else: t += 1
    return trades, z
def net_returns(P, trades, cost=COST):
    p = P.values; return np.array([p[x] / p[e] - 1 - 2 * cost for e, x in trades])
def control_p(P, n, hold, mean_actual, eligible, cost=COST, draws=1000):
    p = P.values; lp = np.log(p); idx = np.where(eligible & (np.arange(len(p)) < len(p) - hold - 1))[0]; out = []
    for _ in range(draws):
        s = rng.choice(idx, size=n); out.append(np.mean(np.exp(lp[s + hold] - lp[s]) - 1 - 2 * cost))
    out = np.array(out); return float((out >= mean_actual).mean()), float(out.mean())

ACC = {}; RES = {}
for a in ('btc', 'eth'):
    P = hist[a]; ih = P.index; ph = P.values
    dpx = daily_px(a); off_h = known((dpx < dpx.rolling(150).mean()).astype(float).where(dpx.rolling(150).mean().notna()), ih)
    off = np.isfinite(off_h) & (off_h == 1.0)
    r1 = np.log(P).diff(); sig_d = np.sqrt((r1 ** 2).ewm(span=720, min_periods=240).mean() * 24).values
    print(f'\n================ {a.upper()}  hourly, hold 72 h, net of {COST*1e4:.0f} bp per side')
    for name, reg in (('K1 OFF only', off), ('K2 all regimes', None)):
        tr, z = burst_trades(P, 72, 2.0, 72, reg)
        net = net_returns(P, tr); ent = ih[[e for e, _ in tr]]
        eras = {k: dict(n=int(((ent >= lo) & (ent <= hi)).sum()), mean=float(net[(ent >= lo) & (ent <= hi)].mean()) if ((ent >= lo) & (ent <= hi)).any() else np.nan) for k, (lo, hi) in (('E1 2018-22', ('2018-01-01', '2022-12-31')), ('E2 2023-26', ('2023-01-01', '2026-10-05')))}
        elig = np.isfinite(z) & (reg if reg is not None else True)
        cp, cm_ = control_p(P, len(tr), 72, net.mean(), elig) if len(tr) else (np.nan, np.nan)
        sweep = {c: float((net + 2 * COST - 2 * c * 1e-4).mean()) for c in (0, 5, 10, 20)}
        # plateau neighbours
        nb = {}
        for lab, (w_, th_) in {'48h z2': (48, 2.0), '96h z2': (96, 2.0), '72h z1.5': (72, 1.5), '72h z2.5': (72, 2.5)}.items():
            t2, _ = burst_trades(P, w_, th_, w_ if w_ in (48, 96) else 72, reg); n2 = net_returns(P, t2); nb[lab] = (float(n2.mean()) if len(n2) else np.nan, len(n2))
        # sleeve returns
        ret = np.zeros(len(ph)); expo = np.zeros(len(ph))
        for e, x in tr:
            w = WSL * min(1.0, SIG_REF / sig_d[e]) if np.isfinite(sig_d[e]) else 0.0
            seg = ph[e + 1:x + 1] / ph[e:x] - 1; ret[e + 1:x + 1] += w * seg; ret[e] -= w * COST; ret[x] -= w * COST; expo[e + 1:x + 1] += w
        sl = pd.Series(ret, ih); sld = sl.groupby(sl.index.date).sum(); sld.index = pd.to_datetime(sld.index)
        ss = M.summary(sld.loc['2018-03-01':]); s_comb = sld.reindex(main.index).fillna(0.0) * ((1.0 - e0).clip(lower=0.0) / WSL).clip(upper=1.0)
        cs = {k: M.summary((main + s_comb).loc[x:y]) for k, (x, y) in (('E1', E1), ('E2', E2), ('full', FULL))}
        RES[(a, name)] = dict(n=len(tr), mean=float(net.mean()), t=float(tstat(pd.Series(net))), win=float((net > 0).mean()), eras=eras, ctrl=(cp, cm_), nb=nb, sweep=sweep, sleeve=dict(cagr=ss['cagr'], sharpe=ss['sharpe'], maxdd=ss['maxdd']),
                              combined=dict(cagr=cs['full']['cagr'], sharpe=cs['full']['sharpe'], maxdd=cs['full']['maxdd'], e1=cs['E1']['sharpe'], e2=cs['E2']['sharpe']))
        r = RES[(a, name)]
        print(f"{name:15s} events {r['n']:3d} | net/event {r['mean']*100:6.2f}% win {r['win']*100:3.0f}% t {r['t']:5.2f} | random-entry control {cm_*100:5.2f}% p={cp:.3f} | cost sweep 0/5/10/20bp: " + ' / '.join(f"{sweep[c]*100:.2f}%" for c in (0, 5, 10, 20)))
        print(f"{'':15s} eras: " + ' | '.join(f"{k} n={v['n']:3d} {v['mean']*100:6.2f}%" for k, v in eras.items()) + ' | plateau: ' + ', '.join(f"{k} {v[0]*100:.2f}% (n={v[1]})" for k, v in nb.items()))
        print(f"{'':15s} sleeve alone (zeros included) {ss['cagr']*100:.1f}% CAGR / Sharpe {ss['sharpe']:.2f} / DD {ss['maxdd']*100:.1f}% | avg inventory {expo.mean():.3f} | combined with main {cs['full']['cagr']*100:.1f}% / {cs['full']['sharpe']:.2f} / {cs['full']['maxdd']*100:.1f}% (E1 {cs['E1']['sharpe']:.2f}, E2 {cs['E2']['sharpe']:.2f}) vs {mb['full']['cagr']*100:.1f}% / {mb['full']['sharpe']:.2f} / {mb['full']['maxdd']*100:.1f}% (E1 {mb['E1']['sharpe']:.2f}, E2 {mb['E2']['sharpe']:.2f})")

# ----- pre-sample daily replication (BTC 2012-2017; ETH 2016-2017)
print('\n================ replication: daily analogue (3-day z >= 2, hold 3 days) on CoinMetrics data before 2018 (never used for this)')
PRE = {}
for a, lo in (('btc', '2012-01-01'), ('eth', '2016-01-01')):
    d = cm_daily(a).loc[:'2017-12-31']; lr = np.log(d).diff(); sg = lr.ewm(span=30, min_periods=15).std().shift(3)
    z3 = (np.log(d / d.shift(3)) / (sg * np.sqrt(3))).values; offd = (d < d.rolling(150).mean()).values; p = d.values
    for name, reg in (('K1 OFF only', offd), ('K2 all regimes', None)):
        tr = []; t = 0
        while t < len(p) - 4:
            if d.index[t] >= pd.Timestamp(lo) and np.isfinite(z3[t]) and z3[t] >= 2 and (reg is None or reg[t]): tr.append((t, t + 3)); t += 3
            else: t += 1
        net = net_returns(d, tr); net30 = net_returns(d, tr, cost=30e-4)
        PRE[(a, name)] = dict(n=len(tr), mean=float(net.mean()) if len(tr) else np.nan, t=float(tstat(pd.Series(net))) if len(tr) > 2 else np.nan, mean30=float(net30.mean()) if len(tr) else np.nan)
        x = PRE[(a, name)]; print(f"{a.upper()} {name:15s} events {x['n']:3d} | net/event {x['mean']*100:6.2f}% (t {x['t']:5.2f}) | at 30 bp per side {x['mean30']*100:6.2f}%")

print('\n--- pre-registered acceptance (BTC decides)')
for name in ('K1 OFF only', 'K2 all regimes'):
    r = RES[('btc', name)]; e = RES[('eth', name)]; pre = PRE[('btc', name)]
    acc = dict(mean_pos_t2=bool(r['mean'] > 0 and r['t'] > 2), beats_random=bool(r['ctrl'][0] < 0.05), eras_pos=bool(all(v['mean'] > 0 for v in r['eras'].values() if v['n'] >= 10)),
               replication=bool(pre['mean'] > 0 and e['mean'] > 0), plateau=bool(all(v[0] > 0 for v in r['nb'].values() if np.isfinite(v[0]))),
               combined=bool(r['combined']['cagr'] > mb['full']['cagr'] and r['combined']['e1'] >= mb['E1']['sharpe'] - 0.02 and r['combined']['e2'] >= mb['E2']['sharpe'] - 0.02 and r['combined']['maxdd'] >= mb['full']['maxdd'] - 0.02))
    acc['ACCEPT'] = all(acc.values()); ACC[name] = acc; print(f'{name:16s}', {k: bool(v) for k, v in acc.items()})
OUT['acceptance'] = {k: {kk: bool(vv) for kk, vv in v.items()} for k, v in ACC.items()}
OUT['results'] = {f'{k[0]}|{k[1]}': v for k, v in RES.items()}; OUT['pre_sample'] = {f'{k[0]}|{k[1]}': v for k, v in PRE.items()}
json.dump(OUT, open('results/burst.json', 'w'), indent=1, default=float); print('saved results/burst.json')
