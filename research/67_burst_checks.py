"""Robustness checks for round K's K2 (burst continuation, all regimes) - NOT new trials; K2 passed the pre-registered rules, and the replication rule
('positive mean') was weak in a strongly drifting asset, so this script applies the stricter tests I should have pre-registered:
regime- and year-stratified random-entry controls, execution delay, event concentration, leave-one-year-out, sleeve size, overlap with the main strategy."""
import sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
from quant import metrics as M
import trendcore_spot as T
CM = '/home/user/coinmetrics/data/csv/'
COST = 10e-4; WSL = 0.25; SIG_REF = 0.024
hist = {a: pd.read_parquet(f'data/intraday/{a.upper()}USDT_spot_1h.parquet')['c'] for a in ('btc', 'eth')}
def cm_daily(a): return pd.read_csv(CM + f'{a}.csv', parse_dates=['time']).set_index('time')['PriceUSD'].dropna()
def daily_px(a):
    d = hist[a].resample('1D').last().dropna(); cm = cm_daily(a).loc[:'2017-12-31']
    return pd.concat([cm, d.loc['2018-01-01':] / (d.loc['2017-12-31'] / cm.loc['2017-12-31'])])
def known(daily, idx_h):
    s = daily.copy(); s.index = s.index + pd.Timedelta(days=1)
    return s.reindex(idx_h + pd.Timedelta(hours=1), method='ffill').values
def tstat(x): return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan
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
rng = np.random.default_rng(67)
px2 = pd.concat({a: hist[a] for a in ('btc', 'eth')}, axis=1, sort=True)
oi = pd.read_parquet('data/intraday/BTCUSDT_oi_daily.parquet')['sum_open_interest_value']
Wm, Rm = T.target_weights(px2, oi); rb, gb = T._backtest(Wm, Rm)
main = (1 + rb).groupby(rb.index.date).prod() - 1; main.index = pd.to_datetime(main.index); main = main.loc['2018-03-01':'2026-10-05']
e0 = gb.groupby(gb.index.date).first(); e0.index = pd.to_datetime(e0.index); e0 = e0.reindex(main.index)
E1 = ('2018-03-01', '2022-12-31'); E2 = ('2023-01-01', '2026-10-05'); FULL = ('2018-03-01', '2026-10-05')
mb = {k: M.summary(main.loc[a:b]) for k, (a, b) in (('E1', E1), ('E2', E2), ('full', FULL))}

def shift_trades(tr, d): return [(e + d, x + d) for e, x in tr]
for a in ('btc', 'eth'):
    P = hist[a]; ih = P.index; ph = P.values; N = len(ph)
    dpx = daily_px(a); on_h = known((dpx > dpx.rolling(150).mean()).astype(float).where(dpx.rolling(150).mean().notna()), ih)
    on = np.isfinite(on_h) & (on_h == 1.0)
    tr, z = burst_trades(P, 72, 2.0, 72, None)
    net = net_returns(P, tr); ent = np.array([e for e, _ in tr]); yrs = ih[ent].year
    print(f'\n================ {a.upper()} K2 (all regimes): {len(tr)} events, net {net.mean()*100:.2f}%/event (t {tstat(pd.Series(net)):.2f}); {on[ent].mean()*100:.0f}% of events happen while the daily close is ABOVE its SMA150')
    # --- controls
    lp = np.log(ph); valid = np.where(np.isfinite(z) & (np.arange(N) < N - 73) & (ih >= pd.Timestamp('2018-01-01')))[0]
    def draw(pool_fn):
        out = []
        for _ in range(1000):
            s = np.array([rng.choice(pool_fn(i)) for i in ent]); out.append(np.mean(np.exp(lp[s + 72] - lp[s]) - 1 - 2 * COST))
        return np.array(out)
    pools_reg = {True: valid[on[valid]], False: valid[~on[valid]]}
    c_reg = draw(lambda i: pools_reg[bool(on[i])])
    yrs_all = ih[valid].year; pools_y = {y: valid[yrs_all == y] for y in np.unique(yrs_all)}
    c_yr = draw(lambda i: pools_y[ih[i].year])
    c_both = draw(lambda i: valid[(yrs_all == ih[i].year) & (on[valid] == on[i])] if ((yrs_all == ih[i].year) & (on[valid] == on[i])).any() else valid)
    for lab, c in (('regime-matched (ON/OFF) random entries', c_reg), ('same-calendar-year random entries', c_yr), ('same year AND regime', c_both)):
        print(f"  control: {lab:42s} mean {c.mean()*100:5.2f}%  -> events beat it with p = {(c >= net.mean()).mean():.3f}  (excess {(net.mean()-c.mean())*100:+.2f} pts)")
    # --- execution delay
    for d in (1, 3, 6, 24):
        tr_d = [(e + d, x + d) for e, x in tr if x + d < N]; n_d = net_returns(P, tr_d)
        print(f"  entry delayed {d:2d} h: net {n_d.mean()*100:5.2f}%/event  (win {np.mean(n_d>0)*100:.0f}%)")
    n20 = net + 2 * COST - 2 * 20e-4
    print(f"  at 20 bp per side: {n20.mean()*100:.2f}%/event")
    # --- concentration
    s = pd.Series(net, index=ih[ent]); top5 = s.nlargest(5).sum() / s.sum()
    print(f"  median {s.median()*100:.2f}%, best {s.max()*100:.1f}%, worst {s.min()*100:.1f}%; top-5 events = {top5*100:.0f}% of total return; events without the best 5: mean {(s.sum()-s.nlargest(5).sum())/(len(s)-5)*100:.2f}%")
    by = s.groupby(s.index.year).agg(['count', 'mean', 'sum']); by['mean'] *= 100; by['sum'] *= 100
    print('  by year (n / mean % / sum %): ' + ' | '.join(f"{y}: {int(r['count'])}/{r['mean']:.1f}/{r['sum']:.0f}" for y, r in by.iterrows()))
    loyo = {y: float(s[s.index.year != y].mean() * 100) for y in by.index}
    print('  leave-one-year-out mean %: ' + ', '.join(f'{y}: {v:.2f}' for y, v in loyo.items()))
    if a == 'btc':
        # --- sleeve size and overlap
        e0_at = e0.reindex(pd.DatetimeIndex(ih[ent].normalize()), method='ffill').values
        print(f"  main-strategy exposure at the event entry (start of that day): mean {np.nanmean(e0_at):.2f}; events with main exposure > 0.5: {np.nanmean(e0_at>0.5)*100:.0f}%")
        sig_d = np.sqrt((np.log(P).diff() ** 2).ewm(span=720, min_periods=240).mean() * 24).values
        for wmax in (0.25, 0.50, 1.00):
            ret = np.zeros(N)
            for e, x in tr:
                w = wmax * min(1.0, SIG_REF / sig_d[e]); ret[e + 1:x + 1] += w * (ph[e + 1:x + 1] / ph[e:x] - 1); ret[e] -= w * COST; ret[x] -= w * COST
            sl = pd.Series(ret, ih); sld = sl.groupby(sl.index.date).sum(); sld.index = pd.to_datetime(sld.index)
            fac = ((1.0 - e0).clip(lower=0.0) / wmax).clip(upper=1.0); comb = main + sld.reindex(main.index).fillna(0.0) * fac
            cs = {k: M.summary(comb.loc[x:y]) for k, (x, y) in (('E1', E1), ('E2', E2), ('full', FULL))}; ss = M.summary(sld.loc['2018-03-01':])
            print(f"  sleeve max weight {int(wmax*100):3d}%: alone {ss['cagr']*100:5.1f}% / Sharpe {ss['sharpe']:.2f} / DD {ss['maxdd']*100:.1f}%  | combined {cs['full']['cagr']*100:5.1f}% / {cs['full']['sharpe']:.2f} / {cs['full']['maxdd']*100:.1f}% (E1 {cs['E1']['sharpe']:.2f}, E2 {cs['E2']['sharpe']:.2f}) vs main {mb['full']['cagr']*100:.1f}% / {mb['full']['sharpe']:.2f} / {mb['full']['maxdd']*100:.1f}%")
        print('  correlation of sleeve daily P&L with main strategy:', f"{sld.reindex(main.index).fillna(0).corr(main):+.2f}")

# --- pre-sample controls (daily): same-year random entries
print('\n================ pre-sample replication with same-year random-entry controls (daily, hold 3 days)')
for a, lo in (('btc', '2012-01-01'), ('eth', '2016-01-01')):
    d = cm_daily(a).loc[lo:'2017-12-31']; lr = np.log(d).diff(); sg = lr.ewm(span=30, min_periods=15).std().shift(3)
    z3 = (np.log(d / d.shift(3)) / (sg * np.sqrt(3))).values; p = d.values; lpp = np.log(p); tr = []; t = 0
    while t < len(p) - 4:
        if np.isfinite(z3[t]) and z3[t] >= 2: tr.append((t, t + 3)); t += 3
        else: t += 1
    ev = np.array([e for e, _ in tr]); net = np.array([p[x] / p[e] - 1 - 2 * COST for e, x in tr]); yy = d.index.year
    valid = np.where(np.isfinite(z3) & (np.arange(len(p)) < len(p) - 4))[0]; pools = {y: valid[yy[valid] == y] for y in np.unique(yy[valid])}
    out = []
    for _ in range(1000):
        s = np.array([rng.choice(pools[yy[i]]) for i in ev]); out.append(np.mean(np.exp(lpp[s + 3] - lpp[s]) - 1 - 2 * COST))
    out = np.array(out)
    print(f"{a.upper()} {lo[:4]}-2017: {len(tr)} events, net {net.mean()*100:.2f}%/event vs same-year random entries {out.mean()*100:.2f}%  -> p = {(out >= net.mean()).mean():.3f}, excess {(net.mean()-out.mean())*100:+.2f} pts")
