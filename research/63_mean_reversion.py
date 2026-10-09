"""Round I (pre-registered in research/PREREGISTRATION.md BEFORE this script was run): mean reversion as the partner for flat / non-trending markets.
Spot-only, long-only.  I0 information scan (descriptive); M1-M5 strategies (5 primary trials, BTC primary / ETH replication)."""
import sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
from quant import metrics as M
import trendcore_spot as T

CM = '/home/user/coinmetrics/data/csv/'
COST = 10e-4; WSL = 0.25
rng = np.random.default_rng(63)
OUT = {}
hist = {a: pd.read_parquet(f'data/intraday/{a.upper()}USDT_spot_1h.parquet') for a in ('btc', 'eth')}
PH = {a: hist[a]['c'] for a in hist}

def daily_px(a):
    """CoinMetrics daily close until 2017-12-31 (pre-sample), Binance spot daily close afterwards, joined at 2017-12-31 (same end-of-UTC-day convention)."""
    d = PH[a].resample('1D').last().dropna()
    cm = pd.read_csv(CM + f'{a}.csv', parse_dates=['time']).set_index('time')['PriceUSD'].dropna().loc[:'2017-12-31']
    scale = d.loc['2017-12-31'] / cm.loc['2017-12-31']
    return pd.concat([cm, d.loc['2018-01-01':] / scale])
def daily_vol(a): return hist[a]['qv'].resample('1D').sum()
def eff_ratio(P, n=30): return (P - P.shift(n)).abs() / P.diff().abs().rolling(n).sum()

# ----------------------------------------------------------------------------- main strategy (official spot) for combinations
px2 = pd.concat({a: PH[a] for a in ('btc', 'eth')}, axis=1, sort=True)
oi = pd.read_parquet('data/intraday/BTCUSDT_oi_daily.parquet')['sum_open_interest_value']
Wm, Rm = T.target_weights(px2, oi); rb, gb = T._backtest(Wm, Rm)
main = (1 + rb).groupby(rb.index.date).prod() - 1; main.index = pd.to_datetime(main.index); main = main.loc['2018-03-01':'2026-10-05']
e0 = gb.groupby(gb.index.date).first(); e0.index = pd.to_datetime(e0.index); e0 = e0.reindex(main.index)
E1 = ('2018-03-01', '2022-12-31'); E2 = ('2023-01-01', '2026-10-05'); FULL = ('2018-03-01', '2026-10-05')
mb = {k: M.summary(main.loc[a:b]) for k, (a, b) in (('E1', E1), ('E2', E2), ('full', FULL))}
print(f"main strategy 2018-03..2026-10: {mb['full']['cagr']*100:.1f}% / {mb['full']['sharpe']:.2f} / {mb['full']['maxdd']*100:.1f}%  | E1 Sharpe {mb['E1']['sharpe']:.2f} | E2 Sharpe {mb['E2']['sharpe']:.2f}")

# ============================================================================= I0 information scan
print('\n================ I0 information scan: excess forward return after displacement (bp, t) vs 20 bp round-trip cost')
NS = (1, 6, 24, 72, 168); HS = (1, 6, 24, 72)
def known_daily_to_hourly(daily_series, index_h):
    s = daily_series.copy(); s.index = s.index + pd.Timedelta(days=1)                 # known at the end of the day
    return s.reindex(index_h + pd.Timedelta(hours=1), method='ffill').set_axis(index_h)
def scan(a):
    Ph = PH[a].loc['2020-01-01':]; lp = np.log(Ph); r1 = lp.diff(); sig = r1.ewm(span=720, min_periods=240).std().shift(1)
    dP = daily_px(a); off_d = (dP < dP.rolling(150).mean()).astype(float).where(dP.rolling(150).mean().notna()); rng_d = (eff_ratio(dP) < 0.25).astype(float).where(eff_ratio(dP).notna())
    regs = {'all': pd.Series(1.0, index=Ph.index), 'OFF': known_daily_to_hourly(off_d, Ph.index), 'range': known_daily_to_hourly(rng_d, Ph.index)}
    res = {}
    for side in ('dip', 'rip'):
        for rg, mask_s in regs.items():
            mask = (mask_s == 1.0).values
            tab = {}
            for n in NS:
                z = ((lp - lp.shift(n)) / (sig * np.sqrt(n))).values
                for h in HS:
                    f = (lp.shift(-h) - lp).values
                    valid = mask & np.isfinite(f)
                    base = f[valid].mean(); sd = f[valid].std()
                    ev = np.where(valid & np.isfinite(z) & ((z <= -2) if side == 'dip' else (z >= 2)))[0]
                    keep = []; last = -10**9; gap = max(n, h)
                    for i in ev:
                        if i - last >= gap: keep.append(i); last = i
                    if len(keep) < 20: tab[(n, h)] = (np.nan, np.nan, len(keep)); continue
                    fe = f[keep]; ex = fe.mean() - base
                    tab[(n, h)] = (ex * 1e4, ex / (sd / np.sqrt(len(keep))), len(keep))
            res[(side, rg)] = tab
    return res
SC = {a: scan(a) for a in ('btc', 'eth')}
def show(a, side, rg):
    tab = SC[a][(side, rg)]
    rows = []
    for n in NS:
        rows.append(f"  z over {n:3d}h | " + ' | '.join(f"h={h:2d}: {tab[(n,h)][0]:+6.0f}bp ({tab[(n,h)][1]:+5.1f}) n={tab[(n,h)][2]:4d}" if np.isfinite(tab[(n, h)][0]) else f"h={h:2d}:    n/a        " for h in HS))
    print(f'{a.upper()} {side}s, regime {rg}:'); print('\n'.join(rows))
for a in ('btc', 'eth'):
    show(a, 'dip', 'all'); show(a, 'rip', 'all')
for a in ('btc', 'eth'):
    show(a, 'dip', 'range')
big = []
for a in SC:
    for (side, rg), tab in SC[a].items():
        for (n, h), (ex, t, k) in tab.items():
            if np.isfinite(t) and abs(t) >= 3: big.append((a, side, rg, n, h, ex, t, k))
print(f"\ncells with |t| >= 3 (of {sum(len(t) for a in SC for t in SC[a].values())} cells; ~0.6 expected by chance if independent):")
for b in big: print(f"  {b[0].upper()} {b[1]} {b[2]:5s} z over {b[3]:3d}h, hold {b[4]:2d}h: {b[5]:+7.1f} bp  t {b[6]:+5.1f}  n={b[7]}  {'<- beats 20bp cost' if (b[1]=='dip' and b[5] > 20) or (b[1]=='rip' and b[5] < -20) else ''}")
OUT['scan_cells_t3'] = [dict(asset=b[0], side=b[1], regime=b[2], n=b[3], h=b[4], bp=b[5], t=b[6], k=b[7]) for b in big]

# ============================================================================= strategies
def engine(P, enter, exit_cond, max_hold):
    """P: price array; decisions at close t; returns list of (entry_idx, exit_idx) and pos array (held over t+1)."""
    N = len(P); pos = np.zeros(N); trades = []; inp = False; days = 0; ent = 0
    for t in range(N):
        if inp:
            days += 1
            if exit_cond[t] or days >= max_hold: trades.append((ent, t)); inp = False
        if (not inp) and enter[t]: inp = True; ent = t; days = 0
        pos[t] = 1.0 if inp else 0.0
    if inp: trades.append((ent, N - 1))
    return trades, pos
def trade_table(P, trades, idx, cost=COST):
    P = np.asarray(P, float); rows = []
    for e, x in trades:
        if x <= e: continue
        g = P[x] / P[e] - 1; rows.append(dict(entry=idx[e], bars=x - e, ret=g - 2 * cost, gross=g))
    return pd.DataFrame(rows)
def tstat(x): return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan
def control_p(P, tr, cost=COST, draws=1000):
    P = np.asarray(P, float); lp = np.log(P); N = len(P); d = tr.bars.values; out = []
    valid = np.arange(0, N - d.max() - 1)
    for _ in range(draws):
        s = rng.choice(valid, size=len(d)); out.append(np.mean(np.exp(lp[s + d] - lp[s]) - 1 - 2 * cost))
    out = np.array(out); return float((out >= tr.ret.mean()).mean()), float(out.mean())
def sleeve_daily(pos_series, r_series, w=WSL, cost=COST):
    held = pos_series.shift(1).fillna(0.0) * w
    return held * r_series - held.diff().abs().fillna(held.abs()) * cost
def combo(sleeve_d):
    s = sleeve_d.reindex(main.index).fillna(0.0) * ((1.0 - e0).clip(lower=0.0) / WSL).clip(upper=1.0)
    comb = main + s
    return {k: M.summary(comb.loc[a:b]) for k, (a, b) in (('E1', E1), ('E2', E2), ('full', FULL))}, s

RES = {}
ERAS = {'pre 2012-17': ('2012-01-01', '2017-12-31'), 'E1 2018-22': ('2018-01-01', '2022-12-31'), 'E2 2023-26': ('2023-01-01', '2026-10-05')}
def report(label, tr, P, sleeve_d, ctrl, a, daily_sleeve_ok=True):
    if len(tr) == 0: print(f'{label:30s} no trades'); return None
    info = dict(n=len(tr), mean=float(tr.ret.mean()), win=float((tr.ret > 0).mean()), t=float(tstat(tr.ret)), avg_bars=float(tr.bars.mean()))
    eras = {}
    for en, (lo, hi) in ERAS.items():
        x = tr[(tr.entry >= lo) & (tr.entry <= hi)]
        eras[en] = dict(n=len(x), mean=float(x.ret.mean()) if len(x) else np.nan)
    sweep = {c: float((tr.gross - 2 * c * 1e-4).mean()) for c in (0, 5, 10, 20)}
    cs, s = combo(sleeve_d)
    print(f"{label:30s} trades {info['n']:4d} (avg {info['avg_bars']:5.1f} bars) | net/trade {info['mean']*100:6.2f}% win {info['win']*100:3.0f}% t {info['t']:5.2f} | vs random entry: control {ctrl[1]*100:5.2f}% p={ctrl[0]:.3f}" if ctrl else f"{label:30s} trades {info['n']:4d} | net/trade {info['mean']*100:6.2f}% win {info['win']*100:3.0f}% t {info['t']:5.2f}")
    print(f"{'':30s} eras: " + ' | '.join(f"{k} n={v['n']:3d} {v['mean']*100:6.2f}%" if v['n'] else f"{k} n=  0" for k, v in eras.items()) + f" | cost sweep (per side 0/5/10/20bp) mean: " + ' / '.join(f"{sweep[c]*100:.2f}%" for c in (0, 5, 10, 20)))
    print(f"{'':30s} combined with main: {cs['full']['cagr']*100:5.1f}% / {cs['full']['sharpe']:4.2f} / {cs['full']['maxdd']*100:6.1f}%  (E1 Sharpe {cs['E1']['sharpe']:.2f}, E2 {cs['E2']['sharpe']:.2f})  vs main {mb['full']['cagr']*100:.1f}% / {mb['full']['sharpe']:.2f} / {mb['full']['maxdd']*100:.1f}%")
    acc = dict(mean_pos_t2=bool(info['mean'] > 0 and info['t'] > 2), beats_random=bool(ctrl is not None and ctrl[0] < 0.05),
               eras_pos=bool(all((v['mean'] > 0) for v in eras.values() if v['n'] >= 5)),
               sharpe_ok=bool(cs['E1']['sharpe'] >= mb['E1']['sharpe'] - 0.02 and cs['E2']['sharpe'] >= mb['E2']['sharpe'] - 0.02), dd_ok=bool(cs['full']['maxdd'] >= mb['full']['maxdd'] - 0.02))
    acc['ACCEPT'] = all(acc.values())
    return dict(info=info, eras=eras, sweep=sweep, combined=dict(cagr=cs['full']['cagr'], sharpe=cs['full']['sharpe'], maxdd=cs['full']['maxdd'], e1=cs['E1']['sharpe'], e2=cs['E2']['sharpe']), acc=acc)

ACC = {}
for a in ('btc', 'eth'):
    print(f'\n================ {a.upper()}  (net of {COST*1e4:.0f} bp per side)')
    P = daily_px(a); idx = P.index; Pv = P.values; r = P.pct_change()
    ma20, sd20 = P.rolling(20).mean(), P.rolling(20).std(); z20 = ((P - ma20) / sd20).values
    er = eff_ratio(P).values; rngm = np.isfinite(er) & (er < 0.25)
    ok = np.isfinite(z20)
    # M1 / M2
    for name, enter in (('M1 daily dip, range regime', ok & (z20 < -2) & rngm), ('M2 daily dip, no regime', ok & (z20 < -2))):
        tr_, pos = engine(Pv, enter, np.where(ok, z20 >= 0, False), 10); tr = trade_table(Pv, tr_, idx)
        ctrl = control_p(Pv, tr) if len(tr) else None
        sl = sleeve_daily(pd.Series(pos, idx), r)
        RES[(a, name)] = report(name, tr, P, sl, ctrl, a)
        if RES[(a, name)] and name.startswith('M1'): ACC[f'{a} {name}'] = RES[(a, name)]['acc']
        if RES[(a, name)] and name.startswith('M2'): ACC[f'{a} {name}'] = RES[(a, name)]['acc']
    # M4 volume-conditioned (Binance data only)
    v = daily_vol(a).reindex(idx); lr = np.log(P).diff()
    sig = lr.ewm(span=30, min_periods=15).std().shift(3)
    z3 = (np.log(P / P.shift(3)) / (sig * np.sqrt(3))).values
    rv3 = (v.rolling(3).mean() / v.shift(3).rolling(30).median()).values
    for name, cond in (('M4 low-volume dip (3d)', np.isfinite(rv3) & (rv3 < 1.0)), ('(control) high-volume dip', np.isfinite(rv3) & (rv3 > 1.5))):
        enter = np.isfinite(z3) & (z3 <= -1.5) & cond
        tr_, pos = engine(Pv, enter, np.where(np.isfinite(z3), z3 >= 0, False), 5); tr = trade_table(Pv, tr_, idx)
        ctrl = control_p(Pv, tr) if len(tr) else None
        sl = sleeve_daily(pd.Series(pos, idx), r)
        RES[(a, name)] = report(name, tr, P, sl, ctrl, a)
        if RES[(a, name)] and name.startswith('M4'): ACC[f'{a} {name}'] = RES[(a, name)]['acc']
    # M3 6h
    P6 = PH[a].resample('6h').last().dropna(); i6 = P6.index; p6 = P6.values
    ma, sd = P6.rolling(20).mean(), P6.rolling(20).std(); z6 = ((P6 - ma) / sd).values
    er_d = eff_ratio(daily_px(a))
    s_known = er_d.copy(); s_known.index = s_known.index + pd.Timedelta(days=1); er6 = s_known.reindex(i6 + pd.Timedelta(hours=6), method='ffill').values
    enter = np.isfinite(z6) & (z6 < -2) & np.isfinite(er6) & (er6 < 0.25)
    tr_, pos = engine(p6, enter, np.where(np.isfinite(z6), z6 >= 0, False), 20); tr = trade_table(p6, tr_, i6)
    ctrl = control_p(p6, tr) if len(tr) else None
    r6 = P6.pct_change(); w6 = pd.Series(pos, i6).shift(1).fillna(0.0) * WSL
    sl6 = (w6 * r6 - w6.diff().abs().fillna(w6.abs()) * COST); sl = (1 + sl6).groupby(sl6.index.date).prod() - 1; sl.index = pd.to_datetime(sl.index)
    RES[(a, 'M3 6h dip, range regime')] = report('M3 6h dip, range regime', tr, P6, sl, ctrl, a)
    if RES[(a, 'M3 6h dip, range regime')]: ACC[f'{a} M3 6h dip, range regime'] = RES[(a, 'M3 6h dip, range regime')]['acc']
    # M5 grid (hourly)
    Ph = PH[a]; ph = Ph.values; ih = Ph.index; N = len(ph)
    sk = er_d.copy(); sk.index = sk.index + pd.Timedelta(days=1); erh = sk.reindex(ih + pd.Timedelta(hours=1), method='ffill').values
    sma = Ph.rolling(480).mean().values
    units = np.zeros(4); lvl_px = np.zeros(4); centre = np.nan; ret = np.zeros(N); expo = np.zeros(N); trd = []
    spacing, alloc, stop = 0.025, 0.0625, 0.05
    for t in range(1, N):
        ret[t] += units.sum() * (ph[t] - ph[t - 1])
        inv = units.sum() > 0
        if not inv: centre = sma[t]
        if np.isfinite(centre):
            L = centre * (1 - spacing * np.arange(1, 5))
            if inv and ph[t] < L[-1] * (1 - stop):
                for k in range(4):
                    if units[k] > 0:
                        ret[t] -= units[k] * ph[t] * COST; trd.append((ih[t], ph[t] / lvl_px[k] - 1 - 2 * COST, 'stop')); units[k] = 0.0
            else:
                for k in range(4):
                    if units[k] > 0 and ph[t] >= L[k] * (1 + spacing):
                        ret[t] -= units[k] * ph[t] * COST; trd.append((ih[t], ph[t] / lvl_px[k] - 1 - 2 * COST, 'tp')); units[k] = 0.0
                    elif units[k] == 0 and np.isfinite(erh[t]) and erh[t] < 0.25 and ph[t] <= L[k]:
                        units[k] = alloc / ph[t]; lvl_px[k] = ph[t]; ret[t] -= alloc * COST
        expo[t] = units.sum() * ph[t]
    sl = pd.Series(ret, ih); sld = (1 + sl).groupby(sl.index.date).prod() - 1; sld.index = pd.to_datetime(sld.index)
    tr = pd.DataFrame(trd, columns=['entry', 'ret', 'kind']); tr['bars'] = 1; tr['gross'] = tr.ret + 2 * COST
    tp = (tr.kind == 'tp').mean() if len(tr) else np.nan
    RES[(a, 'M5 grid, range regime')] = report('M5 grid, range regime', tr, Ph, sld, None, a)
    if RES[(a, 'M5 grid, range regime')]:
        ACC[f'{a} M5 grid, range regime'] = RES[(a, 'M5 grid, range regime')]['acc']
        ss = M.summary(sld.loc['2018-01-01':]); print(f"{'':30s} grid sleeve alone (25% budget, zeros included): {ss['cagr']*100:.1f}% CAGR, Sharpe {ss['sharpe']:.2f}, maxDD {ss['maxdd']*100:.1f}%; take-profit exits {tp*100:.0f}% of round trips; average inventory {expo.mean():.3f} of capital")

print('\n--- pre-registered acceptance (BTC decides; ETH is replication)')
for k, v in ACC.items(): print(f'{k:36s}', {kk: bool(vv) for kk, vv in v.items()})
OUT['acceptance'] = {k: {kk: bool(vv) for kk, vv in v.items()} for k, v in ACC.items()}
OUT['results'] = {f'{k[0]}|{k[1]}': (None if v is None else {kk: vv for kk, vv in v.items() if kk != 'acc'}) for k, v in RES.items()}
json.dump(OUT, open('results/mean_reversion.json', 'w'), indent=1, default=float)
print('saved results/mean_reversion.json')
