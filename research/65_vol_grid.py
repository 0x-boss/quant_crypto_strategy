"""Round J (pre-registered in research/PREREGISTRATION.md BEFORE this script was run): volatility-scaled long-only spot grid.
V1 c=1.0 range regime | V2 c=0.5, V3 c=2.0 (plateau neighbours of V1) | V4 V1 + quiet filter | V5 uptrend pullback grid.  BTC primary, ETH replication."""
import sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
from quant import metrics as M
import trendcore_spot as T

CM = '/home/user/coinmetrics/data/csv/'
COST = 10e-4; WSL = 0.25; ALLOC = 0.0625; SIG_REF = 0.024
rng = np.random.default_rng(65)
OUT = {}
hist = {a: pd.read_parquet(f'data/intraday/{a.upper()}USDT_spot_1h.parquet')['c'] for a in ('btc', 'eth')}

def daily_px(a):
    d = hist[a].resample('1D').last().dropna()
    cm = pd.read_csv(CM + f'{a}.csv', parse_dates=['time']).set_index('time')['PriceUSD'].dropna().loc[:'2017-12-31']
    scale = d.loc['2017-12-31'] / cm.loc['2017-12-31']
    return pd.concat([cm, d.loc['2018-01-01':] / scale])
def eff_ratio(P, n=30): return (P - P.shift(n)).abs() / P.diff().abs().rolling(n).sum()
def known(daily, idx_h):
    s = daily.copy(); s.index = s.index + pd.Timedelta(days=1)                   # a daily value is known at the end of its day
    return s.reindex(idx_h + pd.Timedelta(hours=1), method='ffill').values

# ----- main strategy for combinations
px2 = pd.concat({a: hist[a] for a in ('btc', 'eth')}, axis=1, sort=True)
oi = pd.read_parquet('data/intraday/BTCUSDT_oi_daily.parquet')['sum_open_interest_value']
Wm, Rm = T.target_weights(px2, oi); rb, gb = T._backtest(Wm, Rm)
main = (1 + rb).groupby(rb.index.date).prod() - 1; main.index = pd.to_datetime(main.index); main = main.loc['2018-03-01':'2026-10-05']
e0 = gb.groupby(gb.index.date).first(); e0.index = pd.to_datetime(e0.index); e0 = e0.reindex(main.index)
E1 = ('2018-03-01', '2022-12-31'); E2 = ('2023-01-01', '2026-10-05'); FULL = ('2018-03-01', '2026-10-05')
mb = {k: M.summary(main.loc[a:b]) for k, (a, b) in (('E1', E1), ('E2', E2), ('full', FULL))}
print(f"main strategy 2018-03..2026-10: {mb['full']['cagr']*100:.1f}% / {mb['full']['sharpe']:.2f} / {mb['full']['maxdd']*100:.1f}%  | E1 Sharpe {mb['E1']['sharpe']:.2f} | E2 Sharpe {mb['E2']['sharpe']:.2f}")

def run_grid(ph, sma, sig, ok, c, cost=COST):
    N = len(ph); units = [0.0] * 4; pbuy = [0.0] * 4
    ret = np.zeros(N); expo = np.zeros(N); trades = []
    centre = np.nan; s = np.nan; L = None
    for t in range(1, N):
        held = units[0] + units[1] + units[2] + units[3]
        ret[t] += held * (ph[t] - ph[t - 1])
        inv = held > 0
        if not inv:
            centre = sma[t]; s = c * sig[t]
            L = [centre * (1 - s * k) for k in (1, 2, 3, 4)] if (centre == centre and s == s) else None
        if L is not None:
            p = ph[t]
            if inv and p < L[3] * (1 - 2 * s):
                for k in range(4):
                    if units[k] > 0:
                        ret[t] -= units[k] * p * cost; trades.append((t, p / pbuy[k] - 1 - 2 * cost, 1)); units[k] = 0.0
            else:
                for k in range(4):
                    if units[k] > 0:
                        if p >= L[k] * (1 + s):
                            ret[t] -= units[k] * p * cost; trades.append((t, p / pbuy[k] - 1 - 2 * cost, 0)); units[k] = 0.0
                    elif ok[t] and p <= L[k] and sig[t] == sig[t]:
                        size = ALLOC * min(1.0, SIG_REF / sig[t]); units[k] = size / p; pbuy[k] = p; ret[t] -= size * cost
        expo[t] = (units[0] + units[1] + units[2] + units[3]) * ph[t]
    return ret, expo, trades

def tstat(x): return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan
ERAS = {'E1 2018-22': ('2018-01-01', '2022-12-31'), 'E2 2023-26': ('2023-01-01', '2026-10-05')}
ACC = {}; RES = {}
for a in ('btc', 'eth'):
    P = hist[a]; ih = P.index; ph = P.values
    r1 = np.log(P).diff(); sig_s = np.sqrt((r1 ** 2).ewm(span=720, min_periods=240).mean() * 24); sig = sig_s.values
    quiet = (sig_s <= sig_s.rolling(8760, min_periods=4000).median().shift(1)).values
    sma = P.rolling(480).mean().values
    dpx = daily_px(a); er_h = known(eff_ratio(dpx), ih); on_h = known((dpx > dpx.rolling(200).mean()).astype(float).where(dpx.rolling(200).mean().notna()), ih)
    reg_range = np.isfinite(er_h) & (er_h < 0.25); reg_on = np.isfinite(on_h) & (on_h == 1.0)
    variants = {'V1 c=1.0 range': (1.0, reg_range), 'V2 c=0.5 range': (0.5, reg_range), 'V3 c=2.0 range': (2.0, reg_range),
                'V4 c=1.0 range+quiet': (1.0, reg_range & quiet), 'V5 c=1.0 uptrend pullback': (1.0, reg_on)}
    print(f'\n================ {a.upper()}  (net of {COST*1e4:.0f} bp per side; level size 6.25% x min(1, 0.024/sigma_d))')
    for name, (c, ok) in variants.items():
        ret, expo, trd = run_grid(ph, sma, sig, ok, c)
        sl = pd.Series(ret, ih); sld = sl.groupby(sl.index.date).sum(); sld.index = pd.to_datetime(sld.index)
        tr = pd.DataFrame(trd, columns=['t', 'ret', 'stop']); tr['exit'] = ih[tr.t.values] if len(tr) else pd.NaT
        tr['gross'] = tr.ret + 2 * COST
        d = sld.loc['2018-03-01':'2026-10-05']; t_day = tstat(d)
        info = dict(n=len(tr), mean=float(tr.ret.mean()), tp=float(1 - tr.stop.mean()), t_day=float(t_day))
        eras = {k: dict(n=int(((tr.exit >= lo) & (tr.exit <= hi)).sum()), mean=float(tr.ret[(tr.exit >= lo) & (tr.exit <= hi)].mean()) if ((tr.exit >= lo) & (tr.exit <= hi)).any() else np.nan) for k, (lo, hi) in ERAS.items()}
        sweep = {cb: float(tr.gross.mean() - 2 * cb * 1e-4) for cb in (0, 5, 10, 20)}
        ss = M.summary(sld.loc['2018-03-01':]); duty = float(ok[np.isfinite(sig)].mean())
        # combination
        s_comb = sld.reindex(main.index).fillna(0.0) * ((1.0 - e0).clip(lower=0.0) / WSL).clip(upper=1.0)
        cs = {k: M.summary((main + s_comb).loc[x:y]) for k, (x, y) in (('E1', E1), ('E2', E2), ('full', FULL))}
        # random-activation control (same duty cycle, 10-day blocks)
        ctrl = None
        if name.split()[0] in ('V1', 'V4', 'V5'):
            nblk = int(np.ceil(len(ph) / 240)); stats = []
            for _ in range(100):
                okr = np.repeat(rng.random(nblk) < duty, 240)[:len(ph)]
                rr, _, _ = run_grid(ph, sma, sig, okr, c)
                x = pd.Series(rr, ih); xd = x.groupby(x.index.date).sum(); stats.append(xd.loc[pd.Timestamp('2018-03-01').date():].mean())
            stats = np.array(stats); ctrl = (float((stats >= d.mean()).mean()), float(stats.mean()))
        RES[(a, name)] = dict(info=info, eras=eras, sweep=sweep, combined=dict(cagr=cs['full']['cagr'], sharpe=cs['full']['sharpe'], maxdd=cs['full']['maxdd'], e1=cs['E1']['sharpe'], e2=cs['E2']['sharpe']), ctrl=ctrl, sleeve=dict(cagr=ss['cagr'], sharpe=ss['sharpe'], maxdd=ss['maxdd']))
        print(f"{name:26s} trips {info['n']:5d} (target {info['tp']*100:3.0f}%) | net/trip {info['mean']*100:6.2f}% | day-level t {t_day:5.2f} | duty {duty*100:3.0f}% | avg inventory {expo.mean():.3f}"
              + (f" | vs random activation: control mean {ctrl[1]*1e4:.2f} bp/day vs {d.mean()*1e4:.2f} bp/day, p={ctrl[0]:.2f}" if ctrl else ''))
        print(f"{'':26s} eras: " + ' | '.join(f"{k} n={v['n']:4d} {v['mean']*100:6.2f}%" for k, v in eras.items()) + f" | cost sweep 0/5/10/20bp: " + ' / '.join(f"{sweep[cb]*100:.2f}%" for cb in (0, 5, 10, 20))
              + f" | sleeve alone {ss['cagr']*100:.1f}% / Sharpe {ss['sharpe']:.2f} / DD {ss['maxdd']*100:.1f}%")
        print(f"{'':26s} combined with main: {cs['full']['cagr']*100:5.1f}% / {cs['full']['sharpe']:4.2f} / {cs['full']['maxdd']*100:6.1f}% (E1 {cs['E1']['sharpe']:.2f}, E2 {cs['E2']['sharpe']:.2f}) vs main {mb['full']['cagr']*100:.1f}% / {mb['full']['sharpe']:.2f} / {mb['full']['maxdd']*100:.1f}% (E1 {mb['E1']['sharpe']:.2f}, E2 {mb['E2']['sharpe']:.2f})")
    for name in ('V1 c=1.0 range', 'V4 c=1.0 range+quiet', 'V5 c=1.0 uptrend pullback'):
        r = RES[(a, name)]; i = r['info']
        plateau = (RES[(a, 'V2 c=0.5 range')]['info']['mean'] > 0 and RES[(a, 'V3 c=2.0 range')]['info']['mean'] > 0) if name.startswith('V1') else True
        acc = dict(t_day_gt2=bool(i['t_day'] > 2), trip_mean_pos=bool(i['mean'] > 0), beats_random=bool(r['ctrl'] is not None and r['ctrl'][0] < 0.05),
                   eras_pos=bool(all(v['mean'] > 0 for v in r['eras'].values() if v['n'] >= 20)), sharpe_ok=bool(r['combined']['e1'] >= mb['E1']['sharpe'] - 0.02 and r['combined']['e2'] >= mb['E2']['sharpe'] - 0.02),
                   dd_ok=bool(r['combined']['maxdd'] >= mb['full']['maxdd'] - 0.02), plateau=bool(plateau))
        acc['ACCEPT'] = all(acc.values()); ACC[f'{a} {name}'] = acc
print('\n--- pre-registered acceptance (BTC decides; ETH is replication)')
for k, v in ACC.items(): print(f'{k:36s}', {kk: bool(vv) for kk, vv in v.items()})
OUT['acceptance'] = {k: {kk: bool(vv) for kk, vv in v.items()} for k, v in ACC.items()}
OUT['results'] = {f'{k[0]}|{k[1]}': v for k, v in RES.items()}
json.dump(OUT, open('results/vol_grid.json', 'w'), indent=1, default=float)
print('saved results/vol_grid.json')
