"""Round F (pre-registered in research/PREREGISTRATION.md before this script was run): participation.

Part 1  descriptive frontier of the official spot-only strategy: exposure dial, constant BTC core, idle-cash yield (no acceptance rule)
Part 2  two trials: crowding overlays that also scale UP when positioning is light (F1 open interest, F2 funding); F3 = F1 x F2 only if both pass
"""
import sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
from quant import metrics as M
import trendcore_spot as T

px = pd.concat({a: pd.read_parquet(f'data/intraday/{a.upper()}USDT_spot_1h.parquet')['c'] for a in ('btc', 'eth')}, axis=1, sort=True)
oi = pd.read_parquet('data/intraday/BTCUSDT_oi_daily.parquet')['sum_open_interest_value']
fund = pd.read_parquet('data/intraday/BTCUSDT_funding.parquet')['fundingRate']
START = '2018-03-01'
OI_ORIG = T.oi_multiplier
OUT = {}

def to_daily(rb):
    d = (1 + rb).groupby(rb.index.date).prod() - 1
    d.index = pd.to_datetime(d.index); return d

def run(asset_vol=0.45, target_vol=0.40, mult=None, use_oi=True):
    """mult: function(index) -> multiplier Series that REPLACES the official overlay (None = official)."""
    T.ASSET_VOL = asset_vol
    T.oi_multiplier = OI_ORIG if mult is None else (lambda index, oi_, strength=0.5: mult(index))
    try:
        W, R = T.target_weights(px, oi if use_oi else None, target_vol=target_vol)
        rb, gb = T._backtest(W, R)
    finally:
        T.ASSET_VOL = 0.45; T.oi_multiplier = OI_ORIG
    r = to_daily(rb).loc[START:]
    e = gb.groupby(gb.index.date).mean(); e.index = pd.to_datetime(e.index)
    return r, e.reindex(r.index)

base, e_base = run()
chk, _ = T.backtest(px, oi, start=START)
assert np.allclose(chk.values, base.values), 'baseline mismatch'
FULL = (START, '2026-10-05'); E1 = ('2018-03-01', '2022-12-31'); E2 = ('2023-01-01', '2026-10-05')
def S(r, w): return M.summary(r.loc[w[0]:w[1]])
def line(name, r, e=None, wins=(('full', FULL), ('2018-22', E1), ('2023+', E2))):
    o = []
    for lab, w in wins:
        s = S(r, w); o.append(f"{lab}: {s['cagr']*100:5.1f}% / {s['sharpe']:4.2f} / {s['maxdd']*100:6.1f}%")
    ex = f' | avg exposure {e.loc[FULL[0]:FULL[1]].mean():.2f}' if e is not None else ''
    print(f'{name:38s}' + ' | '.join(o) + ex)

# ----------------------------------------------------------------------------- Part 1
print('================ PART 1 (descriptive): what does a different risk budget buy?  [CAGR / Sharpe / maxDD]')
rows = []
for tv in (0.40, 0.60):
    for av in (0.30, 0.45, 0.60, 0.80, 1.00, 1.50):
        r, e = run(av, tv)
        line(f'ASSET_VOL {av:.2f} / port vol tgt {tv:.2f}', r, e)
        s = S(r, FULL)
        rows.append(dict(asset_vol=av, target_vol=tv, cagr=s['cagr'], sharpe=s['sharpe'], maxdd=s['maxdd'], vol=s['vol'], avg_exposure=float(e.mean()),
                         flat_share=float((e < 0.05).mean()), sh_2023=S(r, E2)['sharpe'], cagr_2023=S(r, E2)['cagr'], dd_2023=S(r, E2)['maxdd'], cagr_1822=S(r, E1)['cagr'], sh_1822=S(r, E1)['sharpe']))
dial = pd.DataFrame(rows); dial.to_csv('results/participation_dial.csv', index=False)
o = S(base, FULL)
free = dial[(dial.cagr > o['cagr']) & (dial.sharpe >= o['sharpe'] - 0.06) & (dial.maxdd >= -0.22)]
print(f"\n'free' dial points (CAGR > official {o['cagr']*100:.1f}%, Sharpe >= {o['sharpe']-0.06:.2f}, maxDD >= -22%):")
print(free[['asset_vol', 'target_vol', 'cagr', 'sharpe', 'maxdd', 'avg_exposure']].round(3).to_string(index=False) if len(free) else '  none')

btc = px['btc'].resample('1D').last().pct_change().reindex(base.index).fillna(0.0)
print('\n--- constant BTC core c + (1-c) x official')
for c in (0.0, 0.10, 0.20, 0.30):
    r = (1 - c) * base + c * btc
    line(f'core {int(c*100):2d}% BTC', r)
    s = S(r, FULL); OUT.setdefault('core', {})[f'{int(c*100)}'] = dict(cagr=s['cagr'], sharpe=s['sharpe'], maxdd=s['maxdd'])
print('\n--- idle-cash yield (optional lever, NOT in the headline numbers)')
for y in (0.0, 0.04, 0.045):
    r = base + y / 365 * (1 - e_base.clip(upper=1.0))
    line(f'idle yield {y*100:.1f}% p.a.', r)
    s = S(r, FULL); OUT.setdefault('yield', {})[f'{y}'] = dict(cagr=s['cagr'], sharpe=s['sharpe'], maxdd=s['maxdd'])
print(f"\nBTC buy & hold same window: ", end=''); sb = S(btc, FULL); print(f"{sb['cagr']*100:.1f}% / {sb['sharpe']:.2f} / {sb['maxdd']*100:.1f}%")

# ----------------------------------------------------------------------------- Part 2
print('\n================ PART 2 (2 trials): crowding overlays that also scale UP when positioning is light')
def zscore_daily(x, win=365):
    return ((x - x.rolling(win).mean()) / x.rolling(win).std()).shift(1)                 # known at the previous day's close
z_oi = zscore_daily(np.log(oi))
fx = fund.resample('1D').mean().rolling(30).mean()
z_f = zscore_daily(fx)
def mult_from(z, strength, lo=-2.0):
    m = 1 - strength * z.clip(lo, 2.0) / 2
    return lambda index: m.reindex(index, method='ffill').fillna(1.0)
def off_oi(index): return OI_ORIG(index, oi)
def mult_f2(strength):
    mf = 1 - strength * z_f.clip(-2.0, 2.0) / 2
    return lambda index: off_oi(index) * mf.reindex(index, method='ffill').fillna(1.0)
def mult_f3(strength):
    mf = 1 - strength * z_f.clip(-2.0, 2.0) / 2; mo = 1 - strength * z_oi.clip(-2.0, 2.0) / 2
    return lambda index: mo.reindex(index, method='ffill').fillna(1.0) * mf.reindex(index, method='ffill').fillna(1.0)

WA = ('2021-09-01', '2022-12-31'); WB = ('2023-01-01', '2026-05-23'); WF = ('2021-09-01', '2026-05-23'); WFW = ('2026-05-24', '2026-10-05')
def wl(r):
    return {k: S(r, w) for k, w in (('A', WA), ('B', WB), ('full', WF), ('fwd', WFW))}
def pline(name, d):
    print(f"{name:30s}" + ' | '.join(f"{k}: {d[k]['cagr']*100:5.1f}% / {d[k]['sharpe']:4.2f} / {d[k]['maxdd']*100:6.1f}%" for k in ('A', 'B', 'full', 'fwd')))
ref = wl(base); pline('baseline (official)', ref)
res = {'baseline': {k: dict(cagr=v['cagr'], sharpe=v['sharpe'], maxdd=v['maxdd']) for k, v in ref.items()}}
def accepted(d, strengths):
    c = {}
    c['cagr_up'] = d[0.5]['full']['cagr'] > ref['full']['cagr']
    c['sharpe_both_eras'] = d[0.5]['A']['sharpe'] >= ref['A']['sharpe'] and d[0.5]['B']['sharpe'] >= ref['B']['sharpe']
    c['dd_ok'] = d[0.5]['full']['maxdd'] >= ref['full']['maxdd'] - 0.02
    c['plateau'] = all(d[s]['A']['sharpe'] >= ref['A']['sharpe'] - 0.02 and d[s]['B']['sharpe'] >= ref['B']['sharpe'] - 0.02 for s in (0.25, 0.75))
    c['ACCEPT'] = all(c.values()); return c
ACC = {}
for name, fn in (('F1 OI both ways', lambda s: mult_from(z_oi, s)), ('F2 + funding both ways', mult_f2)):
    d = {}
    for st in (0.25, 0.5, 0.75):
        r, e = run(mult=fn(st)); d[st] = wl(r); pline(f'{name} strength {st}', d[st])
        if st == 0.5: res[name + ' daily'] = r
    ACC[name] = accepted(d, (0.25, 0.5, 0.75))
    print('   acceptance:', {k: bool(v) for k, v in ACC[name].items()})
    res[name] = {str(st): {k: dict(cagr=v['cagr'], sharpe=v['sharpe'], maxdd=v['maxdd']) for k, v in d[st].items()} for st in d}
if all(v['ACCEPT'] for v in ACC.values()):
    d = {}
    for st in (0.25, 0.5, 0.75):
        r, e = run(mult=mult_f3(st)); d[st] = wl(r); pline(f'F3 F1 x F2 strength {st}', d[st])
    ACC['F3'] = accepted(d, (0.25, 0.5, 0.75)); print('   acceptance:', {k: bool(v) for k, v in ACC['F3'].items()})
else:
    print('F3 not run (requires both F1 and F2 accepted)')
OUT['acceptance'] = {k: {kk: bool(vv) for kk, vv in v.items()} for k, v in ACC.items()}
OUT['part2'] = {k: v for k, v in res.items() if not k.endswith('daily')}
OUT['dial_free_points'] = free[['asset_vol', 'target_vol', 'cagr', 'sharpe', 'maxdd']].to_dict('records') if len(free) else []
# context: how often the overlays were away from 1.0
mo = (1 - 0.5 * z_oi.clip(-2, 2) / 2).loc[WF[0]:WF[1]]
print(f"\ncontext: F1 multiplier on {WF[0]}..{WF[1]}: mean {mo.mean():.2f}, share of days >1.05: {(mo > 1.05).mean()*100:.0f}%, <0.95: {(mo < 0.95).mean()*100:.0f}%")
mf = (1 - 0.5 * z_f.clip(-2, 2) / 2).loc[WF[0]:WF[1]]
print(f"         F2 multiplier: mean {mf.mean():.2f}, share >1.05: {(mf > 1.05).mean()*100:.0f}%, <0.95: {(mf < 0.95).mean()*100:.0f}%;  corr(z_oi, z_funding) = {z_oi.corr(z_f):.2f}")
json.dump(OUT, open('results/participation.json', 'w'), indent=1, default=float)
print('saved results/participation*.{json,csv}')
