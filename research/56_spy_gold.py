"""Round E (pre-registered in research/PREREGISTRATION.md BEFORE this script was run).

Idea (user): when SPY closes below its 200-day EMA (equity risk-off, which "drags crypto too") hold gold.
Spot-only, long-only, gross <= 100 % of capital.  Nothing below is tuned: EMA span 200 and the variants S0-S3 are the
ones written down in advance; sub-period blocks are calendar blocks fixed here before any result was seen.

Part A  does the mechanism exist?  (GLD 2004-, BTC 2014-, ETH 2016-; regime s=1 vs s=0, block-bootstrap CI, sub-periods)
Part B  strategy variants on the final BTC+ETH 6h spot strategy (trendcore_spot.py):
          S0 baseline | S1 crypto x0 when s=1 (cash) | S2 crypto unchanged, idle capital -> gold when s=1 | S3 crypto x0 and 100 % gold
Timing  s is computed on SPY trading days after the close, carried over weekends/holidays and applied from 00:00 UTC of the
        NEXT calendar day (so nothing that happens on day d can influence the position held on day d).
"""
import sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
from quant import metrics as M
import trendcore_spot as T

OUT = {}
END = pd.Timestamp('2026-10-06')

# ------------------------------------------------------------------ regime
spy = pd.read_parquet('data/macro_SPY.parquet')['c']
s_td = (spy < spy.ewm(span=200, adjust=False).mean()).astype(float)             # known after that day's US close
cal = pd.date_range(spy.index[0], END, freq='D')
S = s_td.reindex(cal).ffill().shift(1)                                           # regime applied to UTC day d = known after close of d-1
S.name = 's'
print(f'SPY < EMA200 on {s_td.mean()*100:.1f}% of trading days since 1993; since 2018-03: {S.loc["2018-03-01":].mean()*100:.1f}% of calendar days')

# ------------------------------------------------------------------ assets
gld = pd.read_parquet('data/macro_GLD.parquet')['c']
gld_ret = gld.pct_change().dropna()
P = pd.read_parquet('data/panel_price.parquet')
px = pd.concat({a: pd.read_parquet(f'data/intraday/{a.upper()}USDT_spot_1h.parquet')['c'] for a in ('btc', 'eth')}, axis=1, sort=True)
bin_d = px.resample('1D').last()                                                 # close at end of UTC day, stamped d (same convention as CoinMetrics)
def crypto_daily(a, start):
    cm = P[a].dropna().loc[start:]                                               # CoinMetrics until 2026-05-23
    ext = bin_d[a].dropna().loc[cm.index[-1] + pd.Timedelta(days=1):]
    # splice with a ratio so the join has no jump
    k = bin_d[a].dropna().loc[cm.index[-1]] / cm.iloc[-1] if cm.index[-1] in bin_d.index else 1.0
    return pd.concat([cm, ext / k]).pct_change().dropna()
btc_d = crypto_daily('btc', '2014-01-01')
eth_d = crypto_daily('eth', '2016-01-01')
ov = bin_d['btc'].pct_change().loc['2020-01-01':'2026-05-23'].corr(P['btc'].pct_change().loc['2020-01-01':'2026-05-23'])
print(f'splice check: daily-return correlation Binance vs CoinMetrics BTC 2020-2026.05 = {ov:.4f}')

# ------------------------------------------------------------------ Part A
BLOCKS_GOLD = [('2004-11', '2010-12'), ('2011-01', '2016-12'), ('2017-01', '2021-12'), ('2022-01', '2026-10')]
BLOCKS_CRYPTO = [('2014-01', '2016-12'), ('2017-01', '2019-12'), ('2020-01', '2022-12'), ('2023-01', '2026-10')]

def boot_diff(r, s, ann, n=4000, block=20, seed=1):
    r, s = np.asarray(r, float), np.asarray(s, float)
    N = len(r); nb = int(np.ceil(N / block)); rng = np.random.default_rng(seed)
    d_mean, d_sh = [], []
    for _ in range(n // 500):
        st = rng.integers(0, N, (500, nb))
        idx = ((st[:, :, None] + np.arange(block)[None, None, :]) % N).reshape(500, -1)[:, :N]
        rr, ss = r[idx], s[idx]
        n1, n0 = ss.sum(1), (1 - ss).sum(1)
        m1 = (rr * ss).sum(1) / np.maximum(n1, 1); m0 = (rr * (1 - ss)).sum(1) / np.maximum(n0, 1)
        v1 = ((rr - m1[:, None]) ** 2 * ss).sum(1) / np.maximum(n1 - 1, 1); v0 = ((rr - m0[:, None]) ** 2 * (1 - ss)).sum(1) / np.maximum(n0 - 1, 1)
        ok = (n1 > 20) & (n0 > 20)
        d_mean.append(((m1 - m0) * ann)[ok]); d_sh.append((m1 / np.sqrt(v1) * np.sqrt(ann) - m0 / np.sqrt(v0) * np.sqrt(ann))[ok])
    return np.concatenate(d_mean), np.concatenate(d_sh)

def regime_split(name, ret, ann, blocks, trading_calendar=False):
    s = S.reindex(ret.index).ffill() if not trading_calendar else S.reindex(ret.index)
    df = pd.DataFrame({'r': ret, 's': s}).dropna()
    a, b = df[df.s == 1].r, df[df.s == 0].r
    row = dict(asset=name, n1=len(a), n0=len(b), mean1=a.mean() * ann, mean0=b.mean() * ann,
               sh1=a.mean() / a.std() * np.sqrt(ann), sh0=b.mean() / b.std() * np.sqrt(ann), cum1=(1 + a).prod() - 1, cum0=(1 + b).prod() - 1)
    dm, dsh = boot_diff(df.r.values, df.s.values, ann)
    row.update(diff=row['mean1'] - row['mean0'], ci_lo=np.percentile(dm, 5), ci_hi=np.percentile(dm, 95), p_same_sign=float((np.sign(dm) == np.sign(row['mean1'] - row['mean0'])).mean()),
               sh_diff=row['sh1'] - row['sh0'], sh_ci_lo=np.percentile(dsh, 5), sh_ci_hi=np.percentile(dsh, 95))
    subs = []
    for lo, hi in blocks:
        d = df.loc[lo:hi]
        a_, b_ = d[d.s == 1].r, d[d.s == 0].r
        if len(a_) >= 60 and len(b_) >= 60:
            subs.append(dict(block=f'{lo[:4]}-{hi[:4]}', n1=len(a_), d=(a_.mean() - b_.mean()) * ann, m1=a_.mean() * ann, m0=b_.mean() * ann))
        else:
            subs.append(dict(block=f'{lo[:4]}-{hi[:4]}', n1=len(a_), d=np.nan, m1=np.nan, m0=np.nan))
    row['subs'] = subs
    return row

print('\n================ PART A: does the mechanism exist? ================')
A = [regime_split('GLD (trading days, 252/yr)', gld_ret.loc['2004-11-19':], 252, BLOCKS_GOLD, True),
     regime_split('BTC (calendar days, 365/yr)', btc_d.loc['2014-01-01':], 365, BLOCKS_CRYPTO),
     regime_split('ETH (calendar days, 365/yr)', eth_d.loc['2016-01-01':], 365, BLOCKS_CRYPTO)]
for r in A:
    print(f"\n{r['asset']}: days s=1: {r['n1']}, s=0: {r['n0']}")
    print(f"  ann. mean return  s=1 {r['mean1']*100:7.1f}%   s=0 {r['mean0']*100:7.1f}%   diff {r['diff']*100:+7.1f} pts  90% CI [{r['ci_lo']*100:+.1f}, {r['ci_hi']*100:+.1f}]  (share of bootstrap draws with same sign {r['p_same_sign']:.2f})")
    print(f"  Sharpe            s=1 {r['sh1']:7.2f}    s=0 {r['sh0']:7.2f}    diff {r['sh_diff']:+.2f}  90% CI [{r['sh_ci_lo']:+.2f}, {r['sh_ci_hi']:+.2f}]   cumulative return in s=1 days {r['cum1']*100:+.0f}%  s=0 days {r['cum0']*100:+.0f}%")
    for sb in r['subs']:
        if np.isnan(sb['d']): print(f"    sub-period {sb['block']}: too few days in one state (s=1 days: {sb['n1']}) - not counted")
        else: print(f"    sub-period {sb['block']}: s=1 {sb['m1']*100:+7.1f}%  s=0 {sb['m0']*100:+7.1f}%  diff {sb['d']*100:+7.1f} pts  (s=1 days: {sb['n1']})")

def sign_stable(r, want):                       # want = +1 (gold better) / -1 (crypto worse)
    d = [sb['d'] for sb in r['subs'] if not np.isnan(sb['d'])]
    return (np.sign(r['diff']) == want) and len(d) >= 2 and all(np.sign(x) == want for x in d)
gold_ok = sign_stable(A[0], +1)
btc_ok = sign_stable(A[1], -1)
eth_ok = sign_stable(A[2], -1)
mech = gold_ok and btc_ok and eth_ok
print(f"\nPre-registered mechanism test: gold better in s=1 with stable sign: {gold_ok} | BTC worse: {btc_ok} | ETH worse: {eth_ok}  ->  MECHANISM SUPPORTED: {mech}")
OUT['partA'] = [{k: (v if k != 'subs' else [{kk: (None if isinstance(vv, float) and np.isnan(vv) else vv) for kk, vv in sb.items()} for sb in v]) for k, v in r.items()} for r in A]
OUT['mechanism_supported'] = bool(mech)

# ------------------------------------------------------------------ Part B
print('\n================ PART B: strategy variants (BTC+ETH 6h spot strategy) ================')
oi = pd.read_parquet('data/intraday/BTCUSDT_oi_daily.parquet')['sum_open_interest_value']
W, R = T.target_weights(px, oi)
r0b, g0b = T._backtest(W, R)
def to_daily(rb):
    d = (1 + rb).groupby(rb.index.date).prod() - 1
    d.index = pd.to_datetime(d.index); return d
START = '2018-03-01'
base = to_daily(r0b).loc[START:]
chk, _ = T.backtest(px, oi, start=START)
assert np.allclose(chk.values, base.values), 'baseline mismatch with trendcore_spot.backtest'
e_d = g0b.groupby(g0b.index.date).mean(); e_d.index = pd.to_datetime(e_d.index); e_d = e_d.loc[START:]

s_day = S.reindex(base.index).fillna(0.0)
s_bar = pd.Series(S.reindex(W.index.normalize()).values, index=W.index).fillna(0.0)
s_next = s_bar.shift(-1).ffill()                                  # regime in force over the bar in which the weights are held
W1 = W.mul(1.0 - s_next, axis=0)
r1b, g1b = T._backtest(W1, R)
S1 = to_daily(r1b).loc[START:]

def gold_calendar(ret_td):                                                       # daily gold return per UTC day; weekend / holiday = 0
    x = ret_td.reindex(pd.date_range(base.index[0], base.index[-1], freq='D')).fillna(0.0)
    return x
def gold_sleeve(wt, gr, cost_bps, band=0.05):
    wl, rv = wt.values.astype(float), gr.reindex(wt.index).fillna(0.0).values
    out = np.zeros(len(wl)); w_prev = 0.0
    for t in range(len(wl)):
        tgt = wl[t]
        if tgt != 0.0 and abs(tgt - w_prev) <= band: tgt = w_prev
        out[t] = tgt * rv[t] - abs(tgt - w_prev) * cost_bps / 1e4
        w_prev = tgt * (1 + rv[t])
    return pd.Series(out, index=wt.index)

GLDc = gold_calendar(gld_ret)
idle = (1.0 - e_d.reindex(base.index)).clip(lower=0.0)
S2 = base + gold_sleeve(s_day * idle, GLDc, 5.0)
S3 = S1 + gold_sleeve(s_day * 1.0, GLDc, 5.0)

E1 = ('2018-03-01', '2022-12-31'); E2 = ('2023-01-01', '2026-10-05'); FULL = (START, '2026-10-05')
V = {'S0 baseline': base, 'S1 crypto x0 when s=1 (cash)': S1, 'S2 idle -> GLD when s=1': S2, 'S3 crypto x0, 100% GLD when s=1': S3}
def block(v, lo, hi):
    s_ = M.summary(v.loc[lo:hi]); return s_
rows = {}
for k, v in V.items():
    rows[k] = {lab: block(v, *w) for lab, w in {'era1': E1, 'era2': E2, 'full': FULL}.items()}
def fmt(k):
    o = []
    for lab in ('era1', 'era2', 'full'):
        s_ = rows[k][lab]; o.append(f"{lab}: CAGR {s_['cagr']*100:5.1f}% Sh {s_['sharpe']:4.2f} DD {s_['maxdd']*100:6.1f}%")
    return ' | '.join(o)
for k in V: print(f'{k:34s}{fmt(k)}')
gb = M.summary(gld_ret.reindex(base.index).fillna(0.0).loc[FULL[0]:FULL[1]])
print(f"GLD buy&hold same window (weekends 0): CAGR {gb['cagr']*100:.1f}% Sharpe {gb['sharpe']:.2f} DD {gb['maxdd']*100:.1f}%")

# ---- was the strategy already flat when SPY < EMA200?
e_full = e_d.reindex(base.index)
print('\n--- what the baseline was doing while SPY < EMA200')
for lab, mask in (('s=1', s_day == 1), ('s=0', s_day == 0)):
    rr = base[mask]
    print(f"  {lab}: {int(mask.sum())} days ({mask.mean()*100:.0f}%), mean exposure {e_full[mask].mean()*100:.0f}%, share of days invested (>5%) {(e_full[mask] > 0.05).mean()*100:.0f}%, "
          f"compounded return {((1+rr).prod()-1)*100:+.0f}%, ann. mean {rr.mean()*365*100:+.1f}%")

# ---- episode table (maximal runs of s=1 since 2018-03)
ep = []
run_start = None
sv = s_day
for d, val in sv.items():
    if val == 1 and run_start is None: run_start = d
    if (val == 0 or d == sv.index[-1]) and run_start is not None:
        end = d - pd.Timedelta(days=1) if val == 0 else d
        ep.append((run_start, end)); run_start = None
btc_full = bin_d['btc'].pct_change().reindex(base.index).fillna(0.0)
eth_full = bin_d['eth'].pct_change().reindex(base.index).fillna(0.0)
tab = []
for a, b in ep:
    sl = slice(a, b)
    tab.append(dict(start=a.date(), end=b.date(), days=(b - a).days + 1, BTC=(1 + btc_full.loc[sl]).prod() - 1, ETH=(1 + eth_full.loc[sl]).prod() - 1,
                    baseline=(1 + base.loc[sl]).prod() - 1, exposure=e_full.loc[sl].mean(), S1=(1 + S1.loc[sl]).prod() - 1,
                    GLD=(1 + GLDc.loc[sl]).prod() - 1, S2=(1 + S2.loc[sl]).prod() - 1, S3=(1 + S3.loc[sl]).prod() - 1))
tab = pd.DataFrame(tab)
pd.set_option('display.width', 250)
print(f'\n--- SPY < EMA200 episodes since 2018-03 ({len(tab)} runs; compounded return over the run)')
t2 = tab.copy()
for c in ('BTC', 'ETH', 'baseline', 'S1', 'GLD', 'S2', 'S3'): t2[c] = (t2[c] * 100).round(1)
t2['exposure'] = (t2['exposure'] * 100).round(0)
print(t2.to_string(index=False))
big = tab[tab.days >= 20]
print(f"\nepisodes >= 20 days: {len(big)}; whipsaw runs (< 20 days): {len(tab)-len(big)}, total flip count {2*len(tab)}  (~{2*len(tab)/ (len(base)/365):.1f} per year)")
print('S1 - baseline compounded return per episode (pts):', ((tab['S1'] - tab['baseline']) * 100).round(1).tolist())
print('S3 - baseline compounded return per episode (pts):', ((tab['S3'] - tab['baseline']) * 100).round(1).tolist())

# ---- acceptance
def accept(name, v, partA):
    b = rows['S0 baseline']; x = rows[name]
    c = dict(partA=bool(partA),
             full_sharpe_up=x['full']['sharpe'] > b['full']['sharpe'],
             lower_maxdd=x['full']['maxdd'] > b['full']['maxdd'],
             era2_ok=x['era2']['sharpe'] >= b['era2']['sharpe'] - 0.02,
             era1_positive=x['era1']['sharpe'] > b['era1']['sharpe'])
    c['ACCEPT'] = all(c.values()); return c
print('\n--- pre-registered acceptance')
ACC = {}
for k in list(V)[1:]:
    ACC[k] = accept(k, V[k], mech)
    b = rows['S0 baseline']; x = rows[k]
    print(f"{k:34s}Sharpe full {b['full']['sharpe']:.2f}->{x['full']['sharpe']:.2f} | maxDD {b['full']['maxdd']*100:.1f}->{x['full']['maxdd']*100:.1f} | era1 Sharpe {b['era1']['sharpe']:.2f}->{x['era1']['sharpe']:.2f} | era2 {b['era2']['sharpe']:.2f}->{x['era2']['sharpe']:.2f}  => {ACC[k]}")
OUT['variants'] = {k: {lab: {kk: float(vv) for kk, vv in rows[k][lab].items() if kk in ('cagr', 'sharpe', 'maxdd', 'vol', 'sortino', 'calmar')} for lab in rows[k]} for k in V}
OUT['acceptance'] = {k: {kk: bool(vv) for kk, vv in v.items()} for k, v in ACC.items()}
winners = [k for k, v in ACC.items() if v['ACCEPT']]
OUT['winners'] = winners
print('\nwinners:', winners if winners else 'NONE (no PAXG repeat is run, per the pre-registration)')

# ---- robustness lines (stress tests, not trials) -- always printed so the verdict can be judged
print('\n--- stress / robustness (not additional trials)')
S3_lag = S1 * 0 + 0
def lagged_gold(extra):
    w = (s_day.shift(extra).fillna(0.0))
    return gold_sleeve(w, GLDc, 5.0)
for lag in (1, 3):
    w = s_day.shift(lag).fillna(0.0)
    s3 = S1 + gold_sleeve(w, GLDc, 5.0)
    sh = {lab: M.summary(s3.loc[a:b])['sharpe'] for lab, (a, b) in {'era1': E1, 'era2': E2, 'full': FULL}.items()}
    print(f'  S3 with gold entered {lag} day(s) later than the signal: Sharpe era1 {sh["era1"]:.2f} era2 {sh["era2"]:.2f} full {sh["full"]:.2f}')
sh_cost = {}
for cb in (5.0, 25.0):
    s3 = S1 + gold_sleeve(s_day * 1.0, GLDc, cb)
    print(f'  S3 with gold cost {cb:.0f} bp one-way: Sharpe full {M.summary(s3.loc[FULL[0]:FULL[1]])["sharpe"]:.2f}')
yr = pd.DataFrame({k: v.groupby(v.index.year).apply(lambda x: (1 + x).prod() - 1) * 100 for k, v in V.items()})
yr['GLD'] = GLDc.groupby(GLDc.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
yr['BTC'] = btc_full.groupby(btc_full.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
print('\ncalendar-year returns %:'); print(yr.round(1).to_string())
OUT['years'] = yr.round(2).to_dict()
yr.round(2).to_csv('results/spy_gold_years.csv')
tab.to_csv('results/spy_gold_episodes.csv', index=False)
pd.DataFrame(V).to_csv('results/spy_gold_daily.csv')

# ---- PAXG repeat of the winner (only if there is one)
if winners:
    print('\n--- PAXG repeat of the winner (24/7 venue; 25 bp; data from 2020-09)')
    pg = pd.read_parquet('data/intraday/PAXGUSDT_spot_1d.parquet')['c']
    pgr = pg.pct_change().reindex(base.index)
    pgr = pgr.fillna(0.0)
    for k in winners:
        if k.startswith('S2'): v = base + gold_sleeve((s_day * idle).where(pg.reindex(base.index).notna(), 0.0), pgr, 25.0)
        else: v = S1 + gold_sleeve((s_day * 1.0).where(pg.reindex(base.index).notna(), 0.0), pgr, 25.0)
        b_ = M.summary(base.loc['2020-09-01':]); x_ = M.summary(v.loc['2020-09-01':])
        print(f"  {k} with PAXG: 2020-09+ CAGR {x_['cagr']*100:.1f}% Sharpe {x_['sharpe']:.2f} DD {x_['maxdd']*100:.1f}%  vs baseline {b_['cagr']*100:.1f}% / {b_['sharpe']:.2f} / {b_['maxdd']*100:.1f}%")
json.dump(OUT, open('results/spy_gold.json', 'w'), indent=1, default=float)
print('\nsaved results/spy_gold*.{json,csv}')
