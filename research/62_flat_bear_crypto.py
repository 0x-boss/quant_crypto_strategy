"""Round H (pre-registered in research/PREREGISTRATION.md BEFORE this script was run): making money with crypto itself in flat / bearish regimes.
Track A (spot-only): H1 alt rotation while BTC is OFF.  Track B (NOT spot-only; fully collateralised perp short, gross<=1): H2 trend-down short sleeve."""
import sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
from quant import metrics as M
from quant import xs_perp as X
import trendcore_spot as T

K, ANN = T.K, T.ANN
OUT = {}
px = pd.concat({a: pd.read_parquet(f'data/intraday/{a.upper()}USDT_spot_1h.parquet')['c'] for a in ('btc', 'eth')}, axis=1, sort=True)
oi = pd.read_parquet('data/intraday/BTCUSDT_oi_daily.parquet')['sum_open_interest_value']
def to_daily(rb):
    d = (1 + rb).groupby(rb.index.date).prod() - 1; d.index = pd.to_datetime(d.index); return d
Wl, Rl = T.target_weights(px, oi); r_main_bar, g_main_bar = T._backtest(Wl, Rl)
main_all = to_daily(r_main_bar)

# =========================================================================== TRACK A: H1 alt rotation while BTC is OFF
print('================ TRACK A (spot-only): alt rotation while BTC is OFF   [net of 25 bp per side; sleeve weight 25 % of capital]')
C, QV, F = X.load_panels()
U = X.universe(C, QV, 20, 90); U.loc[:, [c for c in ('BTCUSDT', 'ETHUSDT') if c in U.columns]] = False
btc = C['BTCUSDT']; off = (btc < btc.rolling(150).mean()).values
A1 = ('2020-09-01', '2022-12-31'); A2 = ('2023-01-01', '2026-10-05')
def run_h1(score_name, picks=5, w_pick=0.05, start='2020-09-01', cost=25e-4):
    if score_name == 'EW basket': S = pd.DataFrame(1.0, index=C.index, columns=C.columns).where(U)
    else: S = X.scores(C, F, score_name).where(U)
    R = C.pct_change(fill_method=None).fillna(0.0).values; idx = C.index; Tn, N = R.shape
    w = np.zeros(N); ret = np.zeros(Tn); gross = np.zeros(Tn)
    for t in range(1, Tn):
        pr = float(w @ R[t]); ret[t] = pr
        if w.sum() > 0: w = w * (1 + R[t]) / (1 + pr)
        if idx[t] >= pd.Timestamp(start):
            tgt = w.copy()
            if not off[t]: tgt = np.zeros(N)
            elif idx[t].weekday() == 6:
                s = S.iloc[t].dropna(); tgt = np.zeros(N)
                if len(s) >= 10:
                    if score_name == 'EW basket': chosen = list(s.index); wp = 0.25 / len(chosen)
                    else: chosen = list(s.nlargest(picks).index); wp = w_pick
                    for c in chosen: tgt[C.columns.get_loc(c)] = wp
            ret[t] -= np.abs(tgt - w).sum() * cost; w = tgt
        gross[t] = w.sum()
    return pd.Series(ret, index=idx).loc[start:], pd.Series(gross, index=idx).loc[start:]
btc_d = btc.pct_change().loc['2020-09-01':]
off_l = pd.Series(off, index=C.index).shift(1).fillna(False).astype(bool).loc['2020-09-01':]       # regime known at the prior close
print(f"BTC buy & hold while OFF (2020-09+): {off_l.mean()*100:.0f}% of days, cumulative {((1+btc_d[off_l]).prod()-1)*100:+.0f}%, daily-mean annualised {btc_d[off_l].mean()*365*100:+.0f}%")
main = main_all.loc['2020-09-01':'2026-10-05']
mb = {k: M.summary(main.loc[a:b]) for k, (a, b) in (('E1', A1), ('E2', A2), ('full', ('2020-09-01', '2026-10-05')))}
print(f"main strategy {mb['full']['cagr']*100:.1f}% / {mb['full']['sharpe']:.2f} / {mb['full']['maxdd']*100:.1f}%  | E1 Sharpe {mb['E1']['sharpe']:.2f} | E2 Sharpe {mb['E2']['sharpe']:.2f}")
e0 = g_main_bar.groupby(g_main_bar.index.date).first(); e0.index = pd.to_datetime(e0.index)
ACC = {}; OUT['H1'] = {}
for name in ('EW basket (descriptive)', 'MOM30', 'REV7', 'LOWVOL'):
    key = 'EW basket' if name.startswith('EW') else name
    r, g = run_h1(key)
    act = r[g.shift(1).fillna(0) > 0] / 0.25                     # active-day return scaled to 100 % sleeve
    row = {}
    for lab, (a, b) in (('E1', A1), ('E2', A2)):
        x = act.loc[a:b]; row[lab] = dict(days=len(x), mean_ann=float(x.mean() * 365), t=float(x.mean() / x.std() * np.sqrt(len(x))) if len(x) > 5 and x.std() > 0 else np.nan, cum=float((1 + r.loc[a:b]).prod() - 1))
    pooled_t = float(act.mean() / act.std() * np.sqrt(len(act))) if len(act) > 5 else np.nan
    comb = main + r.reindex(main.index).fillna(0.0)
    cs = {k: M.summary(comb.loc[a:b]) for k, (a, b) in (('E1', A1), ('E2', A2), ('full', ('2020-09-01', '2026-10-05')))}
    print(f"{name:24s} active days {len(act):4d} | active-day mean (ann., at 100%) E1 {row['E1']['mean_ann']*100:+7.0f}% (t {row['E1']['t']:5.2f}) E2 {row['E2']['mean_ann']*100:+7.0f}% (t {row['E2']['t']:5.2f}) pooled t {pooled_t:5.2f} | sleeve cum (25% wt) E1 {row['E1']['cum']*100:+6.1f}% E2 {row['E2']['cum']*100:+6.1f}% | combined {cs['full']['cagr']*100:5.1f}% / {cs['full']['sharpe']:4.2f} / {cs['full']['maxdd']*100:6.1f}% (E1 {cs['E1']['sharpe']:.2f}, E2 {cs['E2']['sharpe']:.2f})")
    if not name.startswith('EW'):
        acc = dict(t_gt2=bool(pooled_t > 2), both_eras_pos=bool(row['E1']['mean_ann'] > 0 and row['E2']['mean_ann'] > 0),
                   sharpe_ok=bool(cs['E1']['sharpe'] >= mb['E1']['sharpe'] - 0.02 and cs['E2']['sharpe'] >= mb['E2']['sharpe'] - 0.02), dd_ok=bool(cs['full']['maxdd'] >= mb['full']['maxdd'] - 0.02))
        acc['ACCEPT'] = all(acc.values()); ACC['H1 ' + name] = acc
    OUT['H1'][name] = dict(row=row, pooled_t=pooled_t, combined=dict(cagr=cs['full']['cagr'], sharpe=cs['full']['sharpe'], maxdd=cs['full']['maxdd']))

# =========================================================================== TRACK B: H2 trend-down short sleeve (perp, collateralised)
print('\n================ TRACK B (NOT spot-only; collateralised perp short, long+short gross <= 1): H2 trend-down short sleeve on BTC+ETH')
perp = pd.concat({a: pd.read_parquet(f'data/intraday/{a.upper()}USDT_perp_1h.parquet')['c'] for a in ('btc', 'eth')}, axis=1, sort=True)
fund = pd.concat({a: pd.read_parquet(f'data/intraday/{a.upper()}USDT_funding.parquet')['fundingRate'] for a in ('btc', 'eth')}, axis=1, sort=True)
P6 = px.resample('6h').last().dropna(how='all'); Q = 1.0 / P6                                    # signals on inverse spot price
rv2 = (np.log(px).diff() ** 2).resample('6h').sum().reindex(P6.index)
vol = lambda span: (rv2.ewm(span=span * K, min_periods=10).mean() ** 0.5) * np.sqrt(ANN)
score_dn = T._trend_scores(Q); acc_w, cnt = 0, 0
for g in T.GATES:
    f = (score_dn * (Q > Q.rolling(g * K).mean())).fillna(0.0)
    for vs in T.VOL_SPANS:
        acc_w = acc_w + (f * T.ASSET_VOL / vol(vs)).div(2).fillna(0.0).clip(upper=1.0); cnt += 1
Ws0 = acc_w / cnt
Rp = perp.resample('6h').last().pct_change().reindex(P6.index)                                    # perp 6h returns (2020-01+)
Fb = fund.resample('6h').sum().reindex(P6.index).fillna(0.0)                                      # funding settled in the bar (8h rate sums)
def bt_short(W, R, Fd, cost_bps=10.0, band=0.05, price=True, funding=True, costs=True):
    Wv = W.shift(1).fillna(0.0).values; Rv = R.fillna(0.0).values; Fv = Fd.values
    ret = np.zeros(len(Wv)); gross = np.zeros(len(Wv)); prev = np.zeros(Wv.shape[1]); comp = np.zeros((len(Wv), 3))
    for t in range(len(Wv)):
        tgt = np.where((np.abs(Wv[t] - prev) <= band) & (Wv[t] != 0.0), prev, Wv[t])
        c = np.abs(tgt - prev).sum() * cost_bps / 1e4 if costs else 0.0
        p = -float(tgt @ Rv[t]) if price else 0.0; fu = float(tgt @ Fv[t]) if funding else 0.0     # short gains when price falls, receives positive funding
        ret[t] = p + fu - c; gross[t] = tgt.sum(); comp[t] = (p, fu, -c)
        prev = tgt * (1 + Rv[t]) / (1 + ret[t])
    return pd.Series(ret, index=R.index), pd.Series(gross, index=R.index), pd.DataFrame(comp, index=R.index, columns=['price', 'funding', 'cost'])
S0 = Ws0.loc['2019-12-01':]; R0 = Rp.loc['2019-12-01':]; F0 = Fb.loc['2019-12-01':]
r0, _, _ = bt_short(S0, R0, F0)
rv = r0.ewm(span=30 * K, min_periods=10).std() * np.sqrt(ANN)
Wsh = S0.mul((T.TARGET_VOL / rv).clip(upper=1.0).fillna(0.0), axis=0)
L = Wl.sum(axis=1).reindex(Wsh.index).fillna(0.0); Ssum = Wsh.sum(axis=1)
Wsh = Wsh.mul(((1.0 - L).clip(lower=0.0) / Ssum.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0), axis=0)
H = ('2020-01-01', '2022-12-31'), ('2023-01-01', '2026-10-05'), ('2020-01-01', '2026-10-05')
main2 = main_all.loc['2020-01-01':'2026-10-05']
mb2 = {k: M.summary(main2.loc[a:b]) for k, (a, b) in zip(('E1', 'E2', 'full'), H)}
print(f"main strategy (2020-01+): {mb2['full']['cagr']*100:.1f}% / {mb2['full']['sharpe']:.2f} / {mb2['full']['maxdd']*100:.1f}%  | E1 Sharpe {mb2['E1']['sharpe']:.2f} | E2 Sharpe {mb2['E2']['sharpe']:.2f}")
OUT['H2'] = {}
rs_bar, gs_bar, comp = bt_short(Wsh.loc['2020-01-01':], Rp.loc['2020-01-01':], Fb.loc['2020-01-01':])
sleeve = to_daily(rs_bar).loc['2020-01-01':'2026-10-05']
print(f"short sleeve standalone (1x): E1 {((1+sleeve.loc[H[0][0]:H[0][1]]).prod()-1)*100:+.1f}%  E2 {((1+sleeve.loc[H[1][0]:H[1][1]]).prod()-1)*100:+.1f}%  | avg short gross {gs_bar.mean():.3f}, days with a short > 5%: {(gs_bar.groupby(gs_bar.index.date).max() > 0.05).mean()*100:.0f}%")
cd = comp.resample('1D').sum()
print(f"   decomposition (sum of per-bar fractions): price {cd.price.sum()*100:+.1f}%  funding {cd.funding.sum()*100:+.1f}%  costs {cd.cost.sum()*100:+.1f}%  [E1 price {cd.price.loc[:'2022-12-31'].sum()*100:+.1f}% / E2 {cd.price.loc['2023-01-01':].sum()*100:+.1f}%]")
yrs = sleeve.groupby(sleeve.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
print('   sleeve by calendar year %:', {int(k): round(float(v), 1) for k, v in yrs.items()})
print(f"   correlation of sleeve with main strategy (daily): {sleeve.corr(main2):+.2f}")
for m in (0.5, 1.0, 2.0):
    Wm = Wsh * m; Ssum = Wm.sum(axis=1); Wm = Wm.mul(((1.0 - L).clip(lower=0.0) / Ssum.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0), axis=0)
    rm_bar, gm, _ = bt_short(Wm.loc['2020-01-01':], Rp.loc['2020-01-01':], Fb.loc['2020-01-01':])
    comb = main2 + to_daily(rm_bar).loc['2020-01-01':'2026-10-05'].reindex(main2.index).fillna(0.0)
    cs = {k: M.summary(comb.loc[a:b]) for k, (a, b) in zip(('E1', 'E2', 'full'), H)}
    sl = to_daily(rm_bar).loc['2020-01-01':'2026-10-05']
    pos_eras = all(((1 + sl.loc[a:b]).prod() - 1) > 0 for a, b in H[:2])
    print(f"   combined with main, sleeve x{m}: {cs['full']['cagr']*100:5.1f}% / {cs['full']['sharpe']:4.2f} / {cs['full']['maxdd']*100:6.1f}%  (E1 Sharpe {cs['E1']['sharpe']:.2f}, E2 {cs['E2']['sharpe']:.2f})  sleeve positive in both eras: {pos_eras}")
    acc = dict(sleeve_pos_both=bool(pos_eras), cagr_up=bool(cs['full']['cagr'] > mb2['full']['cagr']), sharpe_ok=bool(cs['E1']['sharpe'] >= mb2['E1']['sharpe'] - 0.02 and cs['E2']['sharpe'] >= mb2['E2']['sharpe'] - 0.02),
               dd_ok=bool(cs['full']['maxdd'] >= mb2['full']['maxdd'] - 0.02)); acc['ACCEPT'] = all(acc.values())
    OUT['H2'][f'x{m}'] = dict(cagr=cs['full']['cagr'], sharpe=cs['full']['sharpe'], maxdd=cs['full']['maxdd'], e1=cs['E1']['sharpe'], e2=cs['E2']['sharpe'], acc=acc)
    if m == 1.0: ACC['H2 short sleeve (1x)'] = acc
# by BTC regime: what the short sleeve earned in OFF / ON (lagged)
off_d = pd.Series(off, index=C.index).shift(1).fillna(False).astype(bool).reindex(sleeve.index).fillna(False)
print(f"   short sleeve P&L while BTC OFF: {((1+sleeve[off_d]).prod()-1)*100:+.1f}%   while ON: {((1+sleeve[~off_d]).prod()-1)*100:+.1f}%")
print('\n--- pre-registered acceptance')
for k, v in ACC.items(): print(f'{k:28s}', v)
OUT['acceptance'] = ACC
json.dump(OUT, open('results/flat_bear_crypto.json', 'w'), indent=1, default=float)
print('saved results/flat_bear_crypto.json')
