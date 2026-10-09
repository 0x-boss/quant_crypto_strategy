"""Round L (pre-registered): burst continuation (z72 >= 2, hold 72 h) on 10 other coins, same-year random-entry controls."""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
rng = np.random.default_rng(68)
COINS = ['BTC', 'ETH', 'ADA', 'AVAX', 'BNB', 'DOGE', 'DOT', 'LINK', 'LTC', 'SOL', 'TRX', 'XRP']
rows = []; pooled_ex = []; pooled_n = []
for c in COINS:
    P = pd.read_parquet(f'data/intraday/{c}USDT_spot_1h.parquet')['c']; cost = (10 if c in ('BTC', 'ETH') else 25) * 1e-4
    lp = np.log(P.values); N = len(P); sig = np.log(P).diff().ewm(span=720, min_periods=240).std().shift(1).values
    z = np.full(N, np.nan); z[72:] = (lp[72:] - lp[:-72]) / (sig[72:] * np.sqrt(72))
    t0 = int(np.searchsorted(P.index.values, np.datetime64('2020-09-01'))); tr = []; t = max(72, t0)
    while t < N - 73:
        if np.isfinite(z[t]) and z[t] >= 2: tr.append(t); t += 72
        else: t += 1
    if len(tr) < 5: rows.append((c, len(tr), np.nan, np.nan, np.nan, np.nan)); continue
    ev = np.array(tr); net = np.exp(lp[ev + 72] - lp[ev]) - 1 - 2 * cost
    yrs = P.index.year.values; valid = np.where(np.isfinite(z) & (np.arange(N) >= t0) & (np.arange(N) < N - 73))[0]; pools = {y: valid[yrs[valid] == y] for y in np.unique(yrs[valid])}
    ctrl = np.array([np.mean(np.exp(lp[(s := np.array([rng.choice(pools[yrs[i]]) for i in ev])) + 72] - lp[s]) - 1 - 2 * cost) for _ in range(500)])
    rows.append((c, len(ev), net.mean(), net.mean() / (net.std(ddof=1) / np.sqrt(len(ev))), net.mean() - ctrl.mean(), (ctrl >= net.mean()).mean()))
    if c not in ('BTC', 'ETH'): pooled_ex.append((net.mean() - ctrl.mean(), len(ev)))
print(f"{'coin':5s} {'events':>6s} {'net/event':>10s} {'t':>6s} {'excess vs same-year random':>28s} {'p':>6s}")
for c, n, m, t, ex, p in rows: print(f"{c:5s} {n:6d} {m*100:9.2f}% {t:6.2f} {ex*100:+27.2f} pts {p:6.3f}" if np.isfinite(m) else f"{c:5s} {n:6d}  too few")
alts = [r for r in rows if r[0] not in ('BTC', 'ETH') and np.isfinite(r[2])]
pos = sum(1 for r in alts if r[4] > 0); w = np.array([n for _, n in pooled_ex]); ex = np.array([e for e, _ in pooled_ex])
print(f"\n10 other coins: positive excess in {pos} of {len(alts)}; event-weighted mean excess {np.average(ex, weights=w)*100:+.2f} pts; simple mean {ex.mean()*100:+.2f} pts; total events {int(w.sum())}")
print('verdict (pre-set reading):', 'SUPPORTS' if (np.average(ex, weights=w) > 0 and pos >= 7) else ('CONTRADICTS' if (pos < 5 or np.average(ex, weights=w) <= 0) else 'INCONCLUSIVE'))
