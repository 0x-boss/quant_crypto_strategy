"""Chart for round E: equity of baseline vs SPY<EMA200 variants, with the SPY-bear spells shaded, and the strategy's own exposure."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import trendcore_spot as T

d = pd.read_csv('results/spy_gold_daily.csv', index_col=0, parse_dates=True)
spy = pd.read_parquet('data/macro_SPY.parquet')['c']
s = (spy < spy.ewm(span=200, adjust=False).mean()).astype(float)
S = s.reindex(pd.date_range(spy.index[0], '2026-10-06', freq='D')).ffill().shift(1).reindex(d.index).fillna(0.0)
px = pd.concat({a: pd.read_parquet(f'data/intraday/{a.upper()}USDT_spot_1h.parquet')['c'] for a in ('btc', 'eth')}, axis=1, sort=True)
oi = pd.read_parquet('data/intraday/BTCUSDT_oi_daily.parquet')['sum_open_interest_value']
W, R = T.target_weights(px, oi); _, g = T._backtest(W, R)
e = g.groupby(g.index.date).mean(); e.index = pd.to_datetime(e.index); e = e.reindex(d.index)

def shade(ax):
    x = S.values; i = 0
    while i < len(x):
        if x[i] == 1:
            j = i
            while j + 1 < len(x) and x[j + 1] == 1: j += 1
            ax.axvspan(S.index[i], S.index[j] + pd.Timedelta(days=1), color='tab:red', alpha=0.13, lw=0)
            i = j + 1
        else: i += 1
fig, ax = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.3]))
lab = {'S0 baseline': ('Baseline (BTC+ETH, spot-only)', 'k', 2.2), 'S1 crypto x0 when s=1 (cash)': ('S1: crypto -> cash when SPY < EMA200', 'tab:blue', 1.2),
       'S2 idle -> GLD when s=1': ('S2: idle capital -> gold', 'tab:orange', 1.2), 'S3 crypto x0, 100% GLD when s=1': ('S3: crypto -> 100 % gold', 'goldenrod', 1.6)}
for c, (n, col, lw) in lab.items():
    ax[0].plot(d.index, (1 + d[c]).cumprod(), label=n, color=col, lw=lw)
ax[0].set_yscale('log'); shade(ax[0]); ax[0].set_title('Round E: hold gold when SPY < its 200-day EMA?  (red = SPY below EMA200)'); ax[0].legend(loc='upper left'); ax[0].grid(alpha=.3); ax[0].set_ylabel('equity (log)')
ax[1].fill_between(e.index, e.values * 100, color='tab:green', alpha=.6, lw=0); shade(ax[1]); ax[1].set_ylabel('baseline exposure, %'); ax[1].set_ylim(0, 105); ax[1].grid(alpha=.3)
ax[1].set_title('The baseline is already mostly out of the market during SPY-bear spells (average exposure 9 % vs 32 % otherwise)', fontsize=10)
plt.tight_layout(); plt.savefig('results/spy_gold_equity.png', dpi=130)
print('saved')
