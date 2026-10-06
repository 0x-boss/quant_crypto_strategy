"""Chart: spot-only CAGR / drawdown frontier of the SAME signals at different risk budgets (round F)."""
import numpy as np, pandas as pd, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
g = pd.read_csv('results/dial_grid.csv')
core = json.load(open('results/participation.json'))['core']
fig, ax = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
a = ax[0]
for tv, col, ls in ((0.30, 'tab:green', '-'), (0.40, 'tab:blue', '-'), (0.60, 'tab:red', '--')):
    d = g[g.target_vol == tv].sort_values('asset_vol')
    a.plot(-d.maxdd * 100, d.cagr * 100, ls, color=col, marker='o', ms=5, lw=1.4, label=f'portfolio vol brake {tv:.2f}' + (' (official)' if tv == 0.40 else ''))
    if tv == 0.40:
        for _, r in d.iterrows():
            a.annotate(f'{r.asset_vol:.2f}', (-r.maxdd * 100, r.cagr * 100), textcoords='offset points', xytext=(6, -11), fontsize=8, color=col)
o = g[(g.asset_vol == 0.45) & (g.target_vol == 0.40)].iloc[0]; v2 = g[(g.asset_vol == 0.60) & (g.target_vol == 0.40)].iloc[0]
a.scatter([-o.maxdd * 100], [o.cagr * 100], s=190, facecolors='none', edgecolors='k', lw=2, zorder=5, label='official (ASSET_VOL 0.45)')
a.scatter([-v2.maxdd * 100], [v2.cagr * 100], s=190, facecolors='none', edgecolors='tab:orange', lw=2.4, zorder=5, label='ASSET_VOL 0.60 (balanced)')
cx = [-core[k]['maxdd'] * 100 for k in ('0', '10', '20', '30')]; cy = [core[k]['cagr'] * 100 for k in ('0', '10', '20', '30')]
a.plot(cx, cy, ':', color='grey', marker='s', ms=5, label='+ constant BTC core 0/10/20/30 %')
a.set_xlabel('max drawdown, % (2018-03 -> 2026-10)'); a.set_ylabel('CAGR, %'); a.grid(alpha=.3); a.legend(fontsize=8, loc='lower right')
a.set_title('CAGR vs max drawdown (labels: ASSET_VOL on the blue line)', fontsize=11)
b = ax[1]
for tv, col in ((0.30, 'tab:green'), (0.40, 'tab:blue'), (0.60, 'tab:red')):
    d = g[g.target_vol == tv].sort_values('asset_vol'); b.plot(d.asset_vol, d.sharpe, marker='o', color=col, label=f'brake {tv:.2f}')
b.axhline(0.71, color='k', ls=':', lw=1); b.text(0.31, 0.73, 'BTC buy & hold Sharpe 0.71', fontsize=8)
b.set_ylim(0.6, 1.6); b.set_xlabel('per-asset risk budget ASSET_VOL'); b.set_ylabel('Sharpe (2018-03+)'); b.grid(alpha=.3); b.legend(fontsize=8)
b.set_title('Sharpe is ~flat: the dial buys CAGR with drawdown', fontsize=11)
fig.suptitle('Spot-only TrendCore: same signals, different risk budgets (2018-03 -> 2026-10)', fontsize=12)
plt.tight_layout(); plt.savefig('results/frontier_spot.png', dpi=130); print('saved')
