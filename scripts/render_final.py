"""FINAL spot-only version, all history, no idle yield: equity curve, rolling Sharpe curve, monthly heatmap + annual returns."""
import os, sys, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quant import metrics as M
from quant.intraday import spot_only, run, hourly_prices
START = "2018-03-01"
px = hourly_prices(); o = run(spot_only(start=START, cash_rate=0.0), px); r = o["total"]
btc = px["btc"].resample("1D").last().pct_change().reindex(r.index).fillna(0)
r.to_frame("ret").assign(gross=o["gross"]).to_csv("results/final_spot_daily.csv")
SURF, INK, INK2, GRID, BLUE, ORANGE, RED, MID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6", "#eb6834", "#e34948", "#f0efec"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .7})
s, sb = M.summary(r), M.summary(btc)
print(f"FINAL {r.index[0].date()}..{r.index[-1].date()}: CAGR {s['cagr']*100:.1f}% Sharpe {s['sharpe']:.2f} Sortino {s['sortino']:.2f} MaxDD {s['maxdd']*100:.1f}% Vol {s['vol']*100:.1f}% Calmar {s['calmar']:.2f}")
print(f"BTC B&H  same period: CAGR {sb['cagr']*100:.1f}% Sharpe {sb['sharpe']:.2f} MaxDD {sb['maxdd']*100:.1f}%  | avg exposure {o['gross'].mean():.2f}")
for lab, a in [("2018-03+", "2018-03-01"), ("2020-03+", "2020-03-01"), ("2022+", "2022-01-01"), ("2023+", "2023-01-01")]:
    x = M.summary(r.loc[a:]); print(f"  {lab}: CAGR {x['cagr']*100:5.1f}% Sharpe {x['sharpe']:4.2f} MaxDD {x['maxdd']*100:6.1f}%")
OOS = pd.Timestamp("2026-05-24")
# 1 equity
fig, ax = plt.subplots(figsize=(11, 5.2)); eq, eb = M.equity(r), M.equity(btc)
ax.plot(eq, color=BLUE, lw=2, label="TrendCore spot-only"); ax.plot(eb, color=ORANGE, lw=1.4, label="BTC buy & hold"); ax.set_yscale("log")
ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50]); ax.set_yticklabels(["0.5x", "1x", "2x", "5x", "10x", "20x", "50x"]); ax.axvspan(OOS, r.index[-1], color=BLUE, alpha=.10, lw=0)
ax.annotate(f"{eq.iloc[-1]:.1f}x", (r.index[-1], eq.iloc[-1]), xytext=(6, 0), textcoords="offset points", color=BLUE, va="center", fontweight="bold")
ax.annotate(f"{eb.iloc[-1]:.1f}x", (r.index[-1], eb.iloc[-1]), xytext=(6, 0), textcoords="offset points", color=ORANGE, va="center", fontweight="bold")
ax.legend(loc="upper left", frameon=False)
ax.set_title(f"Growth of $1, {r.index[0]:%Y-%m} to {r.index[-1]:%Y-%m} (log, spot-only, no leverage, no idle yield) - CAGR {s['cagr']*100:.1f}%, Sharpe {s['sharpe']:.2f}, max DD {s['maxdd']*100:.1f}%  |  BTC: {sb['cagr']*100:.1f}%, {sb['sharpe']:.2f}, {sb['maxdd']*100:.1f}%", loc="left", fontsize=9.5)
ax.set_xlim(r.index[0], r.index[-1] + pd.Timedelta(days=90)); fig.tight_layout(); fig.savefig("results/final_equity.png", dpi=150); plt.close(fig)
# 2 rolling sharpe
fig, ax = plt.subplots(figsize=(11, 4.4))
rs = r.rolling(365).mean() / r.rolling(365).std() * np.sqrt(365); rb = btc.rolling(365).mean() / btc.rolling(365).std() * np.sqrt(365)
ax.plot(rs, color=BLUE, lw=1.8, label="TrendCore spot-only (rolling 1y Sharpe)"); ax.plot(rb, color=ORANGE, lw=1.1, label="BTC buy & hold")
ax.axhline(0, color=INK2, lw=.8); ax.axhline(1.5, color=INK2, lw=.8, ls="--"); ax.text(rs.index[-1], 1.58, "1.5 target", fontsize=8, color=INK2, ha="right")
ax.axhline(s["sharpe"], color=BLUE, lw=.8, ls=":"); ax.text(rs.index[0], s["sharpe"] + .05, f"full-history Sharpe {s['sharpe']:.2f}", fontsize=8, color=BLUE)
ax.legend(loc="lower left", frameon=False); ax.set_title("Rolling 1-year Sharpe ratio", loc="left", fontsize=10.5); ax.set_ylabel("Sharpe")
fig.tight_layout(); fig.savefig("results/final_sharpe.png", dpi=150); plt.close(fig)
# 3 heatmap
m = (1 + r).resample("ME").prod() - 1; tbl = m.to_frame("r").assign(y=m.index.year, mo=m.index.month).pivot(index="y", columns="mo", values="r") * 100
ann = r.groupby(r.index.year).apply(lambda g: (1 + g).prod() - 1) * 100
cmap = LinearSegmentedColormap.from_list("div", [RED, MID, BLUE]); lim = 30; norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
fig, ax = plt.subplots(figsize=(12.5, 0.62 * len(tbl) + 1.6)); ax.imshow(np.clip(tbl.values, -lim, lim), cmap=cmap, norm=norm, aspect="auto")
for i in range(len(tbl)):
    for j in range(12):
        v = tbl.iloc[i, j]
        if np.isfinite(v): ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8.5, color="#ffffff" if abs(v) > 22 else INK)
ax.set_xticks(range(12)); ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]); ax.set_yticks(range(len(tbl))); ax.set_yticklabels(tbl.index)
ax.grid(False); ax.tick_params(length=0); ax.xaxis.tick_top()
for sp in ax.spines.values(): sp.set_visible(False)
ax2 = ax.twinx(); ax2.set_ylim(ax.get_ylim()); ax2.set_yticks(range(len(tbl))); ax2.set_yticklabels([f"{ann.loc[y]:+.1f}%" for y in tbl.index], fontweight="bold"); ax2.tick_params(length=0); ax2.grid(False)
for sp in ax2.spines.values(): sp.set_visible(False)
ax2.text(1.02, 1.0 + .5 / len(tbl) * 1.3, "Year", transform=ax2.transAxes, ha="left", fontsize=9, color=INK2, fontweight="bold")
ax.set_title("Monthly returns (%), spot-only   -   blue = gain, red = loss, colour capped at +/-30%", loc="left", fontsize=10.5, pad=26)
fig.tight_layout(); fig.savefig("results/final_monthly_heatmap.png", dpi=150); plt.close(fig)
# annual vs BTC table
ab = btc.groupby(btc.index.year).apply(lambda g: (1 + g).prod() - 1) * 100
print(pd.DataFrame({"TrendCore": ann.round(1), "BTC": ab.round(1)}).T.to_string())
