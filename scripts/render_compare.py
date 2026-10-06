import sys, warnings, importlib.util
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
warnings.filterwarnings("ignore"); sys.path.insert(0, "."); 
from quant import metrics as M
from quant.intraday import hourly_prices
import trendcore_spot as T
spec = importlib.util.spec_from_file_location("c51", "research/51_btc_signal_swap.py"); c51 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c51)
START = pd.Timestamp("2020-09-01")
oi = pd.read_parquet("data/intraday/BTCUSDT_oi_daily.parquet")["sum_open_interest_value"]
px = hourly_prices(("btc", "eth", "sol")).sort_index()
P = px.resample("6h").last().dropna(how="all"); rv2 = (np.log(px).diff() ** 2).resample("6h").sum().reindex(P.index)
S = {}
S["BTC only"], _ = c51.swap_strategy(P, rv2, 4, ["btc"], "own", oi)
S["BTC+ETH (final strategy)"], _ = T.backtest(px[["btc", "eth"]], oi, start="2018-03-01")
S["ETH+SOL on BTC signal"], _ = c51.swap_strategy(P, rv2, 4, ["eth", "sol"], "swap", oi)
S = {k: v.loc[START:] for k, v in S.items()}
btc = px["btc"].resample("1D").last().pct_change().loc[START:].fillna(0)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COL = {"BTC only": "#2a78d6", "BTC+ETH (final strategy)": "#eb6834", "ETH+SOL on BTC signal": "#1baf7a"}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .7})
rows = []
for k, r in S.items():
    s = M.summary(r); s22 = M.summary(r.loc["2022-01-01":]); rows.append((k, s["cagr"], s["sharpe"], s["maxdd"], s22["cagr"], s22["sharpe"]))
    print(f"{k:26s} CAGR {s['cagr']*100:5.1f}% Sharpe {s['sharpe']:4.2f} Sortino {s['sortino']:4.2f} MaxDD {s['maxdd']*100:6.1f}% | 2022+ CAGR {s22['cagr']*100:5.1f}% Sharpe {s22['sharpe']:4.2f}")
sb = M.summary(btc); print(f"{'BTC buy & hold':26s} CAGR {sb['cagr']*100:5.1f}% Sharpe {sb['sharpe']:4.2f} MaxDD {sb['maxdd']*100:6.1f}%")
sub = "2020-09 to 2026-10 | spot-only, no leverage, no idle yield | ETH+SOL: SOL is a known survivor (hindsight)"
# 1 equity
fig, ax = plt.subplots(figsize=(11, 5.4))
for k, r in S.items():
    e = M.equity(r); ax.plot(e, color=COL[k], lw=2, label=f"{k}  ({e.iloc[-1]:.1f}x)")
ax.plot(M.equity(btc), color="#9a9a96", lw=1, ls="--", label=f"BTC buy & hold  ({M.equity(btc).iloc[-1]:.1f}x)")
ax.set_yscale("log"); ax.set_yticks([1, 2, 5, 10, 20]); ax.set_yticklabels(["1x", "2x", "5x", "10x", "20x"]); ax.legend(loc="upper left", frameon=False)
ax.set_title("Growth of $1 (log)\n" + sub, loc="left", fontsize=10); fig.tight_layout(); fig.savefig("results/cmp_equity.png", dpi=150); plt.close(fig)
# 2 rolling sharpe
fig, ax = plt.subplots(figsize=(11, 4.4))
for k, r in S.items(): ax.plot(r.rolling(365).mean() / r.rolling(365).std() * np.sqrt(365), color=COL[k], lw=1.7, label=k)
ax.axhline(0, color=INK2, lw=.8); ax.axhline(1.5, color=INK2, lw=.8, ls="--"); ax.legend(loc="lower left", frameon=False, ncol=3); ax.set_ylabel("rolling 1y Sharpe")
ax.set_title("Rolling 1-year Sharpe (dashed = 1.5)", loc="left", fontsize=10.5); fig.tight_layout(); fig.savefig("results/cmp_sharpe.png", dpi=150); plt.close(fig)
# 3 annual returns
ann = pd.DataFrame({k: r.groupby(r.index.year).apply(lambda g: (1 + g).prod() - 1) * 100 for k, r in S.items()})
fig, ax = plt.subplots(figsize=(11, 4.6)); x = np.arange(len(ann)); w = .26
for i, (k, c) in enumerate(COL.items()):
    b = ax.bar(x + (i - 1) * (w + .01), ann[k].values, w, color=c, label=k)
    for rect in b: h = rect.get_height(); ax.annotate(f"{h:+.0f}", (rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2 if h >= 0 else -2), textcoords="offset points", ha="center", va="bottom" if h >= 0 else "top", fontsize=7.5, color=INK2)
ax.axhline(0, color=INK2, lw=.9); ax.set_xticks(x); ax.set_xticklabels([str(y) + (" YTD" if y == 2026 else "") + ("\n(from Sep)" if y == 2020 else "") for y in ann.index]); ax.grid(axis="x", visible=False)
ax.legend(loc="upper right", frameon=False); ax.set_ylabel("calendar-year return (%)"); ax.set_title("Annual returns (%)", loc="left", fontsize=10.5); fig.tight_layout(); fig.savefig("results/cmp_annual.png", dpi=150); plt.close(fig)
# 4 monthly heatmaps (3 stacked)
cmap = LinearSegmentedColormap.from_list("d", ["#e34948", "#f0efec", "#2a78d6"]); norm = TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30)
fig, axs = plt.subplots(3, 1, figsize=(12, 13.2))
for ax, (k, r) in zip(axs, S.items()):
    m = (1 + r).resample("ME").prod() - 1; t = m.to_frame("r").assign(y=m.index.year, mo=m.index.month).pivot(index="y", columns="mo", values="r") * 100
    a = r.groupby(r.index.year).apply(lambda g: (1 + g).prod() - 1) * 100
    ax.imshow(np.clip(t.values, -30, 30), cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(t)):
        for j in range(12):
            v = t.iloc[i, j]
            if np.isfinite(v): ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="#fff" if abs(v) > 22 else INK)
    ax.set_xticks(range(12)); ax.set_xticklabels(list("JFMAMJJASOND")); ax.set_yticks(range(len(t))); ax.set_yticklabels(t.index); ax.grid(False); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax2 = ax.twinx(); ax2.set_ylim(ax.get_ylim()); ax2.set_yticks(range(len(t))); ax2.set_yticklabels([f"{a.loc[y]:+.0f}%" for y in t.index], fontweight="bold"); ax2.tick_params(length=0); ax2.grid(False)
    for sp in ax2.spines.values(): sp.set_visible(False)
    ax.set_title(f"{k} - monthly returns (%), year total at right", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig("results/cmp_monthly.png", dpi=140); plt.close(fig)
