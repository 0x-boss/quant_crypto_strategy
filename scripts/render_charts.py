"""Equity curve, monthly-return heatmap and annual returns for TrendCore-I + carry.  python scripts/render_charts.py"""
import os, sys, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quant import metrics as M

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, RED, MID = "#2a78d6", "#eb6834", "#e34948", "#f0efec"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .7})
d = pd.read_csv("results/intraday_daily.csv", index_col=0, parse_dates=True)
r = d["total"]
btc = pd.read_parquet("data/intraday/BTCUSDT_spot_1h.parquet")["c"].resample("1D").last().pct_change().reindex(r.index).fillna(0)
OOS = pd.Timestamp("2026-05-24")

# ---------------------------------------------------------------- 1 equity curve
fig, ax = plt.subplots(figsize=(11, 5.2))
eq, eb = M.equity(r), M.equity(btc)
ax.plot(eq, color=BLUE, lw=2, label="TrendCore-I + carry"); ax.plot(eb, color=ORANGE, lw=1.4, label="BTC buy & hold")
ax.set_yscale("log"); ax.set_yticks([1, 3, 10, 30, 100]); ax.set_yticklabels(["1x", "3x", "10x", "30x", "100x"])
ax.axvspan(OOS, r.index[-1], color=BLUE, alpha=.10, lw=0); ax.text(OOS, ax.get_ylim()[0] * 1.15, " forward OOS", fontsize=8, color=INK2)
ax.annotate(f"{eq.iloc[-1]:.0f}x", (r.index[-1], eq.iloc[-1]), xytext=(6, 0), textcoords="offset points", color=BLUE, va="center", fontweight="bold")
ax.annotate(f"{eb.iloc[-1]:.0f}x", (r.index[-1], eb.iloc[-1]), xytext=(6, 0), textcoords="offset points", color=ORANGE, va="center", fontweight="bold")
ax.legend(loc="upper left", frameon=False)
s = M.summary(r); sb = M.summary(btc)
ax.set_title(f"Growth of $1, 2020-03 to 2026-10 (log)  -  CAGR {s['cagr']*100:.1f}%, Sharpe {s['sharpe']:.2f}, max drawdown {s['maxdd']*100:.1f}%   |   BTC: {sb['cagr']*100:.1f}%, {sb['sharpe']:.2f}, {sb['maxdd']*100:.1f}%",
             loc="left", fontsize=10.5, color=INK)
ax.set_xlim(r.index[0], r.index[-1] + pd.Timedelta(days=90)); fig.tight_layout(); fig.savefig("results/chart_equity.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- 2 monthly heatmap (+ annual column)
m = (1 + r).resample("ME").prod() - 1
tbl = m.to_frame("r").assign(y=m.index.year, mo=m.index.month).pivot(index="y", columns="mo", values="r") * 100
ann = r.groupby(r.index.year).apply(lambda g: (1 + g).prod() - 1) * 100
cmap = LinearSegmentedColormap.from_list("div", [RED, MID, BLUE]); lim = 40
norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
fig, ax = plt.subplots(figsize=(12.5, 4.6))
ax.imshow(np.clip(tbl.values, -lim, lim), cmap=cmap, norm=norm, aspect="auto")
for i, y in enumerate(tbl.index):
    for j in range(12):
        v = tbl.iloc[i, j]
        if np.isfinite(v):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8.5, color="#ffffff" if abs(v) > 28 else INK)
ax.set_xticks(range(12)); ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax.set_yticks(range(len(tbl))); ax.set_yticklabels(tbl.index); ax.grid(False)
for sp in ax.spines.values(): sp.set_visible(False)
ax.tick_params(length=0); ax.xaxis.tick_top()
ax2 = ax.twinx(); ax2.set_ylim(ax.get_ylim()); ax2.set_yticks(range(len(tbl)))
ax2.set_yticklabels([f"{ann.loc[y]:+.1f}%" for y in tbl.index], fontweight="bold"); ax2.tick_params(length=0); ax2.grid(False)
for sp in ax2.spines.values(): sp.set_visible(False)
ax2.text(1.0, 1.03, "Year", transform=ax2.transAxes, ha="center", fontsize=9, color=INK2, fontweight="bold")
ax.set_title("Monthly returns (%)   -   blue = gain, red = loss, colour capped at +/-40%", loc="left", fontsize=10.5, pad=26, color=INK)
fig.tight_layout(); fig.savefig("results/chart_monthly_heatmap.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- 3 annual returns vs BTC
ab = btc.groupby(btc.index.year).apply(lambda g: (1 + g).prod() - 1) * 100
yrs = list(ann.index); x = np.arange(len(yrs)); w = .38
fig, ax = plt.subplots(figsize=(11, 4.8))
b1 = ax.bar(x - w / 2 - .01, ann.values, w, color=BLUE, label="TrendCore-I + carry"); b2 = ax.bar(x + w / 2 + .01, ab.values, w, color=ORANGE, label="BTC buy & hold")
ax.axhline(0, color=INK2, lw=.9)
for bars in (b1, b2):
    for b in bars:
        h = b.get_height(); ax.annotate(f"{h:+.0f}%", (b.get_x() + b.get_width() / 2, h), xytext=(0, 3 if h >= 0 else -3), textcoords="offset points",
                                        ha="center", va="bottom" if h >= 0 else "top", fontsize=8.5, color=INK2)
ax.set_xticks(x); ax.set_xticklabels([f"{y}" + (" YTD" if y == 2026 else "") + ("\n(from Mar)" if y == 2020 else "") for y in yrs])
ax.set_ylabel("calendar-year return (%)"); ax.legend(loc="upper right", frameon=False); ax.grid(axis="x", visible=False)
ax.set_ylim(min(ab.min(), ann.min()) * 1.25, max(ab.max(), ann.max()) * 1.12)
ax.set_title("Annual returns - TrendCore-I + carry vs BTC (2022: -6% vs -64%)", loc="left", fontsize=10.5, color=INK)
fig.tight_layout(); fig.savefig("results/chart_annual.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- combined sheet
from PIL import Image
ims = [Image.open(f"results/{n}.png") for n in ("chart_equity", "chart_monthly_heatmap", "chart_annual")]
W = max(i.width for i in ims); H = sum(i.height for i in ims)
sheet = Image.new("RGB", (W, H), SURF); y = 0
for i in ims: sheet.paste(i, ((W - i.width) // 2, y)); y += i.height
sheet.save("results/charts_sheet.png")
print(tbl.round(1).to_string()); print((ann).round(1).to_dict())
