"""Round-8 charts.  python src/r8_charts.py"""
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

import lib
import r8_common as R

BLUE, ORANGE, AQUA, GREY, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a95", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6})
rets = pd.read_pickle(f"{R.OUTDIR}/stack_returns.pkl")
base, chosen, prot = rets["base+none"], rets["base+rollq80"], rets["vcap+exp80"]

# ---- 1. equity + underwater
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.5, 7), sharex=True, gridspec_kw=dict(height_ratios=[3, 2], hspace=0.08))
for r, lab, col, ls, lw in ((base, "Baseline: 6-1 momentum, top 10", GREY, "--", 1.4),
                            (chosen, "Chosen stack: same book + own-vol scaling (rolling 80th pct)", BLUE, "-", 1.7),
                            (prot, "Max-protection cell: vol-capped book + own-vol scaling (median falls to 2%)", AQUA, "-", 1.5)):
    x = r.loc["2012":]
    a1.plot((1 + x).cumprod(), color=col, ls=ls, lw=lw, label=lab)
    eq = (1 + x).cumprod()
    a2.plot(eq / eq.cummax() - 1, color=col, ls=ls, lw=lw)
a1.set_yscale("log"); a1.grid(True); a1.set_ylabel("growth of 1 (log)"); a1.legend(frameon=False, loc="upper left", fontsize=9)
a2.grid(True); a2.set_ylabel("drawdown"); a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
for ax in (a1, a2):
    ax.axvline(pd.Timestamp("2022-01-01"), color="#c9c8c3", lw=1)
a2.text(pd.Timestamp("2022-02-01"), -0.57, "holdout ->", fontsize=8.5, color=MUTED)
a1.set_title("Round 8: lighter drawdown at the same median month - net of 0.2% round trip", loc="left", fontsize=11.5)
fig.savefig(f"{R.OUTDIR}/chart_r8_equity_drawdown.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---- 2. frontier: HOLD median month vs HOLD max drawdown across every family config
files = [f for f in glob.glob(f"{R.OUTDIR}/trials_*.csv") if not any(k in f for k in ("run1", "run2", "_prerestart"))]
t = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).dropna(subset=["hold_med_m", "hold_maxdd"])
fig, ax = plt.subplots(figsize=(8.6, 5.4))
ax.add_patch(plt.Rectangle((-0.31, 0.05), 0.31, 0.04, color=BLUE, alpha=0.08, lw=0))
ax.add_patch(plt.Rectangle((-0.31, 0.05), 0.31, 0.04, fill=False, ec=INK, lw=1.2))
ax.scatter(t.hold_maxdd, t.hold_med_m, s=13, color="#8fb4e8", alpha=0.65, lw=0.3, ec=SURF, label=f"{len(t)} configs from the 9 families")
def pt(r, lab, col, mk, dx, dy):
    s = R.evaluate(r, show=False)["hold"]
    ax.scatter([s["maxdd"]], [s["med_m"]], s=90, color=col, marker=mk, ec=INK, lw=0.8, zorder=5)
    ax.annotate(lab, (s["maxdd"], s["med_m"]), xytext=(dx, dy), textcoords="offset points", fontsize=8.8, color=INK)
pt(base, "baseline", GREY, "D", 8, 6)
pt(chosen, "chosen stack: same median, lighter DD", BLUE, "o", -215, 12)
pt(prot, "max protection: DD -32%, median 2%", AQUA, "s", 8, -14)
ax.text(-0.305, 0.0865, "target: median >= 5%\nand HOLD drawdown <= 31%", fontsize=8.8, va="top")
ax.set_xlim(-0.66, -0.12); ax.set_ylim(-0.04, 0.09); ax.grid(True)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0)); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.set_xlabel("HOLDOUT 2022-26 max drawdown  (lighter ->)"); ax.set_ylabel("HOLDOUT median calendar month")
ax.legend(frameon=False, loc="lower right", fontsize=8.8)
ax.set_title("Median month vs drawdown: every config sits on one frontier, the target box is empty", loc="left", fontsize=11)
fig.savefig(f"{R.OUTDIR}/chart_r8_frontier.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---- 3. heat maps (baseline vs chosen)
cmap = LinearSegmentedColormap.from_list("div", ["#c0302f", "#e34948", "#f4b6b1", "#f0efec", "#a9c6ee", "#2a78d6", "#1b4f9a"])
norm = TwoSlopeNorm(vmin=-0.15, vcenter=0.0, vmax=0.15)
fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw=dict(wspace=0.32))
for ax, (r, title) in zip(axes, ((base, "Baseline: 6-1 momentum, top 10"), (chosen, "Chosen stack (own-vol scaling, rolling 80th pct)"))):
    r = r.loc["2012":]
    m = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    cnt = r.groupby([r.index.year, r.index.month]).count()
    m = m[cnt >= 10]
    g = m.unstack().reindex(columns=range(1, 13))
    yr = (1 + r).groupby(r.index.year).prod() - 1
    years = list(g.index)
    ax.imshow(np.ma.masked_invalid(g.values), cmap=cmap, norm=norm, aspect="auto")
    for i, y in enumerate(years):
        for j in range(12):
            v = g.values[i, j]
            if np.isfinite(v) and v >= 0.05:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.1))
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=7.5, color="white" if v > 0.09 else INK)
            elif np.isfinite(v) and v <= -0.10:
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=7.5, color="white")
        ax.text(12.0, i, f"{yr.loc[y]*100:+.0f}%", ha="left", va="center", fontsize=8.5)
    ax.set_xticks(range(12)); ax.set_xticklabels(list("JFMAMJJASOND")); ax.set_yticks(range(len(years))); ax.set_yticklabels(years)
    ax.axhline(years.index(2022) - 0.5, color=INK, lw=1.0, ls=(0, (4, 3)))
    ax.text(12.0, -0.85, "year", fontsize=8.5, color=MUTED)
    mm = m.values
    ax.set_title(f"{title}\nmedian month {np.median(mm)*100:.1f}% | months >= 5%: {np.mean(mm>=0.05)*100:.0f}% | worst {mm.min()*100:.0f}%", loc="left", fontsize=10.5)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
cax = fig.add_axes([0.35, 0.065, 0.3, 0.016])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=[-0.15, -0.075, 0, 0.075, 0.15])
cb.ax.set_xticklabels(["-15%", "-7.5%", "0", "+7.5%", "+15%"]); cb.outline.set_visible(False)
fig.suptitle("Monthly returns, net of 0.2% round trip. Outlined = months at or above 5%; dashed line = start of the 2022 holdout", x=0.01, ha="left", fontsize=11.5)
fig.subplots_adjust(top=0.84, bottom=0.17, left=0.04, right=0.95)
fig.savefig(f"{R.OUTDIR}/chart_r8_heatmap.png", dpi=150); plt.close(fig)
print("ok", len(t), "configs plotted")
