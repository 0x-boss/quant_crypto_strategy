"""Monthly-return heat maps (year x month, plus calendar-year total) for the two headline candidates.  python src/chart_heatmap.py
Diverging scale: red (loss) <-> grey (0) <-> blue (gain), clipped at +-15 %.  Outlined cells = months that hit the 5 % target."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

import lib

INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
cmap = LinearSegmentedColormap.from_list("div", ["#c0302f", "#e34948", "#f4b6b1", "#f0efec", "#a9c6ee", "#2a78d6", "#1b4f9a"])
norm = TwoSlopeNorm(vmin=-0.15, vcenter=0.0, vmax=0.15)
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9.5})

streams = pd.read_pickle(f"{lib.RES}/final_streams.pkl")["base"]
picks = [("BLEND9_EW", "Blend of 9 sleeves (equal capital)"), ("MOM_6_1_k10", "6-1 momentum, top 10, 5-day stagger")]
fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw=dict(wspace=0.32))
for ax, (k, title) in zip(axes, picks):
    r = streams[k].loc["2012":]
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
        ax.text(12.0, i, f"{yr.loc[y]*100:+.0f}%", ha="left", va="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(12)); ax.set_xticklabels(list("JFMAMJJASOND"))
    ax.set_yticks(range(len(years))); ax.set_yticklabels(years)
    ax.axhline(years.index(2022) - 0.5, color=INK, lw=1.0, ls=(0, (4, 3)))
    ax.text(12.0, -0.85, "year", fontsize=8.5, color=MUTED, ha="left")
    mm = m.values
    ax.set_title(f"{title}\nmedian month {np.median(mm)*100:.1f}% | months >= 5%: {np.mean(mm>=0.05)*100:.0f}% | worst {mm.min()*100:.0f}%",
                 loc="left", fontsize=10.5)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
cax = fig.add_axes([0.35, 0.065, 0.3, 0.016])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=[-0.15, -0.075, 0, 0.075, 0.15])
cb.ax.set_xticklabels(["-15%", "-7.5%", "0", "+7.5%", "+15%"]); cb.outline.set_visible(False)
fig.suptitle("Monthly returns, net of 0.2% round trip. Outlined cells = months at or above the 5% target; dashed line = start of the 2022 holdout",
             x=0.01, ha="left", fontsize=11.5)
fig.subplots_adjust(top=0.84, bottom=0.17, left=0.04, right=0.95)
fig.savefig(f"{lib.RES}/chart_monthly_heatmap.png", dpi=150)
print("ok")
