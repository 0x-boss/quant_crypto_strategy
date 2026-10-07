"""Charts for the report (matplotlib, palette from the dataviz reference palette; slots 1-3 + neutral grey).   python src/charts.py"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lib

BLUE, ORANGE, AQUA, GREY, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a95", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": "#c9c8c3",
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})

# ------------------------------------------------------------------ 1. every registered configuration vs the target box
t = pd.read_csv(f"{lib.RES}/trials.csv").dropna(subset=["dev_sharpe", "hold_sharpe"])
grp = {"A_daily_rank": ("Daily factor ranks (liquid pool)", BLUE), "G_composite": ("Daily factor ranks (liquid pool)", BLUE),
       "B_slots": ("Daily factor ranks (liquid pool)", BLUE), "D_ml": ("Daily factor ranks (liquid pool)", BLUE),
       "C_intraday": ("Intraday 5-min (top-10 picks)", AQUA), "F_broad": ("Broad illiquid pool (top 1000-3000)", ORANGE),
       "H_letf_appendix": ("Leveraged-ETF appendix", GREY)}
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
for ax, part, title in ((axes[0], "dev", "DEV 2012-2021"), (axes[1], "hold", "HOLDOUT 2022-2026")):
    ax.axvspan(1.5, 3, ymin=0, ymax=1, color="#2a78d6", alpha=0.07, lw=0)
    ax.axhspan(0.05, 0.08, color="#2a78d6", alpha=0.07, lw=0)
    ax.add_patch(plt.Rectangle((1.5, 0.05), 1.5, 0.03, fill=False, ec=INK, lw=1.2))
    seen = set()
    for fam, (lab, col) in grp.items():
        g = t[t.family == fam]
        ax.scatter(g[f"{part}_sharpe"], g[f"{part}_med_m"], s=14, color=col, alpha=0.75, lw=0.4, ec=SURF, label=None if lab in seen else lab)
        seen.add(lab)
    ax.axhline(0, color="#c9c8c3", lw=0.8); ax.axvline(0, color="#c9c8c3", lw=0.8)
    ax.set_xlim(-3.2, 2.2); ax.set_ylim(-0.06, 0.07)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("annualised Sharpe (net of 0.2% round trip)")
    ax.grid(True)
    ax.text(1.55, 0.0655, "target: Sharpe > 1.5\nand median month > 5%", fontsize=8.5, color=INK, va="top")
axes[0].set_ylabel("median calendar-month return")
axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9, markerscale=1.6)
fig.suptitle(f"All {len(t)} registered configurations: none lands in the target box", x=0.01, ha="left", fontsize=12.5)
fig.tight_layout(rect=(0, 0.06, 1, 0.94))
fig.savefig(f"{lib.RES}/chart_trials_vs_target.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ 2. intraday cost wall
g = pd.read_csv(f"{lib.RES}/intraday_gross_edge.csv")
best = g.groupby(["entry", "part"]).gross_bps.max().unstack()
fig, ax = plt.subplots(figsize=(8.5, 4))
x = np.arange(len(best)); w = 0.36
ax.bar(x - w / 2, best["dev"], w - 0.04, color=BLUE, label="DEV 2016-2021 (best signal)")
ax.bar(x + w / 2, best["hold"], w - 0.04, color=ORANGE, label="HOLDOUT 2022-2026 (best signal)")
ax.axhline(20, color=INK, lw=1.3); ax.text(len(best) - 0.5, 21.2, "round-trip cost 20 bp", ha="right", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(best.index); ax.set_xlabel("entry time (ET), exit at the close")
ax.set_ylabel("mean gross return per trade (bp)"); ax.set_ylim(0, 26); ax.grid(True, axis="y")
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.set_title("Intraday: even the best of 7 signals earns < 10 bp before costs", loc="left", fontsize=11.5)
fig.tight_layout(); fig.savefig(f"{lib.RES}/chart_intraday_cost_wall.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------ 3. equity curves + monthly distribution
streams = pd.read_pickle(f"{lib.RES}/final_streams.pkl")["base"]
sl = pd.read_parquet(f"{lib.RES}/sleeves.parquet")
curves = {"BLEND9_EW": ("Blend of 9 sleeves (equal capital)", BLUE), "MOM_6_1_k10": ("6-1 momentum, top 10, weekly hold", ORANGE),
          "COMP_C2_k25_h21_top1000": ("Composite mom+rev+52wk-high, top 25 of 1000", AQUA)}
fig, ax = plt.subplots(figsize=(10, 4.8))
for k, (lab, col) in curves.items():
    r = streams[k].loc["2012":]
    ax.plot((1 + r).cumprod(), color=col, lw=1.6, label=lab)
ax.plot((1 + sl["SPY_bh"].loc["2012":]).cumprod(), color=GREY, lw=1.4, ls="--", label="SPY buy & hold (no fee)")
ax.axvline(pd.Timestamp("2022-01-01"), color="#c9c8c3", lw=1)
ax.text(pd.Timestamp("2022-02-01"), 1.12, "holdout ->", fontsize=8.5, color=MUTED)
ax.set_yscale("log"); ax.grid(True); ax.set_ylabel("growth of 1 (log)")
ax.set_title("Best honest candidates vs SPY - net of 0.2% round trip", loc="left", fontsize=11.5)
ax.legend(frameon=False, loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(f"{lib.RES}/chart_equity.png", dpi=150); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True)
for ax, k in zip(axs, ("BLEND9_EW", "MOM_6_1_k10")):
    m = lib.monthly(streams[k].loc["2012":])
    ax.hist(m.values, bins=np.arange(-0.25, 0.32, 0.01), color=BLUE if k == "BLEND9_EW" else ORANGE, lw=0)
    ax.axvline(0.05, color=INK, lw=1.3); ax.text(0.052, ax.get_ylim()[1] * 0.92, "5% target", fontsize=8.5)
    ax.axvline(m.median(), color=MUTED, lw=1.1, ls="--"); ax.text(m.median() + 0.003, ax.get_ylim()[1] * 0.75, f"median {m.median()*100:.1f}%", fontsize=8.5, color=MUTED)
    ax.set_title(curves[k][0], loc="left", fontsize=10.5); ax.grid(True, axis="y")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
axs[0].set_ylabel("number of months")
fig.suptitle("Monthly returns 2012-2026", x=0.01, ha="left", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(f"{lib.RES}/chart_monthly_dist.png", dpi=150); plt.close(fig)
print("charts written")
