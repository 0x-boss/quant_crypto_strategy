"""Run TrendCore end-to-end: metrics, yearly table, charts.   python run_backtest.py"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quant import metrics as M
from quant.strategy import Config, run

os.makedirs("results", exist_ok=True)
cfg = Config()
r, det, W, P, R = run(cfg, details=True)
idx = r.index

bench = {
    "BTC buy&hold": R["btc"].reindex(idx).fillna(0.0),
    "ETH buy&hold": R["eth"].reindex(idx).fillna(0.0),
    "50/50 BTC/ETH monthly rebal": (0.5 * R["btc"] + 0.5 * R["eth"]).reindex(idx).fillna(0.0),
}
windows = {"2016-2026 (full)": "2016-01-01", "2018-2026": "2018-01-01", "2020-2026": "2020-01-01", "2022-2026": "2022-01-01"}

rows = []
metrics = {}
for wname, a in windows.items():
    s = M.summary(r.loc[a:])
    metrics[wname] = s
    rows.append((f"TrendCore  {wname}", s))
for bname, br in bench.items():
    rows.append((f"{bname}  2016-2026", M.summary(br)))
print(f"{'':44s}{'CAGR':>8s}{'Sharpe':>8s}{'Sortino':>8s}{'MaxDD':>8s}{'Vol':>7s}{'Calmar':>8s}")
for n, s in rows:
    print(f"{n:44s}{s['cagr']*100:7.1f}%{s['sharpe']:8.2f}{s['sortino']:8.2f}{s['maxdd']*100:7.1f}%{s['vol']*100:6.1f}%{s['calmar']:8.2f}")

yr = M.yearly(r)
print("\nCalendar years:\n", yr.round(3).to_string())
print(f"\navg gross exposure {det['gross'].mean():.2f} | time flat {(det['gross']<0.05).mean()*100:.0f}% | "
      f"avg turnover {det['turnover'].sum()/len(det)*365:.1f}x/yr | cost {det['cost'].sum()/len(det)*365*100:.2f}%/yr | "
      f"financing {det['financing'].sum()/len(det)*365*100:.2f}%/yr | max gross {det['gross'].max():.2f}")

with open("results/metrics.json", "w") as f:
    json.dump(dict(config=cfg.to_dict(), windows=metrics,
                   benchmarks={k: M.summary(v) for k, v in bench.items()},
                   yearly=yr.to_dict(orient="index")), f, indent=1, default=float)
out = pd.DataFrame({"ret": r, "equity": M.equity(r), "gross": det["gross"], "w_btc": W["btc"].reindex(idx), "w_eth": W["eth"].reindex(idx)})
out.to_csv("results/equity.csv")
W.loc[idx, ["btc", "eth"]].tail(10).to_csv("results/latest_target_weights.csv")

# ---------------- charts
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})
C = {"tc": "#1f6feb", "btc": "#f2a900", "eth": "#6e7681", "dd": "#d1242f"}
fig, ax = plt.subplots(4, 1, figsize=(11, 14), gridspec_kw={"height_ratios": [3, 1.3, 1.3, 1.3]}, sharex=False)
ax[0].plot(M.equity(r), color=C["tc"], lw=1.8, label="TrendCore")
ax[0].plot(M.equity(bench["BTC buy&hold"]), color=C["btc"], lw=1.1, alpha=.9, label="BTC buy & hold")
ax[0].plot(M.equity(bench["ETH buy&hold"]), color=C["eth"], lw=1.0, alpha=.8, label="ETH buy & hold")
ax[0].set_yscale("log")
ax[0].set_title("Growth of $1 (log scale, net of costs & financing)")
ax[0].legend(loc="upper left")
dd = M.drawdown_series(r)
ax[1].fill_between(dd.index, dd * 100, 0, color=C["dd"], alpha=.45, label="TrendCore")
ax[1].plot(M.drawdown_series(bench["BTC buy&hold"]) * 100, color=C["btc"], lw=.9, label="BTC buy & hold")
ax[1].set_title("Drawdown (%)")
ax[1].legend(loc="lower left")
roll = (r.rolling(365).mean() / r.rolling(365).std() * np.sqrt(365))
ax[2].plot(roll, color=C["tc"], lw=1.3, label="TrendCore rolling 1y Sharpe")
ax[2].plot((bench["BTC buy&hold"].rolling(365).mean() / bench["BTC buy&hold"].rolling(365).std() * np.sqrt(365)), color=C["btc"], lw=.9, label="BTC")
ax[2].axhline(1.5, color="k", ls="--", lw=.8)
ax[2].axhline(0, color="k", lw=.6)
ax[2].set_title("Rolling 1-year Sharpe (dashed = 1.5 target)")
ax[2].legend(loc="upper right")
ax[3].fill_between(det.index, det["gross"], 0, color=C["tc"], alpha=.5)
ax[3].set_title("Gross exposure (x NAV)")
plt.tight_layout()
plt.savefig("results/equity_drawdown.png")
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
y = yr["ret"] * 100
by = pd.Series({k: M.summary(bench["BTC buy&hold"].loc[str(k)])["total"] * 100 if len(bench["BTC buy&hold"].loc[str(k)]) > 60 else (1 + bench["BTC buy&hold"].loc[str(k)]).prod() * 100 - 100 for k in yr.index})
xs = np.arange(len(y))
ax[0].bar(xs - .2, y.values, .4, color=C["tc"], label="TrendCore")
ax[0].bar(xs + .2, by.values, .4, color=C["btc"], label="BTC")
ax[0].set_xticks(xs)
ax[0].set_xticklabels(y.index.astype(str), rotation=45)
ax[0].set_yscale("symlog", linthresh=100)
ax[0].set_title("Calendar-year return (%), symlog")
ax[0].legend()
m = (1 + r).resample("ME").prod() - 1
piv = m.to_frame("r").assign(y=m.index.year, mth=m.index.month).pivot(index="y", columns="mth", values="r") * 100
im = ax[1].imshow(piv.values, cmap="RdYlGn", vmin=-25, vmax=25, aspect="auto")
ax[1].set_xticks(range(12))
ax[1].set_xticklabels(list("JFMAMJJASOND"))
ax[1].set_yticks(range(len(piv)))
ax[1].set_yticklabels(piv.index)
ax[1].set_title("Monthly returns (%)")
ax[1].grid(False)
plt.tight_layout()
plt.savefig("results/yearly_monthly.png")
print("charts saved to results/")
