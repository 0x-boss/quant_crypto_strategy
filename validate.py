"""Robustness / anti-overfitting validation of TrendCore.  Run:  python validate.py

Writes results/validation.json and prints a report.  Sections
  A  start-date & sub-period sensitivity
  B  stress tests: costs x1/x2/x4, +1/+2 day execution lag, financing 20 %, no leverage
  C  placebo: circularly shifted positions (keeps exposure level & autocorrelation, breaks timing)
  D  block bootstrap confidence intervals
  E  neighbouring-parameter plateau (gate, vol span, vol target, leverage cap)
  F  walk-forward selection among a fixed grid (selection uses only past data)
  G  cross-asset generalisation: identical rule applied to assets it was never developed on
  H  trial-adjusted (deflated) Sharpe
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from quant import metrics as M
from quant.data import load_panels, liquid_universe
from quant.engine import run_backtest
from quant.strategies import build_weights_bagged
from quant.strategy import Config, run, target_weights

OUT = {}
os.makedirs("results", exist_ok=True)


def fmt(s):
    return (f"CAGR {s['cagr']*100:5.1f}%  Sharpe {s['sharpe']:4.2f}  MaxDD {s['maxdd']*100:6.1f}%  "
            f"Vol {s['vol']*100:4.1f}%  Calmar {s['calmar']:4.2f}")


def header(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


cfg = Config()
r, det, W, P, R = run(cfg, details=True)
END = r.index[-1]

# ------------------------------------------------------------------ A
header("A. Sub-period / start-date sensitivity (same rules, nothing re-fitted)")
A = {}
for y in [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]:
    s = M.summary(r.loc[f"{y}-01-01":])
    A[f"from_{y}"] = s
    print(f"from {y}: {fmt(s)}")
blocks = {"2016-2017 (bubble)": ("2016-01-01", "2017-12-31"), "2018 bear": ("2018-01-01", "2018-12-31"),
          "2019-2021": ("2019-01-01", "2021-12-31"), "2022 bear": ("2022-01-01", "2022-12-31"),
          "2023-2024": ("2023-01-01", "2024-12-31"), "2025-26": ("2025-01-01", str(END.date()))}
for k, (a, b) in blocks.items():
    s = M.summary(r.loc[a:b])
    A[k] = s
    print(f"{k:20s}: ret {s['total']*100:7.1f}%  Sharpe {s['sharpe']:5.2f}  MaxDD {s['maxdd']*100:6.1f}%")
roll = (r.rolling(365).mean() / r.rolling(365).std() * np.sqrt(365)).dropna()
rr = (1 + r).rolling(365).apply(np.prod, raw=True).dropna() - 1
A["rolling365"] = dict(frac_positive=float((rr > 0).mean()), frac_over_50pct=float((rr > 0.5).mean()),
                       median_ret=float(rr.median()), p10_ret=float(rr.quantile(0.1)), worst_ret=float(rr.min()),
                       sharpe_median=float(roll.median()), frac_sharpe_over_1_5=float((roll > 1.5).mean()))
print("rolling 1y windows:", {k: round(v, 3) for k, v in A["rolling365"].items()})
OUT["A_windows"] = A

# ------------------------------------------------------------------ B
header("B. Stress tests (full sample from 2016)")
B = {}
for lab, kw in [("base (10bp BTC/ETH one-way)", {}), ("costs x2 (20bp)", dict(tier1_bps=20.0)),
                ("costs x4 (40bp)", dict(tier1_bps=40.0)), ("+1 day execution lag", dict(lag=1)),
                ("+2 day execution lag", dict(lag=2)), ("financing 20%", dict(fin_rate=0.20)),
                ("no leverage (cap 1.0)", dict(max_lev=1.0)), ("costs x2 + 1d lag + fin 20%", dict(tier1_bps=20.0, lag=1, fin_rate=0.20)),
                ("idle cash earns 4%", dict(cash_rate=0.04))]:
    c = Config(**{**cfg.to_dict(), **kw})
    rs = run(c)
    s = M.summary(rs)
    B[lab] = s
    print(f"{lab:32s}: {fmt(s)}")
OUT["B_stress"] = B

# ------------------------------------------------------------------ C
header("C. Placebo: circularly shifted positions (timing destroyed, exposure profile kept)")
R2 = R[["btc", "eth"]]
W2 = W[["btc", "eth"]]
true_sh = M.summary(run_backtest(W2, R2, **cfg.engine_kwargs()).loc[cfg.start:])["sharpe"]
rng = np.random.default_rng(7)
T = len(W2)
sh = []
idx0 = W2.index.get_loc(pd.Timestamp(cfg.start)) if pd.Timestamp(cfg.start) in W2.index else 0
for k in rng.integers(180, T - 180, 400):
    Ws = pd.DataFrame(np.roll(W2.values, k, axis=0), index=W2.index, columns=W2.columns)
    rs = run_backtest(Ws, R2, **cfg.engine_kwargs()).loc[cfg.start:]
    sh.append(M.summary(rs)["sharpe"])
sh = np.array(sh)
p = float((sh >= true_sh).mean())
OUT["C_placebo"] = dict(true_sharpe=true_sh, placebo_mean=float(sh.mean()), placebo_p95=float(np.quantile(sh, .95)),
                        placebo_max=float(sh.max()), p_value=p)
print(f"true Sharpe {true_sh:.2f} | placebo mean {sh.mean():.2f}, 95th pct {np.quantile(sh,.95):.2f}, max {sh.max():.2f} | p = {p:.4f}")

# ------------------------------------------------------------------ D
header("D. Block-bootstrap (20d blocks, 2000 draws), full sample")
bs = M.block_bootstrap_sharpe(r, 2000, 20, seed=1)
rng = np.random.default_rng(3)
x = r.values
nb = int(np.ceil(len(x) / 20))
cg, dd = [], []
for _ in range(1000):
    s = rng.integers(0, len(x) - 20, nb)
    z = np.concatenate([x[j:j + 20] for j in s])[:len(x)]
    eq = np.cumprod(1 + z)
    cg.append(eq[-1] ** (365 / len(z)) - 1)
    dd.append((eq / np.maximum.accumulate(eq) - 1).min())
OUT["D_bootstrap"] = dict(sharpe_ci90=[float(np.quantile(bs, .05)), float(np.quantile(bs, .95))],
                          p_sharpe_gt_1=float((bs > 1).mean()), p_sharpe_gt_1_5=float((bs > 1.5).mean()),
                          cagr_ci90=[float(np.quantile(cg, .05)), float(np.quantile(cg, .95))],
                          maxdd_ci90=[float(np.quantile(dd, .05)), float(np.quantile(dd, .95))])
print(json.dumps(OUT["D_bootstrap"], indent=1))
# same for the post-2018 and post-2022 windows
for a in ["2018-01-01", "2022-01-01"]:
    b2 = M.block_bootstrap_sharpe(r.loc[a:], 2000, 20, seed=2)
    print(f"from {a[:4]}: Sharpe 90% CI [{np.quantile(b2,.05):.2f}, {np.quantile(b2,.95):.2f}]  P(Sharpe>1)={np.mean(b2>1):.2f}")
    OUT["D_bootstrap"][f"sharpe_ci90_from_{a[:4]}"] = [float(np.quantile(b2, .05)), float(np.quantile(b2, .95))]

# ------------------------------------------------------------------ E
header("E. Neighbouring-parameter plateau (each row changes ONE choice)")
E = {}
for lab, kw in [("base", {}), ("gates (200,)", dict(gates=(200,))), ("gates (100,)", dict(gates=(100,))), ("gates (50,100,150,200,250)", dict(gates=(50, 100, 150, 200, 250))),
                ("vol spans (30,)", dict(vol_spans=(30,))), ("vol spans (10,20,30,60,90)", dict(vol_spans=(10, 20, 30, 60, 90))),
                ("signal mom only", dict(signals=("mom",))), ("signal sma only", dict(signals=("sma",))), ("signal ewmac only", dict(signals=("ewmac",))),
                ("target vol 0.25", dict(target_vol=0.25)), ("target vol 0.30", dict(target_vol=0.30)), ("target vol 0.40", dict(target_vol=0.40)),
                ("max lev 1.0", dict(max_lev=1.0)), ("max lev 2.0", dict(max_lev=2.0)), ("band 0", dict(band=0.0)), ("BTC only", dict(assets=("btc",)))]:
    c = Config(**{**cfg.to_dict(), **kw})
    rs = run(c)
    s = M.summary(rs)
    s18 = M.summary(rs.loc["2018-01-01":])
    E[lab] = dict(full=s, from2018=s18)
    print(f"{lab:30s}: {fmt(s)} | from 2018: Sharpe {s18['sharpe']:4.2f} CAGR {s18['cagr']*100:5.1f}%")
OUT["E_plateau"] = E
shs = np.array([v["full"]["sharpe"] for v in E.values()])
print(f"-> Sharpe across {len(shs)} neighbours: min {shs.min():.2f}, median {np.median(shs):.2f}, max {shs.max():.2f}")

# ------------------------------------------------------------------ F
header("F. Walk-forward selection (expanding window, choose on trailing 2y Sharpe, hold 6 months)")
grid = []
for g in [(100,), (200,), (100, 150, 200)]:
    for sg in [("mom",), ("sma",), ("mix",)]:
        for tv in [0.30, 0.35]:
            grid.append(Config(**{**cfg.to_dict(), "gates": g, "signals": sg, "target_vol": tv}))
rets = pd.concat([run(c) for c in grid], axis=1)
rets.columns = range(len(grid))
starts = pd.date_range("2019-01-01", "2026-05-23", freq="6MS")
chosen, parts = [], []
for a, b in zip(starts[:-1], list(starts[1:]) + [pd.Timestamp("2026-06-30")]):
    past = rets.loc[a - pd.Timedelta(days=730):a - pd.Timedelta(days=1)]
    best = (past.mean() / past.std()).idxmax()
    chosen.append(best)
    parts.append(rets.loc[a:b - pd.Timedelta(days=1), best])
wf = pd.concat(parts)
base_same = r.loc[wf.index[0]:]
s_wf, s_base = M.summary(wf), M.summary(base_same)
print(f"walk-forward-selected : {fmt(s_wf)}")
print(f"fixed TrendCore       : {fmt(s_base)}")
print(f"grid of {len(grid)} configs; same-window Sharpe range of the grid: "
      f"{(rets.loc[wf.index[0]:].mean()/rets.loc[wf.index[0]:].std()*np.sqrt(365)).min():.2f} .. "
      f"{(rets.loc[wf.index[0]:].mean()/rets.loc[wf.index[0]:].std()*np.sqrt(365)).max():.2f}")
OUT["F_walkforward"] = dict(walkforward=s_wf, fixed=s_base, n_configs=len(grid))

# ------------------------------------------------------------------ G
header("G. Cross-asset generalisation: identical rule on assets NOT used in development")
Pm, Vm = load_panels()
Rm = Pm.pct_change(fill_method=None)
Uliq = liquid_universe(Pm, Vm, top_n=30, min_adv=5e6, min_hist=250)
rows = []
for a in [c for c in Pm.columns if c not in ("btc", "eth") and Uliq[c].any()]:
    first = Uliq[a].idxmax()
    last = Pm[a].last_valid_index()
    if (last - first).days < 700:
        continue
    Pa, Ra = Pm[[a]], Rm[[a]]                       # single-asset frames (fast)
    U1 = Pa.notna()
    Wa = build_weights_bagged(Pa, Ra, U1, n_min=1, target_vol=0.35, max_lev=1.5,
                              engine_kwargs=dict(tier1_bps=25.0, other_bps=25.0, fin_rate=0.10, band=0.05))
    ra = run_backtest(Wa, Ra, tier1_bps=25.0, other_bps=25.0, band=0.05).loc[first:last]
    bh = Ra[a].loc[first:last].fillna(0.0)
    sr, sb = M.summary(ra), M.summary(bh)
    rows.append(dict(asset=a, start=str(first.date()), end=str(last.date()), rule_sharpe=sr["sharpe"], bh_sharpe=sb["sharpe"],
                     rule_dd=sr["maxdd"], bh_dd=sb["maxdd"], rule_cagr=sr["cagr"], bh_cagr=sb["cagr"]))
G = pd.DataFrame(rows).set_index("asset")
print(G.round(2).to_string())
imp = (G.rule_sharpe > G.bh_sharpe).mean()
ddimp = (G.rule_dd > G.bh_dd).mean()
print(f"\n{len(G)} assets: rule Sharpe > buy&hold in {imp*100:.0f}% | rule maxDD shallower in {ddimp*100:.0f}% | "
      f"median Sharpe rule {G.rule_sharpe.median():.2f} vs B&H {G.bh_sharpe.median():.2f} | median maxDD rule {G.rule_dd.median()*100:.0f}% vs {G.bh_dd.median()*100:.0f}%")
OUT["G_cross_asset"] = dict(n=len(G), frac_sharpe_improved=float(imp), frac_dd_improved=float(ddimp),
                            median_rule_sharpe=float(G.rule_sharpe.median()), median_bh_sharpe=float(G.bh_sharpe.median()),
                            median_rule_dd=float(G.rule_dd.median()), median_bh_dd=float(G.bh_dd.median()),
                            median_rule_cagr=float(G.rule_cagr.median()), median_bh_cagr=float(G.bh_cagr.median()))
G.to_csv("results/cross_asset.csv")

# ------------------------------------------------------------------ H
header("H. Trial-adjusted Sharpe (Bailey & Lopez de Prado deflated Sharpe)")
H = {}
for n_trials, sd in [(1, 0.0), (20, 0.25), (60, 0.25), (150, 0.25), (150, 0.5)]:
    if n_trials == 1:
        v = M.probabilistic_sharpe(r, 0.0)
        print(f"PSR(SR>0)                       = {v:.3f}")
    else:
        v = M.deflated_sharpe(r, n_trials, trial_sr_var=sd ** 2)
        print(f"DSR, {n_trials:3d} trials, Sharpe sd {sd:.2f}  = {v:.3f}")
    H[f"{n_trials}_{sd}"] = v
OUT["H_deflated"] = H

with open("results/validation.json", "w") as f:
    json.dump(OUT, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
print("\nsaved results/validation.json")
