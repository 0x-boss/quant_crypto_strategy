"""Validation of TrendCore-I (+carry).  python validate_intraday.py -> results/validation_intraday.json"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from quant import metrics as M
from quant.intraday import IConfig, run, hourly_prices
px = hourly_prices()
OUT = {}
def fmt(s): return f"CAGR {s['cagr']*100:5.1f}%  Sharpe {s['sharpe']:4.2f}  Sortino {s['sortino']:4.2f}  MaxDD {s['maxdd']*100:6.1f}%  Vol {s['vol']*100:4.1f}%"
def hdr(t): print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)
base = run(IConfig(), px)
r = base["total"]; END = r.index[-1]
hdr("A. Windows (nothing re-fitted)")
A = {}
for lab, a, b in [("2020-03..2026-10", "2020-03-01", None), ("2021+", "2021-01-01", None), ("2022+", "2022-01-01", None), ("2023+", "2023-01-01", None),
                  ("2024+", "2024-01-01", None), ("NEW forward OOS 2026-05-24..", "2026-05-24", None)]:
    s = M.summary(r.loc[a:b]); A[lab] = s; print(f"{lab:30s}: {fmt(s)}")
for y in range(2020, 2027):
    g = r.loc[str(y)]; print(f"  {y}: ret {((1+g).prod()-1)*100:6.1f}%  Sharpe {(g.mean()/g.std()*np.sqrt(365)) if len(g)>60 else float('nan'):5.2f}  maxDD {M.max_drawdown(g)*100:6.1f}%")
for lab, c in [("trend only", IConfig(carry=False)), ("daily bars (k=1) + carry", IConfig(k=1)), ("close-to-close vol", IConfig(rv_vol=False))]:
    s = M.summary(run(c, px)["total"].loc["2020-03-01":]); s22 = M.summary(run(c, px)["total"].loc["2022-01-01":]); A[lab] = (s, s22)
    print(f"{lab:30s}: full Sharpe {s['sharpe']:4.2f} CAGR {s['cagr']*100:5.1f}% | 2022+ Sharpe {s22['sharpe']:4.2f} CAGR {s22['cagr']*100:5.1f}%")
OUT["A"] = A
hdr("B. Stress")
B = {}
for lab, kw in [("costs x2 (20bp)", dict(tier1_bps=20.0)), ("costs x4 (40bp)", dict(tier1_bps=40.0)), ("+1 bar lag (6h)", dict(lag_bars=1)), ("+4 bars lag (24h)", dict(lag_bars=4)),
                ("financing 20%", dict(fin_rate=0.20)), ("no leverage (cap 1.0)", dict(max_lev=1.0)), ("x2 costs + 6h lag + 20% fin", dict(tier1_bps=20.0, lag_bars=1, fin_rate=0.20)),
                ("carry margin 50% (conservative)", dict(carry_margin=0.5))]:
    o = run(IConfig(**kw), px)["total"]; s = M.summary(o.loc["2020-03-01":]); s22 = M.summary(o.loc["2022-01-01":]); B[lab] = (s, s22)
    print(f"{lab:32s}: full {fmt(s)} | 2022+ Sharpe {s22['sharpe']:4.2f} CAGR {s22['cagr']*100:5.1f}%")
OUT["B"] = B
hdr("C. Plateau - one change at a time")
C = {}
for lab, kw in [("k=2 (12h)", dict(k=2)), ("k=3 (8h)", dict(k=3)), ("k=6 (4h)", dict(k=6)), ("gates (200,)", dict(gates=(200,))), ("gates (100,)", dict(gates=(100,))), ("vol spans (30,)", dict(vol_spans=(30,))),
                ("target vol 0.25", dict(target_vol=0.25)), ("target vol 0.30", dict(target_vol=0.30)), ("target vol 0.40", dict(target_vol=0.40)), ("max lev 1.0", dict(max_lev=1.0)), ("max lev 2.0", dict(max_lev=2.0)), ("band 0", dict(band=0.0))]:
    o = run(IConfig(**kw), px)["total"]; s = M.summary(o.loc["2020-03-01":]); s22 = M.summary(o.loc["2022-01-01":]); C[lab] = (s, s22)
    print(f"{lab:22s}: full {fmt(s)} | 2022+ Sharpe {s22['sharpe']:4.2f}")
OUT["C"] = C
sh = np.array([v[0]["sharpe"] for v in C.values()]); sh22 = np.array([v[1]["sharpe"] for v in C.values()])
print(f"-> full Sharpe over {len(sh)} neighbours: min {sh.min():.2f} median {np.median(sh):.2f} max {sh.max():.2f} | 2022+ Sharpe: min {sh22.min():.2f} median {np.median(sh22):.2f} max {sh22.max():.2f}")
hdr("D. Placebo: circularly shift the target WEIGHTS against asset returns at bar level (timing destroyed, exposure profile kept)")
from quant.intraday import target_weights_bars
from quant.engine import run_backtest
P_, R_, W_, ek_ = target_weights_bars(px, IConfig())
def total_from(Wx):
    rr, dd = run_backtest(Wx, R_, return_details=True, **ek_)
    dly = lambda x: (1 + x).groupby(x.index.date).prod() - 1
    t = dly(rr); t.index = pd.to_datetime(t.index); g = dd["gross"].groupby(dd.index.date).mean(); g.index = pd.to_datetime(g.index)
    car_ = base["carry"].reindex(t.index).fillna(0.0)
    return (t + (1 - g.clip(upper=1.0)).reindex(t.index).fillna(1.0) * car_).loc["2020-03-01":]
true = M.summary(total_from(W_))["sharpe"]
rng = np.random.default_rng(5); sh_p = []
for kshift in rng.integers(4 * 120, len(W_) - 4 * 120, 120):
    Ws = pd.DataFrame(np.roll(W_.values, kshift, axis=0), index=W_.index, columns=W_.columns)
    sh_p.append(M.summary(total_from(Ws))["sharpe"])
sh_p = np.array(sh_p)
print(f"true Sharpe {true:.2f} | placebo mean {sh_p.mean():.2f}, p95 {np.quantile(sh_p,.95):.2f}, max {sh_p.max():.2f}, p-value {(sh_p>=true).mean():.3f} (120 shifts)")
OUT["D"] = dict(true=true, mean=float(sh_p.mean()), p95=float(np.quantile(sh_p, .95)), max=float(sh_p.max()), p=float((sh_p >= true).mean()))
hdr("E. Block bootstrap Sharpe 90% CI")
for lab, a in [("2020-03+", "2020-03-01"), ("2022+", "2022-01-01")]:
    b = M.block_bootstrap_sharpe(r.loc[a:], 2000, 20, seed=4)
    print(f"{lab}: [{np.quantile(b,.05):.2f}, {np.quantile(b,.95):.2f}]  P(>1)={np.mean(b>1):.2f} P(>1.5)={np.mean(b>1.5):.2f}")
    OUT[f"E_{lab}"] = [float(np.quantile(b, .05)), float(np.quantile(b, .95)), float(np.mean(b > 1)), float(np.mean(b > 1.5))]
json.dump(OUT, open("results/validation_intraday.json", "w"), indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
pd.DataFrame({"trend": base["trend"], "carry": base["carry"], "gross": base["gross"], "total": r}).to_csv("results/intraday_daily.csv")
print("\nsaved results/validation_intraday.json")
