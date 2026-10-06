"""Validation of the SPOT-ONLY preset (quant.intraday.spot_only).  python validate_spot.py"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(__file__))
from quant import metrics as M
from quant.intraday import spot_only, run, target_weights_bars, hourly_prices
from quant.engine import run_backtest
px = hourly_prices(); OUT = {}
def fmt(s): return f"CAGR {s['cagr']*100:5.1f}%  Sharpe {s['sharpe']:4.2f}  Sortino {s['sortino']:4.2f}  MaxDD {s['maxdd']*100:6.1f}%  Vol {s['vol']*100:4.1f}%"
def hdr(t): print("\n" + "="*95 + f"\n{t}\n" + "="*95)
base = run(spot_only(), px); r = base["total"]
hdr("A. Windows")
for lab, a in [("2020-03+", "2020-03-01"), ("2021+", "2021-01-01"), ("2022+", "2022-01-01"), ("2023+", "2023-01-01"), ("forward OOS 2026-05-24+", "2026-05-24")]:
    print(f"{lab:26s}: {fmt(M.summary(r.loc[a:]))}")
for y in range(2020, 2027):
    g = r.loc[str(y)]; print(f"  {y}: {((1+g).prod()-1)*100:6.1f}%  maxDD {M.max_drawdown(g)*100:6.1f}%")
print("with 4% idle-cash stablecoin yield:", fmt(M.summary(run(spot_only(cash_rate=0.04), px)["total"].loc["2020-03-01":])), "| 2022+:", fmt(M.summary(run(spot_only(cash_rate=0.04), px)["total"].loc["2022-01-01":])))
hdr("B. Stress")
for lab, kw in [("costs x2", dict(tier1_bps=20.0)), ("costs x4", dict(tier1_bps=40.0)), ("+6h lag", dict(lag_bars=1)), ("+24h lag", dict(lag_bars=4)), ("x2 costs + 24h lag", dict(tier1_bps=20.0, lag_bars=4))]:
    o = run(spot_only(**kw), px)["total"]; print(f"{lab:22s}: full {fmt(M.summary(o.loc['2020-03-01':]))} | 2022+ Sharpe {M.summary(o.loc['2022-01-01':])['sharpe']:4.2f}")
hdr("C. Plateau (one change at a time)")
sh, sh22 = [], []
for lab, kw in [("k=1 (daily bars)", dict(k=1)), ("k=2", dict(k=2)), ("k=3", dict(k=3)), ("k=6", dict(k=6)), ("gates (200,)", dict(gates=(200,))), ("gates (100,)", dict(gates=(100,))), ("vol spans (30,)", dict(vol_spans=(30,))),
                ("target vol 0.25", dict(target_vol=0.25)), ("target vol 0.60", dict(target_vol=0.60)), ("OI strength 0", dict(oi_strength=0.0)), ("OI strength 1.0", dict(oi_strength=1.0)), ("band 0", dict(band=0.0)), ("BTC only", dict(assets=("btc",)))]:
    o = run(spot_only(**kw), px if "assets" not in kw else None)["total"]; a, b = M.summary(o.loc["2020-03-01":]), M.summary(o.loc["2022-01-01":]); sh.append(a["sharpe"]); sh22.append(b["sharpe"])
    print(f"{lab:20s}: full {fmt(a)} | 2022+ Sharpe {b['sharpe']:4.2f}")
print(f"-> Sharpe over {len(sh)} neighbours: full min {min(sh):.2f} med {np.median(sh):.2f} max {max(sh):.2f} | 2022+ min {min(sh22):.2f} med {np.median(sh22):.2f} max {max(sh22):.2f}")
hdr("D. Placebo: weights shifted against returns")
P_, R_, W_, ek_ = target_weights_bars(px, spot_only())
def tot(Wx):
    rr = run_backtest(Wx, R_, **ek_); d = (1 + rr).groupby(rr.index.date).prod() - 1; d.index = pd.to_datetime(d.index); return d.loc["2020-03-01":]
true = M.summary(tot(W_))["sharpe"]; rng = np.random.default_rng(3); pl = []
for k in rng.integers(480, len(W_) - 480, 120): pl.append(M.summary(tot(pd.DataFrame(np.roll(W_.values, k, axis=0), index=W_.index, columns=W_.columns)))["sharpe"])
pl = np.array(pl); print(f"true Sharpe {true:.2f} | placebo mean {pl.mean():.2f} p95 {np.quantile(pl,.95):.2f} max {pl.max():.2f} p={(pl>=true).mean():.3f}")
hdr("E. Bootstrap")
for lab, a in [("2020-03+", "2020-03-01"), ("2022+", "2022-01-01")]:
    b = M.block_bootstrap_sharpe(r.loc[a:], 2000, 20, seed=4); print(f"{lab}: Sharpe 90% CI [{np.quantile(b,.05):.2f}, {np.quantile(b,.95):.2f}]  P(>1)={np.mean(b>1):.2f}")
pd.DataFrame({"ret": r, "gross": base["gross"]}).to_csv("results/spot_only_daily.csv")
print("\nsaved")
