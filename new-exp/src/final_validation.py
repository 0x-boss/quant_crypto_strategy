"""Final validation of the best honest candidates: stress (costs, lag, reselection), placebo, bootstrap, deflated Sharpe,
survivorship drag, forward window.     python src/final_validation.py   ->  results/final_*.csv / .parquet / .json"""
import json
import time

import numpy as np
import pandas as pd

import lib
import signals
import strategies as st

t0 = time.time()
P = lib.load_panels()
F = signals.features(P, mkt=P["C"]["SPY"].pct_change())
S = signals.build_signal_menu(F)

# ---------------------------------------------------------------- scenarios
scen = {
    "base": dict(),
    "cost_x2 (0.4% RT)": dict(fee=2 * lib.FEE),
    "cost_x4 (0.8% RT)": dict(fee=4 * lib.FEE),
    "lag +1 day": dict(lag=1),
    "reselect quarterly": dict(reselect="Q"),
    "reselect yearly": dict(reselect="Y"),
}
streams = {}
for name, kw in scen.items():
    d = st.build_all(P, F, S, **kw)
    d["BLEND9_EW"] = st.blend(d)
    streams[name] = d
    print(name, f"{time.time()-t0:.0f}s", flush=True)
pd.to_pickle(streams, f"{lib.RES}/final_streams.pkl")

rows = []
for sname, d in streams.items():
    for k, r in d.items():
        r = r.loc["2012":]
        a, b, c = lib.perf(lib.split(r, "dev")), lib.perf(lib.split(r, "hold")), lib.perf(lib.split(r, "all"))
        rows.append(dict(scenario=sname, strategy=k, dev_sh=a["sharpe"], dev_med=a["med_m"], dev_mean=a["mean_m"], hold_sh=b["sharpe"],
                         hold_med=b["med_m"], hold_mean=b["mean_m"], all_sh=c["sharpe"], all_med=c["med_m"], all_mean=c["mean_m"],
                         all_dd=c["maxdd"], all_cagr=c["cagr"]))
tab = pd.DataFrame(rows)
tab.to_csv(f"{lib.RES}/final_scenarios.csv", index=False)
pd.set_option("display.width", 220)
fin = ["BLEND9_EW", "MOM_6_1_k10", "COMP_C2_k25_h21_top1000", "MOM_res12_1_k25"]
print(tab[tab.strategy.isin(fin)].round(3).to_string())

# ---------------------------------------------------------------- bootstrap / DSR / survivorship / forward window
ntr, var_sr = lib.trial_stats()
print("registered trials:", ntr, "var of per-period Sharpe across trials:", var_sr)
out = {}
for k in fin:
    r = streams["base"][k].loc["2012":]
    boot = lib.block_bootstrap(r, n=2000, block=21)
    medb = lib.block_bootstrap(r, n=2000, block=21, fn=lambda s: np.median(pd.Series(s).groupby(np.arange(len(s)) // 21).apply(lambda x: (1 + x).prod() - 1)))
    dsr = lib.deflated_sharpe(lib.split(r, "dev"), ntr, var_sr)
    dsr_h = lib.deflated_sharpe(lib.split(r, "hold"), ntr, var_sr)
    sp = r.loc["2026-06-11":]
    out[k] = dict(sharpe_all=lib.perf(r)["sharpe"], boot_sharpe_p5=float(np.percentile(boot, 5)), boot_sharpe_p95=float(np.percentile(boot, 95)),
                  p_sharpe_gt_1_5=float((boot > 1.5).mean()), p_median_gt_5pct=float((medb > 0.05).mean()),
                  dsr_dev=dsr, dsr_hold=dsr_h, n_trials=ntr,
                  fwd_window=lib.perf(sp) if len(sp) > 20 else None, fwd_ret=float((1 + sp).prod() - 1), fwd_days=len(sp))
json.dump(out, open(f"{lib.RES}/final_stats.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)[:3000])

# ---------------------------------------------------------------- placebo: random ranks (same universe, k, h, costs)
M3 = lib.pit_universe(P, 300)
M10 = lib.pit_universe(P, 1000)
cols = list(M10.columns[M10.any()])
rng = np.random.default_rng(7)
pl = {}
for name, (M, k, h) in {"MOM_6_1_k10": (M3, 10, 5), "COMP_C2_k25_h21_top1000": (M10, 25, 21)}.items():
    sh_dev, sh_hold, sh_all = [], [], []
    for i in range(100):
        rnd = pd.DataFrame(rng.random((len(P["C"]), len(cols))), index=P["C"].index, columns=cols)
        r = lib.weights_backtest(st.topk(rnd, M, k, h, cols), P["O"][cols]).loc["2012":]
        sh_dev.append(lib.perf(lib.split(r, "dev"))["sharpe"]); sh_hold.append(lib.perf(lib.split(r, "hold"))["sharpe"]); sh_all.append(lib.perf(r)["sharpe"])
    act = streams["base"][name].loc["2012":]
    pl[name] = dict(placebo_mean_all=float(np.mean(sh_all)), placebo_max_all=float(np.max(sh_all)), actual_all=lib.perf(act)["sharpe"],
                    p_value_all=float((np.array(sh_all) >= lib.perf(act)["sharpe"]).mean()),
                    placebo_mean_hold=float(np.mean(sh_hold)), actual_hold=lib.perf(lib.split(act, "hold"))["sharpe"],
                    p_value_hold=float((np.array(sh_hold) >= lib.perf(lib.split(act, "hold"))["sharpe"]).mean()))
    print(name, pl[name], flush=True)
json.dump(pl, open(f"{lib.RES}/final_placebo.json", "w"), indent=1)
print("done", f"{time.time()-t0:.0f}s")
