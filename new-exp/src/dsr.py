"""Deflated Sharpe Ratio of the finalists with the trial dispersion taken from the comparable family (liquid-pool daily strategies),
N = number of registered configs (conservative: trials are strongly correlated).   python src/dsr.py"""
import json, numpy as np, pandas as pd
import lib
t = pd.read_csv(f"{lib.RES}/trials.csv")
fam = t[t.family.isin(["A_daily_rank", "G_composite", "B_slots", "D_ml"])].dropna(subset=["dev_sharpe"])
var_ann = fam.dev_sharpe.var()
streams = pd.read_pickle(f"{lib.RES}/final_streams.pkl")["base"]
out = {"n_family": len(fam), "n_all": len(t), "annual_sharpe_std_across_family": float(np.sqrt(var_ann))}
for k in ["BLEND9_EW", "MOM_6_1_k10", "COMP_C2_k25_h21_top1000", "MOM_res12_1_k25"]:
    r = streams[k].loc["2012":]
    for part in ("dev", "hold", "all"):
        for n_name, n in (("N_family", len(fam)), ("N_all", len(t))):
            sr0_ann = lib.expected_max_sr(n, var_ann / 252) * np.sqrt(252)
            out[f"{k}|{part}|{n_name}"] = dict(dsr=lib.deflated_sharpe(lib.split(r, part), n, var_ann / 252), sr0_annual=float(sr0_ann),
                                              sharpe=lib.perf(lib.split(r, part))["sharpe"])
json.dump(out, open(f"{lib.RES}/final_dsr.json", "w"), indent=1, default=float)
for k, v in out.items():
    print(k, {a: round(b, 3) for a, b in v.items()} if isinstance(v, dict) else v)
