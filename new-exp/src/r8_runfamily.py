"""Round-8 family runner.   python src/r8_runfamily.py <module> [top_n_for_perturbation_test]

A family module (src/r8_f_<name>.py) defines
    FAMILY   : short id string
    CONFIGS  : list of dict(name=..., **params)  -- the pre-registered grid (everything in it is run and logged)
    apply(c, W, p) -> W2      overlay on weights W (decision-day weights, see r8_common conventions), p = one CONFIGS entry
  and, for signal/selection families only, instead of apply():  weights(c, p) -> W   (build the whole book from scratch).
The runner: loads the cache, builds the baseline, runs EVERY config, logs every config to results/r8/trials_<FAMILY>.csv, ranks them with the
pre-registered DEV-only objective (r8_common.dev_objective), runs the perturbation look-ahead guard on the best finite-scoring configs,
writes results/r8/<FAMILY>_table.csv and <FAMILY>_summary.json and prints them.
"""
import importlib
import json
import sys
import time
import traceback

import numpy as np
import pandas as pd

import r8_common as R

modname = sys.argv[1]
topn = int(sys.argv[2]) if len(sys.argv) > 2 else 3
mod = importlib.import_module(modname)
fam = mod.FAMILY
c = R.load()
W0 = R.base_weights(c)
r0 = R.run(c, W0)
base = R.evaluate(r0, "BASELINE")


def build(cc, p):
    if hasattr(mod, "weights"):
        return mod.weights(cc, p)
    return mod.apply(cc, R.base_weights(cc), p)


rows, t0 = [], time.time()
for p in mod.CONFIGS:
    try:
        W = build(c, p)
        assert (W.sum(axis=1) <= 1 + 1e-6).all(), "gross exposure > 100 %"
        assert (W.fillna(0) >= -1e-12).all().all(), "negative weights (short) not allowed"
        r = R.run(c, W)
        st = R.log_trial(fam, p["name"], {k: v for k, v in p.items() if k != "name"}, r)
        score = R.dev_objective(st, base)
        rows.append(dict(name=p["name"], score=score, avg_gross=r.attrs["avg_gross"],
                         turn=float(r.attrs["turnover"].loc["2012":].mean()), **{f"{k}_{m}": st[k][m] for k in ("dev", "hold", "all") for m in
                         ("sharpe", "med_m", "mean_m", "maxdd", "calmar", "worst_m", "gt5")}))
    except Exception as e:  # noqa
        print("CONFIG FAILED", p.get("name"), repr(e)[:200])
        traceback.print_exc(limit=2)
        rows.append(dict(name=p["name"], score=-np.inf, error=repr(e)[:200]))
    print(f"  done {p['name']:40s} {time.time()-t0:5.0f}s", flush=True)

tab = pd.DataFrame(rows).sort_values("score", ascending=False)
tab.to_csv(f"{R.OUTDIR}/{fam}_table.csv", index=False)
pd.set_option("display.width", 250)
cols = ["name", "score", "dev_sharpe", "dev_med_m", "dev_maxdd", "hold_sharpe", "hold_med_m", "hold_maxdd", "all_sharpe", "all_maxdd", "avg_gross", "turn"]
print("\n=== ranked by pre-registered DEV objective (HOLD shown for information, never used to select) ===")
print(tab[[x for x in cols if x in tab.columns]].round(3).to_string(index=False))
print("\nBaseline:", R.fmt_row("baseline", base))

summary = dict(family=fam, n_configs=len(mod.CONFIGS), baseline={k: base[k] for k in base}, top=[])
for _, row in tab.head(topn).iterrows():
    if not np.isfinite(row["score"]):
        continue
    p = next(q for q in mod.CONFIGS if q["name"] == row["name"])
    print(f"\nperturbation guard for {row['name']}")
    ok, worst = R.check_perturbation(lambda cc: build(cc, p), c, dates=("2014-03-14", "2017-10-02", "2020-06-17", "2022-06-14", "2025-08-12"))
    summary["top"].append(dict(name=row["name"], params=p, dev_score=float(row["score"]), lookahead_guard_ok=bool(ok), worst_diff=worst,
                               stats={k: row[k] for k in row.index if k.startswith(("dev_", "hold_", "all_"))}))
json.dump(summary, open(f"{R.OUTDIR}/{fam}_summary.json", "w"), indent=1, default=float)
print("summary ->", f"{R.OUTDIR}/{fam}_summary.json")
