"""Round 8 stage 2: pre-registered factorial stack (see R8_STACK_PLAN.md).   python src/r8_stack.py"""
import json
import numpy as np
import pandas as pd

import lib
import r8_common as R
import r8_f_sizing as SZ
import r8_f_voltarget as VT

ANN = np.sqrt(252)
P_VCAP = next(p for p in SZ.CONFIGS if p["name"] == "eq_k10_vcap100")
P_EXP80 = next(p for p in VT.CONFIGS if p["name"] == "own20_exp80")


def roll_scale(c, W, q=0.8, win=504, minp=252):
    r = R.run(c, W)
    sig = R.lagged(r).rolling(20).std() * ANN
    star = sig.rolling(win, min_periods=minp).quantile(q)
    s = (star / sig).replace([np.inf, -np.inf], np.nan).clip(upper=1.0).fillna(1.0)
    return W.mul(s.reindex(W.index).fillna(1.0), axis=0)


def select_layer(c, A):
    return R.base_weights(c) if A == "base" else SZ.weights(c, P_VCAP)


def expo_layer(c, W, B):
    if B == "none":
        return W
    if B == "exp80":
        return VT.apply(c, W, P_EXP80)
    if B == "rollq80":
        return roll_scale(c, W)
    raise ValueError(B)


def dest_layer(c, W, C):
    if C == "cash":
        return W
    W2 = W.copy()
    free = (1.0 - W.fillna(0.0).sum(axis=1)).clip(lower=0.0)
    W2["BIL"] = free.where(c["O"]["BIL"].reindex(W.index).notna(), 0.0)
    return W2


def build(c, cell):
    A, B, C = cell
    return dest_layer(c, expo_layer(c, select_layer(c, A), B), C)


CELLS = [("base", "none", "cash"), ("vcap", "none", "cash")]
for A in ("base", "vcap"):
    for B in ("exp80", "rollq80"):
        for C in ("cash", "BIL"):
            CELLS.append((A, B, C))
name = lambda cell: "+".join(cell[:2]) + ("+BIL" if cell[2] == "BIL" else "")

EXWIN = ("2021-02-12", "2021-05-10")


def sharpe_ex(r, a, b, win=EXWIN):
    x = r.loc[a:b]
    x = x[(x.index < pd.Timestamp(win[0])) | (x.index > pd.Timestamp(win[1]))]
    return x.mean() / x.std() * ANN


if __name__ == "__main__":
    c = R.load()
    W0 = R.base_weights(c)
    r0 = R.run(c, W0)
    base = R.evaluate(r0, "BASELINE")
    base2x = R.run(c, W0, fee=2 * R.FEE)
    b2x = sharpe_ex(base2x, *lib.DEV) if False else base2x.loc[lib.DEV[0]:lib.DEV[1]].mean() / base2x.loc[lib.DEV[0]:lib.DEV[1]].std() * ANN
    bex = sharpe_ex(r0, *lib.DEV)
    rows, rets = [], {}
    for cell in CELLS:
        W = build(c, cell)
        r = R.run(c, W)
        r2 = R.run(c, W, fee=2 * R.FEE)
        st = R.log_trial("stack", name(cell), dict(A=cell[0], B=cell[1], C=cell[2]), r)
        obj = R.dev_objective(st, base)
        d2 = r2.loc[lib.DEV[0]:lib.DEV[1]]
        s2 = d2.mean() / d2.std() * ANN
        sex = sharpe_ex(r, *lib.DEV)
        elig = np.isfinite(obj) and (s2 >= b2x + 0.05) and (sex >= bex + 0.02)
        rets[name(cell)] = r
        rows.append(dict(cell=name(cell), n_comp=sum([cell[0] != "base", cell[1] != "none", cell[2] != "cash"]), obj=obj, eligible=bool(elig),
                         dev_sh=st["dev"]["sharpe"], dev_med=st["dev"]["med_m"], dev_dd=st["dev"]["maxdd"], dev_sh_2x=s2, dev_sh_exwin=sex,
                         avg_gross=r.attrs["avg_gross"], hold_sh=st["hold"]["sharpe"], hold_med=st["hold"]["med_m"], hold_dd=st["hold"]["maxdd"],
                         hold_cagr=st["hold"]["cagr"], all_dd=st["all"]["maxdd"]))
        print(f"  {name(cell):22s} obj {obj:7.3f} elig {elig}", flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{R.OUTDIR}/stack_table.csv", index=False)
    pd.to_pickle(rets, f"{R.OUTDIR}/stack_returns.pkl")
    pd.set_option("display.width", 250)
    print(f"\nBASELINE DEV Sh {base['dev']['sharpe']:.3f} (2x {b2x:.3f}, ex-window {bex:.3f}) DD {base['dev']['maxdd']:.3f} | HOLD Sh {base['hold']['sharpe']:.3f} med {base['hold']['med_m']:.4f} DD {base['hold']['maxdd']:.3f}")
    print(tab.round(3).to_string(index=False))
    el = tab[tab.eligible].sort_values("obj", ascending=False)
    if el.empty:
        print("NO CELL IS ELIGIBLE")
        json.dump(dict(chosen=None), open(f"{R.OUTDIR}/stack_choice.json", "w"))
    else:
        top = el.iloc[0]
        near = el[el.obj >= top.obj - 0.02].sort_values(["n_comp", "obj"], ascending=[True, False])
        chosen = near.iloc[0]["cell"]
        print("\nCHOSEN (DEV only):", chosen, "| eligible cells:", list(el.cell))
        json.dump(dict(chosen=chosen, eligible=list(el.cell)), open(f"{R.OUTDIR}/stack_choice.json", "w"))
