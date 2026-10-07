"""Round 7 - composite and sector-neutral rank signals, regime switch. Composite recipes are fixed here a priori (equal-weight z-scores).
python src/composites.py"""
import numpy as np, pandas as pd
import lib, signals

P = lib.load_panels()
meta = lib.stock_meta()
sector = meta["ssi"].reindex(P["C"].columns).fillna("NA")
spy = P["C"]["SPY"]
gate = (spy > spy.rolling(200).mean())
F = signals.features(P, mkt=spy.pct_change())
S = signals.build_signal_menu(F)
O = P["O"]
rows = []

def z(df, M):
    d = df.where(M)
    return d.sub(d.mean(axis=1), axis=0).div(d.std(axis=1), axis=0).clip(-3, 3)

def sector_neutral(df, M):
    d = df.where(M)
    out = d.copy()
    for sct in sector.unique():
        cs = sector.index[sector == sct]
        cs = [c for c in cs if c in d.columns]
        if len(cs) < 5: continue
        sub = d[cs]
        out[cs] = sub.sub(sub.mean(axis=1), axis=0)
    return out

for top_n in (300, 1000):
    M = lib.pit_universe(P, top_n)
    cols = list(M.columns[M.any()])
    Mc = M[cols]; Oc = O[cols]
    zz = {k: z(S[k][cols], Mc) for k in ["res_mom_12_1", "mom_6_1", "rev5", "rev_res5", "high52", "lowvol", "clv_high", "rev1_atr", "idiovol_hi", "lowbeta"]}
    comps = {
        "C1_mom+rev": zz["res_mom_12_1"] + zz["rev_res5"],
        "C2_mom+rev+52wh": zz["res_mom_12_1"] + zz["rev_res5"] + zz["high52"],
        "C3_mom+rev+52wh+lowvol": zz["res_mom_12_1"] + zz["rev_res5"] + zz["high52"] + zz["lowvol"],
        "C4_mom6+rev1atr+clv": zz["mom_6_1"] + zz["rev1_atr"] + zz["clv_high"],
        "C5_mom+idiovol": zz["res_mom_12_1"] + zz["idiovol_hi"],
        "SN_mom12": sector_neutral(S["res_mom_12_1"][cols], Mc),
        "SN_mom6": sector_neutral(S["mom_6_1"][cols], Mc),
        "SN_rev5": sector_neutral(S["rev_res5"][cols], Mc),
        "SN_C1": sector_neutral(S["res_mom_12_1"][cols], Mc).pipe(lambda a: z(a, Mc)) + sector_neutral(S["rev_res5"][cols], Mc).pipe(lambda a: z(a, Mc)),
    }
    for name, sg in comps.items():
        for k in (10, 25):
            for h in (5, 21):
                s = sg.where(Mc)
                rank = s.rank(axis=1, ascending=False, method="first")
                W = (((rank <= k) & s.notna()).astype(float) / k).rolling(h, min_periods=1).mean()
                for g_on in (False, True):
                    Wg = W.mul(gate.shift(1).reindex(W.index).fillna(False).astype(float), axis=0) if g_on else W
                    r = lib.weights_backtest(Wg, Oc)
                    lib.log_trial("G_composite", name, dict(top_n=top_n, k=k, h=h, gate=g_on), r)
                    d, o = lib.perf(lib.split(r, "dev")), lib.perf(lib.split(r, "hold"))
                    rows.append(dict(top_n=top_n, sig=name, k=k, h=h, gate=g_on, dev_sh=d["sharpe"], dev_mean=d["mean_m"], dev_med=d["med_m"],
                                     hold_sh=o["sharpe"], hold_mean=o["mean_m"], hold_med=o["med_m"], dev_dd=d["maxdd"], hold_dd=o["maxdd"]))
    print(top_n, "done", flush=True)
df = pd.DataFrame(rows); df.to_csv(f"{lib.RES}/composites.csv", index=False)
pd.set_option("display.width", 220)
print(df.sort_values("dev_sh", ascending=False).head(20).round(3).to_string())
print(df.sort_values("hold_sh", ascending=False).head(15).round(3).to_string())
df["min_sh"] = df[["dev_sh", "hold_sh"]].min(axis=1)
print(df.sort_values("min_sh", ascending=False).head(15).round(3).to_string())
