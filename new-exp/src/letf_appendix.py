"""APPENDIX (violates the spirit of 'no leverage': leveraged / inverse ETFs embed 2-3x daily leverage).
Fixed a priori, simple rules, to show how far embedded leverage alone gets towards the target.   python src/letf_appendix.py"""
import numpy as np, pandas as pd
import lib

T = ["TQQQ", "SQQQ", "SOXL", "SOXS", "UPRO", "TECL", "TNA", "FAS", "SPXL", "QQQ", "SPY"]
d = lib.DATA + "/panels/"
g = lambda c: pd.read_parquet(d + f"raw_{c}.parquet", columns=T)
rc, ac = g("close"), g("adj_close")
f = ac / rc
O, C = g("open") * f, ac
cal = C["SPY"].dropna().index
O, C = O.reindex(cal).loc["2010-06":], C.reindex(cal).loc["2010-06":]
qqq = C["QQQ"]; sma200 = qqq.rolling(200).mean()
bull = (qqq > sma200)
res = {}

def run(W, label):
    r = lib.weights_backtest(W, O[W.columns])
    res[label] = r
    lib.log_trial("H_letf_appendix", label, {}, r)
    lib.print_report(r, label)

z = pd.DataFrame(0.0, index=C.index, columns=T)
W = z.copy(); W["TQQQ"] = bull.astype(float); run(W[["TQQQ"]], "L1 TQQQ when QQQ>SMA200")
W = z.copy(); W["SOXL"] = bull.astype(float); run(W[["SOXL"]], "L2 SOXL when QQQ>SMA200")
W = z.copy(); W["TQQQ"] = bull.astype(float) * 0.5; W["SOXL"] = bull.astype(float) * 0.5; run(W[["TQQQ", "SOXL"]], "L2b 50/50 TQQQ+SOXL when QQQ>SMA200")
lev = ["TQQQ", "SOXL", "UPRO", "TECL", "TNA", "FAS"]
mom = C[lev] / C[lev].shift(63) - 1
ok = (C[lev] > C[lev].rolling(100).mean())
best = mom.where(ok).fillna(-9).idxmax(axis=1).where(ok.any(axis=1))
W = pd.DataFrame(0.0, index=C.index, columns=lev)
for t in lev: W[t] = (best == t).astype(float)
run(W, "L3 top-1 63d momentum of 6 LETFs (own SMA100 filter)")
W = z.copy(); W["TQQQ"] = bull.astype(float); W["SQQQ"] = (~bull).astype(float) * 0.5; run(W[["TQQQ", "SQQQ"]], "L4 TQQQ in bull / 50% SQQQ in bear")
run(pd.DataFrame({"TQQQ": 1.0}, index=C.index), "ref TQQQ buy&hold")
run(pd.DataFrame({"SPY": 1.0}, index=C.index), "ref SPY buy&hold")
pd.DataFrame(res).to_parquet(f"{lib.RES}/letf_appendix.parquet")
