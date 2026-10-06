"""Market-neutral cross-sectional perp sleeve (see research/PREREGISTRATION.md).  Daily USDT-perp data incl. delisted symbols."""
from __future__ import annotations
import glob, os
import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(__file__), "..", "data", "perp_daily")
# non-crypto / pegged contracts (equity, commodity, FX, stable) excluded from the universe
NON_CRYPTO = set("""AAPLUSDT ADBEUSDT AAOIUSDT ACNUSDT AGPUUSDT AGTUSDT TSLAUSDT NVDAUSDT AMZNUSDT GOOGLUSDT METAUSDT MSFTUSDT COINUSDT MSTRUSDT
SNDKUSDT SOXLUSDT SPCXUSDT SKHYNIXUSDT SKHYUSDT MUUSDT KORUUSDT CRCLUSDT INTCUSDT MRVLUSDT EWYUSDT SAMSUNGUSDT DRAMUSDT SNXXUSDT PAXGUSDT
XAUUSDT XAGUSDT XPTUSDT XPDUSDT CLUSDT BZUSDT NATGASUSDT EURUSDT GBPUSDT USDCUSDT BUSDUSDT TUSDUSDT FDUSDUSDT USDPUSDT SPXUSDT QQQUSDT""".split())


def load_panels():
    C, QV, F = {}, {}, {}
    for f in glob.glob(f"{D}/*.parquet"):
        name = os.path.basename(f)[:-8]
        if name.endswith("_funding"):
            F[name[:-8]] = pd.read_parquet(f)["fundingRate"].resample("1D").sum()
        else:
            d = pd.read_parquet(f)
            C[name] = d["c"]; QV[name] = d["qv"]
    C = pd.DataFrame(C).sort_index(); QV = pd.DataFrame(QV).reindex(C.index); F = pd.DataFrame(F).reindex(index=C.index, columns=C.columns)
    return C, QV, F


def universe(C, QV, top_n=20, min_hist=90):
    adv = QV.rolling(30, min_periods=20).median()
    ok = C.notna().cumsum() >= min_hist
    adv = adv.where(ok & C.notna())
    adv = adv.drop(columns=[c for c in adv.columns if c in NON_CRYPTO])
    rk = adv.rank(axis=1, ascending=False)
    return (rk <= top_n)


def scores(C, F, name):
    r1 = C.pct_change(fill_method=None)
    if name == "MOM30": return C / C.shift(30) - 1
    if name == "MOM90": return C / C.shift(90) - 1
    if name == "REV7": return -(C / C.shift(7) - 1)
    if name == "LOWVOL": return -r1.rolling(30, min_periods=20).std()
    if name == "FUND14": return -F.rolling(14, min_periods=10).mean()
    if name == "MULTI":
        return sum(scores(C, F, n).rank(axis=1, pct=True) for n in ("MOM30", "MOM90", "REV7", "LOWVOL", "FUND14")) / 5
    raise ValueError(name)


def run_sleeve(C, QV, F, name, top_n=20, quint=0.2, cost_bps=10.0, target_vol=0.20, max_lev=2.0, rebal_weekday=6, start="2020-12-01"):
    """Decision at close of day t (weekday==6 = Sunday, i.e. executed 00:00 Monday); positions held until next rebalance."""
    U = universe(C, QV, top_n)
    S = scores(C, F, name).where(U)
    R = C.pct_change(fill_method=None)
    Rv = R.fillna(0.0).values; Fv = F.fillna(0.0).values
    idx = C.index; T, N = Rv.shape
    w = np.zeros(N); ret = np.zeros(T); cost = np.zeros(T); fund = np.zeros(T); gross = np.zeros(T)
    raw = np.zeros(T)                                     # unscaled sleeve return history for vol scaling
    for t in range(1, T):
        # P&L of positions held over day t (decided earlier)
        pr = float(w @ Rv[t]); fd = -float(w @ Fv[t])
        ret[t] = pr + fd; fund[t] = fd
        V = 1.0 + pr
        if V > 0: w = w * (1 + Rv[t]) / V
        if idx[t].weekday() == rebal_weekday and idx[t] >= pd.Timestamp(start):
            s = S.iloc[t].dropna()
            tgt = np.zeros(N)
            if len(s) >= 10:
                k = max(int(round(len(s) * quint)), 1)
                lo = s.nsmallest(k).index; hi = s.nlargest(k).index
                for c in hi: tgt[C.columns.get_loc(c)] = 0.5 / k
                for c in lo: tgt[C.columns.get_loc(c)] = -0.5 / k
                # vol scaling from trailing 30d of the sleeve's own (unscaled-equivalent) returns
                hist = raw[max(0, t - 30):t]
                if t > 60 and hist.std() > 0:
                    sc = min(max_lev, target_vol / (hist.std() * np.sqrt(365)))
                    tgt = tgt * sc
            cost[t] = np.abs(tgt - w).sum() * cost_bps / 1e4
            ret[t] -= cost[t]; w = tgt
        gross[t] = np.abs(w).sum()
        # unscaled proxy of sleeve return for vol estimation: realised return / current scale (use gross as scale proxy)
        raw[t] = ret[t] / max(gross[t] if gross[t] > 0 else 1.0, 1e-9) if gross[t] > 0 else 0.0
        raw[t] = raw[t] * 1.0
    out = pd.DataFrame({"ret": ret, "funding": fund, "cost": cost, "gross": gross}, index=idx)
    return out.loc[start:]


def run_tsmom(C, QV, F, long_short=True, top_n=20, cost_bps=10.0, target_vol=0.25, asset_vol=0.20, max_lev=2.0, rebal_weekday=6, start="2020-12-01"):
    """Round B: time-series trend across the point-in-time perp universe (see PREREGISTRATION.md)."""
    U = universe(C, QV, top_n)
    R = C.pct_change(fill_method=None)
    sg = (np.sign(C / C.shift(30) - 1) + np.sign(C / C.shift(90) - 1) + np.sign(C / C.shift(180) - 1)) / 3.0
    if not long_short: sg = sg.clip(lower=0)
    vol = R.rolling(30, min_periods=20).std() * np.sqrt(365)
    Rv = R.fillna(0.0).values; Fv = F.fillna(0.0).values
    idx = C.index; T, N = Rv.shape
    w = np.zeros(N); ret = np.zeros(T); cost = np.zeros(T); fund = np.zeros(T); gross = np.zeros(T); raw = np.zeros(T)
    for t in range(1, T):
        pr = float(w @ Rv[t]); fd = -float(w @ Fv[t]); ret[t] = pr + fd; fund[t] = fd
        V = 1.0 + pr
        if V > 0: w = w * (1 + Rv[t]) / V
        if idx[t].weekday() == rebal_weekday and idx[t] >= pd.Timestamp(start):
            u = U.iloc[t].values & np.isfinite(sg.iloc[t].values) & np.isfinite(vol.iloc[t].values)
            tgt = np.zeros(N)
            if u.sum() >= 5:
                tgt[u] = sg.iloc[t].values[u] * (asset_vol / vol.iloc[t].values[u]) / u.sum()
                hist = raw[max(0, t - 30):t]
                if t > 60 and hist.std() > 0: tgt = tgt * min(max_lev, target_vol / (hist.std() * np.sqrt(365)))
            cost[t] = np.abs(tgt - w).sum() * cost_bps / 1e4; ret[t] -= cost[t]; w = tgt
        gross[t] = np.abs(w).sum()
        raw[t] = ret[t] / gross[t] * 0.5 if gross[t] > 0 else 0.0     # rough unscaled-equivalent for vol estimate
    return pd.DataFrame({"ret": ret, "funding": fund, "cost": cost, "gross": gross}, index=idx).loc[start:]
