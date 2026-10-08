"""Round-8 shared harness.  EVERY round-8 experiment must go through these functions so that definitions, costs, periods and
look-ahead conventions are identical.

CONVENTIONS (the look-ahead contract)
  * A weights DataFrame W (date x ticker) has, in row t, the target weights DECIDED AT THE CLOSE OF DAY t from data through t.
    They are executed at the OPEN of t+1 and held to the next open.  `run()` does the shift; you never shift W yourself.
  * Any overlay factor / filter / regime flag s[t] that multiplies row t of W must therefore be computed from data through the
    close of day t only (rolling windows ending at t are fine; shift(-1), centered windows, full-sample statistics are NOT).
  * Cost: 0.1 % per side on traded notional (0.2 % round trip) unless a stress test says otherwise.  Gross exposure <= 100 %.
  * DEV = 2012-2021, HOLD = 2022-2026-10.  Select on DEV only.  HOLD is for reporting finalists.
"""
import json
import os

import numpy as np
import pandas as pd

import lib

CACHE = os.path.join(lib.DATA, "r8_cache")
OUTDIR = os.path.join(lib.RES, "r8")
os.makedirs(OUTDIR, exist_ok=True)
FEE = lib.FEE
FEATURES = ["mom_6_1", "mom_12_1", "res_mom_12_1", "high52", "low52", "atr14", "vol20", "vol60", "idio_vol60", "beta60", "sma20", "sma50",
            "sma200", "ret1", "ret5", "ret21", "ret63", "ret126", "rvol", "clv", "gap", "res21"]
PRICES = ["O", "H", "L", "C", "V", "DV", "rawC"]


def load(names=None):
    """Return dict: prices O,H,L,C (adjusted), V, DV (dollar volume), rawC; features (see FEATURES); masks M300, M1000 (bool PIT universes,
    STOCK only, leveraged/inverse excluded); 'vix' (DataFrame ^VIX, ^VIX3M, ffilled to trading days); 'sector' (Series ssi by ticker).
    All frames are float32, index = trading days 2008-2026-10-07, columns = 2596 research tickers (stocks + a few ETFs: SPY QQQ IWM DIA TLT GLD
    BIL SHY ... are in the columns of the price frames).  `names` restricts what is loaded (saves RAM)."""
    want = set(names) if names else set(PRICES + FEATURES + ["M300", "M1000"])
    c = {}
    for k in sorted(want):
        df = pd.read_parquet(os.path.join(CACHE, f"{k}.parquet"))
        c[k] = df.astype(bool) if k.startswith("M") else df
    c["vix"] = pd.read_parquet(os.path.join(CACHE, "vix.parquet"))
    c["sector"] = pd.read_csv(os.path.join(CACHE, "sector.csv"), index_col=0)["ssi"].fillna("NA")
    return c


def base_weights(c, k=10, h=5, sig="mom_6_1", mask="M300"):
    """The baseline: each day buy the top-k of `sig` inside the PIT universe `mask`, equal weight, average the last h daily books."""
    s = c[sig].where(c[mask])
    rank = s.rank(axis=1, ascending=False, method="first")
    sleeve = ((rank <= k) & s.notna()).astype(float) / k
    return sleeve.rolling(h, min_periods=1).mean()


def run(c, W, fee=FEE, lag=0):
    """Backtest target weights W (see conventions).  Returns daily net returns (open-to-open) Series, attrs['turnover'] = daily sum|dw|."""
    cols = [x for x in W.columns if x in c["O"].columns]
    W = W[cols].reindex(c["O"].index).fillna(0.0)
    r = lib.weights_backtest(W, c["O"][cols].astype(np.float64), fee=fee, lag=lag)
    r.attrs["avg_gross"] = float(W.sum(axis=1).loc["2012":].mean())
    return r


def _period(r, a, b):
    x = r.loc[a:b].dropna()
    if len(x) < 60 or x.std() == 0:
        return dict(n=len(x), sharpe=np.nan, med_m=np.nan, mean_m=np.nan, maxdd=np.nan, cagr=np.nan, calmar=np.nan, worst_m=np.nan,
                    gt5=np.nan, vol=np.nan, ulcer=np.nan)
    m = lib.monthly(x)
    eq = (1 + x).cumprod()
    dd = eq / eq.cummax() - 1
    yrs = len(x) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    return dict(n=len(x), sharpe=x.mean() / x.std() * np.sqrt(252), med_m=m.median(), mean_m=m.mean(), maxdd=dd.min(), cagr=cagr,
                calmar=cagr / abs(dd.min()) if dd.min() < 0 else np.nan, worst_m=m.min(), gt5=(m > 0.05).mean(),
                vol=x.std() * np.sqrt(252), ulcer=float(np.sqrt((dd ** 2).mean())))


def evaluate(r, label="", show=True):
    """dict with 'dev', 'hold', 'all' period statistics (Sharpe, median month, max drawdown, ...). Prints a 3-line summary."""
    out = {"dev": _period(r, *lib.DEV), "hold": _period(r, *lib.HOLD), "all": _period(r, "2012-01-01", "2100-01-01")}
    if show:
        print(f"[{label}]")
        for k in ("dev", "hold", "all"):
            p = out[k]
            print(f"   {k:4s} Sh {p['sharpe']:5.2f} | med/mo {p['med_m']*100:5.2f}% | mean/mo {p['mean_m']*100:5.2f}% | >5%mo {p['gt5']*100:3.0f}% | "
                  f"maxDD {p['maxdd']*100:6.1f}% | Calmar {p['calmar']:4.2f} | worst mo {p['worst_m']*100:6.1f}% | vol {p['vol']*100:4.0f}%")
    return out


def log_trial(family, name, params, r, note=""):
    """Append one evaluated configuration to results/r8/trials_<family>.csv (one file per family -> no write races). ALWAYS log, also the bad ones."""
    st = evaluate(r, show=False)
    row = dict(family=family, name=name, params=json.dumps(params, default=str), note=note, avg_gross=r.attrs.get("avg_gross"))
    for part in ("dev", "hold", "all"):
        for k in ("sharpe", "med_m", "mean_m", "maxdd", "calmar", "worst_m", "gt5", "vol", "cagr"):
            row[f"{part}_{k}"] = st[part][k]
    fn = os.path.join(OUTDIR, f"trials_{family}.csv")
    pd.DataFrame([row]).to_csv(fn, mode="a", header=not os.path.exists(fn), index=False)
    return st


def baseline(c=None):
    c = c or load()
    return run(c, base_weights(c))


def dev_objective(st, base):
    """Pre-registered DEV-only selection score.  Higher is better.  A candidate must (i) keep >= 80 % of the baseline DEV mean month and
    >= 85 % of the baseline DEV median month, (ii) improve DEV max drawdown by at least 20 % (relative); then score = DEV Sharpe - 0.5*|DEV maxDD|.
    Otherwise -inf."""
    d, b = st["dev"], base["dev"]
    if not (d["mean_m"] >= 0.8 * b["mean_m"] and d["med_m"] >= 0.85 * b["med_m"] and abs(d["maxdd"]) <= 0.8 * abs(b["maxdd"])):
        return -np.inf
    return d["sharpe"] - 0.5 * abs(d["maxdd"])


def truncate(c, T0):
    """Cut every frame in the cache dict to dates <= T0 (for look-ahead / truncation-invariance tests)."""
    T0 = pd.Timestamp(T0)
    out = {}
    for k, v in c.items():
        out[k] = v.loc[:T0] if isinstance(v, pd.DataFrame) else v
    return out


def check_truncation(build_fn, c, T0="2019-06-28", tol=1e-7):
    """build_fn(c) -> daily return Series computed ONLY from the frames in c.  Compares the series built on the full data with the one built on data
    cut at T0 for all dates <= T0 - 5 trading days.  Differences => your overlay uses information from the future."""
    full = build_fn(c)
    cut = build_fn(truncate(c, T0))
    idx = cut.index[:-5]
    a, b = full.reindex(idx), cut.reindex(idx)
    diff = float((a - b).abs().max())
    ok = diff < tol
    print(f"truncation check at {T0}: max |diff| = {diff:.2e} -> {'OK' if ok else 'LOOK-AHEAD!'}")
    return ok, diff


def month_bootstrap_median_gt(r, thr=0.05, n=4000, seed=0):
    """P(median calendar-month return > thr) under resampling of the months of r (iid months)."""
    m = lib.monthly(r.dropna()).values
    rng = np.random.default_rng(seed)
    med = np.median(rng.choice(m, size=(n, len(m)), replace=True), axis=1)
    return float((med > thr).mean())


def fmt_row(label, st):
    d, h, a = st["dev"], st["hold"], st["all"]
    return (f"{label:44s} DEV Sh {d['sharpe']:.2f} med {d['med_m']*100:4.1f}% DD {d['maxdd']*100:5.1f}% | HOLD Sh {h['sharpe']:.2f} "
            f"med {h['med_m']*100:4.1f}% DD {h['maxdd']*100:5.1f}% | ALL Sh {a['sharpe']:.2f} DD {a['maxdd']*100:5.1f}%")


def lagged(r):
    """Strategy returns KNOWN at the close of day t.  run() indexes a return by the day it starts (open d -> open d+1), so at the close of t the
    latest known return is the one indexed t-1.  Any overlay that uses the strategy's own realised returns MUST use lagged(r), never r."""
    return r.shift(1)


def perturb(c, d0, seed=0, sigma=0.15, sigma_first=0.6):
    """Copy of the cache dict in which EVERY price/feature/VIX frame is multiplied by random log-normal noise on all rows after d0 and every mask is
    inverted after d0.  Anything decided on days <= d0 that changes under this perturbation depends on the future."""
    rng = np.random.default_rng(seed)
    d0 = pd.Timestamp(d0)
    out = {}
    for k, v in c.items():
        if not isinstance(v, pd.DataFrame):
            out[k] = v
            continue
        a = v.copy()
        idx = a.index > d0
        if idx.any():
            if v.dtypes.iloc[0] == bool:
                a.loc[idx] = ~a.loc[idx]
            else:
                sig = np.full((int(idx.sum()), 1), sigma)
                sig[0, 0] = sigma_first            # the very next row is shaken hardest: a one-day look-ahead cannot hide
                noise = np.exp(rng.normal(0, 1, (int(idx.sum()), a.shape[1])) * sig).astype(np.float32)
                a.loc[idx] = a.loc[idx].values * noise
        out[k] = a
    return out


def check_perturbation(build_W, c, dates=("2013-05-21", "2014-03-14", "2016-01-20", "2017-10-02", "2019-12-18", "2020-06-17", "2022-06-14", "2023-02-21", "2025-08-12"), tol=1e-9, verbose=True):
    """THE look-ahead guard.  build_W(c) -> weights DataFrame computed ONLY from the frames in c (recompute everything you use from c, including
    any backtest of the strategy's own returns).  For each date d0 the future (rows > d0) of every frame is destroyed with noise; the weights of
    all rows <= d0 must be unchanged.  Returns (ok, worst_abs_diff)."""
    W = build_W(c)
    worst = 0.0
    for i, d0 in enumerate(dates):
        W2 = build_W(perturb(c, d0, seed=i))
        a, b = W.loc[:d0], W2.reindex(W.index).loc[:d0]
        cols = a.columns.union(b.columns)
        diff = float((a.reindex(columns=cols).fillna(0.0) - b.reindex(columns=cols).fillna(0.0)).abs().to_numpy().max())
        worst = max(worst, diff)
        if verbose:
            print(f"   perturbation after {d0}: max |dW| on earlier rows = {diff:.2e}")
    ok = worst < tol
    print(f"perturbation check: {'OK' if ok else 'LOOK-AHEAD DETECTED'} (worst {worst:.2e})")
    return ok, worst


if __name__ == "__main__":
    c = load()
    r = baseline(c)
    st = evaluate(r, "BASELINE MOM_6_1 top10 stagger5 (top300)")
    print("turnover/day", r.attrs["turnover"].loc["2012":].mean(), "avg gross", r.attrs["avg_gross"])
    ok, d = check_truncation(lambda cc: run(cc, base_weights(cc)), c)
    check_perturbation(base_weights, c)
    print("HOLD P(median>5%) by month bootstrap:", month_bootstrap_median_gt(r.loc[lib.HOLD[0]:lib.HOLD[1]]))
