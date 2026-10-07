"""Core library for new-exp: panels, point-in-time universe, engines, metrics, trial registry.

Conventions (read carefully - they are the lookahead guards):
  * Every array is indexed [date, ticker]; row t of a *signal / decision* array uses data through the CLOSE of day t only.
  * A decision made at close t is executed at the OPEN of t+1 (engines do the shift themselves).
  * Prices are dividend/split adjusted (adj_close/close factor applied to O, H, L). Dollar volume uses raw close x volume.
  * Fees are charged per side on traded notional (default 0.001 = 0.1 %  ->  0.2 % round trip).
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from numba import njit
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

FEE = 0.001  # per side
DEV = ("2012-01-01", "2021-12-31")
HOLD = ("2022-01-01", "2026-12-31")


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def stock_meta():
    stk = json.load(open(os.path.join(DATA, "binance_stocks.json")))["data"]
    m = pd.DataFrame(stk).set_index("s")
    return m[~m.index.duplicated()]


def build_panels(force=False):
    """Concatenate downloaded chunks into wide parquet panels (cached)."""
    d = os.path.join(DATA, "panels")
    os.makedirs(d, exist_ok=True)
    if not force and os.path.exists(os.path.join(d, "close.parquet")):
        return
    parts = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(DATA, "daily_raw", "chunk_*.parquet")))]
    parts = [p for p in parts if len(p)]
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df[df["date"] >= "2005-01-01"].drop_duplicates(["date", "ticker"])
    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        w = df.pivot(index="date", columns="ticker", values=c).sort_index()
        w.to_parquet(os.path.join(d, f"raw_{c}.parquet"))
    print("panels built", w.shape)


def load_panels(start="2008-01-01", cols="research"):
    """Return dict of adjusted O,H,L,C and raw close / volume / dollar volume (DataFrames date x ticker).
    cols="research": only tickers that were *ever* inside the point-in-time top-1000 by dollar volume (a superset of every
    universe used; names never that liquid can never be selected, so this subsetting changes no result - it only saves RAM).
    cols=None: all tickers."""
    d = os.path.join(DATA, "panels")
    if cols == "research":
        cols = research_cols()
    g = lambda c: pd.read_parquet(os.path.join(d, f"raw_{c}.parquet"), columns=cols)
    rc, ac = g("close"), g("adj_close")
    f = (ac / rc).where(rc > 0)
    O, H, L = g("open") * f, g("high") * f, g("low") * f
    C = ac.where(rc > 0)
    V = g("volume")
    P = dict(O=O, H=H, L=L, C=C, rawC=rc, V=V)
    # kill bad prints: a >50 % one-day move that is reversed (within 10 %) the next day
    r = C / C.shift(1) - 1
    rn = C.shift(-1) / C - 1
    bad = (r.abs() > 0.5) & (((1 + r) * (1 + rn) - 1).abs() < 0.1)
    for k in ("O", "H", "L", "C", "rawC"):
        P[k] = P[k].mask(bad)
    ok = (P["O"] > 0) & (P["H"] > 0) & (P["L"] > 0) & (P["C"] > 0) & (P["H"] >= P["L"] * 0.999)
    for k in ("O", "H", "L", "C", "rawC"):
        P[k] = P[k].where(ok)
    P["DV"] = (P["rawC"] * V).where(V > 0)
    # restrict dates to genuine trading days (days on which SPY printed)
    cal = P["C"]["SPY"].dropna().index if "SPY" in P["C"] else P["C"].dropna(how="all").index
    cal = cal[cal >= pd.Timestamp(start)]
    for k in P:
        P[k] = P[k].reindex(cal)
    return P


def research_cols():
    fn = os.path.join(DATA, "research_cols.txt")
    if os.path.exists(fn):
        return [x.strip() for x in open(fn) if x.strip()]
    P = load_panels(start="2005-01-01", cols=None)
    M = pit_universe(P, top_n=1000)
    E = pit_universe(P, top_n=150, kinds=("ETF",))
    cols = sorted(set(M.columns[M.any()]) | set(E.columns[E.any()]) | {"SPY", "QQQ", "IWM", "DIA", "TLT", "GLD"})
    open(fn, "w").write("\n".join(cols))
    return cols


def excluded_tickers():
    """Leveraged / inverse / single-stock-ETF products embed leverage -> excluded from the main pool."""
    m = stock_meta()
    bad = m.index[m["ssi"].isin(["Leveraged ETF", "Inverse ETF", "Single-Stock ETF"])]
    pat = ("2X", "3X", "ULTRA", "BULL", "BEAR", "INVERSE", "LEVERAGED", "DAILY TARGET", "-1X", "-2X", "-3X")
    nm = m["n"].fillna("").str.upper()
    bad2 = m.index[nm.apply(lambda s: any(p in s for p in pat))]
    return set(bad) | set(bad2)


def pit_universe(P, top_n=300, win=63, min_price=5.0, min_hist=252, reselect="M", kinds=("STOCK",),
                 exclude=None):
    """Point-in-time universe. Ranked at each reselection date on data through that day's close and valid for
    decisions made on the following trading days until the next reselection. Returns bool DataFrame."""
    DV, rawC, C = P["DV"], P["rawC"], P["C"]
    meta = stock_meta()
    ok_kind = pd.Series(meta["t"].reindex(C.columns).isin(kinds).values, index=C.columns)
    if exclude is None:
        exclude = excluded_tickers()
    ok_kind &= ~pd.Series(C.columns.isin(list(exclude)), index=C.columns)
    med = DV.rolling(win, min_periods=int(win * 0.6)).median()
    hist = C.notna().cumsum()
    elig = (rawC >= min_price) & (hist >= min_hist) & med.notna() & ok_kind.values[None, :]
    score = med.where(elig)
    idx = score.index
    if reselect == "M":
        key = idx.to_period("M")
    elif reselect == "Q":
        key = idx.to_period("Q")
    elif reselect == "Y":
        key = idx.to_period("Y")
    else:
        raise ValueError
    last_of_period = pd.Series(idx, index=idx).groupby(key).transform("max") == pd.Series(idx, index=idx)
    rank = score.rank(axis=1, ascending=False, method="first")
    sel = ((rank <= top_n) & score.notna()).astype(float)
    sel.loc[~last_of_period.values] = np.nan
    # membership chosen at the last day of a period is valid from the NEXT day on; shift by one row, then ffill
    M = sel.shift(1).ffill().fillna(0.0).astype(bool)
    return M


# ----------------------------------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------------------------------
def monthly(r):
    m = (1 + r).resample("ME").prod() - 1
    n = r.resample("ME").count()
    return m[n >= 15]


def maxdd(r):
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def perf(r, label="", ann=252):
    r = r.dropna()
    if len(r) < 30 or r.std() == 0:
        return dict(label=label, n=len(r), sharpe=np.nan, med_m=np.nan, mean_m=np.nan)
    m = monthly(r)
    eq = (1 + r).prod()
    yrs = len(r) / ann
    dn = r[r < 0].std()
    return dict(label=label, n=len(r), months=len(m),
                sharpe=r.mean() / r.std() * np.sqrt(ann),
                sortino=r.mean() / dn * np.sqrt(ann) if dn and dn > 0 else np.nan,
                cagr=eq ** (1 / yrs) - 1,
                med_m=m.median(), mean_m=m.mean(), std_m=m.std(),
                pos_m=(m > 0).mean(), gt5_m=(m > 0.05).mean(),
                maxdd=maxdd(r), vol=r.std() * np.sqrt(ann),
                skew=sps.skew(r), kurt=sps.kurtosis(r, fisher=False))


def fmt(p):
    if p is None or not p or np.isnan(p.get("sharpe", np.nan)):
        return "n/a"
    return (f"Sh {p['sharpe']:.2f} | med/mo {p['med_m']*100:5.2f}% | mean/mo {p['mean_m']*100:5.2f}% | "
            f"CAGR {p['cagr']*100:6.1f}% | DD {p['maxdd']*100:6.1f}% | >5% mo {p['gt5_m']*100:3.0f}% | "
            f"pos mo {p['pos_m']*100:3.0f}% | n {p['n']}")


def split(r, which):
    a, b = {"dev": DEV, "hold": HOLD, "all": ("1990-01-01", "2100-01-01")}[which]
    return r.loc[a:b]


def report(r, label=""):
    out = {}
    for w in ("dev", "hold", "all"):
        out[w] = perf(split(r, w), f"{label}:{w}")
    return out


def print_report(r, label=""):
    o = report(r, label)
    print(f"[{label}]")
    for w in ("dev", "hold", "all"):
        print(f"   {w:5s} {fmt(o[w])}")
    return o


def expected_max_sr(n_trials, var_sr):
    """Expected maximum Sharpe (per-period) of n_trials independent zero-skill strategies (Bailey/Lopez de Prado)."""
    if n_trials <= 1:
        return 0.0
    g = 0.5772156649
    return np.sqrt(var_sr) * ((1 - g) * sps.norm.ppf(1 - 1 / n_trials) + g * sps.norm.ppf(1 - 1 / (n_trials * np.e)))


def deflated_sharpe(r, n_trials, var_sr):
    """Probability that the true Sharpe exceeds the best-of-n_trials noise level. r = daily returns."""
    r = r.dropna()
    T = len(r)
    sr = r.mean() / r.std()
    sk, ku = sps.skew(r), sps.kurtosis(r, fisher=False)
    sr0 = expected_max_sr(n_trials, var_sr)
    den = np.sqrt(max(1e-12, 1 - sk * sr + (ku - 1) / 4 * sr ** 2))
    return float(sps.norm.cdf((sr - sr0) * np.sqrt(T - 1) / den))


def block_bootstrap(r, n=2000, block=21, seed=0, fn=None):
    """Circular block bootstrap of daily returns; returns array of statistic (default annual Sharpe)."""
    rng = np.random.default_rng(seed)
    x = r.dropna().values
    T = len(x)
    nb = int(np.ceil(T / block))
    out = np.empty(n)
    for i in range(n):
        st = rng.integers(0, T, nb)
        idx = (st[:, None] + np.arange(block)[None, :]) % T
        s = x[idx.ravel()[:T]]
        out[i] = fn(s) if fn else s.mean() / s.std() * np.sqrt(252)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# trial registry
# ----------------------------------------------------------------------------------------------------------------------
TRIALS = os.path.join(RES, "trials.csv")


def log_trial(family, name, params, r, note=""):
    """Append one evaluated configuration (always, including the bad ones) to the registry."""
    d = perf(split(r, "dev"))
    h = perf(split(r, "hold"))
    row = dict(family=family, name=name, params=json.dumps(params, default=str),
               dev_sharpe=d.get("sharpe"), dev_med_m=d.get("med_m"), dev_mean_m=d.get("mean_m"),
               hold_sharpe=h.get("sharpe"), hold_med_m=h.get("med_m"), hold_mean_m=h.get("mean_m"), note=note)
    df = pd.DataFrame([row])
    df.to_csv(TRIALS, mode="a", header=not os.path.exists(TRIALS), index=False)


def trial_stats():
    if not os.path.exists(TRIALS):
        return 0, 0.0
    t = pd.read_csv(TRIALS)
    s = t["dev_sharpe"].dropna() / np.sqrt(252)  # per-period
    return len(t), float(s.var()) if len(s) > 2 else 0.0


# ----------------------------------------------------------------------------------------------------------------------
# engines
# ----------------------------------------------------------------------------------------------------------------------
def weights_backtest(W, O, fee=FEE, lag=0):
    """W: DataFrame of target weights, row t = decision at close t (>=0, row sums <= 1).  Executed at open t+1+lag,
    held to the next open.  Returns daily net returns indexed by the execution date (open-to-open)."""
    cols = W.columns
    Wv = W.reindex(O.index).fillna(0.0).values
    Ov = O[cols].values
    T, N = Wv.shape
    Wv = np.vstack([np.zeros((1 + lag, N)), Wv[: T - 1 - lag]])  # position taken at open d = decision at d-1-lag
    assert (Wv.sum(1) <= 1 + 1e-9).all(), "gross exposure > 100 %"
    ratio = np.ones((T, N))
    nxt = Ov[1:] / Ov[:-1]
    nxt = np.where(np.isfinite(nxt), nxt, 1.0)
    ratio[:-1] = nxt
    ret = np.zeros(T)
    turn_series = np.zeros(T)
    drift = np.zeros(N)
    for d in range(T - 1):
        w = Wv[d]
        turn = np.abs(w - drift).sum()
        g = 1.0 - w.sum() + (w * ratio[d]).sum()
        ret[d] = (1 - fee * turn) * g - 1
        turn_series[d] = turn
        drift = w * ratio[d] / g
    out = pd.Series(ret, index=O.index)[:-1]
    out.attrs["turnover"] = pd.Series(turn_series, index=O.index)[:-1]
    return out


@njit(cache=True)
def _slots(O, H, L, C, entry, prio, exit_sig, atr, K, max_hold, stop_atr, trail_atr, tp_atr, fee, hold_gate):
    T, N = O.shape
    cash = 1.0
    shares = np.zeros(K)
    tick = -np.ones(K, dtype=np.int64)
    entry_px = np.zeros(K)
    stop = np.zeros(K)
    tp = np.full(K, np.inf)
    hh = np.zeros(K)
    atr0 = np.zeros(K)
    age = np.zeros(K, dtype=np.int64)
    eq = np.ones(T)
    max_trades = T * K + 10
    tr_i = np.zeros(max_trades, dtype=np.int64)
    tr_e = np.zeros(max_trades, dtype=np.int64)
    tr_x = np.zeros(max_trades, dtype=np.int64)
    tr_r = np.zeros(max_trades)
    nt = 0
    expo = np.zeros(T)
    for d in range(1, T):
        # ---- 1. exits decided at close d-1, executed at open d
        for s in range(K):
            if tick[s] < 0:
                continue
            i = tick[s]
            ex = False
            px = O[d, i]
            if not np.isfinite(px):
                continue
            if exit_sig[d - 1, i] or age[s] >= max_hold:
                ex = True
            elif stop[s] > 0 and px <= stop[s]:
                ex = True  # gapped through the stop
            if ex:
                proceeds = shares[s] * px * (1 - fee)
                cash += proceeds
                tr_i[nt] = i
                tr_e[nt] = d - age[s]
                tr_x[nt] = d
                tr_r[nt] = proceeds / (shares[s] * entry_px[s] / (1 - fee) * 1.0) - 1.0
                nt += 1
                tick[s] = -1
                shares[s] = 0.0
        # ---- 2. entries decided at close d-1, executed at open d
        free = 0
        for s in range(K):
            if tick[s] < 0:
                free += 1
        if free > 0 and hold_gate[d - 1]:
            # equity at this open (mark to open)
            e = cash
            for s in range(K):
                if tick[s] >= 0 and np.isfinite(O[d, tick[s]]):
                    e += shares[s] * O[d, tick[s]]
            # candidates
            cand = np.where(entry[d - 1] & np.isfinite(O[d]) & np.isfinite(prio[d - 1]))[0]
            if cand.size > 0:
                # drop names already held
                keep = np.ones(cand.size, dtype=np.bool_)
                for c in range(cand.size):
                    for s in range(K):
                        if tick[s] == cand[c]:
                            keep[c] = False
                cand = cand[keep]
                if cand.size > 0:
                    order = np.argsort(-prio[d - 1][cand])
                    nfill = min(free, cand.size)
                    for q in range(nfill):
                        i = cand[order[q]]
                        slot = -1
                        for s in range(K):
                            if tick[s] < 0:
                                slot = s
                                break
                        size = min(e / K, cash)
                        if size <= 0:
                            break
                        px = O[d, i]
                        sh = size * (1 - fee) / px
                        cash -= size
                        shares[slot] = sh
                        tick[slot] = i
                        entry_px[slot] = px
                        a = atr[d - 1, i]
                        atr0[slot] = a if np.isfinite(a) else 0.0  # fraction of price
                        stop[slot] = px * (1 - stop_atr * atr0[slot]) if (stop_atr > 0 and atr0[slot] > 0) else 0.0
                        tp[slot] = px * (1 + tp_atr * atr0[slot]) if (tp_atr > 0 and atr0[slot] > 0) else np.inf
                        hh[slot] = px
                        age[slot] = 0
        # ---- 3. intraday: stops / targets (stop checked first = conservative)
        for s in range(K):
            if tick[s] < 0:
                continue
            i = tick[s]
            if not (np.isfinite(L[d, i]) and np.isfinite(H[d, i])):
                continue
            hit = False
            px = 0.0
            if stop[s] > 0 and L[d, i] <= stop[s]:
                hit = True
                px = min(stop[s], O[d, i])
            elif H[d, i] >= tp[s]:
                hit = True
                px = max(tp[s], O[d, i])
            if hit:
                proceeds = shares[s] * px * (1 - fee)
                cash += proceeds
                tr_i[nt] = i
                tr_e[nt] = d - age[s]
                tr_x[nt] = d
                tr_r[nt] = proceeds / (shares[s] * entry_px[s] / (1 - fee)) - 1.0
                nt += 1
                tick[s] = -1
                shares[s] = 0.0
            else:
                if np.isfinite(C[d, i]):
                    if C[d, i] > hh[s]:
                        hh[s] = C[d, i]
                    if trail_atr > 0 and atr0[s] > 0:
                        ns = hh[s] - trail_atr * atr0[s] * entry_px[s]
                        if ns > stop[s]:
                            stop[s] = ns
                age[s] += 1
        # ---- 4. mark to close
        e = cash
        gross = 0.0
        for s in range(K):
            if tick[s] >= 0:
                c = C[d, tick[s]]
                if not np.isfinite(c):
                    c = entry_px[s]
                e += shares[s] * c
                gross += shares[s] * c
        eq[d] = e
        expo[d] = gross / e if e > 0 else 0.0
    return eq, tr_i[:nt], tr_e[:nt], tr_x[:nt], tr_r[:nt], expo


def slots_backtest(P, entry, prio, K=10, exit_sig=None, max_hold=5, stop_atr=0.0, trail_atr=0.0, tp_atr=0.0,
                   atr=None, fee=FEE, gate=None):
    """Event-trade portfolio with K equal slots (each = equity/K at entry; no leverage).
    entry/prio/exit_sig: [date,ticker] decided at the close of the row's date, executed at next open.
    Returns (daily_returns Series, trades DataFrame, exposure Series)."""
    cols = list(P["C"].columns)
    O, H, L, C = (P[k][cols].values for k in "OHLC")
    entry = entry.reindex(index=P["C"].index, columns=cols).fillna(False).values.astype(np.bool_)
    prio = prio.reindex(index=P["C"].index, columns=cols).values.astype(np.float64)
    if exit_sig is None:
        ex = np.zeros_like(entry)
    else:
        ex = exit_sig.reindex(index=P["C"].index, columns=cols).fillna(False).values.astype(np.bool_)
    a = np.zeros_like(O) if atr is None else atr.reindex(index=P["C"].index, columns=cols).values.astype(np.float64)
    g = np.ones(len(P["C"]), dtype=np.bool_) if gate is None else gate.reindex(P["C"].index).fillna(False).values.astype(np.bool_)
    eq, ti, te, tx, tr, expo = _slots(O, H, L, C, entry, prio, ex, a, K, max_hold, stop_atr, trail_atr, tp_atr, fee, g)
    idx = P["C"].index
    r = pd.Series(eq, index=idx).pct_change().fillna(0.0)
    r = r.iloc[1:]
    trades = pd.DataFrame(dict(ticker=np.array(cols)[ti], entry=idx[te], exit=idx[tx], ret=tr))
    return r, trades, pd.Series(expo, index=idx)


# ----------------------------------------------------------------------------------------------------------------------
# feature helpers
# ----------------------------------------------------------------------------------------------------------------------
def atr_pct(P, n=14):
    """ATR as a fraction of close (uses data through close t)."""
    C, H, L = P["C"], P["H"], P["L"]
    pc = C.shift(1)
    tr = np.maximum(np.maximum(H - L, (H - pc).abs()), (L - pc).abs())
    return tr.rolling(n, min_periods=n // 2).mean() / C


def period_end_mask(index, freq="M"):
    """True on the last trading day of each period (decision day for rebalancing)."""
    s = pd.Series(index, index=index)
    return (s.groupby(index.to_period(freq)).transform("max") == s).values


def rebalance(target, freq="M"):
    """Hold the target weights decided on the last trading day of each period until the next period end."""
    t = target.astype(float).copy()
    t.loc[~period_end_mask(target.index, freq)] = np.nan
    return t.ffill().fillna(0.0)
