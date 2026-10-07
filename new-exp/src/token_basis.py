"""Exploratory (small-sample!) look at the Binance stock-token vs underlying basis outside US regular hours.
Token 5m klines (data-api.binance.vision) vs Yahoo 5m prepost bars of the underlying, last ~58 days.  python src/token_basis.py"""
import time, requests, numpy as np, pandas as pd, yfinance as yf
import lib

tk = ["NVDA", "TSLA", "MSTR", "AAPL", "SPY", "QQQ", "GOOGL", "AMD", "INTC", "MU", "AVGO", "META", "COIN", "HOOD", "PLTR", "AMZN", "MSFT"]
end = pd.Timestamp.utcnow().floor("5min")
start = end - pd.Timedelta(days=58)
BASE = "https://data-api.binance.vision"

def klines5(sym):
    rows, st = [], int(start.timestamp() * 1000)
    while True:
        r = requests.get(BASE + "/api/v3/klines", params=dict(symbol=sym, interval="5m", startTime=st, limit=1000), timeout=60).json()
        if not r: break
        rows += r; st = r[-1][6] + 1
        if len(r) < 1000: break
    df = pd.DataFrame(rows).iloc[:, :6]; df.columns = ["t", "o", "h", "l", "c", "v"]
    df["t"] = pd.to_datetime(df.t, unit="ms", utc=True)
    return df.set_index("t").astype(float)

y = yf.download(tk, start=start.tz_convert(None).strftime("%Y-%m-%d"), interval="5m", prepost=True, auto_adjust=False,
                progress=False, group_by="ticker")
y.index = y.index.tz_convert("UTC")
rows = []
for t in tk:
    try:
        b = klines5(f"{t}BUSDT")
    except Exception as e:
        print("skip", t, e); continue
    u = y[t]["Close"].dropna()
    j = pd.DataFrame({"tok": b["c"], "und": u}).dropna()
    et = j.index.tz_convert("America/New_York")
    mod = et.hour * 60 + et.minute
    j["day"] = et.normalize().tz_localize(None)
    j["mod"] = mod
    j["basis"] = j.tok / j.und - 1
    # target: token price at the 09:35 ET bar close minus token now (convergence trade), only pre-market observations 04:00-09:25
    open_px = j[(j["mod"] == 9 * 60 + 35)].set_index("day")["tok"]
    und_open = j[(j["mod"] == 9 * 60 + 35)].set_index("day")["und"]
    pm = j[(j["mod"] >= 4 * 60) & (j["mod"] < 9 * 60 + 30)].copy()
    pm["tok_open"] = pm.day.map(open_px)
    pm["und_open"] = pm.day.map(und_open)
    pm["fwd_tok"] = pm.tok_open / pm.tok - 1
    pm["fwd_und"] = pm.und_open / pm.und - 1
    pm["sym"] = t
    rows.append(pm.dropna(subset=["fwd_tok", "basis"]))
d = pd.concat(rows)
d.to_parquet(f"{lib.RES}/token_basis_premarket.parquet")
print(len(d), "premarket observations over", d.day.nunique(), "days,", d.sym.nunique(), "tokens")
print("basis (token/underlying-1) bps: ", d.basis.describe(percentiles=[.05, .25, .5, .75, .95]).mul(1e4).round(1).to_dict())
import statsmodels.api as sm
X = sm.add_constant(d["basis"] * 1e4)
m = sm.OLS(d["fwd_tok"] * 1e4, X).fit(cov_type="cluster", cov_kwds={"groups": d["day"].astype(str) + d["sym"]})
print(m.params.round(3).to_dict(), "t:", m.tvalues.round(2).to_dict())
for lo, hi in [(-1e9, -0.004), (-0.004, -0.002), (-0.002, 0.002), (0.002, 0.004), (0.004, 1e9)]:
    s = d[(d.basis > lo) & (d.basis <= hi)]
    print(f"basis in ({lo*1e4:.0f},{hi*1e4:.0f}] bps: n={len(s):6d}  mean fwd token ret to 09:35 = {s.fwd_tok.mean()*1e4:7.1f} bps   mean fwd underlying = {s.fwd_und.mean()*1e4:7.1f} bps")
