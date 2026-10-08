"""Round 8 - one-off cache so that parallel research agents do not each rebuild the 6 GB feature set.
Writes data/r8_cache/<name>.parquet (float32 frames, date x ticker) + vix.parquet + sector.csv + meta.json.
    python src/build_r8_cache.py            (needs ~6 GB RAM; run it alone)"""
import json
import os

import numpy as np
import pandas as pd
import yfinance as yf

import lib
import signals

OUT = os.path.join(lib.DATA, "r8_cache")
os.makedirs(OUT, exist_ok=True)
P = lib.load_panels()
spy_ret = P["C"]["SPY"].pct_change()
F = signals.features(P, mkt=spy_ret)
M300 = lib.pit_universe(P, 300)
M1000 = lib.pit_universe(P, 1000)
cols = list(M1000.columns[M1000.any()])


def save(name, df, cols=cols):
    d = df[cols] if cols is not None else df
    d.astype(np.float32).to_parquet(os.path.join(OUT, f"{name}.parquet"))


# prices (dividend/split adjusted), volumes
for k, v in dict(O=P["O"], H=P["H"], L=P["L"], C=P["C"], V=P["V"], DV=P["DV"], rawC=P["rawC"]).items():
    save(k, v, cols=None if False else list(P["C"].columns))   # full research columns incl. ETFs (needed for SPY/GLD/TLT...)
# features used by the baseline and by most overlays
for k in ["mom_6_1", "mom_12_1", "res_mom_12_1", "high52", "low52", "atr14", "vol20", "vol60", "idio_vol60", "beta60", "sma20", "sma50", "sma200",
          "ret1", "ret5", "ret21", "ret63", "ret126", "rvol", "clv", "gap", "res21"]:
    save(k, F[k], cols=list(P["C"].columns))
M300.astype(np.uint8).to_parquet(os.path.join(OUT, "M300.parquet"))
M1000.astype(np.uint8).to_parquet(os.path.join(OUT, "M1000.parquet"))
meta = lib.stock_meta()
meta.reindex(P["C"].columns)[["ssi", "n", "t"]].to_csv(os.path.join(OUT, "sector.csv"))
# VIX (index, not in the stock pool; used only as a regime variable)
v = yf.download(["^VIX", "^VIX3M"], start="2005-01-01", progress=False, auto_adjust=False)["Close"]
v.index = pd.to_datetime(v.index).tz_localize(None).normalize()
v.reindex(P["C"].index).ffill().to_parquet(os.path.join(OUT, "vix.parquet"))
json.dump(dict(dev=lib.DEV, hold=lib.HOLD, fee=lib.FEE, first=str(P["C"].index[0].date()), last=str(P["C"].index[-1].date()),
               n_cols=len(P["C"].columns)), open(os.path.join(OUT, "meta.json"), "w"))
print("cache written", len(P["C"].columns), "columns", P["C"].shape)
