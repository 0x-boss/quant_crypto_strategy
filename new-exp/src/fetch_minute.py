"""Build 5-minute regular-session bars (+ pre/post-market summaries) from the public Hugging Face 1-minute dataset
`mito0o852/OHLCV-1m` (US stocks, Finnhub-sourced, incl. extended hours), for a fixed ticker list and a range of months.

The ticker list must be point-in-time-agnostic (union of every name that was ever liquid enough) so that no future
information selects the data: selection is done later, per date, inside the backtests.

  python -I new-exp/src/fetch_minute.py <tickers.txt> <start YYYY-MM> <end YYYY-MM> [workers]

Output: data/m5/m5_YYYY-MM.parquet   columns: ticker, ts (US/Eastern, naive), o,h,l,c,v,n
        data/m5/ext_YYYY-MM.parquet  per ticker-day: pm_vol, pm_hi, pm_lo, pm_first, pm_last, ah_vol, ah_last
Prices are RAW (not split adjusted) - use only within-day ratios or join to the adjusted daily panel.
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "m5")
TMP = os.environ.get("MIN_TMP", "/tmp/hf_minute")
URL = "https://huggingface.co/datasets/mito0o852/OHLCV-1m/resolve/main/data/ohlcv_{ym}.parquet"


def process(ym, tickers):
    fo = os.path.join(OUT, f"m5_{ym}.parquet")
    fe = os.path.join(OUT, f"ext_{ym}.parquet")
    if os.path.exists(fo) and os.path.exists(fe):
        return ym, "cached"
    os.makedirs(TMP, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(TMP, f"ohlcv_{ym}.parquet")
    for attempt in range(4):
        try:
            with requests.get(URL.format(ym=ym), stream=True, timeout=120) as r:
                if r.status_code == 404:
                    return ym, "missing"
                r.raise_for_status()
                with open(fn, "wb") as f:
                    for ch in r.iter_content(1 << 22):
                        f.write(ch)
            break
        except Exception as e:  # noqa
            time.sleep(3 * (attempt + 1))
    else:
        return ym, "download failed"
    try:
        t = pq.read_table(fn, filters=[("ticker", "in", list(tickers))])
        df = t.to_pandas()
    finally:
        os.remove(fn)
    if df.empty:
        return ym, "empty"
    ts = df["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    df["ts"] = ts
    df["day"] = ts.dt.normalize()
    mod = ts.dt.hour * 60 + ts.dt.minute
    reg = df[(mod >= 570) & (mod < 960)].copy()
    pm = df[(mod >= 240) & (mod < 570)]
    ah = df[(mod >= 960) & (mod < 1200)]
    # --- 5-minute regular-session bars
    reg["b"] = reg["ts"].dt.floor("5min")
    reg = reg.sort_values(["ticker", "ts"])
    g = reg.groupby(["ticker", "b"], sort=False)
    m5 = g.agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"),
               v=("volume", "sum"), n=("open", "size")).reset_index().rename(columns={"b": "ts"})
    m5.to_parquet(fo)
    # --- extended-hours summaries
    pm = pm.sort_values(["ticker", "ts"])
    gp = pm.groupby(["ticker", "day"], sort=False).agg(pm_vol=("volume", "sum"), pm_hi=("high", "max"),
                                                    pm_lo=("low", "min"), pm_first=("open", "first"),
                                                    pm_last=("close", "last"))
    ah = ah.sort_values(["ticker", "ts"])
    ga = ah.groupby(["ticker", "day"], sort=False).agg(ah_vol=("volume", "sum"), ah_last=("close", "last"))
    ext = gp.join(ga, how="outer").reset_index()
    ext.to_parquet(fe)
    return ym, f"ok rows={len(m5)}"


if __name__ == "__main__":
    tk = [x.strip() for x in open(sys.argv[1]) if x.strip()]
    months = pd.period_range(sys.argv[2], sys.argv[3], freq="M").strftime("%Y-%m").tolist()
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    print(len(tk), "tickers", len(months), "months", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(process, ym, tk) for ym in months]
        for f in futs:
            print(*f.result(), flush=True)
    print("DONE", flush=True)
