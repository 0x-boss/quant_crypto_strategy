"""Download full daily OHLCV history (Yahoo Finance via yfinance) for every ticker in
data/binance_stocks.json that Binance flags as available (os == True).

Resumable: each chunk is written to data/daily_raw/chunk_XXXX.parquet (long format).
Run:  python -I new-exp/src/fetch_daily.py
"""
import json, os, re, sys, time
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "daily_raw")
os.makedirs(OUT, exist_ok=True)

stk = json.load(open(os.path.join(ROOT, "data", "binance_stocks.json")))["data"]
rows = [x for x in stk if x.get("os") and x["t"] in ("STOCK", "ETF")
        and re.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?", x["s"])]
tickers = sorted({x["s"] for x in rows})
ymap = {t: t.replace(".", "-") for t in tickers}
print(len(tickers), "tickers", flush=True)

CH = 150
chunks = [tickers[i:i + CH] for i in range(0, len(tickers), CH)]
for ci, ch in enumerate(chunks):
    fn = os.path.join(OUT, f"chunk_{ci:04d}.parquet")
    if os.path.exists(fn):
        continue
    ys = [ymap[t] for t in ch]
    for attempt in range(4):
        try:
            df = yf.download(ys, period="max", interval="1d", auto_adjust=False,
                             group_by="ticker", threads=True, progress=False, timeout=60)
            break
        except Exception as e:  # noqa
            print("retry", ci, attempt, repr(e)[:100], flush=True)
            time.sleep(5 * (attempt + 1))
    else:
        print("FAILED chunk", ci, flush=True)
        continue
    inv = {v: k for k, v in ymap.items()}
    parts = []
    for y in ys:
        try:
            d = df[y].dropna(how="all")
        except KeyError:
            continue
        if d.empty:
            continue
        d = d.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
        d["ticker"] = inv[y]
        d.index.name = "date"
        parts.append(d.reset_index())
    if parts:
        out = pd.concat(parts, ignore_index=True)
        out.to_parquet(fn)
        print(f"chunk {ci}/{len(chunks)} rows={len(out)} tickers={out.ticker.nunique()}", flush=True)
    else:
        pd.DataFrame().to_parquet(fn)
        print(f"chunk {ci} empty", flush=True)
    time.sleep(1)
print("DONE", flush=True)
