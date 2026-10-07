"""Fetch klines (1h and 1d) of the Binance spot *stock tokens* (symbols like NVDABUSDT) from the public
market-data mirror data-api.binance.vision (api.binance.com itself returns HTTP 451 from this region).

Writes data/binance_tokens/<SYMBOL>_<interval>.parquet and data/binance_tokens/_symbols.csv
Run: python -I new-exp/src/fetch_binance_tokens.py
"""
import json, os, re, time
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "binance_tokens")
os.makedirs(OUT, exist_ok=True)
BASE = "https://data-api.binance.vision"

stk = json.load(open(os.path.join(ROOT, "data", "binance_stocks.json")))["data"]
tick = {x["s"]: x for x in stk}

ex = requests.get(BASE + "/api/v3/exchangeInfo", timeout=120).json()["symbols"]
# candidate stock tokens: base = <TICKER>B  quoted in USDT
cands = []
for s in ex:
    if s["quoteAsset"] != "USDT" or not s["baseAsset"].endswith("B"):
        continue
    t = s["baseAsset"][:-1]
    if t in tick:
        cands.append((s["symbol"], t, s["status"]))
pd.DataFrame(cands, columns=["symbol", "ticker", "status"]).to_csv(os.path.join(OUT, "_symbols.csv"), index=False)
print(len(cands), "candidate symbols", flush=True)

COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_vol", "trades",
        "taker_base", "taker_quote", "ignore"]


def klines(sym, interval):
    rows, start = [], 0
    while True:
        for attempt in range(5):
            try:
                r = requests.get(BASE + "/api/v3/klines", params=dict(symbol=sym, interval=interval,
                                 startTime=start, limit=1000), timeout=60)
                r.raise_for_status()
                break
            except Exception as e:  # noqa
                time.sleep(2 * (attempt + 1))
        else:
            return None
        k = r.json()
        if not k:
            break
        rows += k
        start = k[-1][6] + 1
        if len(k) < 1000:
            break
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=COLS)
    for c in ["open", "high", "low", "close", "volume", "quote_vol", "taker_base", "taker_quote"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df.drop(columns="ignore")


def first_open(sym):
    r = requests.get(BASE + "/api/v3/klines", params=dict(symbol=sym, interval="1d", startTime=0, limit=1), timeout=60)
    k = r.json()
    return pd.to_datetime(k[0][0], unit="ms", utc=True) if k else None


keep = []
for sym, t, st in cands:
    fo = first_open(sym)
    # stock tokens were listed in 2025-26; older listings with the same suffix pattern are crypto (BNB, ARB, TRB...)
    if fo is not None and fo >= pd.Timestamp("2025-01-01", tz="UTC"):
        keep.append((sym, t, st))
    else:
        print("skip (crypto-like)", sym, fo, flush=True)
pd.DataFrame(keep, columns=["symbol", "ticker", "status"]).to_csv(os.path.join(OUT, "_symbols.csv"), index=False)
print(len(keep), "stock tokens kept", flush=True)

for sym, t, st in keep:
    for itv in ("1h", "1d"):
        fn = os.path.join(OUT, f"{sym}_{itv}.parquet")
        if os.path.exists(fn):
            continue
        df = klines(sym, itv)
        if df is None:
            print("none", sym, itv, flush=True)
            continue
        df.to_parquet(fn)
    print(sym, t, st, flush=True)
print("DONE")
