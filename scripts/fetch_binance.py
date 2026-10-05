"""Download public Binance history (no API key needed): 1h spot klines, 1h perp klines and 8h funding.

    python scripts/fetch_binance.py [BTCUSDT ETHUSDT ...]

Writes data/intraday/<SYMBOL>_{spot_1h,perp_1h,funding}.parquet.  Requires network access to
api.binance.com and fapi.binance.com (blocked by the default sandbox policy)."""
import os, sys, time
import pandas as pd
import requests

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "intraday")
os.makedirs(OUT, exist_ok=True)
S = requests.Session()


def get(url, params, tries=5):
    for i in range(tries):
        r = S.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (418, 429):
            time.sleep(2 ** i * 5)
        else:
            r.raise_for_status()
    raise RuntimeError(url)


def klines(base, path, symbol, start_ms, interval="1h"):
    rows, t = [], start_ms
    while True:
        k = get(base + path, dict(symbol=symbol, interval=interval, startTime=t, limit=1000))
        if not k:
            break
        rows += k
        t = k[-1][0] + 1
        if len(k) < 1000:
            break
    df = pd.DataFrame(rows, columns="open_time o h l c v close_time qv n tb tq _".split())
    df = df.assign(time=pd.to_datetime(df.open_time, unit="ms")).set_index("time")[["o", "h", "l", "c", "v", "qv", "tb"]].astype(float)
    return df[~df.index.duplicated()]


def funding(symbol, start_ms):
    rows, t = [], start_ms
    while True:
        k = get("https://fapi.binance.com/fapi/v1/fundingRate", dict(symbol=symbol, startTime=t, limit=1000))
        if not k:
            break
        rows += k
        t = k[-1]["fundingTime"] + 1
        if len(k) < 1000:
            break
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df.fundingTime, unit="ms")
    return df.set_index("time")[["fundingRate"]].astype(float)


if __name__ == "__main__":
    start = int(pd.Timestamp("2019-09-01").timestamp() * 1000)   # perp history begins Sep-2019
    for sym in (sys.argv[1:] or ["BTCUSDT", "ETHUSDT"]):
        klines("https://api.binance.com", "/api/v3/klines", sym, start).to_parquet(f"{OUT}/{sym}_spot_1h.parquet")
        klines("https://fapi.binance.com", "/fapi/v1/klines", sym, start).to_parquet(f"{OUT}/{sym}_perp_1h.parquet")
        funding(sym, start).to_parquet(f"{OUT}/{sym}_funding.parquet")
        print("saved", sym)
