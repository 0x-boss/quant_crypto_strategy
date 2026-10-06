"""Download Binance futures 'metrics' (open interest, long/short ratios, taker vol ratio; 5-min) -> daily parquet."""
import io, os, sys, zipfile, datetime as dt, concurrent.futures as cf
import pandas as pd, requests
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "intraday"); S = requests.Session()
def one(a):
    sym, d = a
    for _ in range(3):
        try:
            r = S.get(f"https://data.binance.vision/data/futures/um/daily/metrics/{sym}/{sym}-metrics-{d:%Y-%m-%d}.zip", timeout=60)
            if r.status_code == 404: return None
            if r.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(r.content)); return pd.read_csv(z.open(z.namelist()[0]))
        except requests.RequestException: pass
    return None
for sym in (sys.argv[1:] or ["BTCUSDT", "ETHUSDT"]):
    days = [(sym, dt.date(2020, 9, 1) + dt.timedelta(days=i)) for i in range((dt.date.today() - dt.date(2020, 9, 1)).days)]
    with cf.ThreadPoolExecutor(16) as ex: parts = [p for p in ex.map(one, days) if p is not None]
    df = pd.concat(parts); df["create_time"] = pd.to_datetime(df.create_time); df = df.drop_duplicates("create_time").set_index("create_time").drop(columns="symbol").sort_index()
    df.to_parquet(f"{OUT}/{sym}_metrics_5m.parquet"); print(sym, len(df), df.index[0], df.index[-1], flush=True)
