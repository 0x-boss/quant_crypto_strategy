"""Daily USDT-perp klines for every symbol in the Binance archive (incl. delisted) -> data/perp_daily/<SYM>.parquet
and funding for a list of symbols.   python scripts/fetch_perp_daily.py klines | funding SYM1 SYM2 ..."""
import io, os, sys, json, zipfile, datetime as dt, concurrent.futures as cf
import pandas as pd, requests
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "perp_daily"); os.makedirs(OUT, exist_ok=True)
BASE = "https://data.binance.vision/data/futures/um"; S = requests.Session(); today = dt.date.today()
KC = "open_time o h l c v close_time qv n tb tq ig".split()
def get(url):
    for _ in range(3):
        try:
            r = S.get(url, timeout=60)
            if r.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(r.content)); return z.read(z.namelist()[0])
            if r.status_code == 404: return None
        except requests.RequestException: pass
    return None
def months(a):
    d = a
    while d < today.replace(day=1):
        yield d; d = (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
def kl(sym):
    urls = [f"{BASE}/monthly/klines/{sym}/1d/{sym}-1d-{m:%Y-%m}.zip" for m in months(dt.date(2019, 9, 1))]
    urls += [f"{BASE}/daily/klines/{sym}/1d/{sym}-1d-{today.replace(day=1)+dt.timedelta(days=i):%Y-%m-%d}.zip" for i in range(today.day - 1)]
    parts = []
    for u in urls:
        b = get(u)
        if b:
            df = pd.read_csv(io.BytesIO(b), header=None)
            if not str(df.iloc[0, 0]).isdigit(): df = df.iloc[1:]
            df = df.iloc[:, :12]; df.columns = KC; parts.append(df.astype(float))
    if not parts: return sym, None
    df = pd.concat(parts); ts = df.open_time.astype("int64"); ts = ts.where(ts < 1e14, ts // 1000)
    df.index = pd.to_datetime(ts, unit="ms"); df = df[["o", "h", "l", "c", "v", "qv", "n", "tb"]].sort_index()
    return sym, df[~df.index.duplicated()]
def fu(sym):
    parts = []
    for m in months(dt.date(2019, 9, 1)):
        b = get(f"{BASE}/monthly/fundingRate/{sym}/{sym}-fundingRate-{m:%Y-%m}.zip")
        if b: parts.append(pd.read_csv(io.BytesIO(b)))
    if not parts: return sym, None
    df = pd.concat(parts); df["time"] = pd.to_datetime(df.calc_time.astype("int64"), unit="ms")
    return sym, df.set_index("time")[["last_funding_rate"]].rename(columns={"last_funding_rate": "fundingRate"}).astype(float)
if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "klines":
        syms = json.load(open(sys.argv[2]))
        with cf.ThreadPoolExecutor(32) as ex:
            for i, (s, df) in enumerate(ex.map(kl, syms)):
                if df is not None: df.to_parquet(f"{OUT}/{s}.parquet")
                if i % 50 == 0: print(i, s, flush=True)
    else:
        with cf.ThreadPoolExecutor(32) as ex:
            for s, df in ex.map(fu, sys.argv[2:]):
                if df is not None: df.to_parquet(f"{OUT}/{s}_funding.parquet")
