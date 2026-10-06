"""Download Binance public history from the official archive https://data.binance.vision (no API key needed).

    python scripts/fetch_vision.py            # default symbol list
Writes data/intraday/<SYM>_{spot_1h,perp_1h,funding}.parquet  (UTC timestamps = bar open / funding time).
Monthly zips through the last full month, daily zips for the current month."""
import io, os, sys, zipfile, datetime as dt, concurrent.futures as cf
import pandas as pd, requests

BASE = "https://data.binance.vision/data"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "intraday")
os.makedirs(OUT, exist_ok=True)
SYMS = sys.argv[1:] or ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT",
                        "LTCUSDT", "AVAXUSDT", "TRXUSDT", "DOTUSDT"]
S = requests.Session()
today = dt.date.today()
KCOL = "open_time o h l c v close_time qv n tb tq ig".split()


def fetch(url):
    for _ in range(4):
        try:
            r = S.get(url, timeout=60)
            if r.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                return z.read(z.namelist()[0])
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
    return None


def months(start):
    d = start
    while d < today.replace(day=1):
        yield d
        d = (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1)


def days_this_month():
    d = today.replace(day=1)
    while d < today:
        yield d
        d += dt.timedelta(days=1)


def urls(market, kind, sym, interval=None):
    root = f"{BASE}/{market}/{{p}}/{kind}/{sym}" + (f"/{interval}" if interval else "")
    stem = f"{sym}-{interval}" if interval else f"{sym}-{kind}"
    out = [f"{root.format(p='monthly')}/{stem}-{m:%Y-%m}.zip" for m in months(dt.date(2019, 9, 1))]
    out += [f"{root.format(p='daily')}/{stem}-{d:%Y-%m-%d}.zip" for d in days_this_month()]
    return out


def read_k(b):
    df = pd.read_csv(io.BytesIO(b), header=None)
    if not str(df.iloc[0, 0]).isdigit():
        df = df.iloc[1:]
    df = df.iloc[:, :12]; df.columns = KCOL
    df = df.astype(float)
    ts = df.open_time.astype("int64")
    ts = ts.where(ts < 1e14, ts // 1000)                       # newer archive files use microseconds
    df.index = pd.to_datetime(ts, unit="ms")
    return df[["o", "h", "l", "c", "v", "qv", "tb"]]


def read_f(b):
    df = pd.read_csv(io.BytesIO(b))
    df["time"] = pd.to_datetime(df.calc_time.astype("int64"), unit="ms")
    return df.set_index("time")[["last_funding_rate"]].rename(columns={"last_funding_rate": "fundingRate"}).astype(float)


def grab(us, reader):
    with cf.ThreadPoolExecutor(12) as ex:
        parts = [reader(b) for b in ex.map(fetch, us) if b]
    if not parts:
        return None
    df = pd.concat(parts).sort_index()
    return df[~df.index.duplicated()]


for sym in SYMS:
    for name, market, kind, itv, reader in [("spot_1h", "spot", "klines", "1h", read_k), ("perp_1h", "futures/um", "klines", "1h", read_k),
                                            ("funding", "futures/um", "fundingRate", None, read_f)]:
        df = grab(urls(market, kind, sym, itv), reader)
        if df is None:
            print(sym, name, "no data"); continue
        df.to_parquet(f"{OUT}/{sym}_{name}.parquet")
        print(f"{sym:9s} {name:8s} {len(df):7d} rows  {df.index[0]} -> {df.index[-1]}", flush=True)
