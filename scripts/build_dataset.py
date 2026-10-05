"""Build a compact daily panel (price, reported spot volume, market cap) from the
CoinMetrics community-data CSV dump (https://github.com/coinmetrics/data).

Usage: python scripts/build_dataset.py /path/to/coinmetrics/data/csv
Output: data/panel_{price,volume,mcap}.parquet  (date x asset)
"""
import sys, glob, os
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "/home/user/coinmetrics/data/csv"
out = os.path.join(os.path.dirname(__file__), "..", "data")
cols = ["time", "PriceUSD", "CapMrktCurUSD", "volume_reported_spot_usd_1d"]
price, vol, mcap = {}, {}, {}
for f in sorted(glob.glob(os.path.join(src, "*.csv"))):
    name = os.path.basename(f)[:-4]
    try:
        d = pd.read_csv(f, usecols=lambda c: c in cols, parse_dates=["time"]).set_index("time")
    except Exception:
        continue
    # PriceUSD stamped day t is the 00:00-UTC-(t+1) close.  ReferenceRateUSD stamped t+1 is the *same* instant,
    # so it must not be used as a fill (it would duplicate the last day and create a fake zero-return row).
    p = d["PriceUSD"].dropna() if "PriceUSD" in d else pd.Series(dtype=float)
    # volume / market cap are kept for every asset (base-ticker aliases such as avax -> avaxc are
    # resolved later), price only for assets with a usable history
    if "volume_reported_spot_usd_1d" in d and d["volume_reported_spot_usd_1d"].notna().sum() > 100:
        vol[name] = d["volume_reported_spot_usd_1d"]
    if "CapMrktCurUSD" in d and d["CapMrktCurUSD"].notna().sum() > 100:
        mcap[name] = d["CapMrktCurUSD"]
    if len(p) >= 400:
        price[name] = p
# volume for tokens that trade under a different base ticker than the CoinMetrics price series
ALIAS = {"avaxc": "avax", "vet_eth": "vet", "shib_eth": "shib", "matic_eth": "matic", "zil_eth": "zil",
         "icx_eth": "icx", "qtum_eth": "qtum", "leo_eth": "leo", "lrc_eth": "lrc", "ae_eth": "ae"}
keep = set(price) | set(ALIAS.values())
vol = {k: v for k, v in vol.items() if k in keep}
for nm, dct in (("price", price), ("volume", vol), ("mcap", mcap)):
    df = pd.DataFrame(dct).sort_index().astype("float64")
    df.index.name = "date"
    df.to_parquet(os.path.join(out, f"panel_{nm}.parquet"))
    print(nm, df.shape, df.index.min().date(), df.index.max().date())
