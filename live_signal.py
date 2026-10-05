"""Print TrendCore's target portfolio for the next day from a CSV of daily closes.

    python live_signal.py prices.csv

CSV: columns `date,btc,eth` - one row per UTC day, close at 00:00 UTC (the CoinMetrics convention: the row
dated D holds the close of day D).  Provide >= 400 rows (the longest indicator needs ~260 days of warm-up).
Output weights are fractions of NAV (long-only; 0 = flat); 'gross' is total exposure (cap 2.0x).
"""
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
from quant.strategy import Config, target_weights

if len(sys.argv) != 2:
    sys.exit(__doc__)
P = pd.read_csv(sys.argv[1], parse_dates=["date"]).set_index("date").sort_index()[["btc", "eth"]].astype(float)
if len(P) < 400 or P.index.to_series().diff().dropna().ne(pd.Timedelta(days=1)).any():
    sys.exit("need >= 400 consecutive daily rows without gaps")
R = P.pct_change(fill_method=None)
W = target_weights(P, R, Config())
w = W.iloc[-1]
print(f"as of close {P.index[-1].date()} -> target weights for the next 24h")
print(f"  BTC  {w['btc']*100:6.1f}% of NAV\n  ETH  {w['eth']*100:6.1f}% of NAV\n  gross {w.abs().sum()*100:5.1f}%  cash {max(0, 1-w.abs().sum())*100:5.1f}%")
print("(rebalance only if a target differs from the current holding by more than 5% of NAV, or the target is 0)")
