"""TrendCore - SPOT-ONLY, LONG-ONLY, NO-LEVERAGE trend strategy on BTC + ETH (self-contained).

Rules
  * Decisions every 6 hours (K=4 bars/day); every look-back below is in days x K bars.
  * Trend score per asset in [0,1] = average of 3 families: momentum votes (14/30/60/90/180d), price>SMA votes
    (20/50/100/200d), Carver EWMAC (8/32, 16/64, 32/128, 64/256d).
  * Regime gate: hold an asset only while price > its SMA(g); weights averaged over g in {100,150,200}d.
  * Sizing: inverse realised vol (from HOURLY returns, EWMA span 20/30/60d averaged) -> equal risk per asset.
  * Portfolio: scale to 40% annualised vol using the strategy's own trailing vol, scalar capped at 1.0 (no leverage).
  * Optional overlay: exposure x (1 - 0.5*clip(z,0,2)/2), z = 365d z-score of log BTC futures open interest
    (pass `oi=None` to disable; the overlay only reads a data series, it never trades a derivative).
  * Total exposure never exceeds 100% of capital.  Costs: 10 bp per side, 5%-of-NAV no-trade band.

Input : hourly closes, DataFrame with columns ['btc','eth'] and a UTC DatetimeIndex (Binance spot 1h works).
Output: target weights (fractions of capital) per 6h bar, and a backtest.
"""
import numpy as np
import pandas as pd

K = 4                                   # bars per day (6h)
ANN = 365 * K
GATES = (100, 150, 200)
VOL_SPANS = (20, 30, 60)
TARGET_VOL = 0.40
ASSET_VOL = 0.45
COST_BPS = 10.0
BAND = 0.05
OI_STRENGTH = 0.5
MOM_LB = (14, 30, 60, 90, 180)
SMA_LB = (20, 50, 100, 200)
EWMAC = ((8, 32, 10.6), (16, 64, 7.5), (32, 128, 5.3), (64, 256, 3.75))   # (fast, slow, forecast scalar), days


def _trend_scores(P):
    mom = sum((P / P.shift(L * K) > 1.0).astype(float).where(P.shift(L * K).notna()) for L in MOM_LB) / len(MOM_LB)
    sma = 0
    for L in SMA_LB:
        m = P.rolling(L * K, min_periods=L * K).mean()
        sma = sma + (P > m).astype(float).where(m.notna())
    sma = sma / len(SMA_LB)
    dvol = np.log(P).diff().ewm(span=30 * K, min_periods=10).std()
    fc = 0
    for f, s, sc in EWMAC:
        raw = (P.ewm(span=f * K, min_periods=f * K).mean() - P.ewm(span=s * K, min_periods=s * K).mean()) / (P * dvol)
        fc = fc + (raw * sc).clip(-20, 20)
    ew = (fc / len(EWMAC)).clip(lower=0) / 10.0
    return (mom + sma + ew.clip(upper=1.0)) / 3.0


def _backtest(W, R, cost_bps=COST_BPS, band=BAND, lag=0):
    """Weights decided at bar t are held over bar t+1.  Returns per-bar net return and gross exposure."""
    Wl = W.shift(1 + lag).fillna(0.0).values
    Rv = R.fillna(0.0).values
    ret, gross = np.zeros(len(Wl)), np.zeros(len(Wl))
    prev = np.zeros(Wl.shape[1])
    for t in range(len(Wl)):
        tgt = Wl[t]
        if band > 0:                                           # no-trade band (exits to zero always execute)
            tgt = np.where((np.abs(tgt - prev) <= band) & (tgt != 0.0), prev, tgt)
        cost = np.abs(tgt - prev).sum() * cost_bps / 1e4
        pr = float(tgt @ Rv[t])
        ret[t], gross[t] = pr - cost, tgt.sum()
        prev = tgt * (1.0 + Rv[t]) / (1.0 + ret[t])            # drift to next bar
    return pd.Series(ret, index=R.index), pd.Series(gross, index=R.index)


def oi_multiplier(index, oi, strength=OI_STRENGTH):
    """oi: daily Series of BTC futures open interest in USD (UTC date index)."""
    x = np.log(oi)
    z = ((x - x.rolling(365).mean()) / x.rolling(365).std()).shift(1)       # known at previous day's close
    return (1 - strength * z.reindex(index, method="ffill").clip(0, 2) / 2).fillna(1.0)


def target_weights(px1h, oi=None, target_vol=TARGET_VOL):
    """Target weights (fractions of capital, long-only, sum <= 1) for every 6h bar close."""
    px1h = px1h[["btc", "eth"]]
    P = px1h.resample("6h").last().dropna(how="all")
    R = P.pct_change()
    rv2 = (np.log(px1h).diff() ** 2).resample("6h").sum().reindex(P.index)         # realised variance from hourly data
    vol = lambda span: (rv2.ewm(span=span * K, min_periods=10).mean() ** 0.5) * np.sqrt(ANN)
    score = _trend_scores(P)
    n = P.notna().sum(axis=1).clip(lower=1)
    acc, cnt = 0, 0
    for g in GATES:
        f = (score * (P > P.rolling(g * K).mean())).fillna(0.0)
        for vs in VOL_SPANS:
            acc = acc + (f * ASSET_VOL / vol(vs)).div(n, axis=0).fillna(0.0).clip(upper=1.0)
            cnt += 1
    W = acc / cnt
    r0, _ = _backtest(W, R)                                                         # unscaled strategy returns
    rv = r0.ewm(span=30 * K, min_periods=10).std() * np.sqrt(ANN)
    W = W.mul((target_vol / rv).clip(upper=1.0).fillna(0.0), axis=0)                # vol target, never > 1x
    if oi is not None:
        W = W.mul(oi_multiplier(W.index, oi), axis=0)
    gross = W.sum(axis=1)
    return W.mul((1.0 / gross).clip(upper=1.0).fillna(1.0), axis=0), R              # hard cap: 100% of capital


def backtest(px1h, oi=None, start="2018-03-01"):
    """Daily net returns of the strategy (UTC days)."""
    W, R = target_weights(px1h, oi)
    r, gross = _backtest(W, R)
    daily = (1 + r).groupby(r.index.date).prod() - 1
    daily.index = pd.to_datetime(daily.index)
    return daily.loc[start:], W


def stats(r):
    eq = (1 + r).cumprod()
    yrs = len(r) / 365
    return dict(cagr=eq.iloc[-1] ** (1 / yrs) - 1, sharpe=r.mean() / r.std() * np.sqrt(365),
                maxdd=(eq / eq.cummax() - 1).min(), vol=r.std() * np.sqrt(365))


if __name__ == "__main__":
    # data: hourly spot closes (Binance, data.binance.vision) + optional daily BTC open interest (USD)
    px = pd.concat({a: pd.read_parquet(f"data/intraday/{a.upper()}USDT_spot_1h.parquet")["c"] for a in ("btc", "eth")}, axis=1, sort=True)
    oi = pd.read_parquet("data/intraday/BTCUSDT_oi_daily.parquet")["sum_open_interest_value"]
    r, W = backtest(px, oi)
    s = stats(r)
    print(f"{r.index[0].date()} -> {r.index[-1].date()}: CAGR {s['cagr']*100:.1f}%  Sharpe {s['sharpe']:.2f}  MaxDD {s['maxdd']*100:.1f}%  Vol {s['vol']*100:.1f}%")
    print("latest target weights:", W.iloc[-1].round(3).to_dict())
