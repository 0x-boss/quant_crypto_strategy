"""Feature / signal library. Every feature at row t uses data through the close of day t (no look-ahead).
Higher value = more attractive to BUY unless the name says otherwise."""
import numpy as np
import pandas as pd


def features(P, mkt=None):
    O, H, L, C, V, DV = P["O"], P["H"], P["L"], P["C"], P["V"], P["DV"]
    F = {}
    pc = C.shift(1)
    r1 = C / pc - 1
    F["ret1"] = r1
    for n in (2, 3, 5, 10, 21, 63, 126):
        F[f"ret{n}"] = C / C.shift(n) - 1
    F["mom_12_1"] = C.shift(21) / C.shift(252) - 1
    F["mom_6_1"] = C.shift(21) / C.shift(126) - 1
    F["mom_3_0"] = C / C.shift(63) - 1
    F["high52"] = C / C.rolling(252, min_periods=200).max()
    F["low52"] = C / C.rolling(252, min_periods=200).min()
    F["sma20"] = C / C.rolling(20).mean() - 1
    F["sma50"] = C / C.rolling(50).mean() - 1
    F["sma200"] = C / C.rolling(200, min_periods=150).mean() - 1
    tr = np.maximum(np.maximum(H - L, (H - pc).abs()), (L - pc).abs())
    F["atr14"] = tr.rolling(14, min_periods=7).mean() / C
    F["vol20"] = r1.rolling(20, min_periods=15).std()
    F["vol60"] = r1.rolling(60, min_periods=40).std()
    F["gap"] = O / pc - 1                      # known at the open of day t, i.e. before the close of t
    F["intraday"] = C / O - 1                  # open->close of day t
    F["overnight21"] = (O / pc - 1).rolling(21, min_periods=15).mean()
    F["intraday21"] = (C / O - 1).rolling(21, min_periods=15).mean()
    F["rvol"] = V / V.rolling(20, min_periods=15).mean().shift(1)
    F["dvol_rank_chg"] = DV.rolling(5).mean() / DV.rolling(63, min_periods=40).mean()
    F["clv"] = (C - L) / (H - L).replace(0, np.nan)          # close location value in day range
    F["max21"] = r1.rolling(21, min_periods=15).max()
    F["min21"] = r1.rolling(21, min_periods=15).min()
    # RSI(2) (Wilder-free simple version, Connors style)
    d = C.diff()
    up = d.clip(lower=0).rolling(2).mean()
    dn = (-d.clip(upper=0)).rolling(2).mean()
    F["rsi2"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    # market beta / residual returns vs a market proxy (equal-weight mean of the panel if no proxy given)
    if mkt is None:
        mkt = r1.mean(axis=1)
    mv = mkt.rolling(60, min_periods=40).var()
    cov = r1.mul(mkt, axis=0).rolling(60, min_periods=40).mean() - r1.rolling(60, min_periods=40).mean().mul(
        mkt.rolling(60, min_periods=40).mean(), axis=0)
    beta = cov.div(mv, axis=0)
    F["beta60"] = beta
    res1 = r1 - beta.shift(1).mul(mkt, axis=0)
    F["res1"] = res1
    F["res5"] = res1.rolling(5).sum()
    F["res_mom_12_1"] = res1.shift(21).rolling(231, min_periods=150).sum()
    F["res21"] = res1.rolling(21, min_periods=15).sum()
    F["idio_vol60"] = res1.rolling(60, min_periods=40).std()
    return F


def build_signal_menu(F):
    """Name -> signal DataFrame (higher = buy). Mix of trend, reversal, volume, risk - all fixed a priori."""
    S = {}
    S["mom_12_1"] = F["mom_12_1"]
    S["mom_6_1"] = F["mom_6_1"]
    S["mom_3_0"] = F["mom_3_0"]
    S["high52"] = F["high52"]
    S["res_mom_12_1"] = F["res_mom_12_1"]
    S["rev1"] = -F["ret1"]
    S["rev5"] = -F["ret5"]
    S["rev21"] = -F["ret21"]
    S["rev1_atr"] = -F["ret1"] / F["atr14"]
    S["rev_res1"] = -F["res1"]
    S["rev_res5"] = -F["res5"]
    S["rev_gap"] = -F["gap"]
    S["gap_up"] = F["gap"]
    S["rev_lowclv"] = -F["clv"]
    S["clv_high"] = F["clv"]
    S["vol_surge_dir"] = F["rvol"] * np.sign(F["ret1"])
    S["vol_surge_rev"] = -F["rvol"] * np.sign(F["ret1"])
    S["lowvol"] = -F["vol60"]
    S["highvol"] = F["vol60"]
    S["max_rev"] = -F["max21"]
    S["max_pos"] = F["max21"]
    S["lowbeta"] = -F["beta60"]
    S["highbeta"] = F["beta60"]
    S["idiovol_hi"] = F["idio_vol60"]
    S["overnight_mom"] = F["overnight21"]
    S["intraday_mom"] = F["intraday21"]
    return S
