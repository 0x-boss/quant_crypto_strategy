"""Family D - walk-forward gradient-boosting ranker (hyper-parameters fixed a priori, refit once a year, expanding window).

  python -I new-exp/src/ml_walkforward.py [top_n] [label_h]
Predictions for year Y come only from a model trained on labels that were fully realised before Jan 1 of Y
(embargo = label horizon + 5 days).
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import lib
import signals

top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
LH = int(sys.argv[2]) if len(sys.argv) > 2 else 5
P = lib.load_panels()
M = lib.pit_universe(P, top_n=top_n)
mkt = P["C"]["SPY"].pct_change() if "SPY" in P["C"] else None
F = signals.features(P, mkt=mkt)
cols = list(M.columns[M.any()])
Mc = M[cols]
O = P["O"][cols]

feat_names = ["ret1", "ret2", "ret3", "ret5", "ret10", "ret21", "ret63", "ret126", "mom_12_1", "mom_6_1", "high52", "low52",
              "sma20", "sma50", "sma200", "atr14", "vol20", "vol60", "gap", "intraday", "overnight21", "intraday21",
              "rvol", "dvol_rank_chg", "clv", "max21", "min21", "rsi2", "beta60", "res1", "res5", "res21", "idio_vol60"]
raw_keep = {"atr14", "vol20", "vol60", "idio_vol60", "beta60"}   # level matters -> keep a raw copy as well


def xs_rank(df):
    return df.where(Mc).rank(axis=1, pct=True) - 0.5


X = {}
for f in feat_names:
    Fd = F[f][cols]
    X[f + "_r"] = xs_rank(Fd).astype(np.float32)
    if f in raw_keep:
        X[f + "_raw"] = Fd.where(Mc).clip(*Fd.where(Mc).stack().quantile([0.01, 0.99]).values).astype(np.float32)
del F
# forward label: open t+1 -> open t+1+LH
fwd = O.shift(-(1 + LH)) / O.shift(-1) - 1
lab = fwd.where(Mc)
lab = lab.sub(lab.mean(axis=1), axis=0)
lo, hi = lab.stack().quantile([0.005, 0.995]).values
lab = lab.clip(lo, hi)

dates = O.index
names = list(X.keys())
stk = pd.concat({k: v.stack() for k, v in X.items()}, axis=1)
stk["y"] = lab.stack()
stk = stk.dropna(subset=names)
print("rows", len(stk), "features", len(names), flush=True)
d_idx = stk.index.get_level_values(0)

params = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=3000, feature_fraction=0.7,
              bagging_fraction=0.5, bagging_freq=1, lambda_l2=50.0, verbose=-1, num_threads=4)
NROUND = 250
pred = pd.Series(np.nan, index=stk.index)
t0 = time.time()
for Y in range(2012, 2027):
    cut = pd.Timestamp(f"{Y}-01-01") - pd.tseries.offsets.BDay(LH + 5)
    tr = stk[(d_idx < cut) & stk["y"].notna()]
    te_mask = (d_idx >= pd.Timestamp(f"{Y}-01-01")) & (d_idx <= pd.Timestamp(f"{Y}-12-31"))
    if len(tr) < 200000 or not te_mask.any():
        continue
    tr = tr.iloc[::2]
    mdl = lgb.train(params, lgb.Dataset(tr[names].values.astype(np.float32), tr["y"].values), NROUND)
    pred[te_mask] = mdl.predict(stk.loc[te_mask, names].values.astype(np.float32))
    imp = pd.Series(mdl.feature_importance("gain"), index=names).sort_values(ascending=False)
    print(Y, f"train {len(tr)}", f"{time.time()-t0:.0f}s", "top feats:", list(imp.index[:5]), flush=True)

pr = pred.unstack().reindex(index=dates, columns=cols)
pr.to_parquet(f"{lib.RES}/ml_pred_top{top_n}_lh{LH}.parquet")


def topk_weights(sig, k, h):
    s = sig.where(Mc)
    rank = s.rank(axis=1, ascending=False, method="first")
    sleeve = ((rank <= k) & s.notna()).astype(float) / k
    return sleeve.rolling(h, min_periods=1).mean()


for k in (10, 25):
    for h in (1, 5, 10):
        r = lib.weights_backtest(topk_weights(pr, k, h), O)
        lib.log_trial("D_ml", f"gbm_lh{LH}", dict(top_n=top_n, k=k, h=h), r)
        lib.print_report(r.loc["2012":], f"GBM top{top_n} label{LH}d  k={k} h={h}")
