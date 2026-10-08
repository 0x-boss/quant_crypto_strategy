"""The perturbation guard must (a) pass the clean baseline, (b) pass a correct lagged-return overlay, (c) FAIL a one-day-peeking overlay and
an un-lagged own-return overlay.  python tests/test_r8_perturbation.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, pandas as pd
import r8_common as R

c = R.load(["O", "C", "mom_6_1", "M300", "atr14", "vol60"])
DATES = ("2014-03-14", "2016-01-20", "2020-06-17", "2022-06-14", "2025-08-12")

def clean(cc):
    return R.base_weights(cc)

def correct_voltarget(cc):
    W = R.base_weights(cc)
    r = R.run(cc, W)
    vol = R.lagged(r).rolling(20).std() * np.sqrt(252)
    s = (0.45 / vol).clip(upper=1.0).fillna(1.0)
    return W.mul(s, axis=0)

def leaky_peek(cc):                      # uses tomorrow's close
    W = R.base_weights(cc)
    s = (cc["C"].shift(-1)["SPY"] > cc["C"]["SPY"]).astype(float)
    return W.mul(s, axis=0)

def leaky_unlagged_return(cc):           # uses r[t] = open t -> open t+1, unknown at the close of t
    W = R.base_weights(cc)
    r = R.run(cc, W)
    vol = r.rolling(20).std() * np.sqrt(252)
    s = (0.45 / vol).clip(upper=1.0).fillna(1.0)
    return W.mul(s, axis=0)

assert R.check_perturbation(clean, c, DATES, verbose=False)[0], "clean baseline flagged"
assert R.check_perturbation(correct_voltarget, c, DATES, verbose=False)[0], "correct overlay flagged"
assert not R.check_perturbation(leaky_peek, c, DATES, verbose=False)[0], "peeking overlay NOT detected"
assert not R.check_perturbation(leaky_unlagged_return, c, DATES, verbose=False)[0], "un-lagged return overlay NOT detected"
print("r8 perturbation guard tests OK")
