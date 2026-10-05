"""Causality tests: the weight decided at day T must not depend on any data after T."""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quant.data import load_panels
from quant.engine import run_backtest
from quant.strategy import Config, target_weights


def test_truncation_invariance():
    P, _ = load_panels()
    R = P.pct_change(fill_method=None)
    cfg = Config()
    full = target_weights(P, R, cfg)[["btc", "eth"]]
    for d in ["2017-06-30", "2019-03-15", "2021-11-10", "2023-08-17", "2025-10-10"]:
        Pt = P.loc[:d]
        part = target_weights(Pt, Pt.pct_change(fill_method=None), cfg)[["btc", "eth"]]
        a, b = full.loc[d].values, part.loc[d].values
        assert np.allclose(a, b, atol=1e-9), (d, a, b)
    print("truncation invariance OK (weights at T identical with and without future data)")


def test_future_shock_does_not_change_past_pnl():
    """Perturb prices after T: P&L up to and including T must be unchanged."""
    P, _ = load_panels()
    cfg = Config()
    T = pd.Timestamp("2022-05-01")
    R = P.pct_change(fill_method=None)
    r1 = run_backtest(target_weights(P, R, cfg), R, **cfg.engine_kwargs())
    P2 = P.copy()
    P2.loc[P2.index > T, ["btc", "eth"]] *= 3.0   # absurd future
    R2 = P2.pct_change(fill_method=None)
    r2 = run_backtest(target_weights(P2, R2, cfg), R2, **cfg.engine_kwargs())
    assert np.allclose(r1.loc[:T].values, r2.loc[:T].values, atol=1e-12)
    print("future-shock invariance OK")


def test_weights_are_long_only_and_capped():
    P, _ = load_panels()
    R = P.pct_change(fill_method=None)
    W = target_weights(P, R, Config())
    assert (W.values >= -1e-12).all()
    assert W.abs().sum(axis=1).max() <= 2.0 + 1e-9
    print("long-only and gross <= 2x OK; max gross", round(W.abs().sum(axis=1).max(), 3))


if __name__ == "__main__":
    test_truncation_invariance()
    test_future_shock_does_not_change_past_pnl()
    test_weights_are_long_only_and_capped()
