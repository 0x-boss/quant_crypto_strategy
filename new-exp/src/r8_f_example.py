"""EXAMPLE family module (also used to test the runner): SPY trend gate.  Copy this file to start a new family."""
import numpy as np
import pandas as pd
import r8_common as R

FAMILY = "example"
CONFIGS = [dict(name=f"spy_sma{n}_cut{cut}", n=n, cut=cut) for n in (100, 200) for cut in (0.0, 0.5)]


def apply(c, W, p):
    spy = c["C"]["SPY"]
    on = (spy > spy.rolling(p["n"]).mean())              # known at the close of day t -> multiplies row t (executed t+1 open)
    s = on.astype(float) + (1 - on.astype(float)) * p["cut"]   # exposure 1 when on, `cut` when off
    return W.mul(s, axis=0)
