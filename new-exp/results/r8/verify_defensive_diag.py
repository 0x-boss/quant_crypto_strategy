import sys
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import numpy as np, pandas as pd
import r8_common as R
import r8_f_defensive as mod
exec(open("/home/user/quant_crypto_strategy/new-exp/results/r8/verify_defensive_main.py").read().split("# Part B")[0].replace("print(", "(lambda *a, **k: None)("))
D = np.abs(Wown - Wmod_v)
i, j = np.unravel_index(np.argmax(D), D.shape)
print("worst at", dates[i], cols[j], Wown[i, j], Wmod_v[i, j], "s_own", s_own[i], "sig", sig[i])
rows = np.where(D.max(1) > 1e-9)[0]
print("n rows with diff>1e-9:", len(rows), dates[rows[:10]], dates[rows[-5:]])
colsd = np.where(D.max(0) > 1e-9)[0]
print("cols:", [cols[k] for k in colsd][:20])
mask = m2012.copy(); mask[-1] = False
print("max diff rows>=2012 excl. last row (star=0.2501):", float(D[mask].max()))
print("max diff rows>=2012 excl. last row (star=exact):", float(np.abs(Wown2 - Wmod_v)[mask].max()))
print("NaN count in module W (raw):", int(Wmod.isna().sum().sum()), "rows with any NaN:", list(Wmod.index[Wmod.isna().any(axis=1)][:5]), Wmod.index[Wmod.isna().any(axis=1)][-3:].tolist())
print("module W last-row sum:", float(Wmod.iloc[-1].sum()), "second-last:", float(Wmod.iloc[-2].sum()), " own last", float(Wown[-1].sum()))
