import sys
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import r8_common as R
import r8_f_ddbreaker as mod
c = R.load()
p = next(q for q in mod.CONFIGS if q["name"] == "w126_t10_c50_d10")
R.lagged = lambda r: r          # MUTANT: un-lagged strategy returns (in-process monkeypatch only)
mod._CACHE.clear()
ok, worst = R.check_perturbation(lambda cc: mod.apply(cc, R.base_weights(cc), p), c, dates=("2021-03-10", "2020-03-12", "2025-03-20"), tol=1e-9)
print("MUTANT (no lag) detected as look-ahead:", not ok, worst)
