import sys, time
sys.path.insert(0, "/home/user/quant_crypto_strategy/new-exp/src")
import r8_common as R
import r8_f_sizing as mod
c = R.load()
p = next(q for q in mod.CONFIGS if q["name"] == "k10_iv60_cap20_vcap100_sec3")
print(p)
dates = ("2021-03-10", "2020-03-12", "2025-03-20", "2026-06-25", "2018-11-15", "2012-09-13", "2015-08-24", "2023-10-19")
print([pd for pd in dates if __import__("pandas").Timestamp(pd) in c["C"].index])
t0 = time.time()
ok, worst = R.check_perturbation(lambda cc: mod.weights(cc, p), c, dates=dates)
print("RESULT", ok, worst, "secs", time.time() - t0)
