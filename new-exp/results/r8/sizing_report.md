R8 FAMILY SIZING - report (module src/r8_f_sizing.py, 31 configs logged: 26 pre-registered + 5 post-hoc; log results/r8/trials_sizing.csv)
Question: can position weights / concentration cut the idiosyncratic blow-ups of MOM_6_1_k10 (DEV Sh 0.78, med 2.15%, DD -53.2% | HOLD Sh 0.83, med 5.61%, DD -46.0%)?
RESULT: only ONE ingredient works - excluding names whose annualised vol60 is above ~80-100%.  Inverse-vol weights, K>10, sector caps, top-1000 mixing do not
  pass the DEV objective on their own.  Nothing passes the plan's HOLD acceptance rule (HOLD DD <= -31%, HOLD median >= 5%).
DEV-ONLY RANKING (score = DEV Sh - 0.5|DEV DD|; finite only if DEV mean>=80%, median>=85% of base and DD cut >=20%); all guard-OK (perturbation worst diff 0.0):
  1 ph_k10_iv60_cap20_vcap80_sec3 [POST-HOC] 0.721 | DEV Sh 0.91 med 2.64% DD -38.5% | HOLD Sh 0.86 med 3.45% DD -40.6% | 2xfee DEV Sh 0.80 | lag1 0.93
  2 k10_iv60_cap20_vcap100_sec3 [pre-reg. winner] 0.718 | DEV Sh 0.92 med 2.44% DD -40.8% | HOLD Sh 0.89 med 4.57% DD -42.7% | 2xfee 0.815 | lag1 0.928
  3 ph_k10_iv60_cap20_vcap100 [POST-HOC] 0.717 | DEV Sh 0.92 med 2.40% DD -41.0% | HOLD Sh 0.90 med 4.70% DD -43.9% | 2xfee 0.822 | lag1 0.931
  (pre-reg. #2 eq_k10_vcap80 0.704: DEV Sh 0.90 DD -39.4% | HOLD Sh 0.92 med 4.48% DD -42.8% | 2xfee 0.809 | lag1 0.912).  Baseline DEV Sh: 2xfee 0.704, lag1 0.768.
  The post-hoc #1 beats the pre-registered winner by 0.003 = a tie; carry the pre-registered one (#2) forward.  Components: vcap100 (annualised vol60 <= 100%, K=10,
  replaced by next-ranked name) + inverse-vol60 weights (cap 20%) + max 3 names/sector.  Sector cap adds nothing (score 0.7173 without it); inverse-vol adds ~+0.05 DEV Sh at vcap100.
MECHANISM (window return base -> winner, SPY): 2021-02..05 -50.8% -> -23.0% (SPY +6.3%; held vol60 187% -> 70%: GME/AMC-type meme names, 45% of held names >100% vol in 2021);
  2025-02..04 -41.0% -> -37.4% (SPY -18.1%); 2020-02..03 -44.1% -> -40.1% (SPY -29.2%); 2026-06..07 -37.2% -> -22.6% (SPY -2%); 2023-08..10 -37.7% -> -27.3% (SPY -8.5%).
  So the filter repairs the idiosyncratic/high-vol blow-ups (2021, 2026, 2023) but NOT the factor/market unwinds (COVID 2020, Feb-Apr 2025: still -40%).  The winner's DEV max DD (-40.8%) is
  the COVID window; its HOLD max DD (-42.7%) is Feb-Apr 2025.  In 2025-26 the cap binds hard (49%/43% of baseline names exceed 100% vol), so upside of high-vol winners is also cut.
CAVEATS (read these): (1) DEV edge is one episode: excluding 2021-02-12..05-10 the DEV Sharpe is 1.06 vs 1.05 baseline; 2012-2019 only: Sh 0.83 vs 0.70, DD -31.0% vs -36.3% (modest, real);
  2022-24: Sh 0.54 vs 0.60 (worse), 2025-26: 1.32 vs 1.10 (better).  (2) HOLD median month falls 5.61% -> 4.6% (post-hoc/other caps 3.5-4.7%); month-bootstrap P(HOLD median>5%) 0.55 -> 0.34.
  (3) turnover/day 0.109 -> 0.130 (inverse-vol re-weighting); costs 2x leave DEV Sh 0.815 (baseline 0.704) so the edge survives fees; lag1 does not hurt (0.928).
PLATEAU: vol-cap dimension (K=10 equal weight, DEV Sh / DD): 120% 0.82/-44.0% (score -inf, DD cut <20%), 100% 0.87/-42.8% (-inf by 0.005 of DD), 80% 0.90/-39.4% (0.704), 70% post-hoc 0.90/-38.8% (0.710),
  60% post-hoc 0.86/-38.9% (0.661).  Smooth, broad plateau 70-100% with the DEV optimum at 70-80%; HOLD gets worse as the cap tightens (HOLD Sh 0.98/0.92/0.75/0.64 for 100/80/70/60%).
  Winner's neighbours: k15_iv60_cap20_vcap100 0.647 (DEV Sh 0.84 - more names dilutes), ph_k10_iv60_cap20_vcap100 0.717, ph_..vcap80_sec3 0.721, eq_k10_vcap80 0.704 -> plateau is good.
NEGATIVE RESULTS (all -inf): inverse-vol (vol20/vol60/atr14 x cap 15/20/25%) DEV Sh 0.79-0.83 but DD -54..-57% (no DD cut: blow-ups are bought at lower weight but still bought);
  K=15/20/30: DD -47.7/-43.8/-38.3% but DEV Sh 0.76/0.74/0.73 and HOLD Sh 0.73/0.72/0.60 (dilutes the premium); sector cap 3/2 alone: DD unchanged -53.2%;
  top-1000 mix 30/50% and pure top-1000: worse in HOLD (Sh 0.72/0.63/0.47, pure M1000 HOLD median -0.3%).
VERDICT: sizing alone cannot hit the user's target.  It lifts Sharpe (DEV 0.78->0.92, HOLD 0.83->0.89) and trims DD (DEV -53%->-41%, HOLD -46%->-43%) but loses the >5% HOLD median (4.6%) and
  leaves the factor-unwind drawdowns (-40%) intact.  Useful as a COMPONENT (vol60-ann cap 100%, optionally 80%) for the stack, to be combined with a regime/vol-target/ddbreaker overlay.
