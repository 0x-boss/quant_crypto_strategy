R8 family DEFENSIVE (src/r8_f_defensive.py, log logs/r8_defensive.log, trials_defensive.csv = 33 pre-registered + 4 post-hoc rows)
Idea: where does the capital freed by a risk-reducing rule go?  (T) W = s*W0 + (1-s)*D, s = min(1, sigma*/vol20 of lagged own returns), sigma* = DEV median
0.2501 (hard-coded, audited; "E" = causal expanding median); D = cash/BIL/GLD/TLT/USMV/SPLV/IEF/50-50 GLD+TLT/50-50 USMV+GLD.  (M) static 90/10, 80/20, 70/30 mixes.
(V) inverse-vol dynamic mix (20d lagged sleeve vol vs 20d ETF vol), uncapped or defensive <= 30%.  ETF weight = 0 before listing (freed capital -> cash).
Grid: 33 pre-registered (9 trigF, 5 trigE, 13 mixes, 6 ivol).  ivolcap30_{gld,tlt,usmv} == mix70_{gld,tlt,usmv} exactly (cap always binds: sleeve vol >> ETF vol) -> 30 distinct.
Baseline: DEV Sh 0.776 med 2.15% mean 2.24% DD -53.2% | HOLD Sh 0.825 med 5.61% DD -46.0%.  DEV floors: mean >= 1.79%, med >= 1.83%, DD <= -42.5%.

RESULT: 6/33 configs have a finite DEV score (5 distinct); the 3 DEV-ranked finalists all pass the perturbation guard (worst diff 0):
  trigF_usmv_gld  score .728  DEV Sh 0.886 med 2.05% DD -31.6% | HOLD Sh 1.105 med 3.77% DD -25.5% | ALL Sh 0.96 DD -32.4%
  trigF_usmv      score .715  DEV Sh 0.898 med 2.11% DD -36.6% | HOLD Sh 0.997 med 2.53% DD -25.2%
  trigF_splv      score .710  DEV Sh 0.900 med 2.13% DD -38.1% | HOLD Sh 0.986 med 2.96% DD -26.5%
Stress of the DEV-best (trigF_usmv_gld): DEV Sh 0.886 -> 0.779 at 2x fees (baseline 0.704), 0.904 with lag=1 (baseline 0.768).  Turnover 0.106/day (baseline 0.109).
Acceptance rule (HOLD, informational): DD ok (-25.5%), Sharpe > baseline in DEV and HOLD ok, but HOLD median 3.8% < 5%: NO config with a finite DEV score keeps it.

MECHANISM (compound return in window; baseline -> cash twin trigF_cash -> trigF_usmv_gld; stock exposure s the day before / mean in window):
  2021-02-12..05-10  -50.8% -> -17.3% -> -14.6%  (s .45/.26; USMV +7.2%, GLD +0.2%)    2025-02-12..04-04  -41.0% -> -11.4% -> -12.5%  (s .36/.25; USMV -7.3%, GLD +4.4%)
  2020-02-19..03-18  -44.1% -> -21.4% -> -30.6%  (s .76/.45; USMV -26.6%, GLD -7.9%)    2026-06-02..07-28  -37.2% -> -12.7% -> -15.4%  (s .36/.31; USMV +2.5%, GLD -11.1%)
  2023-08-01..10-30  -37.7% -> -22.1% -> -22.4%  (s .64/.52)
The drawdown cut is done by the TRIGGER (vol clustering, same as voltarget), not by the destination.  Same average DEV exposure without timing (post-hoc ph_const17_usmv_gld,
17% USMV+GLD) gives DEV Sh 0.80, DD -45.2%.  The destination is second order and sign-unstable: it helps when uncorrelated (2021: +2.7pp vs cash) and hurts in a market crash
(COVID: USMV -26.6% turns -21.4% into -30.6%; GLD sold off too).  Its main effect is its OWN RETURN: DEV ann. USMV 13.9% (Sh 1.02), SPLV 12.7%, TLT 4.5%, IEF 2.6%, GLD 0.9%;
HOLD USMV 6.2%, GLD 18.3% (Sh 1.01), TLT -9.5%, IEF -2.2%.  USMV wins DEV, GLD wins HOLD (trigF_gld HOLD Sh 1.17, but it misses the DEV floors): regime luck, not protection.
BIL CARRY (risk-free carry, NOT alpha): BIL ann. return DEV 0.45%, HOLD 3.89%.  trigF_bil vs same-trigger cash twin trigF_cash: DEV Sh 0.860 vs 0.870 (the 0.1%/side fee on
the BIL leg eats the ZIRP carry), HOLD Sh 0.973 vs 0.935 (+0.04, pure carry).  SGOV starts 2020-06, not usable in DEV.  With cash the trigger fails the DEV mean floor (1.72%).
Static mixes cut DD only to -39..-49% (70/30: -39% to -40%, 80/20 -43..-45%), Sharpe .78-.85, HOLD med 3.6-4.9% (90/10: 5.1-5.3% but DD -42%): none beats the trigger.
Uncapped inverse-vol is the Sharpe/DD champion but kills the median: ivol_usmv DEV Sh 1.12 DD -33% med 1.75% mean 1.53%; ivol_gld_tlt DEV Sh 0.92 DD -16.9% med 0.97% (both fail the floors).

PLATEAU: Sharpe/DD plateau is good, the identity of the winner is NOT robust.  All 9 fixed-trigger destinations: DEV Sh 0.857-0.900, DD -32..-38%, no-floor score .697-.731
(cash twin .709); trigF_tlt (.731) would win without floors.  Eligibility hangs on the DEV mean-month floor 1.79%: usmv_gld 1.84, tlt 1.78, gld_tlt 1.76, ief 1.75, gld 1.74, cash 1.72.
Neighbours of the winner: trigF_usmv .898/-36.6% (.715), trigF_splv .900/-38.1% (.710), trigE_usmv .891/-36.5% (.709) all finite; post-hoc sigma* neighbours: expanding median
(ph_trigE_usmv_gld) .880/-31.6% (.725), DEV q65 .880/-33.9% (.714), q80 .87/-37.3% (.684, but HOLD med 5.5%, DD -34.4%, ALL DD -39.5%): smooth monotone, sigma* is the median-vs-DD dial.

POST-HOC (flagged posthoc=True in CONFIGS, logged note "post-hoc", never used to select): ph_trigE_usmv_gld, ph_trigF65_usmv_gld, ph_trigF80_usmv_gld, ph_const17_usmv_gld.
CAVEATS: 33 configs on one DEV window; the 0.01pp-level floor cliffs decide eligibility; the HOLD gain of the winner (Sh 1.1) is mostly GLD's +18%/yr, not selectable evidence.
VERDICT: The family beats the baseline on the DEV objective (0.728 vs 0.510 raw for the baseline, which itself fails the DD condition) but adds only ~+0.02 over putting the freed capital
in cash (0.709) and nothing in market-crash windows; the destination changes the return path, not the protection.  It does NOT restore the HOLD median > 5%.  Use the trigger, treat the destination as a
low-conviction ~free carry/return sleeve (USMV+GLD diversifies the DEV/HOLD regimes); BIL is carry only.
