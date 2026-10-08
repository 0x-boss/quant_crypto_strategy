R8 FAMILY REGIME - factor-aware regime filters (module src/r8_f_regime.py, 42 configs logged in results/r8/trials_regime.csv)
Baseline MOM_6_1_k10: DEV Sh 0.78 med 2.15% DD -53.2% | HOLD Sh 0.83 med 5.61% DD -46.0%.  Book is multiplied by a {1, 0.5, 0} flag with hysteresis.

1. PRE-REGISTERED GRID (31 configs: momentum-factor trailing return/vol, held & universe breadth, dispersion, VIX level/term structure,
   Daniel-Moskowitz bear+vol rule, held-names 5d shock, votes, composites): ALL 31 score -inf on the DEV objective. Run-1 logs kept in
   results/r8/*_run1_prereg.*. Closest misses: vote_fac4 (DD -40.6%, but mean month 1.51% < 1.79% floor, DEV Sh 0.63), hb50_step (DD -41.9%,
   med 1.51%), disp_step (DD -33.8%, mean 1.53%, med 1.57%, Sh 0.73).  dm_* rules never trigger in DEV (0.4% of days) -> no DEV information.
2. WHY (DEV-only diagnostic, next-day Sharpe of the unscaled book by flag state, unconditional 0.78): in the deep "flat" states it is 1.5-5.0
   (wml21 2.6, wmlvol21 1.8, disp 1.9, vix_a 3.5, shock 5.0): stress/crash states are followed by V-shaped rebounds, so cutting to 0 destroys
   the mean month. Only the early/medium "half" states carry information (disp half -0.17, shock half -1.6, vts_a 0.40, wmlvol63 0.58).
3. POST-HOC (appended, named ph_*, 11 configs, flagged): 8 = half-ONLY counterparts of the informative indicators + 1 union vote, then 3 plateau neighbours.
   Written after seeing run-1 DEV and (printed) HOLD columns of run 1 -> treat as exploratory, selection bias is real.
   Only dispersion survives: flag = expanding percentile of the IQR/1.349 of 21d returns of the PIT top-300 (Stivers-Sun 2010).
4. FINALISTS (DEV objective; perturbation guard OK, worst |dW| 0; truncation check OK at 2019-06 and 2024-03)
   ph_disp_half80_x75 half at pct>=0.80, back to 1 below 0.75: DEV Sh 0.908 med 2.18% DD -32.5% | HOLD Sh 0.842 med 4.09% DD -31.4%
   ph_disp_half80     half at 0.80, exit 0.70:                  DEV Sh 0.866 med 1.97% DD -34.7% | HOLD Sh 0.890 med 3.53% DD -32.5%
   ph_disp_half85     half at 0.85, exit 0.75:                  DEV Sh 0.830 med 2.10% DD -35.5% | HOLD Sh 0.817 med 3.99% DD -32.4%
5. COST/LAG of the DEV-best (ph_disp_half80_x75): 2x fees DEV Sh 0.807 (baseline 2x 0.704); lag=1 DEV Sh 0.846 (baseline lag1 0.768).
   Turnover 0.114/day vs 0.109; avg gross DEV ~0.88, HOLD ~0.71.
6. MECHANISM (return in the five windows, baseline -> best): 2021-02..05 -50.8 -> -30.8% (flag half on 77% of days); 2025-02..04 -41.0 -> -27.7;
   2020-02..03 -44.1 -> -20.3; 2026-06..07 -37.2 -> -19.7; 2023-08..10 -37.7 -> -30.7 (flag mostly off, dispersion was low: unwind NOT caught).
   Dispersion of 21d returns jumps 1.5-2.6x during factor unwinds (0.09-0.16 vs 0.06 normal in Feb-Mar 2021) and is visible while SPY is up.
   Not just 2021: DEV excluding Feb-Jun 2021 Sharpe 0.911 vs 0.849 and maxDD -32.0% vs -44.5%. 63 half spells in DEV, 24.5% of days.
7. PLATEAU (DEV Sh / maxDD / objective): enter0.80 exit0.75 0.908/-32.5/0.746; 0.80/0.70 0.866/-34.7/0.692; 0.85/0.75 0.830/-35.5/0.652;
   0.75/0.65 0.832/-35.0/-inf (mean month 1.790% vs floor 1.791%); 0.80/0.60 0.792/-34.7/-inf (mean 1.71%); 0.90/0.80 0.826/-46.2/-inf (DD cliff).
   So the Sharpe 0.79-0.91 and DD -32..-36% gain holds across enter 0.75-0.85 / any exit; the formal pass is borderline at the edges.
8. WHAT FAILS THE USER'S TARGET: HOLD median month falls 5.61% -> 4.09% (month-bootstrap P(median>5%) 0.55 -> 0.11) because the book is half-sized
   on 58% of HOLD days (DEV 25%): the expanding percentile of dispersion drifts up after 2021 (2026: 90% of days half) - a non-stationarity seen
   in HOLD, NOT used to tune anything. HOLD Sharpe 0.84 vs 0.83 is a tie, not an improvement. Upside is given up in strong years (2020 72% vs 105%,
   2024 32% vs 49%, 2026 73% vs 102%; 2021 wins 52% vs 31%).  Lighter DD (-31% HOLD vs -46%) is the real, repeatable gain.
VERDICT: no pre-registered regime filter beats the baseline on the DEV objective. One post-hoc, single-indicator dispersion half-cut does (DEV Sh
+0.13, DD -53 -> -33) and is causal and cost-robust, but it trades the >5% HOLD median away; candidate for the stack only as a drawdown component,
ideally re-registered with a rolling (e.g. 3y) dispersion percentile before any HOLD look.
