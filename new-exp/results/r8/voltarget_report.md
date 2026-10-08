R8 family VOLTARGET  (src/r8_f_voltarget.py, log logs/r8_voltarget.log, trials_voltarget.csv = 30 rows)
Idea: W * min(1, sigma*/sigma_hat); sigma_hat = own lagged-return vol (10/20/40/60d, EWMA .94/.97), vol of held names, xs dispersion of 21d returns.
Grid: 30 pre-registered configs (24 with sigma* = DEV 50/65/80th pct of sigma_hat of the unscaled book, hard-coded; 6 with causal expanding pct). No post-hoc configs.
Baseline: DEV Sh 0.776 med 2.15% DD -53.2% | HOLD Sh 0.825 med 5.61% DD -46.0%.

RESULT: 15/30 configs have a finite DEV objective; 29/30 beat the baseline DEV Sharpe; ALL 30 cut DEV maxDD. NONE keeps the HOLD median >= 5% (max 4.85%).
DEV-ranked finalists (all 3 pass the perturbation guard, worst diff 0):
  own10_q65   (10d own vol, sigma*=0.3012)  DEV Sh 0.908 med 1.95% DD -32.7% | HOLD Sh 0.795 med 2.98% DD -30.1% | avg gross 0.82
  own20_q65   (20d own vol, sigma*=0.3075)  DEV Sh 0.885 med 2.02% DD -31.2% | HOLD Sh 0.928 med 3.47% DD -27.5% | avg gross 0.82
  own20_exp80 (20d, expanding 80th pct)     DEV Sh 0.880 med 2.07% DD -31.2% | HOLD Sh 0.921 med 4.85% DD -35.8% | avg gross 0.90
Stress of the DEV-best (own10_q65): DEV Sh 0.908 -> 0.809 at 2x fees (baseline 0.704), 0.926 with lag=1 (baseline 0.768). Turnover 0.105/day (baseline 0.109).

MECHANISM (return inside the five windows, baseline -> own10_q65; mean exposure s in window; s the day before the window):
  2021-02-12..05-10  -50.8% -> -20.5%  (s .32; .62 before)     2025-02-12..04-04  -41.0% -> -17.0%  (s .30; .46)
  2020-02-19..03-18  -44.1% -> -22.6%  (s .49; .89)            2026-06-02..07-28  -37.2% -> -15.5%  (s .36; .35)
  2023-08-01..10-30  -37.7% -> -29.6%  (s .66; .73)
Vol clustering does the work: momentum unwinds are preceded/accompanied by a vol spike, the lagged 10-20d vol cuts the book to ~30-50% within days, so
the unwind is roughly halved in 4 of 5 windows (weak in 2023-Q3: slow grind, vol never spiked; COVID: the first leg is taken at s=.89).
It is timing, not deleveraging: a constant 0.885 scale of the baseline (same average DEV exposure) gives DEV Sh 0.776, DD -48.1%, HOLD med 4.99%.
The overlay's DEV worst drawdown is the slow 2018-01-23..12-24 bleed (-32.7%; baseline over the same span -34.2%): barely touched, Q4-2018 -35.1% -> -29.4%.
COST: the same scaling shrinks every calm-and-strong month. DEV median/mean month 2.15/2.24% -> 1.95/1.97%; HOLD median 5.61% -> 2.98-4.85%, HOLD mean 3.49% -> 2.1-3.2%.
Fixed DEV levels are too tight for HOLD (strategy vol 0.43-0.48 median in HOLD vs 0.25 in DEV): avg exposure in HOLD 0.69 (own10_q65) -> median month 3.0%.
The causal expanding level (own20_exp80) keeps exposure at 0.85 in HOLD and the median at 4.85% (month-bootstrap P(median>5%) 0.45 vs 0.55 baseline).

PLATEAU (DEV Sharpe / DEV maxDD; q = 50/65/80): own10 .883/.908/.884 DD -33/-33/-35 | own20 .870/.885/.868 | own40 .829/.835/.832 | own60 .785/.784/.780 (DD -32/-35/-42).
Flat in q, monotone worse with longer windows (10 ~ 20 > 40 > 60): fast crashes want fast vol. Neighbours of the winner: own10_q80 obj .711, own20_q65 .729
(own10_q50 is -inf only because its DEV mean month 1.77% misses the 80%-of-baseline floor 1.79% by 0.02pp). EWMA .94 ~ own20 (.86); EWMA .97 ~ .81.
held20 (.84, DD -34%) is comparable; xs dispersion21 is weak (DEV Sh .76-.79, DD -43..-50%, no real protection). All q50 configs fail a DEV floor (mean month; DD for disp21).

CAVEATS: (1) the DEV winner (10d) is NOT best in HOLD (Sh 0.795 < baseline 0.825) while 20d is (0.93): the 10d-vs-20d choice is inside DEV noise. (2) Fixed sigma*
is a DEV in-sample level (one number); the expanding configs need no DEV constant and give the same DEV result (.880 vs .868 at q80). (3) 5 of 15 finite configs
miss HOLD Sharpe > baseline; HOLD maxDD <= -31% (plan acceptance) is met by 6/15 but only with HOLD median <= 3.8%.

VERDICT: voltarget delivers the drawdown (maxDD DEV -53% -> -31..-33%, HOLD -46% -> -28..-36%) and the Sharpe (+0.1 DEV, +0.1 HOLD for 20d) but it does NOT keep the >5% HOLD median; it is a
Sharpe/DD component, not a median-month component. Best compromise for the stack: own20_exp80-style (high percentile, causal level). Nothing selected on HOLD.
