# Round 8, stage 2 - the stack (pre-registered BEFORE it was run; follows the critic's review of the 269 family configs)

What the families established (all verified free of look-ahead by independent re-implementation; every verifier rated them "use with caution"):
* vol clustering lets a lagged own-vol scale-down cut the fast unwinds (voltarget); a vol60 <= 100 % per-name cap removes the idiosyncratic blow-ups (sizing).
  Destination of freed capital (defensive), DD breakers, dispersion half-cut, 52-week-high filters, exits: no independent value / not robust.
* The HOLD median is mechanically coupled to exposure: HOLD median ~ -5.7 % + 10.9 % x average gross; DD relief ~ 8 pp per 10 pp of gross.
  No config of the 269 has HOLD median >= 5 % together with HOLD DD <= -31 %. The baseline itself passes "median > 5 %" by ONE month (29 of 57).

## The factorial (10 distinct cells, nothing else is run for selection)
A selection layer : `base` (equal-weight top-10 by 6-1 momentum)  |  `vcap` (same, names with annualised vol60 > 100 % excluded and replaced by the next-ranked; = sizing config eq_k10_vcap100, no inverse-vol, no sector cap)
B exposure layer  : `none`  |  `exp80` (voltarget own20_exp80: s = min(1, sigma*/sigma_hat), sigma_hat = 20-day vol of the stacked book's own lagged returns, sigma* = causal EXPANDING 80th percentile)  |  `rollq80` (same, sigma* = ROLLING 504-day 80th percentile, 252 min)
C destination     : `cash`  |  `BIL` (freed capital held in the T-bill ETF; fees on that leg are paid; BIL carry is risk-free carry, not alpha)
(C is irrelevant when B = none.)  Because sigma* is a percentile of the stacked book's own vol, every B layer is automatically recalibrated on the book it sits on.

## Selection (DEV only)
eligible = pre-registered `dev_objective` finite (keeps >= 80 % of baseline DEV mean month, >= 85 % of the median, cuts DEV max DD >= 20 %)
           AND DEV Sharpe at 2x fees >= baseline's 2x-fee DEV Sharpe + 0.05
           AND DEV Sharpe with the 2021-02-12..2021-05-10 window removed >= baseline's same-window-removed Sharpe + 0.02.
chosen   = eligible cell with the highest DEV objective; within 0.02 prefer fewer components. If none is eligible the answer is "no stack passes".

## Final evaluation of the chosen cell (HOLD is looked at ONCE for it; the other cells' HOLD numbers are printed for information only)
1. look-ahead: R.check_perturbation (9 dates) on the full cell builder + truncation check.
2. acceptance restated in paired, low-noise terms (the absolute 5 % median is a one-month statistic):
   (a') HOLD CAGR retention >= 0.95 of the baseline, Calmar and Ulcer better, paired month-bootstrap CI of (stack - baseline) for the median month and the share of months >= 5 %;
        the absolute HOLD median and P(median > 5 %) are reported as information.
   (b)  HOLD max DD and ALL max DD at least one-third smaller than the baseline.     (c) Sharpe better than the baseline in BOTH DEV and HOLD.
3. timing placebo: 200 circular shifts (>= 63 days, 21-day blocks) of the exposure series applied to the unscaled selection-layer book -> Sharpe / DD distribution vs the actual timing.
4. cost and lag stress: 2x, 4x fees, lag 1; DEV Sharpe without 2020 and 2021.
5. generalisation: the same B layer on sibling books (k = 20; 12-1 momentum; top-1000 universe; residual momentum) - paired change in Sharpe and DD, DEV and HOLD.
Do not iterate on the stack after seeing HOLD; anything further is a new, separately registered round.
