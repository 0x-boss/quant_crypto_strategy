R8 family DDBREAKER - drawdown circuit breakers on the strategy's own lagged equity (src/r8_f_ddbreaker.py)
Setup: eq = cumprod(1+lagged(r)) of the UNSCALED baseline book; dd = eq/rolling-max(win) - 1 (peak restarts at each re-entry); dd <= -thr -> exposure cNN (NN% of book) until re-risk.
Grid: 33 pre-registered (win 126/252 x thr 10/15/20 x cut 50/0 x re half|d10 + centre 252/15: high,d5,d21,mkt for cut 50 and 0 + ladder) + 5 POST-HOC probes (PH_*). All logged in results/r8/trials_ddbreaker.csv.
Look-ahead: perturbation guard OK (worst 0.0) for all runner-guarded configs; extra truncation checks (T0 2019-06-28, 2021-03-15) on both finalists: max |diff| 0.
Baseline: DEV Sh 0.78 med 2.15% DD -53.2% | HOLD Sh 0.83 med 5.61% DD -46.0%.

DEV-objective ranking (pre-registered grid; only 3 of 33 pass the filters; w126/w252 are identical books at thr 10%):
 1. w126_t10_c50_d10 (= w252_t10_c50_d10)  DEV Sh 0.86 med 2.07% DD -40.6% | HOLD Sh 0.85 med 4.46% DD -33.9%   obj 0.655
 3. w252_t15_c50_d21                       DEV Sh 0.80 med 2.07% DD -38.1% | HOLD Sh 0.73 med 5.61% DD -42.0%   obj 0.611
 Post-hoc PH_w126_t10_c25_d10 scores 0.669 (DEV Sh 0.87, DD -40.7%) but is post-hoc, margin tiny, HOLD med 2.5% -> not promoted.
Winner robustness (DEV Sh): 2x fees 0.770 (base 0.704); lag=1 0.852 (base 0.768). HOLD: 2x fees 0.779 (base 0.775), lag1 0.907 (base 0.848). Turnover 11.5%/d vs 10.9%.
 Paired monthly-block bootstrap of DEV Sharpe gain: +0.08, 90% CI [-0.06,+0.21], P(>0)=0.82 -> NOT significant. Constant scaling to same avg gross (0.92): DEV Sh 0.78, DD -49.6% (so timing adds ~+0.08 Sh, ~-9pp DD).

Mechanism (winner vs baseline compound return in the five windows; avg exposure in brackets):
 2021-02-12..05-10 -50.8% -> -37.7% (0.72) | 2025-02-12..04-04 -41.0% -> -24.6% (0.80) | 2020-02-19..03-18 -44.1% -> -29.7% (0.67)
 2026-06-02..07-28 -37.2% -> -29.8% (0.62) | 2023-08-01..10-30 -37.7% -> -32.2% (0.80).  (2nd finalist: -35.0, -30.8, -32.8, -30.2, -35.0)
 The DEV max DD is still the Feb-May 2021 unwind (-40.6%): the breaker trips after the first -10% and a 10-day half-exposure stint is too short for a 3-month unwind.
 It works as a DD-triggered, lagged vol-target: baseline vol on cut days 66% (DEV) / 78% (HOLD) vs 32% / 48% off, while mean daily return is about the same on cut days (9.9 vs 11.8 bp DEV; 23.9 vs 17.1 bp HOLD) - it removes variance, it does not avoid negative drift.
 Episodes: 30 in DEV (12% of days cut), 31 in HOLD (26% of days cut); ann vol DEV 37.5->31.9%, HOLD 57.5->46.2%.
Whipsaw cost: top-quartile months DEV 13.1->12.1%, HOLD 20.5->17.3%; bottom quartile DEV -8.3->-7.2%, HOLD -13.9->-10.5%. Months >=5% that drop below 5%: 3 (DEV), 3 (HOLD); HOLD median 5.61->4.46%;
 month-bootstrap P(HOLD median > 5%) falls 0.55 -> 0.25. Biggest single-episode cost: cut 2020-09-09 (base +16.9% in 10 days), 2021-10-05 (+15.1%); biggest save: 2020-02-27 (-22%).
 Slow re-entry is fatal: cash (c0) with half / high / ladder re-entry misses the V-rebounds: DEV med 0.0-1.0%, HOLD med 0.0-0.6%, avg gross 0.39-0.76 (w252_t15_c0_high DEV Sh 0.43, HOLD 0.23); at c50 half/high still give DEV med 1.5-1.9%, HOLD 2.3-3.3%.
 Cash (c0) with d10 keeps Sharpe (DEV 0.86) but median 1.7% (<85% of base) so it fails the filter; at thr 15% the mkt and d5 re-entries leave the 2021 DD at -54.5% (no help).

Plateau (neighbours of w126_t10_c50_d10; DEV Sh / DEV DD / DEV med):
 thr: 12.5%* 0.795/-50.4/2.1 | 15% 0.763/-54.1/2.0 | 20% 0.803/-52.3/2.0   -> CLIFF: DD benefit exists only at the 10% trigger (single-episode dependence on 2021)
 re-risk: half 0.778/-41.8/1.7 (fails median) | d5* 0.827/-43.4/2.0 | d21* 0.846/-44.5/2.0   -> Sharpe plateau ok, DD gain 16-18% (<20% filter)
 exposure: 75%* 0.822/-46.9 | 50% 0.858/-40.6 | 25%* 0.873/-40.7 | 0% 0.859/-43.5/1.7    -> Sharpe 0.82-0.87 all > base 0.78; (* = post-hoc probe)
 win 126 vs 252: identical at thr 10% (peak restarts at each re-entry), so window is not a real parameter.

Verdict vs R8 acceptance rule: (a) HOLD median >= 5%: FAIL (4.46%); (b) DD cut by 1/3: FAIL (DEV -24%, HOLD -26%, ALL -40.6% vs target -36%); (c) Sharpe > base in DEV and HOLD: pass but HOLD only +0.02.
The family gives lighter DD and a modest Sharpe lift in DEV, but it trades the HOLD median for it. Its effect is vol-reduction and likely overlaps with the voltarget family - treat as a candidate stack component only after checking that overlap.
