# Round 8 - keep the >5 % median month of the 6-1 momentum sleeve, but cut the drawdown and lift the Sharpe

Written BEFORE any round-8 result was seen (the one result already known: a plain SPY>SMA gate *hurts* - see "Known facts").

## Goal and acceptance rule
Baseline `MOM_6_1_k10` (top-10 of the PIT top-300 by 6-1 momentum, equal weight, average of the last 5 daily books, 0.1 %/side, next-open):

| | Sharpe | median month | max DD | worst month |
|---|---|---|---|---|
| DEV 2012-21 | 0.78 | 2.15 % | -53.2 % | -22 % |
| HOLD 2022-26 | 0.83 | 5.61 % | -46.0 % | -25 % |

A modification is **accepted** only if, on the final out-of-sample look at HOLD, it has
(a) median month >= 5.0 % (note: bootstrap P(baseline HOLD median > 5 %) is only 0.55, so this is a coin flip even for the baseline - report it, do not pretend otherwise),
(b) max DD at least one-third smaller than the baseline (HOLD <= -31 %, ALL <= -36 %), and
(c) Sharpe better than the baseline in BOTH DEV and HOLD (>= 0.95 each is the aspiration).
No leverage (gross <= 100 %), long-only, spot, 0.1 %/side, next-open execution. Nothing here may use HOLD to choose.

## Known facts that shape the search (read them)
* The biggest drawdowns are **momentum-factor unwinds, not market crashes**: 2021-02-12 -> 2021-05-10 (-53.2 %, SPY +6.8 %), 2025-02 -> 04 (-46 %, SPY -16 %),
  2020-02 -> 03 (-44.5 %, SPY -29 %), 2026-06 -> 07 (-39 %, SPY -2 %), 2023-08 -> 10 (-38 %), 2018-Q4 (-36 %). The held names run ~61 % annualised vol (universe ~33 %).
* A plain SPY>SMA100/200 gate (full or half cut) did NOT reduce the DEV max DD (-53.2 % unchanged) and lowered Sharpe in 3 of 4 variants. Do not re-run it as a "new idea".
* Days with r < -5 % (146 of ~3 700) carry 28 % of the squared-return mass: the damage is concentrated in a few violent days/weeks.
* The HOLD median 5.61 % is fragile: scaling the book down mechanically (exposure < 1) lowers every month and will push the median below 5 %. Tail-trimming
  (cutting the loss months, not shrinking every month), better selection, and reducing idiosyncratic noise are what can keep it.
* Turnover is already ~11 % of NAV per day (2.75 %/yr of fees); an overlay that doubles it costs real Sharpe.

## Protocol (identical for every family)
1. Each family is a module `src/r8_f_<name>.py` (see `src/r8_f_example.py`): `FAMILY`, `CONFIGS` (pre-registered grid, 12-40 entries, every entry is run and
   logged), `apply(c, W, p)` (overlay on a weights frame) or `weights(c, p)` (build the book from scratch). Run it with `python src/r8_runfamily.py r8_f_<name>`.
2. Everything goes through `src/r8_common.py` (cache, `run`, `evaluate`, `log_trial`, `dev_objective`, `check_perturbation`, `lagged`). Read its docstring first.
   LOOK-AHEAD CONTRACT: row t of a weights frame is decided at the close of t with data through t. Strategy-return-based rules must use `lagged(r)`.
   The runner calls `check_perturbation` on the best configs - a LOOK-AHEAD verdict disqualifies the config.
3. Selection uses ONLY the DEV-only objective `dev_objective` (keeps >= 80 % of the baseline DEV mean month, >= 85 % of the DEV median, cuts DEV max DD by >= 20 %,
   then DEV Sharpe - 0.5 |DEV maxDD|). HOLD columns are printed for information; do not tune, re-grid or re-rank after looking at HOLD. If you add configs after seeing
   results, they must be appended to CONFIGS (so they are logged) and flagged `note: post-hoc` in your report.
4. Parameters come from priors/literature (cite the idea in a comment), not from searching. Grids are small on purpose: each family reports a *plateau* (are the
   neighbouring grid points of the winner also good?) and a *mechanism* (which drawdown episodes it actually fixes: use the five episode windows above).
5. Cost realism: after the grid, re-run the DEV-best config at 2x costs (`fee=2*R.FEE`) and with `lag=1`, report both.
6. Report honestly. A family that finds nothing says so. Return the structured JSON the workflow asks for.

## Families (one agent each)
| id | question |
|---|---|
| sizing | Can position weights / concentration cut the idiosyncratic blow-ups? inverse-vol (vol20, vol60, ATR) weights with a per-name cap, K in {10,15,20,30}, exclude names with vol60 above a cap, sector cap (max 3 names per `c['sector']`), blends of top-300 and top-1000 universes |
| voltarget | Barroso & Santa-Clara / Daniel & Moskowitz style: scale the book by min(1, sigma*/sigma_hat) where sigma_hat is the strategy's own trailing vol from `lagged(r)` (windows 10/20/40/60 d, sigma* grid chosen as quantiles of the DEV vol distribution so that the book is fully invested most of the time), EWMA versions, cross-sectional-dispersion and held-names-vol triggers |
| regime | Factor-aware regime filters (NOT a plain SPY gate): momentum-factor state (long-short top vs bottom decile trailing return/vol built from `c`), breadth of the held names (share above SMA50/20), dispersion of 21-day returns, VIX level / VIX3M term structure, SPY 24-month return with market vol (Daniel-Moskowitz bear-rebound rule), held-names 5-day return shock; stepwise exposure {0, 0.5, 1} with hysteresis |
| ddbreaker | Drawdown circuit breakers on the strategy's own lagged equity: cut to 50 %/0 % when the DD from the rolling 126/252-day peak exceeds 10/15/20 %, re-risk rules (new high, half-recovery, N days), asymmetric re-entry; evaluate whipsaw cost |
| signal | Better selection: risk-adjusted momentum (return/vol60), residual momentum, momentum with trend-quality filters (close > SMA50/200, high52 > 0.8), absolute-momentum filter (hold a name only if its own 6-1 return > 0, otherwise cash), frog-in-the-pan / smoothness, 6-1 + 12-1 rank blends, avoid names overextended above SMA50, k and stagger h variants |
| exits | Position-level exits as filters on the target weights: drop a name when close < trailing max(close, 20d) - m*ATR (m = 2, 3, 4), close < SMA20/SMA50, 5-day return < -X ATR, with replacement by the next-ranked name vs hold-cash; re-entry rules |
| defensive | What the freed capital does: cash vs BIL (T-bill ETF, accrues interest) vs GLD vs TLT vs USMV/SPLV (low-vol ETFs) vs a 50/50; fixed strategic mixes (e.g. 80 % momentum + 20 % defensive), inverse-vol dynamic mixes using lagged returns. (Use the first config of the winning *regime/voltarget/ddbreaker* style rule as the trigger so this family is testable standalone: trigger = DEV-best simple own-vol scale-down from the voltarget grid with sigma* = DEV median vol.) |

## After the family runs
Independent verifiers re-check each family's DEV-selected finalists (perturbation guard re-run, code review against the contract, cost x2, lag +1, quarterly
re-selection of the universe, plateau). Only verified components enter a **stack** that is built by a pre-registered greedy rule on DEV
(add one verified component at a time, keep it only if the DEV objective improves, max 4 components). The stack - and ONLY the stack - is then evaluated on HOLD once,
with a random-overlay placebo, month-bootstrap of the HOLD median, and a walk-forward re-selection check.
