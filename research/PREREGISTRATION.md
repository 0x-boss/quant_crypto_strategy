# Pre-registration: market-neutral perp sleeve (written BEFORE any result was seen)

Goal: find a return stream with low correlation to the BTC/ETH trend book, so the combination - not any single tuned signal -
lifts risk-adjusted return. Protocol fixed in advance to limit overfitting.

**Data**: Binance USDT-perp daily klines for every symbol in the public archive (including delisted ones: LUNA, FTT, SRM, ...),
daily at 00:00 UTC; actual 8h funding payments. 2020-09 -> 2026-10.

**Universe (point-in-time)**: at each weekly rebalance, the top 20 symbols by trailing 30-day median quote volume, with >= 90 days of
history; non-crypto symbols (equity/index perps) and stable/pegged contracts excluded by an explicit list. Delisted coins stay in
until they stop trading (their final return is realised).

**Construction (identical for every candidate, nothing tuned)**: rebalance every Monday 00:00 UTC; dollar-neutral, long the top quintile
and short the bottom quintile of the score (equal weight inside each leg, 100 % gross per leg before scaling); sleeve scaled to 20 %
annualised vol using trailing 30-day sleeve vol (cap 2x); costs 10 bp one-way on traded notional; funding paid/received per day on the
actual position (long pays positive funding, short receives it).

**Candidates (6 trials total, nothing else will be run on this data)**
1. MOM30  : score = 30-day return
2. MOM90  : score = 90-day return
3. REV7   : score = - 7-day return
4. LOWVOL : score = - 30-day realised vol
5. FUND14 : score = - trailing 14-day mean funding rate (long lowest funding, short highest)
6. MULTI  : equal-weight average of the cross-sectional ranks of 1-5 (defined now, not selected later)

**Acceptance rule for a sleeve**: net Sharpe > 0.5 in BOTH eras (2020-09..2022-12 and 2023-01..2026-10); full-sample t-stat > 2;
|corr| to the TrendCore-I daily returns < 0.3. Anything failing is rejected and reported.

**Combination rule**: accepted sleeves get equal risk budget with the trend book (inverse-vol, fixed 50/50 risk), no optimisation.
Reported with the deflated Sharpe for 6 trials + the ~400 earlier trials.

---
## Round A result (recorded): none of the 6 candidates passed the acceptance rule
MOM30 (Sh 0.35 / 1.38), MOM90 (-0.25 / 0.94), REV7 (-1.26 / -1.22), LOWVOL (1.38 / -0.15), FUND14 (-0.42 / 1.15), MULTI (1.21 / 0.35).
All rejected. MOM30 came closest (full Sharpe 1.00, t 2.4, corr to trend +0.03) but fails era 1; the rule is not relaxed.

## Round B (written BEFORE running): time-series trend on the survivor-corrected perp universe
Motivation: the earlier daily-data tests used survivor-only coins; this universe contains delisted coins (LUNA, FTT, ...), so shorts now
capture collapses. Same universe, weekly Sunday-close rebalance, 10 bp costs, actual funding.
* B1 TSMOM-LS : score = mean(sign(r30), sign(r90), sign(r180)) in [-1, 1]; weight = score x (20 % target vol / own 30d vol) / N; long & short.
* B2 TSMOM-LO : same with score clipped at 0 (long-only).
Both scaled to 25 % ex-ante portfolio vol (trailing 30d, cap 2x). Acceptance: Sharpe > 0.5 in both eras, t > 2.
If accepted and corr to TrendCore-I < 0.6: combined 50/50 by risk with TrendCore-I (fixed, no optimisation).
Trials counted in the family so far: 6 + 2 = 8 (plus ~400 earlier trials on other designs).

---
## Round C (written BEFORE running): BTC signal, higher-beta holding   (spot-only, long-only, gross <= 1, no leverage)
Idea (user): compute the trend/regime signal on BTC only, but hold ETH and/or SOL (or an alt basket) while the signal is on.
Common rules: signal = BTC trend score x BTC SMA-gate (exactly TrendCore: mix of momentum/SMA/EWMAC, gates 100/150/200d bagged);
sizing per bar = f_btc x 0.45 / vol, then 40 % portfolio vol target with scalar capped at 1.0, BTC open-interest overlay 0.5, gross <= 1;
6h bars, hourly realised vol; costs 10 bp BTC/ETH, 25 bp other; 5 % no-trade band.
Variants (7 trials, 6h framework, spot data 2018-/2020-09):
  C0 BTC held (baseline)      C1 ETH, own-vol sizing    C2 ETH, BTC-vol sizing ("swap")
  C3 SOL, own-vol sizing      C4 SOL, BTC-vol sizing    C5 ETH+SOL 50/50 own-vol    C6 ETH+SOL 50/50 BTC-vol
Survivor-corrected check (daily bars, 900-symbol perp archive incl. delisted): C7 = top-3 non-BTC symbols by trailing 30d median quote volume
(point-in-time, weekly rebalanced, equity/commodity perps excluded), equal weight, own-vol sizing, 25 bp costs.
Eras: 2020-09..2022-12 and 2023-01..2026-10 (SOL spot exists from 2020-08).
Acceptance: variant must (a) have Sharpe >= baseline Sharpe in BOTH eras, (b) higher full-period CAGR than baseline, (c) max drawdown no worse than -30 %.
SOL results are reported with an explicit hindsight warning (known survivor); C7 is the decisive, survivor-corrected test.

### Round C result (recorded)
6h framework: all 7 variants fail acceptance (Sharpe era1/era2 vs baseline 1.38/1.30): C1 1.66/0.98, C2 1.56/1.02, C3 2.02/1.09, C4 1.76/1.16, C5 2.32/1.17, C6 1.99/1.26 (misses era 2 by 0.04; CAGR 45.0 % vs 33.1 %, DD -26.7 %).
Survivor-corrected C7 (top-3 non-BTC by volume, weekly, delisted coins included; daily framework): CAGR 17.1 %, Sharpe 0.77, DD -31.2 % vs BTC baseline 28.8 % / 1.19 / -25.4 % - **worse in both eras**. Buy & hold of that basket: CAGR -10 %.
=> REJECTED. The SOL/ETH+SOL gains are hindsight (SOL = known survivor); not adopted.

---
## Round D (written BEFORE running): can idle months be predicted, and does gold on idle capital help?  (spot-only)
Q1 (descriptive): relation between exposure at the start of a month and that month's result (final BTC+ETH strategy, 2018-03+).
Q2 (3 variants): idle capital (1 - exposure) held in spot gold PAXG: G0 none (baseline) | G1 always PAXG | G2 PAXG only while PAXG > its own 200d SMA, else cash.
Costs 25 bp one-way on PAXG weight changes, 5 % no-trade band; daily decisions at the 00:00-UTC close. Eras 2020-09..2022-12 and 2023-01..2026-10.
Acceptance: higher CAGR and Sharpe >= baseline in BOTH eras and max drawdown no more than 3 pts worse. Gold's 2024-25 rally = hindsight risk, reported.

### Round D result (recorded)
Q1: idle months are predictable from exposure at month start (31 of 36 idle months flagged; 8 false alarms; idle-start months: mean +0.9 %, worst -0.8 %); underperformance is NOT: 15 loss months worse than -3 %, 9 began >50 % invested, 6 partly in, 0 idle. Strategy < BTC in 33 of 35 big BTC-up months (by design, ~37 % upside capture).
Q2: gold on idle capital FAILS acceptance. G1 always-PAXG: Sharpe 1.24 / 1.53 (era1 / era2) vs baseline 1.56 / 1.17, DD -29.2 % vs -19.2 %. G2 trend-gated: 1.12 / 1.51, DD -32.5 %. Gain is entirely gold's 2024-25 rally (+30 %, +65 %); era1 worse; drawdown rises ~10 pts. Not adopted.

---
## Round E (written BEFORE running): SPY < 200-EMA as a risk-off regime -> hold gold   (user idea; spot-only)
Regime: s = 1 when SPY close < EMA(200) of SPY closes (pandas ewm span=200, adjust=False; SPY daily closes from Yahoo). Known after the US close of
day t-1, applied to the crypto/gold period starting 00:00 UTC of day t (weekends/holidays carry the last value). No parameter tuning (200 as specified).
Part A - does the mechanism exist? (long history, bootstrap CIs, sub-periods)
  A1 gold (GLD, 2004-2026): return and Sharpe when s=1 vs s=0.   A2 BTC (2014-2026): same.   A3 ETH (2016-2026).
  Mechanism is supported only if gold does better AND crypto does worse when s=1, with the sign stable across sub-periods.
Part B - strategy variants on the final BTC+ETH strategy (6h, spot-only), GLD as gold (PAXG variant for crypto-venue implementability):
  S0 baseline | S1 crypto x0 when s=1 (cash; no gold) | S2 crypto unchanged, idle capital -> gold when s=1 | S3 when s=1 crypto x0 and 100 % gold.
Costs: 10 bp crypto, 5 bp GLD (25 bp PAXG) on regime-flip turnover.  Eras: 2018-03..2022-12 and 2023-01..2026-10.
Acceptance: Part A supports the mechanism AND the variant raises full-period Sharpe and lowers max drawdown AND is not worse in era 2 (Sharpe >= baseline - 0.02)
AND positive effect in era 1.  Trials: 3 variants (S1-S3) + PAXG repeat of the winner only.

### Round E result (recorded; `research/56_spy_gold.py`, `results/spy_gold.{log,json}`, `spy_gold_{episodes,years,daily}.csv`)
Part A (mechanism): **not supported**. Gold (GLD 2004-11..2026-10): s=1 +13.0 %/yr vs s=0 +11.2 % (diff +1.9 pts, 90 % block-bootstrap CI [-17.6, +21.7]; Sharpe 0.54 vs 0.68);
sub-periods 2004-10 -35.3 pts, 2011-16 +41.2, 2017-21 +13.7, 2022-26 +6.1 -> sign not stable. BTC (2014-): -54.4 pts (CI [-139.5, +29.9]); 2014-16 -17.9, 2017-19 -346, 2020-22 -99, **2023-26 +219** -> not stable.
ETH (2016-): **+53.8 pts** (crypto better in s=1; CI [-130.6, +243.0]); the first ETH block holds 2016 only. Gold-better = False, BTC-worse = False, ETH-worse = False -> mechanism fails (it also fails under a looser "full-sample sign" reading because of ETH).
Part B (2018-03..2026-10, Sharpe era1 / era2 / full, max DD): S0 1.49 / 1.17 / 1.36, -20.2 % | S1 1.64 / 0.90 / 1.33, -19.9 % | S2 1.45 / 1.41 / 1.43, -30.5 % | S3 1.59 / 1.18 / 1.42, -27.0 %.
Acceptance (all conditions): S1 fails (full Sharpe down, era 2 -0.27), S2 fails (DD, era 1 Sharpe down), S3 fails (DD +6.8 pts worse), and Part A fails for all -> **0 of 3 accepted; no PAXG repeat**.
Context: s=1 on 23 % of trading days; 35 runs since 2018-03 (27 lasting < 20 days, ~8 flips per year); the baseline had mean exposure 9 % in s=1 days vs 32 % in s=0 days (+14 % compounded in s=1 days vs +890 %).
Stress (not trials): gold entered 1 / 3 days after the signal -> S3 full Sharpe 1.49 / 1.44; gold cost 25 bp -> 1.35.
