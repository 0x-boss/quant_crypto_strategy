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
