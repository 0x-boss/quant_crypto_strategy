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
