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

---
## Round F (written BEFORE running): participation - how much more can the same spot-only strategy earn?   (user: "decent CAGR without the cost of drawdown or Sharpe")
Motivation (from the ledger, not from new results): the strategy's weakness is under-participation, not losses (average exposure ~0.30; ~35 % upside capture; 2024: +37 % vs BTC +121 %).
Part 1 - descriptive frontier (NOT a trial, no acceptance rule; it maps what a different risk budget buys).
  1a exposure dial: per-asset risk target ASSET_VOL in {0.30, 0.45 (official), 0.60, 0.80, 1.00, 1.50} x portfolio vol target in {0.40 (official), 0.60}; gross <= 1 always.
  1b constant BTC core c in {10, 20, 30 %} + (1-c) x official strategy, daily rebalanced.   1c idle-cash yield 4 % p.a. on the idle fraction (optional lever; headline numbers stay without it).
  A dial point is called "free" only if CAGR > official AND Sharpe >= official - 0.06 AND max DD no worse than -22 % (2018-03..2026-10).
Part 2 - two trials, the only mechanism-consistent extensions of the one overlay that worked (open-interest crowding; the official overlay only REDUCES exposure when crowding is high):
  F1 OI crowding both ways: m = 1 - 0.5 * clip(z, -2, 2) / 2 (z = 365d z-score of log BTC futures OI, lagged one day, same construction as the official overlay) - i.e. also scale UP when positioning is light.
  F2 funding crowding both ways, in addition to the official OI overlay: z = 365d z-score of the 30-day mean BTC perp funding rate (daily mean of the 8h rates, lagged one day), m = 1 - 0.5 * clip(z, -2, 2) / 2.
  F3 = F1 x F2, run only if BOTH are accepted individually.
  Gross <= 1 still enforced after the multiplier.  Eras: 2021-09-01..2022-12-31 (first date with 365 days of OI history) and 2023-01-01..2026-05-23; the forward window 2026-05-24..2026-10-05 is reported, not used.
  Acceptance (every condition): full-window CAGR higher; Sharpe >= baseline in BOTH eras; max DD no worse than baseline + 2 pts; and plateau: strengths 0.25 and 0.75 keep Sharpe >= baseline - 0.02 in both eras.
  Caveat stated in advance: OI/funding history is ~5 years, so there are only a handful of independent crowding episodes - low statistical power whatever the result.
Not tested (reasons): stablecoin supply impulse (IC test #6 already negative), valuation signals (MVRV etc., #6), anything needing options / quarterly-futures history or ETF flows (data not available), halving-cycle timing (n = 3).

### Round F result (recorded; `research/57_participation.py`, `58_dial_validation.py`, `59_dial_grid.py`; `results/participation*`, `dial_*`, `frontier_spot.png`)
Part 1 (descriptive): a limited, Sharpe-neutral risk dial exists in the spot-only version (per-asset risk budget ASSET_VOL; the portfolio vol brake only works downward, which is why I had concluded "no dial" - that statement was too strong):
ASSET_VOL 0.30 / 0.45 / 0.60 / 0.80 / 1.00 / 1.50 -> CAGR 22.1 / 32.5 / 38.9 / 40.6 / 40.9 / 42.2 %, Sharpe 1.34 / 1.36 / 1.40 / 1.38 / 1.36 / 1.35, max DD -16.3 / -20.2 / -22.8 / -24.5 / -26.5 / -28.6 % (2018-03..2026-10, brake 0.40). Saturation: average exposure <= 0.38.
By the pre-registered "free" rule (CAGR up, Sharpe >= official - 0.06, DD >= -22 %) only the trivial point (0.45, brake 0.60) qualifies (34.3 % / 1.37 / -20.9 %); ASSET_VOL 0.60 misses the DD bound by 0.8 pt (-22.8 %), so it is offered as a risk-preference choice on the frontier, not as an accepted free improvement.
A constant BTC core is dominated (10 %: 33.6 % / 1.29 / -24.2 %; 30 %: 34.8 % / 1.11 / -39.7 %). Idle yield 4 %: 36.4 % / 1.48 / -19.7 %.
Part 2: F1 not accepted (era-A Sharpe -0.2939 vs -0.2934 baseline, i.e. no effect; era B 1.056 vs 1.039; CAGR +0.3 pt); F2 not accepted (CAGR down, plateau fails; forward window 62.8 % -> 44.0 %). F3 not run.

---
## Round G (written BEFORE running): what to pair with trend when the market is flat or falling?   (spot-only, long-only, gross <= 1; user question)
Diagnostic D1 (not a trial): by BTC regime (OFF = close < SMA150; FLAT = OFF and |90d return| < 20 %; DOWN = OFF and 90d return <= -20 %; ON = otherwise) report share of days, BTC buy&hold return / Sharpe, and what the official strategy earned / how invested it was.
A sleeve may only use capital the main strategy is not using (sleeve weight <= 1 - main exposure).  Daily decisions at the 00:00-UTC close (CoinMetrics PriceUSD / CapMVRVCur stamped t = end of day t), position held over day t+1; costs 10 bp per side; fixed budget 25 % of capital per asset; BTC primary, ETH replication.
Regimes use only prices (per asset): OFF = close < SMA150 (centre of the main gate set); FLAT = OFF and |90d return| < 20 %; DOWN = OFF and 90d return <= -20 %.
Candidates (3 primary trials + 3 ETH replications; nothing else will be run on this):
 G1a dip-buy in FLAT : z = (P - SMA20) / STD20 (price levels, Bollinger-style); enter when z < -2 in FLAT; exit when z >= 0, after 10 days, or when FLAT ends.
 G1b dip-buy in DOWN : same rule inside DOWN.
 G2  value (MVRV)     : hold while CapMVRVCur < 1.0 (price below realised price; canonical threshold, not tuned), any regime.
Eras: pre-sample 2012-01-01..2017-12-31 (BTC; ETH from 2016), 2018-01-01..2022-12-31, 2023-01-01..2026-05-23 (CoinMetrics ends there).  Combination with the official spot strategy (ASSET_VOL 0.45, 2018-03+) uses the sleeve only with idle capital.
Acceptance (all): (1) pooled trade (G1) / episode (G2) mean net return > 0, G1 with t > 2, G2 with >= 75 % of episodes positive; (2) mean net trade / episode return positive in every era that has trades (G1: >= 5 trades; G2: >= 1 episode);
 (3) combined with the main strategy, Sharpe >= main's in both main eras (tolerance -0.02) and max DD no more than 2 pts worse.  Pre-sample cost sensitivity at 30 bp is reported.
Power caveat stated in advance: G2 has only ~4 BTC clusters (2011-12, 2015, 2018-19, 2022-23) and ETH ~6, so any pass is suggestive, not proof.
Already in the ledger and not repeated: short-horizon reversal and crash rebound (#7, #9, #29, #41: no tradable edge), gated shorts (#34: +47 % 2018 / +6 % 2022 but lower Sharpe overall), long/short and perp trend (#3, #49), cross-sectional factors (#48), funding carry (#42/#43, needs perps), gold/SPY (#55, #56).

### Round G result (recorded; `research/61_flat_down.py`, `results/flat_down.{log,json}`)
D1 (2018-03..2026-05-23, regime lagged one day): ON 52.5 % of days (BTC Sharpe 1.49; strategy exposure 0.51), FLAT 24.5 % (BTC -61 % cumulative; strategy -7 %, exposure 0.04), DOWN 22.7 % (BTC -14 %; strategy +5 %, exposure 0.01).
G1a BTC 29 trades mean -1.98 % net (t -1.17), G1b 34 trades +1.16 % (t 0.64), both fail; ETH replications negative; combined with the main strategy: Sharpe 1.31 -> 1.19 / 1.29, max DD -20.2 % -> -28.0 / -26.3 %.
G2 BTC: 14 episodes (>= 5 days), mean +3.16 % net while held, 10 of 14 positive (71 % < the 75 % rule), t 1.71; eras: pre-sample +2.4 % (8 episodes), 2018-22 +4.2 % (6), 2023-26 none; combined 33.5 % / 1.34 / -20.0 % vs main 31.3 % / 1.31 / -20.2 % (era Sharpes 1.50 / 1.10 vs 1.49 / 1.04) - passes every condition except the 75 % rule.
G2 ETH: 19 episodes, mean +0.64 % (t 0.24), long losing episodes in 2018-19 (-35 %, -15 %), combined DD -33.6 %.  **0 of 6 accepted.**  A first draft of D1 labelled regimes with the same-day close (leak); it was fixed before any conclusion was drawn.

---
## Round H (written BEFORE running): making money with crypto itself in flat / bearish regimes   (user: "no yield; I already handle idle cash; I want crypto to earn when the trend book is idle or the market falls")
Constraint note: a long-only spot book cannot profit from a falling market, so two tracks are evaluated and kept strictly apart.  Track A obeys "spot only, long only, no leverage".  Track B (H2) is NOT spot-only (USDT perp short, fully collateralised: short notional <= idle capital, never leverage) and is run only to answer the question; adopting it needs the user's explicit OK.
Delta-neutral carry and stablecoin yield are not pursued (the user treats them as idle-cash yield).
Regime: OFF = BTC close (lagged one day) < SMA150 (centre of the main gate set).  Eras (both tracks): E1 2020-09-01..2022-12-31 (H1) / 2020-01-01..2022-12-31 (H2: perp data start), E2 2023-01-01..2026-10-05.
Descriptive D2 (not a trial): what a survivor-corrected alt basket (point-in-time top-20 by trailing 30d quote volume excl. BTC/ETH, delisted coins included) does while BTC is OFF.
Track A - H1 alt rotation while BTC is OFF (3 trials): each Sunday close pick the top 5 of that universe by {MOM30, REV7 (= -7d return), LOWVOL (= -30d vol)}; equal weight, sleeve weight 25 % of capital (5 % per pick), cash otherwise; exit when BTC turns ON; costs 25 bp per side; prices = perp closes as proxy for spot (spot-tradability caveat).
  Acceptance (all): pooled t-stat of the sleeve's active-day mean net return > 2; mean positive in BOTH eras; combined with the official spot strategy (sleeve only with idle capital): Sharpe >= main - 0.02 in both eras, max DD no more than 2 pts worse.
Track B - H2 trend-down short sleeve on BTC and ETH perps (1 trial): mirror of the long book: trend scores computed on the INVERSE price (1/P) from spot 6h closes (momentum votes, SMA votes, EWMAC all flip sign), gate 1/P > SMA_g(1/P), g in {100,150,200}d bagged, inverse-vol sizing ASSET_VOL 0.45 / own hourly RV (spans 20/30/60d), sleeve vol brake 40 % (scalar <= 1), 5 % no-trade band, 6h bars, 10 bp per side, REAL funding (short receives positive funding) on perp hourly data from 2020-01; short notional scaled so that long + short gross <= 1.
  Acceptance (all): sleeve net P&L positive in BOTH eras; combined CAGR higher than the official strategy; combined Sharpe >= main - 0.02 in both eras; max DD no more than 2 pts worse.  Sizes 0.5x / 1x / 2x of the sleeve are shown (acceptance judged at 1x only).
Trials: 4.  Already in the ledger and not repeated: daily gated short sleeve (#34: +47 % 2018, +6 % 2022, lower Sharpe), long/short BTC-ETH trend (#3), perp trend (#49), cross-sectional factors (#1, #48), dip-buying and value (#61).

### Round H result (recorded; `research/62_flat_bear_crypto.py`, `results/flat_bear_crypto.{log,json}`)
D2: BTC lost 43 % cumulatively on the 43 % of days (since 2020-09) when it was OFF, so being long then is a loser; the equal-weight alt basket lost 36 % of a 25 % sleeve in E1 and 4 % in E2.
H1 (spot-only): MOM30 / REV7 / LOWVOL active-day mean (at 100 % weight, annualised) E1 -121 / -150 / -69 %, E2 +34 / -53 / -21 %; pooled t -0.64 / -1.50 / -0.88; combined with the official strategy: Sharpe 1.33 -> 0.88 / 0.68 / 0.95, DD -19.2 % -> -43.7 / -49.4 / -35.5 %.  Rejected.
H2 (perp short, not spot-only): sleeve -3.2 % in E1, -17.0 % in E2 (price -4.5 %, funding +4.7 %, costs -8.3 %); positive only in 2022 (+19.2 %) and 2025-26; -16.0 % while BTC was ON; combined x0.5 / x1 / x2: Sharpe 1.30 / 1.05 / 0.80 (main 1.39), DD -20.9 / -29.0 / -36.4 % (main -20.2 %).  Rejected.  0 of 4 accepted.

---
## Round I (written BEFORE running): mean reversion as the partner for flat / non-trending markets   (user: "I meant something like mean reversion")
Spot-only, long-only (buy displacement below fair value, sell at fair value; the sell-the-rip side would need shorts).  The sleeve may only use capital the main strategy leaves idle: sleeve weight 25 % x min(1, (1 - start-of-day main exposure) / 0.25).  Costs 10 bp per side unless stated.  BTC primary, ETH replication.
Why a second look: round G tested daily dip-buying only inside "below SMA150" regimes and produced 29 trades - too few to say much.  Here: more horizons, a regime defined by absence of trend (not by price level), volume conditioning and a grid.
Range regime = 30-day efficiency ratio (|P_t - P_{t-30}| / sum |daily changes|) < 0.25 at the last completed daily close (a pre-set value, not tuned).
I0 information scan (descriptive, NOT trials): BTC and ETH hourly spot 2020-01..2026-10; displacement z_n = ln(P_t/P_{t-n}) / (sigma_1h sqrt(n)) with sigma = EWMA(720h) std of hourly log returns, n in {1, 6, 24, 72, 168} h; forward return over h in {1, 6, 24, 72} h; events z <= -2 (dips) and z >= +2 (rips), de-clustered (events at least max(n, h) bars apart); excess = mean forward return over events minus unconditional mean, in bp, with t-stat; all regimes / own-asset OFF (close < SMA150, lagged) / range.  Cells with |t| >= 3 are listed (240 cells: ~0.6 expected by chance).  Round-trip cost to beat: 20 bp.
Strategies (5 primary trials; every parameter set now):
 M1 daily Bollinger dip in the RANGE regime: z = (P - SMA20)/STD20 < -2 at the close; exit when z >= 0 or after 10 days.
 M2 the same without the regime filter (control for whether the regime matters).
 M3 6h Bollinger dip in the RANGE regime: z over 20 bars (5 days) < -2; exit z >= 0 or after 20 bars.
 M4 volume-conditioned dip (liquidity-driven selloffs revert, information-driven continue): daily; 3-day return z <= -1.5 (sigma = EWMA(30d) daily std x sqrt 3) AND mean volume of the last 3 days < median of the previous 30 days; exit after 5 days or when the 3-day z >= 0.  (Control, not a trial: the same with volume ratio > 1.5.)
 M5 long-only grid in the RANGE regime, hourly closes: while flat, centre = 20-day SMA (re-centred hourly), 4 levels 2.5 % apart below the centre, 6.25 % of capital per level; buy a level when the hourly close <= level, sell it when the close >= level x 1.025; all levels stopped out when the close < last level x 0.95; the grid freezes while any level is held and only (re)starts while the range regime holds.
Evaluation: per-trade net return (round trip 2 x cost), pooled and by era (pre-sample 2012-17 for the daily versions on BTC, E1 2018-01..2022-12, E2 2023-01..2026-10), cost sweep 0 / 5 / 10 / 20 bp per side, and a random-entry control (same durations, random start dates, 1000 draws): does buying dips beat buying at random?  Combination with the official spot strategy (2018-03+).
Acceptance (all; judged on BTC): (1) pooled mean net per-trade return > 0 with t > 2 AND beats the random-entry control (p < 0.05); (2) mean net return positive in every era with >= 5 trades; (3) combined with the main strategy: Sharpe >= main - 0.02 in both eras and max DD no more than 2 pts worse.  M5 'trades' are completed level round trips.
Already tested and not repeated: hourly reversal tradability (#41: gross Sharpe ~0), daily cross-sectional reversal (#7), crash rebound (#29), shock breaker (#30), regime x shock (#12), dip-buy in FLAT / DOWN price regimes (round G).

### Round I result (recorded; `research/63_mean_reversion.py`, `64_ratio_scan.py`, `results/mean_reversion.{log,json}`, `ratio_scan.log`)
I0: reversion exists only at 1 hour (BTC dips +7 bp next hour, t 4.0; ETH +6 bp) - below the 20 bp round-trip cost; at 6-72 h dips show no excess; multi-day RIPS continue (BTC 72 h z >= 2 -> +196 bp over 72 h, t 3.0; 168 h z >= 2 -> +247 bp) and weekly DIPS continue down (-250 bp over 72 h).  Only 9 of 240 cells have |t| >= 3 (0.6 expected); one dip cell beats costs (BTC, OFF, 24 h displacement, 1 h hold: +23 bp on 90 events).
Strategies (BTC): M1 64 trades +2.87 % net, t 2.00, positive in pre / E1 / E2 (+5.0 / +0.8 / +3.1 %), but random-entry control +2.98 % (p 0.50); combined Sharpe 1.38 vs 1.36 with DD -23.9 % vs -20.2 % and E1 Sharpe 1.44 vs 1.49.  M2 +1.46 % (t 1.1, p 0.82).  M3 -0.29 % (172 trades, t -0.6).  M4 -2.47 % (19 trades; control high-volume +0.57 %).  M5 grid 2511 round trips -0.21 % (t -2.7), gross -0.01 %.  ETH replications weaker or negative.  0 of 5 accepted.
ETH/BTC ratio scan: no reversion (cheap ETH keeps falling, rich keeps rising 2016-2022; ~0 in 2023-26).
