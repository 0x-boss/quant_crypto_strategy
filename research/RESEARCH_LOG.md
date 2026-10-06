# Research log (trial ledger)

Everything below was run on the same real data (CoinMetrics daily `PriceUSD`, 2010-2026-05-23).  Metrics
are net of costs.  "Dev" = 2016/17-2021, "Hold" = 2022-2026 unless stated.  The ledger exists so the
number of trials behind the final design is on the record (see *Deflated Sharpe* in the README) and so
the dead ends are visible, not just the winner.  ~350 backtests were run in total (round 1: ~180; round 2 - the weak 2022-26 window - added ~170, mostly the 105-rule ceiling grid).

| # | Script | Idea | Outcome |
|---|--------|------|---------|
| 0 | `00_baselines.py` | BTC/ETH buy&hold, BTC SMA / TSMOM rules | BTC B&H Sharpe 1.41 dev / **0.48** hold; simple BTC trend 1.6-1.9 dev / 0.7-0.9 hold. Regime dependence is the central fact of this data. |
| 1 | `01_universe_factors.py` | Cross-sectional momentum / reversal / low-vol, dollar-neutral, top-20 liquid | Momentum Sharpe **negative in hold** (-0.6..-1.3); reversal destroyed by costs. **Rejected.** |
| 2 | `02_trend_portfolio.py` | Diversified long-only trend on top-10/20 liquid alts, vol-targeted | dev 1.7-2.0, **hold 0.0-0.3**, DD -50..-75 %. Alts bleed vs BTC after 2021. **Rejected.** |
| 3 | `03_longshort_btc_eth.py` | Long/short trend on BTC, ETH | Shorts *reduce* Sharpe in both periods (hold 0.5 vs 0.85 long-only). **Rejected.** |
| 4 | `04_diversifiers.py` | ETH/BTC relative trend; breadth gating | ETH/BTC rel. trend hold Sharpe -0.4; breadth gate no help. **Rejected.** |
| 5 | `05_s1_anatomy.py` | Where does the BTC+ETH trend book lose? | 2018 & 2022 whipsaw losses in bear-market rallies; exposure ~0.3. |
| 6 | `06_onchain_ic.py` | 8 on-chain/volume features (stablecoin growth, exchange flows, MVRV, ...) | IC 0.0-0.15, same sign but not significant out of sample; exchange-flow data is **retroactively relabelled** (hindsight risk). **Rejected.** |
| 7 | `07_reversal_gross.py` | Gross vs net of short-term reversal | Gross is *negative* (continuation); breakeven cost 3-14 bp: untradable. **Rejected.** |
| 8 | `08_ewmac.py` | Canonical Carver EWMAC (8/32..64/256) | Same as binary ensemble (1.5 full / 0.55 hold). Kept as one of 3 signal families. |
| 9 | `09_autocorr.py` | 1-5 day autocorrelation / fast TSMOM | AC ~ -0.05 at lag 1; fast TSMOM hold Sharpe < 0.2. **Rejected.** |
| 10 | `10_gold_sleeve.py` | Tokenised-gold (PAXG/XAUT) trend as uncorrelated sleeve | Helps only because gold boomed 2024-25, hurts 2020-21; full-sample Sharpe 1.33 -> 1.35. **Rejected as hindsight.** |
| 11 | `11_ml_walkforward.py` | LightGBM, 18 features, pooled liquid universe, expanding-window walk-forward | OOS rank-IC 0.01-0.03; dollar-neutral gross Sharpe 0.67 -> **-1.9 net**. **Rejected.** |
| 12 | `12_regime_conditional.py` | Forward returns by trend regime x 3-day shock | No cell with |t| > 2 and consistent sign. **Rejected.** |
| 13 | `13_risk_layers.py` | Vol estimator, portfolio vol target, drawdown overlay | Overlay lowers CAGR as much as DD (Calmar flat) - **dropped**. |
| 14 | `14_regime_gate.py` | Price > SMA(100/150/200) gate | Consistent: DD -24 -> -19 %, Sharpe 1.58 -> 1.60-1.68; plateau across lengths. **Adopted (200, bagged 100/150/200).** |
| 15 | `15_universe_breadth.py` | BTC / BTC+ETH / top5 / top10 / top20 liquid universes | BTC+ETH best (1.54); broader universes worse (DD -41..-49 %). **Adopted BTC+ETH.** |
| 16 | `16_alt_satellite.py` | Strictly filtered alt satellite (0/15/30 % budget) | No gain at any budget. **Rejected.** |
| 17 | `17_conviction.py` | Exponent on trend score f^p (p=0.5..3) | Flat (1.49-1.63). Kept linear. |
| 18 | `18_risk_frontier.py` | Vol target 0.30-0.45 x drawdown overlay | Sharpe ~1.5 everywhere; the target is a risk dial. |
| 19 | `19_drawdown_anatomy.py` | Worst drawdowns | Chop drawdowns (mid-2023, mid-2024), not crash exposure. |
| 20 | `20_leverage_caps.py` | Leverage cap 1.0-3.0, fast vol estimators | Sharpe 1.5-1.6 throughout; cap 1.0 (no leverage) still 46 % CAGR / 1.60. Adopted cap 1.5. |
| 21 | `21_bagged.py` | Parameter-bagging over gate x vol span | ~Same as single config (1.61 vs 1.59): chosen as the more defensible one. |
| 22 | `22_efficiency_ratio.py` | Kaufman efficiency-ratio trend-quality filter | Hold-block Sharpe 0.60-1.18 depending on window = noise. **Rejected.** |
| 23 | `23_band.py` | 0-10 % no-trade band | Turnover 17x -> 13x/yr at no cost. **Adopted 5 %.** |
| 24 | `24_leadlag.py` | Alt-basket -> BTC/ETH next-day lead-lag | Sign flips across regimes. **Rejected.** |
| 25 | `25_presample.py` | Same rule, BTC-only, **2011-2015 (never examined)** | Sharpe 1.8-2.0 at 10-60 bp costs, DD -35..-40 % vs -84 % B&H. Supports generalisation. |
| 26 | `26_risk_dial_table.py` | Vol target 15-50 % with the final strategy | Sharpe flat 1.6-1.73; return and DD scale together (table in README). |
| 27 | `27_whipsaw_control.py` | EWMA-smoothed score (span 3/7/14), Schmitt-trigger hysteresis (3 settings) | Smoothing lowers Sharpe (1.66-1.69 vs 1.73); hysteresis +/-0.02 = noise. **Rejected, baseline unchanged.** |
| 28 | `28_stretch_gates.py` | R2: stretch (distance above SMA200) trim; stricter gates (SMA rising, golden cross, +SMA50, 365d mom) | Stretch: no consistent effect. SMA50 second gate looked good (0.78->0.93) but see #33. Rising-SMA / golden-cross / 365d gates **hurt both eras**. |
| 29 | `29_crash_rebound.py` | R2: rebound after liquidation-cascade-style drops (z<-2/-3) | **No rebound** (BTC/ETH, 2022-26: forward 5d -1.2..-2.2 %). Counting bug in the pooled rows (pandas `stack`) inflated n/t; means valid. **Rejected.** |
| 30 | `30_shock_breaker.py` | R2: cut exposure for 3/5/10 days after a vol-adjusted down-day | Adds nothing (0.89-0.94 vs 0.93): trend system already reacts. **Rejected.** |
| 31 | `31_episodes.py` | R2: trade-episode decomposition | 2022-26: only 4 big winners; losses are many small chop episodes (win rate 11 %); big legs are captured (+103 % of BTC's +101 % in 2023-24). |
| 32 | `32_quality_chandelier.py` | R2: slope-t-stat "trend quality" family; chandelier exit; ETH gated by BTC | t-stat family and chandelier: no gain. ETH-needs-BTC *with* SMA50 gate: 1.03 in 2022-26 - **but see #33**. |
| 33 | `33_gate_plateau.py` | R2: plateau check of #28/#32 | **Not a plateau**: fast-gate SMA20/30/50/75/none -> 0.65/0.77/**1.03**/0.91/0.80; ETH-needs-BTC alone 0.80 vs 0.78. The 1.03 is a spike = noise-fit. **Rejected (kept base).** |
| 34 | `34_gated_short.py` | R2: bear-regime short sleeve gated like the long side, x0.5/x1.0 | +47 % in 2018 and +6 % in 2022 but gives it back in bull years: Sharpe 1.50 vs 1.66 (18-21) and 0.58 vs 0.78 (22-26). **Rejected.** |
| 35 | `35_gold_member.py` | R2: tokenised gold (PAXG/XAUT) as equal-risk universe member | 2022-26 Sharpe 0.78 -> 1.21, but 2020-21 2.55 -> 1.91, DD -31 -> -38 %, PAXG ADV only $3-8M until 2025. **Hindsight-flavoured; optional only.** |
| 36 | `36_weights_relstrength.py` | R2: BTC-heavier risk weights; ETH needs ETH/BTC > SMA | Equal risk best; relative-strength gate hurts (0.47-0.68 in 22-26). **Rejected.** |
| 37 | `37_adaptive_speed.py` | R2: walk-forward tilt across 13 sub-signal speeds | Equal weights 0.84 vs adaptive 0.70-0.82 in 22-26. **Rejected.** |
| 38 | `38_ceiling.py` | R2: **hindsight ceiling** - 105 single long-only trend rules on BTC+ETH | 2022-26 Sharpe: **max 1.22**, 90th pct 0.89, median 0.60; era-to-era rank correlation 0.26 (selection = noise). Gate200 lifts the family average 0.49 -> 0.79; no family beats another once gated. **1.5 is unreachable by this family even with hindsight.** |
| 39 | `39_round2_variants.py` | R2: graded variants (yield; gold; both) | 2022-26 Sharpe 0.78 / 0.87 (+4 % idle yield, data-grounded) / 1.21 (+gold) / 1.25 (both). |
| - | (inline) | R2: PyPI (906k names) and npm search for packages bundling funding/intraday data | Only API wrappers / <3.5 MB samples. Network hosts (Binance, Bybit, ...) still 403. |
| - | (inline) | R2: realised yield of yield-bearing stables in the CoinMetrics data | sDAI 4.2 % (2025), sUSDe 6.5 % (2025; 7.1 % annualised since 2024-06): supports a 4 % idle-cash assumption. |
| - | (inline) | R2: deflated Sharpe, ~350 trials | 1.00 (Sharpe dispersion 0.25), 0.80 (0.5). 2022+ window alone: PSR(SR>1)=0.32, PSR(SR>1.5)=0.06. |
| - | (inline) | Day-of-week / month effects | abs(t) < 2.5, unstable. **Rejected.** |

## What this says (round 1)

* The only edge that survived was **bull-regime drift capture with volatility control** (trend + gate +
  vol targeting) on the two deepest-liquidity assets.  No stable alpha was found in cross-sectional
  factors, short-horizon effects, on-chain data, ML, or shorting.
* Strategy Sharpe is ~1.6-1.7x BTC's Sharpe in *every* regime (1.0 -> 1.7 full-sample, 0.5 -> 0.8 since
  2022), i.e. the value-add is real and proportional, but absolute Sharpe depends on the market's drift.

## What round 2 adds (the weak 2022-26 window)

* The 2022-26 weakness is structural, not a bug: BTC buy&hold Sharpe is 0.48 there, only ~4 independent trend legs occurred, and
  a **hindsight search over 105 trend rules tops out at Sharpe 1.22**. Nothing inside "long-only daily trend on BTC+ETH" can
  honestly reach 1.5.
* Everything that looked like an improvement was either a spike (not a plateau) or hurt the other era. The single-config result
  that "worked" (SMA50 second gate + ETH-needs-BTC, 1.03) failed the neighbour check, so it was not adopted.
* The only increments with an economic rationale: (1) idle-cash yield (real, data-grounded, +0.09 Sharpe), (2) an uncorrelated
  asset (tokenised gold; +0.43 Sharpe in-sample, but driven by gold's 2025 rally and thin liquidity - treat as ~+0.05-0.10 forward).

## Round 3 - Binance intraday data (data.binance.vision; the live API is geo-blocked from the sandbox, HTTP 451)

| # | Script | Idea | Outcome |
|---|--------|------|---------|
| 40 | `40_intraday_scan.py` | Hour-of-day, past-k-hour -> next-hour IC, 3-sigma hourly moves (BTC/ETH perp 1h, 2020-26) | Hour-of-day: nothing stable. Next-hour reversal IC -0.05..-0.06 is stable in both eras (|t| 7-11) **but tiny**; after 3-sigma hourly moves price *continues* (+4..+13bp). |
| 41 | `41_hourly_reversion.py` | Tradable hourly reversal (8 variants x BTC/ETH) | **Gross Sharpe ~0 or negative, breakeven cost negative**, net -1..-17 at 2-8bp. **Rejected.** |
| - | `quant/carry.py` | Delta-neutral spot/perp funding carry, BTC and ETH, real funding + basis P&L, fees | 2020-21: 21-30 %/yr; **2022-26: ~5-6 %/yr**, vol ~0.5 % (funding decayed: BTC 30 % in 2021 -> 4 %, 8 %, 12 %, 5 %, 2 % in 2022-26). Behaves like a better cash yield. Daily Sharpe (8-12) is not a real-world Sharpe (exchange/ADL/liquidation risk invisible to daily bars). |
| 42 | `42_multicoin_carry.py` | Rotate carry into the top-k highest-funding of 12 coins | **Worse than always-on BTC/ETH in 2022-26** (+1..3 % vs ~6 %): funding mean-reverts, rotation costs. **Rejected.** |
| 43 | `43_trend_plus_carry.py` | TrendCore extended to 2026-10-05 with Binance closes + carry on idle capital | Genuine forward window 2026-05-24..10-05 (never used in design): TrendCore +21.6 % vs BTC +11.7 %. 2022+ Sharpe 0.88 -> 0.96 with carry. corr(trend, carry) = -0.05. |
| 44 | `44_flow_features.py` | Taker-buy imbalance, perp premium, funding level, perp/spot volume, hourly RV/skew/jump vs 1/3/7d returns | **No stable edge** (|IC| < 0.1, sign flips); ~100 tests so expect spurious hits. **Rejected.** |
| 45 | `45_intraday_trend.py` | Same TrendCore on 12h/8h/6h/4h bars; hourly realised-vol sizing | 2022+ Sharpe 0.89 (24h) -> 0.92/1.04/0.99/0.97; hourly RV +0.04-0.05 in every window. **Adopted: 6h bars + hourly RV** (gentle plateau, not a spike). |
| 46 | `46_oi_overlay.py` | Open-interest crowding overlay (futures OI z-score vs its own year) | Monotone dose-response, better in **both** eras at every strength (2023+: 1.31 -> 1.44 -> 1.50, DD -29.5 -> -22 -> -19 %). **Adopted at the pre-set midpoint 0.5**, not the best-looking value. |
| 47 | `47_sol_eth.py` | Same TrendCore-I (+carry) on ETH, SOL and combinations, common window 2021-03+ (SOL is a known survivor = hindsight) | **BTC alone is best**: Sharpe 1.10 (2021-03+), 1.25 (2022+). ETH 0.94 / 0.68, SOL 0.51 / 0.13 (DD -66 %; 2022: -46 % through the FTX collapse), ETH+SOL 1.02 / 0.57, BTC+ETH+SOL 1.15 / 0.93 (vs BTC+ETH 1.18 / 1.16). Same at 10 bp costs for SOL (0.58 / 0.20). Higher-vol assets are not better under vol targeting: more whipsaw and gap risk. **Keep BTC (+ETH); do not add SOL.** |
| 48 | `48_xs_perp.py` | **Pre-registered** market-neutral cross-sectional perp sleeve, 6 candidates (MOM30/MOM90/REV7/LOWVOL/FUND14/MULTI), survivor-corrected universe (900 archive symbols incl. delisted, top-20 by trailing volume, equities/commodities excluded), weekly, 10bp, real funding | **0 of 6 accepted** (rule: Sharpe > 0.5 in both eras, t > 2). Era Sharpes 20-22 / 23-26: MOM30 0.35/1.38, MOM90 -0.25/0.94, REV7 -1.26/-1.22, LOWVOL 1.38/-0.15, FUND14 -0.42/1.15, MULTI 1.21/0.35. Effects flip with regime. Correlation to trend book ~0 (would diversify if real). |
| 49 | `49_tsmom_perp.py` | **Pre-registered** round B: time-series trend (30/90/180d) across the same survivor-corrected perp universe, long/short and long-only | **Rejected**: LS 0.64/-0.03 (t 0.4); LO 0.79/0.61 but t 1.6 and corr to trend +0.70. Broad perp trend Sharpe 0.2-0.65 < BTC/ETH trend: independent, survivor-corrected confirmation that BTC+ETH is the right universe. |
| - | `results/risk_dial_intraday.csv` | Risk dial for TrendCore-I+carry | Sharpe falls slowly with leverage (2022+: 1.29 -> 0.97 from TV 0.30 -> 0.80) while CAGR rises 27 % -> 44 %, DD -21 % -> -45 %. Return is bought with drawdown, not alpha. |
| 50 | `50_spot_only.py`, `validate_spot.py` | **User constraint: spot only, no leverage, no derivatives.** Re-ran everything with gross <= 1.0, no carry, no financing | Spot-only TrendCore-I (+OI signal overlay): 36.1 % / Sharpe 1.45 / DD -19.2 % (2020-03+), 19.0 % / 1.00 (2022+); +4 % stablecoin earn: 39.8 % / 1.56. The vol target no longer matters (cap binds). Earlier 52.9 % / 1.64 figures are non-compliant. |
| 51 | `51_btc_signal_swap.py` | **Pre-registered (round C)**: BTC trend signal, hold ETH / SOL / ETH+SOL instead of BTC (own-vol or BTC-vol sizing), spot-only, gross<=1 | **0 of 7 accepted** (Sharpe >= BTC baseline in both eras). Era1 Sharpe jumps (1.56-2.32 vs 1.38) but era2 falls (0.98-1.26 vs 1.30). Best: ETH+SOL swap 45.0 % CAGR / 1.50 / DD -26.7 % vs BTC 33.1 % / 1.33 / -21.6 % - higher return, not better risk-adjusted, and SOL is a known survivor. |
| 52 | `52_alt_basket.py` | **Survivor-corrected C7**: BTC signal -> top-3 non-BTC symbols by trailing volume (weekly, point-in-time, incl. delisted LUNA/MATIC) | **Worse than BTC in both eras**: 17.1 % / 0.77 / -31.2 % vs BTC 28.8 % / 1.19 / -25.4 % (daily framework). Basket buy&hold -10 %/yr. Confirms the SOL result is hindsight. **Rejected.** |
| 53 | `53_btc_sol.py` | (post-hoc, user request; NOT pre-registered) BTC+SOL held together on the BTC signal, or each on its own signal | BTC-signal / own-vol: Sharpe 2.38 (2020-22) / 1.33 (2023-26) vs BTC 1.38 / 1.30; CAGR 39.5 % vs 33.1 %, DD -19.9 %. Mechanically passes the round-C rule, but **since 2022 it equals BTC only** (23.3 % / 1.15 vs 23.7 % / 1.12): the whole gain is SOL's 2021 run. Own-signals version: 29.1 % / 1.44. SOL = known survivor. |
| 54 | `54_btc_plus_basket.py` | Survivor-corrected twin of #53: 50 % BTC + 50 % point-in-time top-3 alt basket (incl. dead coins), BTC signal, daily | **Worse than BTC only**: 24.5 % / 1.06 vs 28.8 % / 1.19 (2023-26 Sharpe 0.93 vs 1.13; 2022+ 0.78 vs 0.95). With SOL named in hindsight: 37.1 % / 1.56. Hindsight confirmed. **Not adopted.** |
| 55 | `55_idle_gold.py` | **Pre-registered (round D)**: predict idle/underperforming months; put idle capital into spot gold (PAXG), always or SMA200-gated | Idle months predictable from start-of-month exposure (31/36 flagged); loss months are not (9 of 15 started >50 % invested, 0 idle). Gold on idle capital: era1 Sharpe 1.56 -> 1.24 / 1.12, era2 1.17 -> 1.53 / 1.51, max DD -19 % -> -29 / -33 %: a regime bet on gold's 2024-25 rally (PAXG volume only $1.6-3.9M/day until 2025). **Rejected.** |
| 56 | `56_spy_gold.py` | **Pre-registered (round E)**: SPY < EMA(200) risk-off regime (known after the US close, applied next 00:00 UTC) -> hold gold (GLD; spot-only). Part A mechanism test on GLD 2004+, BTC 2014+, ETH 2016+; Part B on the final BTC+ETH 6h strategy: S1 crypto x0 (cash), S2 idle capital -> GLD, S3 crypto x0 + 100 % GLD | **Mechanism NOT supported, 0 of 3 variants accepted.** A: gold in s=1 vs s=0 +1.9 pts/yr (90 % CI -18..+22; Sharpe 0.54 vs 0.68; sign flips: 2004-10 -35 pts, later blocks +6..+41); BTC -54 pts (CI -140..+30; 2023-26 reverses to +219); ETH +54 pts (crypto *better* in s=1; first ETH block is 2016 only). s=1 on 23 % of days, 35 runs since 2018 (27 shorter than 20 days, ~8 flips/yr). The baseline is already flat in SPY-bear spells: mean exposure 9 % vs 32 % otherwise, invested on only 23 % of s=1 days. B (Sharpe era1 / era2 / full; maxDD): S0 1.49 / 1.17 / 1.36, -20.2 %; S1 1.64 / 0.90 / 1.33, -19.9 % (cuts the Mar-2020 crash but also Mar-2023 and Oct-2023 rallies); S2 1.45 / 1.41 / 1.43, **-30.5 %**; S3 1.59 / 1.18 / 1.42, **-27.0 %** (gold fell with equities in Mar-2020 and 2022; the 2025 gold rally is the era-2 gain). S3 is the only variant clearing the Sharpe tests, +0.06, but costs +7 pts of drawdown and varies 1.35-1.49 with gold entry lag / cost. **Rejected**; no PAXG repeat (no winner). |
