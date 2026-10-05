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
